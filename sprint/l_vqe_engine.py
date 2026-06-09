import math
import itertools
from typing import Optional, List, Tuple

import numpy as np
import networkx as nx
import pennylane as qml
from scipy.optimize import minimize

# ---------------------------------------------------------
# 1. HAMILTONIAN BUILDERS
# ---------------------------------------------------------


def n_bits(k: int) -> int:
    """N = ceil(log2(k)) — number of binary variables per node."""
    return math.ceil(math.log2(k))


def total_qubits(n_nodes: int, k: int) -> int:
    return n_nodes * n_bits(k)


def qubit_index(bit: int, node: int, n_nodes: int) -> int:
    return bit * n_nodes + node


def build_k_community_hamiltonian(graph: nx.Graph, k: int) -> qml.Hamiltonian:
    """Constructs the exact many-body Hamiltonian for k-Community Detection."""
    n_nodes = graph.number_of_nodes()
    m = graph.number_of_edges()
    N = n_bits(k)

    if m == 0:
        return qml.Hamiltonian([0.0], [qml.Identity(0)])

    A = nx.to_numpy_array(graph)
    degrees = np.array([d for _, d in graph.degree()])
    coefficients, observables = [], []

    for u in range(n_nodes):
        for v in range(n_nodes):
            B_uv = A[u, v] - (degrees[u] * degrees[v]) / (2.0 * m)
            if abs(B_uv) < 1e-14:
                continue
            coeff = -1 / (2.0 * m) * B_uv * (0.5**N)

            for subset in itertools.product([0, 1], repeat=N):
                z_wires = []
                for i, apply_z in enumerate(subset):
                    if apply_z:
                        z_wires.append(qubit_index(i, u, n_nodes))
                        z_wires.append(qubit_index(i, v, n_nodes))

                if len(z_wires) == 0:
                    obs = qml.Identity(0)
                else:
                    obs = qml.Z(z_wires[0])
                    for w in z_wires[1:]:
                        obs = obs @ qml.Z(w)

                coefficients.append(coeff)
                observables.append(obs)

    return qml.Hamiltonian(coefficients, observables)


def build_maxcut_hamiltonian(graph: nx.Graph) -> qml.Hamiltonian:
    """Constructs the exact 2-body Hamiltonian for Max-Cut."""
    coeffs, observables = [], []
    for u, v in graph.edges:
        coeffs.append(-0.5)
        observables.append(qml.Identity(u))

        coeffs.append(0.5)
        observables.append(qml.Z(u) @ qml.Z(v))

    return qml.Hamiltonian(coeffs, observables)


# ---------------------------------------------------------
# 2. CLASSICAL BENCHMARKS (BRUTE FORCE)
# ---------------------------------------------------------


def best_known_community_cost(
    graph: nx.Graph, k: int, max_brute_nodes: int = 12
) -> Optional[float]:
    n = graph.number_of_nodes()
    if n > max_brute_nodes:
        return None
    m = graph.number_of_edges()
    if m == 0:
        return 0.0

    A = nx.to_numpy_array(graph)
    degrees = np.array([d for _, d in graph.degree()])
    B = A - np.outer(degrees, degrees) / (2.0 * m)

    best_Q = -np.inf
    for assignment in itertools.product(range(k), repeat=n):
        Q = 0.0
        for u in range(n):
            for v in range(n):
                if assignment[u] == assignment[v]:
                    Q += B[u, v]
        Q /= 2.0 * m
        if Q > best_Q:
            best_Q = Q
    return best_Q


def best_known_maxcut_cost(
    graph: nx.Graph, max_brute_nodes: int = 20
) -> Optional[float]:
    n = graph.number_of_nodes()
    if n > max_brute_nodes:
        return None

    max_cut_val = 0
    for bits in itertools.product([0, 1], repeat=n):
        cut_val = sum(1 for u, v in graph.edges if bits[u] != bits[v])
        if cut_val > max_cut_val:
            max_cut_val = cut_val
    return -float(max_cut_val)  # Return negative because VQE minimizes


# ---------------------------------------------------------
# 3. QUANTUM CIRCUIT (L-VQE ANSATZ)
# ---------------------------------------------------------


def _apply_L0(params, n_q):
    for i in range(n_q):
        qml.RY(params[i], wires=i)


def _apply_entangling_block(params, w1, w2):
    qml.CNOT(wires=[w1, w2])
    qml.RY(params[0], wires=w1)
    qml.RY(params[1], wires=w2)
    qml.CNOT(wires=[w1, w2])
    qml.RY(params[2], wires=w1)
    qml.RY(params[3], wires=w2)


def _apply_L1(params, n_q):
    idx = 0
    for i in range(0, n_q - 1, 2):
        _apply_entangling_block(params[idx : idx + 4], i, i + 1)
        idx += 4
    for i in range(1, n_q - 1, 2):
        _apply_entangling_block(params[idx : idx + 4], i, i + 1)
        idx += 4


def apply_lvqe_circuit(params_flat, n_q, n_layers, no_entanglement=False):
    """The core L-VQE hardware-efficient ansatz."""
    _apply_L0(params_flat[:n_q], n_q)
    idx = n_q
    params_per_layer = 4 * (n_q - 1)
    layer_fn = _apply_L1_no_entanglement if no_entanglement else _apply_L1
    for _ in range(n_layers):
        layer_fn(params_flat[idx : idx + params_per_layer], n_q)
        idx += params_per_layer


def _flat_param_size(n_q: int, n_layers: int) -> int:
    return n_q + n_layers * 4 * (n_q - 1)


def _initial_flat_params(
    n_q: int, n_layers: int, rng: np.random.Generator
) -> np.ndarray:
    p = np.zeros(_flat_param_size(n_q, n_layers))
    p[:n_q] = rng.uniform(0, 2 * np.pi, size=n_q)
    return p


def _expand_params(flat_params: np.ndarray, n_q: int) -> np.ndarray:
    return np.concatenate([flat_params, np.zeros(4 * (n_q - 1))])


# ---------------------------------------------------------
# 4. CUSTOM OPTIMIZERS
# ---------------------------------------------------------


def sequential_minimal_optimization(
    objective_fn, initial_params: np.ndarray, max_evals: int
) -> np.ndarray:
    """
    Sequential Minimal Optimization (SMO) for parameterized quantum circuits.
    Updates parameters analytically one-by-one using the parameter-shift rule.
    Respects a strict function evaluation budget to ensure fair benchmarking.
    """
    params = np.copy(initial_params)
    J = len(params)
    eval_count = 0

    while eval_count < max_evals:
        for j in range(J):
            if eval_count >= max_evals:
                return params

            theta_j = params[j]

            # 1. Evaluate at theta
            L0 = objective_fn(params)
            eval_count += 1
            if eval_count >= max_evals:
                return params

            # 2. Evaluate at theta + pi/2
            params[j] = theta_j + np.pi / 2
            L_plus = objective_fn(params)
            eval_count += 1
            if eval_count >= max_evals:
                return params

            # 3. Evaluate at theta - pi/2
            params[j] = theta_j - np.pi / 2
            L_minus = objective_fn(params)
            eval_count += 1

            # 4. The Corrected Mathematical Minimum
            diff_pm = L_plus - L_minus
            diff_0 = 2 * L0 - L_plus - L_minus

            # Calculate the true phase phi using correctly ordered (a, b) coordinates
            phi = np.arctan2(diff_pm, diff_0)

            # The exact global minimum of this parameter's sine wave is always phi - pi
            theta_new = theta_j + phi - np.pi

            # Wrap to [0, 2pi] to keep the parameter space clean
            params[j] = theta_new % (2 * np.pi)

    return params


# ---------------------------------------------------------
# 5. THE EXECUTION ENGINE
# ---------------------------------------------------------

from qiskit.circuit import QuantumCircuit
import pennylane as qml
from pennylane_qiskit.converter import circuit_to_qiskit


def _build_qiskit_circuit(flat_params, n_q, n_layers, no_entanglement):
    dev_temp = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev_temp)
    def pl_circuit():
        apply_lvqe_circuit(flat_params, n_q, n_layers, no_entanglement)
        return qml.state()

    pl_circuit.construct([], {})
    tape = pl_circuit._tape

    qc = circuit_to_qiskit(tape, register_size=n_q)
    # Don't add measure_all() — EstimatorV2 handles measurements internally
    return qc


from qiskit.quantum_info import SparsePauliOp


from qiskit.quantum_info import SparsePauliOp


def _pl_hamiltonian_to_sparse_pauli(H: qml.Hamiltonian, n_q: int) -> SparsePauliOp:
    """Convert a PennyLane Hamiltonian to a Qiskit SparsePauliOp."""
    pauli_map = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}
    terms = []

    for coeff, op in zip(H.coeffs, H.ops):
        pauli_str = ["I"] * n_q

        # Tensor product of Paulis (e.g. X @ Z) — new PennyLane uses qml.ops.Prod
        if isinstance(op, qml.ops.Prod):
            for sub_op in op.operands:
                wire = sub_op.wires[0]
                pauli_str[wire] = pauli_map.get(type(sub_op).__name__, "I")
        # Single Pauli
        elif not isinstance(op, qml.Identity):
            wire = op.wires[0]
            pauli_str[wire] = pauli_map.get(type(op).__name__, "I")
        # Identity → leave all "I"

        # Qiskit uses reversed qubit ordering
        terms.append(("".join(reversed(pauli_str)), float(coeff)))

    return SparsePauliOp.from_list(terms)


from qiskit.circuit import ParameterVector


def _build_parametrized_qiskit_circuit(n_q, n_layers, no_entanglement):
    """Build a parametrized Qiskit circuit once, to be bound with values later."""
    n_params = _flat_param_size(n_q, n_layers)
    params = ParameterVector("θ", n_params)

    dev_temp = qml.device("default.qubit", wires=n_q)
    # Use dummy values to get the circuit structure
    dummy = np.zeros(n_params)

    @qml.qnode(dev_temp)
    def pl_circuit():
        apply_lvqe_circuit(dummy, n_q, n_layers, no_entanglement)
        return qml.state()

    pl_circuit.construct([], {})
    tape = pl_circuit._tape
    qc = circuit_to_qiskit(tape, register_size=n_q)
    return qc, dummy  # structure only, values are baked in as zeros


def simulate_one_lvqe(
    n_q: int,
    H: qml.Hamiltonian,
    max_layers: int,
    shots: Optional[int],
    max_iter_per_layer: int,
    rng: np.random.Generator,
    device_name: str = "default.qubit",
    backend=None,
    optimizer: str = "COBYLA",
    no_entanglement: bool = False,
) -> dict:
    """
    Executes one full L-VQE run.
    Supports dynamic device injection (lightning.qubit, default.mixed, etc.)
    and toggling between COBYLA and SMO.
    """
    use_ibm = (device_name == "qiskit.ibmq") and (backend is not None)

    if use_ibm:
        from qiskit_ibm_runtime import EstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        qiskit_op = _pl_hamiltonian_to_sparse_pauli(H, n_q)
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        estimator = EstimatorV2(mode=backend)

        # Cache: maps n_layers -> (transpiled_circuit, mapped_observable)
        _circuit_cache = {}

        def cost_fn(flat_params, n_layers):
            if n_layers not in _circuit_cache:
                qc = _build_qiskit_circuit(flat_params, n_q, n_layers, no_entanglement)
                qc_t = pm.run(qc)
                qiskit_op_mapped = qiskit_op.apply_layout(qc_t.layout)
                # Save the layout, not the transpiled circuit (params are baked in)
                _circuit_cache[n_layers] = (qc_t.layout, qiskit_op_mapped)

            layout, qiskit_op_mapped = _circuit_cache[n_layers]

            # Rebuild circuit with current params and apply the cached layout
            qc_current = _build_qiskit_circuit(
                flat_params, n_q, n_layers, no_entanglement
            )
            qc_current_t = pm.run(qc_current)  # still needs transpiling for gate basis

            pub = (qc_current_t, qiskit_op_mapped)
            result = estimator.run([pub]).result()
            val = float(result[0].data.evs)
            print(f"  IBM eval → {val:.6f}")
            return val

    else:
        dev = qml.device(device_name, wires=n_q, shots=shots)

        @qml.qnode(dev)
        def cost_fn_pl(flat_params, n_layers):
            apply_lvqe_circuit(flat_params, n_q, n_layers, no_entanglement)
            return qml.expval(H)

        cost_fn = cost_fn_pl

    cost_history = []
    flat_params = _initial_flat_params(n_q, 0, rng)
    print("started LVQE")

    # Layer Expansion Loop
    for layer in range(max_layers + 1):
        print(f"what {layer}")
        # print(f"  Layer {layer}  ({len(flat_params)} params) ...", end=" ")

        def objective(p, _layer=layer):
            val = float(cost_fn(p, _layer))
            cost_history.append(val)
            return val

        max_it = max_iter_per_layer if layer < max_layers else max_iter_per_layer * 3

        if optimizer.upper() == "SMO":
            print("if smo")
            flat_params = sequential_minimal_optimization(
                objective, flat_params, max_evals=max_it
            )
            print("after smo")
            final_cost = objective(flat_params)
        else:
            result = minimize(
                objective,
                flat_params,
                method="COBYLA",
                options={"maxiter": max_it, "disp": False},
            )
            flat_params = result.x
            final_cost = result.fun

        print(f"cost = {final_cost:.6f}")

        if layer < max_layers:
            flat_params = _expand_params(flat_params, n_q)

    final_cost = float(cost_fn(flat_params, max_layers))

    return {
        "cost_history": cost_history,
        "final_cost": final_cost,
        "final_params": flat_params,
    }


# Add this import to the top of your engine file if it isn't there
from pennylane import qaoa

# ---------------------------------------------------------
# 6. QAOA EXECUTION ENGINE
# ---------------------------------------------------------


def build_mixing_hamiltonian(n_q: int) -> qml.Hamiltonian:
    """Builds the standard X-mixing Hamiltonian for QAOA."""
    coeffs = [1.0] * n_q
    observables = [qml.X(i) for i in range(n_q)]
    return qml.Hamiltonian(coeffs, observables)


def simulate_one_qaoa(
    n_q: int,
    H_cost: qml.Hamiltonian,
    p_steps: int,
    shots: Optional[int],
    max_evals: int,
    rng: np.random.Generator,
    device_name: str = "default.qubit",
) -> dict:
    """
    Executes one run of QAOA for a specific depth p.
    """
    dev = qml.device(device_name, wires=n_q, shots=shots)
    H_mixer = build_mixing_hamiltonian(n_q)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, H_cost)
        qaoa.mixer_layer(alpha, H_mixer)

    @qml.qnode(dev)
    def cost_fn(params):
        # Initial state: Hadamard on all qubits
        for i in range(n_q):
            qml.Hadamard(wires=i)

        # params shape: (2, p) -> params[0] = gammas, params[1] = alphas
        qml.layer(qaoa_layer, p_steps, params[0], params[1])
        return qml.expval(H_cost)

    # Initialize gammas and alphas randomly between [0, 2pi]
    initial_params = rng.uniform(0, 2 * np.pi, size=(2, p_steps))

    # Flatten params for SciPy
    flat_initial = initial_params.flatten()

    cost_history = []

    def objective(p_flat):
        p_reshaped = p_flat.reshape((2, p_steps))
        val = float(cost_fn(p_reshaped))
        cost_history.append(val)
        return val

    result = minimize(
        objective,
        flat_initial,
        method="COBYLA",
        options={"maxiter": max_evals, "disp": False},
    )

    return {
        "cost_history": cost_history,
        "final_cost": result.fun,
        "final_params": result.x.reshape((2, p_steps)),
    }


# ---------------------------------------------------------
# FIXED BUDGET LVQE (For VQE v/s LVQE analysis)
# ---------------------------------------------------------
def simulate_one_lvqe_fixed_budget(
    n_q: int,
    H: qml.Hamiltonian,
    max_layers: int,
    shots: int | None,
    total_budget: int,
    warm_start_iters: int,
    rng: np.random.Generator,
    device_name: str = "default.qubit",
    optimizer: str = "SMO",
) -> dict:
    """
    Executes one full L-VQE run with a strict global evaluation budget.
    Early layers use 'warm_start_iters' (e.g., 200) to find the basin.
    The final layer consumes the entire remaining budget for convergence.
    """
    dev = qml.device(device_name, wires=n_q, shots=shots)

    @qml.qnode(dev)
    def cost_fn(flat_params, n_layers):
        apply_lvqe_circuit(flat_params, n_q, n_layers)
        return qml.expval(H)

    cost_history = []
    flat_params = _initial_flat_params(n_q, 0, rng)

    for layer in range(max_layers + 1):
        print(f"  Layer {layer}  ({len(flat_params)} params) ...", end=" ")

        def objective(p, _layer=layer):
            val = float(cost_fn(p, _layer))
            cost_history.append(val)
            return val

        # --- THE EXACT PAPER ALGORITHM ---
        if layer < max_layers:
            # Step 2 & 5: Stop early (before convergence)
            max_it = warm_start_iters
        else:
            # Step 7: Dump the remaining budget into the final convergence
            max_it = total_budget - len(cost_history)

            # Failsafe in case early layers somehow exceeded the budget
            if max_it <= 0:
                print("Budget exhausted early.")
                break

        # Execute Optimizer
        if optimizer.upper() == "SMO":
            flat_params = sequential_minimal_optimization(
                objective, flat_params, max_evals=max_it
            )
            final_cost = cost_history[-1]
        else:
            result = minimize(
                objective,
                flat_params,
                method="COBYLA",
                options={"maxiter": max_it, "disp": False},
            )
            flat_params = result.x
            final_cost = result.fun

        print(f"cost = {final_cost:.6f} | Total Evals So Far: {len(cost_history)}")

        if layer < max_layers:
            flat_params = _expand_params(flat_params, n_q)

    # Ensure we return the absolute final calculated state
    final_cost = float(cost_fn(flat_params, max_layers))

    return {
        "cost_history": cost_history,
        "final_cost": final_cost,
        "final_params": flat_params,
    }


# ---------------------------------------------------------
# NO ENTANGLEMENT
# ---------------------------------------------------------


def _apply_entangling_block_no_entanglement(params, w1, w2):
    """
    replaces CNOT with T gates.
    T gates are single-qubit
    """
    qml.T(wires=w1)
    qml.T(wires=w2)
    qml.RY(params[0], wires=w1)
    qml.RY(params[1], wires=w2)
    qml.T(wires=w1)
    qml.T(wires=w2)
    qml.RY(params[2], wires=w1)
    qml.RY(params[3], wires=w2)


def _apply_L1_no_entanglement(params, n_q):
    idx = 0
    for i in range(0, n_q - 1, 2):
        _apply_entangling_block_no_entanglement(params[idx : idx + 4], i, i + 1)
        idx += 4
    for i in range(1, n_q - 1, 2):
        _apply_entangling_block_no_entanglement(params[idx : idx + 4], i, i + 1)
        idx += 4
