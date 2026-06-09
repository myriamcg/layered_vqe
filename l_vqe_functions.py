import math
import itertools
from dataclasses import dataclass, field
from tabnanny import verbose
from typing import Optional, Callable

import numpy as np
import networkx as nx
import pennylane as qml
from scipy.optimize import minimize


def n_bits(k: int) -> int:
    """N = ceil(log2(k)) — number of binary variables per node."""
    return math.ceil(math.log2(k))


def total_qubits(n_nodes: int, k: int) -> int:
    # because each node needs log(k) qubits to encode the community they re part of
    return n_nodes * n_bits(k)


def qubit_index(bit: int, node: int, n_nodes: int) -> int:
    return bit * n_nodes + node


def build_k_community_hamiltonian(graph: nx.Graph, k: int) -> qml.Hamiltonian:
    n_nodes = graph.number_of_nodes()
    m = graph.number_of_edges()
    N = n_bits(k)
    n_qubits = total_qubits(n_nodes, k)

    if m == 0:
        # No edges — the Hamiltonian is zero.
        return qml.Hamiltonian([0.0], [qml.Identity(0)])
    A = nx.to_numpy_array(graph)
    degrees = np.array([d for _, d in graph.degree()])
    coefficients = []
    observables = []

    for u in range(n_nodes):
        for v in range(n_nodes):

            B_uv = A[u, v] - (degrees[u] * degrees[v]) / (2.0 * m)
            if abs(B_uv) < 1e-14:
                continue
            coeff = -1 / (2.0 * m) * B_uv * (0.5**N)

            # we generate the 2^N possible Z-term combinations from the Hamiltonian in Eq 12.
            for subset in itertools.product([0, 1], repeat=N):

                z_wires = []

                for i, apply_z in enumerate(subset):
                    if apply_z:
                        z_wires.append(qubit_index(i, u, n_nodes))
                        z_wires.append(qubit_index(i, v, n_nodes))

                if len(z_wires) == 0:
                    # identity
                    obs = qml.Identity(0)  # randomly choose qubit 0 to put identity on
                else:
                    obs = qml.Z(z_wires[0])
                    for w in z_wires[1:]:
                        obs = obs @ qml.Z(w)

                coefficients.append(coeff)
                observables.append(obs)

    return qml.Hamiltonian(coefficients, observables)


def best_known_cost(
    graph: nx.Graph, k: int, max_brute_nodes: int = 12
) -> Optional[float]:
    """
    Compute C_bkv (best known modularity) by brute-force for small graphs,
    returns None for large graphs where brute force is infeasible.
    """
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
        # Eq. 6: Q = (1/2m) * Σ_{u,v} B_{u,v} δ(c_u, c_v)
        # sum over ALL pairs including u==v (B_{u,u} = -d_u²/2m ≤ 0)
        Q = 0.0
        for u in range(n):
            for v in range(n):
                if assignment[u] == assignment[v]:
                    Q += B[u, v]
        Q /= 2.0 * m  # single division — Eq. 6 has exactly one 1/2m
        if Q > best_Q:
            best_Q = Q

    return best_Q


# ─────────────────────────────────────────────────────────────
# 3.  L-VQE CIRCUIT  (same as the one from the notebook)
# ─────────────────────────────────────────────────────────────


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
    for i in range(0, n_q - 1, 2):  # odd pairs
        _apply_entangling_block(params[idx : idx + 4], i, i + 1)
        idx += 4
    for i in range(1, n_q - 1, 2):  # even pairs
        _apply_entangling_block(params[idx : idx + 4], i, i + 1)
        idx += 4


def _apply_lvqe_circuit(params_flat, n_q, n_layers):
    """Apply the full L-VQE ansatz, slicing flat_params on the fly."""
    _apply_L0(params_flat[:n_q], n_q)
    idx = n_q
    params_per_layer = 4 * (n_q - 1)
    for _ in range(n_layers):
        _apply_L1(params_flat[idx : idx + params_per_layer], n_q)
        idx += params_per_layer


def _flat_param_size(n_q: int, n_layers: int) -> int:
    return n_q + n_layers * 4 * (n_q - 1)


def _initial_flat_params(
    n_q: int, n_layers: int, rng: np.random.Generator
) -> np.ndarray:
    """Layer 0: random in [0, 2π].  Layer 1+: zeros (identity init)."""
    p = np.zeros(_flat_param_size(n_q, n_layers))
    p[:n_q] = rng.uniform(0, 2 * np.pi, size=n_q)
    return p


def _expand_params(flat_params: np.ndarray, n_q: int) -> np.ndarray:
    """Append one zero-initialised layer to flat_params."""
    return np.concatenate([flat_params, np.zeros(4 * (n_q - 1))])


# ─────────────────────────────────────────────────────────────
# 4.  ONE-LVQE RUN
# ─────────────────────────────────────────────────────────────


def simulate_one_lvqe(
    n_q: int,
    H: qml.Hamiltonian,
    max_layers: int,
    shots: Optional[int],
    max_iter_per_layer: int,
    rng: np.random.Generator,
    optimizer: str = "cobyla",
    re_estimate_interval: int = 32,
    device_name="lightning.qubit",
) -> dict:
    """
    Execute one full L-VQE run (one random seed).

    Returns
    -------
    dict with keys:
        'cost_history'  : list of floats — every objective evaluation
        'final_cost'    : float — expectation value at end
        'final_params'  : np.ndarray
    """
    dev = qml.device(device_name, wires=n_q, shots=shots)

    @qml.qnode(dev)
    def cost_fn(flat_params, n_layers):
        _apply_lvqe_circuit(flat_params, n_q, n_layers)
        return qml.expval(H)

    cost_history = []
    flat_params = _initial_flat_params(n_q, 0, rng)

    for layer in range(max_layers + 1):
        print(f"  Layer {layer}  ({len(flat_params)} params) ...", end=" ")

        def objective(p, _layer=layer):
            val = float(cost_fn(p, _layer))
            cost_history.append(val)
            return val

        # For the last layer we let COBYLA run until convergence
        # (the paper says "update all parameters until convergence" at the end).
        max_it = max_iter_per_layer if layer < max_layers else max_iter_per_layer * 3

        if optimizer == "smo":
            flat_params = _smo_optimize(
                objective,
                flat_params,
                max_iter_per_layer=max_it,
                re_estimate_interval=re_estimate_interval,
            )
            final_layer_cost = objective(flat_params)
            print(f"cost = {final_layer_cost:.6f}")
        else:
            result = minimize(
                objective,
                flat_params,
                method="COBYLA",
                options={"maxiter": max_it, "disp": False},
            )
            flat_params = result.x

            print(f"cost = {result.fun:.6f}")

        if layer < max_layers:
            flat_params = _expand_params(flat_params, n_q)

    final_cost = float(cost_fn(flat_params, max_layers))

    return {
        "cost_history": cost_history,
        "final_cost": final_cost,
        "final_params": flat_params,
    }


def _smo_optimize(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_iter_per_layer: int,
    re_estimate_interval: int,
) -> np.ndarray:

    params = params.copy()
    J = len(params)
    step = 0

    for _ in range(max_iter_per_layer):
        for j in range(J):
            theta_j = params[j]

            L0 = cost_fn(params)

            params[j] = theta_j + np.pi / 2
            L_plus = cost_fn(params)

            params[j] = theta_j - np.pi / 2
            L_minus = cost_fn(params)

            diff_pm = L_plus - L_minus
            diff_0 = 2 * L0 - L_plus - L_minus

            # a1 = 0.5 * np.sqrt(diff_pm ** 2 + diff_0 ** 2)
            a2 = theta_j - np.arctan2(diff_0, diff_pm)

            theta_new = (a2 + np.pi) % (2 * np.pi)
            params[j] = theta_new

            step += 1

            if step % re_estimate_interval == 0:
                _ = cost_fn(params)

    return params
