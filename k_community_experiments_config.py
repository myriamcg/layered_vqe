import math
import itertools
from dataclasses import dataclass, field
from tabnanny import verbose
from typing import Optional

import numpy as np
import networkx as nx
import pennylane as qml
from scipy.optimize import minimize
from l_vqe_functions import (
    n_bits,
    total_qubits,
    qubit_index,
    build_k_community_hamiltonian,
    simulate_one_lvqe,
    best_known_cost,
)


@dataclass
class ExperimentConfig:
    """
    Configuration from the paper's experimental section.

    Parameters
    ----------
    graph               : NetworkX graph to detect communities in.
    k                   : Number of communities (default 4, as in paper).
    max_layers          : Maximum L-VQE layers ℓ ∈ {0, 1, 2} (default 2).
    shots               : Measurement shots.
                          Paper uses 2000 for the finite-sample experiments.
    max_iter_per_layer  : COBYLA budget per layer expansion (paper: 200,
                          scales linearly with system size in practice).
    n_seeds             : Number of random initialisations (paper: 10).
    verbose             : Print per-layer progress.
    random_seed         : Master seed for reproducibility.
    """

    graph: nx.Graph
    k: int = 4
    max_layers: int = 2
    shots: Optional[int] = 2000
    max_iter_per_layer: int = 200
    n_seeds: int = 10
    verbose: bool = True
    random_seed: Optional[int] = None


@dataclass
class ExperimentResults:
    """
    Used to simulate multiple configurations of experiments to be reproduced from the paper.

    Attributes
    ----------
    rho_best       : Best approximation ratio across all seeds.
    rho_mean       : Mean approximation ratio.
    rho_std        : Standard deviation of approximation ratio.
    rho_per_seed   : Per-seed approximation ratio list.
    cost_per_seed  : Final raw cost per seed (expectation value of H).
    C_bkv          : Best known modularity (brute-force, or None if too large).
    all_histories  : Cost-evaluation histories for every seed.
    n_qubits       : Number of qubits used.
    config         : The ExperimentConfig that produced this result.
    """

    rho_best: float
    rho_mean: float
    rho_std: float
    rho_per_seed: list
    cost_per_seed: list
    C_bkv: Optional[float]
    all_histories: list
    n_qubits: int
    config: ExperimentConfig


def run_lvqe_experiment(
    graph: nx.Graph,
    k: int = 4,
    max_layers: int = 2,
    shots: Optional[int] = 2000,
    max_iter_per_layer: int = 200,
    n_seeds: int = 10,
    random_seed: Optional[int] = None,
    optimizer: str = "cobyla",
    device_name="lightning.qubit",
) -> ExperimentResults:
    """
    Run the full L-VQE k-community detection experiment for more seeds as described in the paper.

    ----------

    Returns
    -------
    ExperimentResults dataclass
    """
    cfg = ExperimentConfig(
        graph, k, max_layers, shots, max_iter_per_layer, n_seeds, random_seed
    )
    n_nodes = graph.number_of_nodes()
    n_q = total_qubits(n_nodes, k)
    N_bits = n_bits(k)

    print("=" * 60)
    print(f"L-VQE  |  nodes={n_nodes}  k={k}  N_bits={N_bits}")
    print(f"       |  n_qubits={n_q}  max_layers={max_layers}")
    print(f"       |  shots={shots}  seeds={n_seeds}")
    print("=" * 60)

    # Build Hamiltonian
    H = build_k_community_hamiltonian(graph, k)

    # Best known cost (used to verify the performance of the algorithm)
    C_bkv = best_known_cost(graph, k, max_brute_nodes=12)
    if C_bkv is not None:
        print(f"C_bkv (brute-force) = {C_bkv:.6f}")
    else:
        print("C_bkv: graph too large for brute-force (>12 nodes)")
    print()

    master_rng = np.random.default_rng(random_seed)

    cost_per_seed = []
    all_histories = []

    for seed_idx in range(n_seeds):

        print(f"─── Seed {seed_idx + 1}/{n_seeds} ───")
        child_rng = np.random.default_rng(master_rng.integers(0, 2**31))

        seed_result = simulate_one_lvqe(
            n_q=n_q,
            H=H,
            max_layers=max_layers,
            shots=shots,
            max_iter_per_layer=max_iter_per_layer,
            rng=child_rng,
            optimizer=optimizer,
            device_name=device_name,
        )
        cost_per_seed.append(seed_result["final_cost"])
        all_histories.append(seed_result["cost_history"])

    # The Hamiltonian is constructed with a minus sign so that *minimising* H
    # *maximises* Q.  Therefore:
    #     the modularity Q = -<H>   (the raw expectation value returned is negative of Q)
    # and the approximation ratio is:
    #     ρ = Q / C_bkv  =  -cost / C_bkv
    modularity_per_seed = [-c for c in cost_per_seed]

    if C_bkv is not None and abs(C_bkv) > 1e-12:
        rho_per_seed = [q / C_bkv for q in modularity_per_seed]
    else:
        best_found = max(modularity_per_seed)
        if abs(best_found) > 1e-12:
            rho_per_seed = [q / best_found for q in modularity_per_seed]
        else:
            rho_per_seed = [1.0] * n_seeds

    rho_arr = np.array(rho_per_seed)
    rho_best = float(rho_arr.max())
    rho_mean = float(rho_arr.mean())
    rho_std = float(rho_arr.std())

    print()
    print("=" * 60)
    print(f"  rho_best = {rho_best:.4f}")
    print(f"  rho_mean = {rho_mean:.4f} ± {rho_std:.4f}")
    print("=" * 60)

    return ExperimentResults(
        rho_best=rho_best,
        rho_mean=rho_mean,
        rho_std=rho_std,
        rho_per_seed=rho_per_seed,
        cost_per_seed=cost_per_seed,
        C_bkv=C_bkv,
        all_histories=all_histories,
        n_qubits=n_q,
        config=cfg,
    )
