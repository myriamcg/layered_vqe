import csv
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from l_vqe_functions import (
    build_k_community_hamiltonian,
    total_qubits,
    best_known_cost,
)
from sprint.l_vqe_engine import simulate_one_lvqe

CSV_PATH = "lvqe_results.csv"

K = 4
MAX_LAYERS = 2
SHOTS = 2000
MAX_ITER = 200
OPTIMIZER = "SMO"

if __name__ == "__main__":
    node_sizes = [7, 8, 9]
    rho_best_values = []

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "n_nodes",
                    "k",
                    "max_layers",
                    "shots",
                    "max_iter_per_layer",
                    "n_qubits",
                    "C_bkv",
                    "final_cost",
                    "modularity",
                    "rho",
                    "optimizer",
                ]
            )

    rng = np.random.default_rng(42)

    for n in node_sizes:
        print(f"\n{'='*50}")
        print(
            f"Running L-VQE on {n}-node gnp graph (k={K}, L={MAX_LAYERS}, optimizer={OPTIMIZER})"
        )
        print(f"{'='*50}")

        G = nx.gnp_random_graph(n, 0.5, seed=42)
        H = build_k_community_hamiltonian(G, K)
        n_q = total_qubits(n, K)
        C_bkv = best_known_cost(G, K, max_brute_nodes=12)

        res = simulate_one_lvqe(
            n_q=n_q,
            H=H,
            max_layers=MAX_LAYERS,
            shots=SHOTS,
            max_iter_per_layer=MAX_ITER,
            rng=rng,
            optimizer=OPTIMIZER,
            device_name="lightning.qubit",
        )

        final_cost = res["final_cost"]
        modularity = -final_cost  # H is built with a minus sign

        if C_bkv is not None and abs(C_bkv) > 1e-12:
            rho = modularity / C_bkv
        elif abs(modularity) > 1e-12:
            rho = 1.0  # no reference available, treat as best found
        else:
            rho = 1.0

        rho_best_values.append(rho)
        print(
            f"  nodes={n}  final_cost={final_cost:.4f}  modularity={modularity:.4f}  rho={rho:.4f}"
        )

        # ── Append immediately ────────────────────────────────────────
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    n,
                    K,
                    MAX_LAYERS,
                    SHOTS,
                    MAX_ITER,
                    n_q,
                    C_bkv,
                    final_cost,
                    modularity,
                    rho,
                    OPTIMIZER,
                ]
            )
        print(f"  ✓ Row saved to {CSV_PATH}")
        # ─────────────────────────────────────────────────────────────

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        node_sizes,
        rho_best_values,
        marker="^",
        color="gold",
        linewidth=2,
        markersize=9,
        label=f"L-VQE ({OPTIMIZER})",
    )
    ax.set_xlabel("Number of nodes", fontsize=13)
    ax.set_ylabel("Approximation Ratio", fontsize=13)
    ax.set_title(
        f"L-VQE vs Number of Nodes (k={K}, L≤{MAX_LAYERS}, {OPTIMIZER})", fontsize=13
    )
    ax.set_xticks(node_sizes)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"lvqe_figure3_{OPTIMIZER}.png", dpi=150)
    plt.show()
    print(f"Plot saved to lvqe_figure3_{OPTIMIZER}.png")
