import networkx as nx
import matplotlib.pyplot as plt
from k_community_experiments_config import run_lvqe_experiment

if __name__ == "__main__":
    node_sizes = [7, 8, 9, 10]
    rho_best_values = []

    for n in node_sizes:
        print(f"\n{'='*50}")
        print(f"Running L-VQE on {n}-node gnp graph (k=4, L=2)")
        print(f"{'='*50}")

        G = nx.gnp_random_graph(n, 0.5, seed=42)

        res = run_lvqe_experiment(
            graph=G,
            k=4,
            max_layers=2,
            shots=2000,
            max_iter_per_layer=200,
            n_seeds=1,  # paper uses 10, maybe we should increase it to reproduce the results
            random_seed=0,
        )

        rho_best_values.append(res.rho_best)
        print(
            f"  nodes={n}  rho_best={res.rho_best:.4f}  "
            f"rho_mean={res.rho_mean:.4f}±{res.rho_std:.4f}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        node_sizes,
        rho_best_values,
        marker="^",
        color="gold",
        linewidth=2,
        markersize=9,
        label="L-VQE (COBYLA)",
    )

    ax.set_xlabel("Number of nodes", fontsize=13)
    ax.set_ylabel("Approximation Ratio", fontsize=13)
    ax.set_title("L-VQE vs Number of Nodes (k=4, L≤2)", fontsize=13)
    ax.set_xticks(node_sizes)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lvqe_figure3.png", dpi=150)
    plt.show()
    print("\nPlot saved to lvqe_figure3.png")
