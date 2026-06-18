import csv
import matplotlib.pyplot as plt

HISTORY_CSV = "csv_files/vqe_noisy_k_comm.csv"
SUMMARY_CSV = "csv_files/vqe_noisy_results_k_comm.csv"
TARGET_FINAL_COST = -0.076563  # cost for the k community problem with the caveman graph

# ---- 1. Load the full cost history (already single-run, since file is overwritten each time) ----
steps, costs = [], []
with open(HISTORY_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        costs.append(-float(row["cost"]))

print(f"Loaded {len(costs)} cost-history points from {HISTORY_CSV}")

# ---- 2. Load summary CSV and pick the row matching the run of interest ----
with open(SUMMARY_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

matches = [r for r in rows if abs(float(r["final_cost"]) - TARGET_FINAL_COST) < 1e-4]
if matches:
    run = matches[-1]
else:
    print(
        "No row matched TARGET_FINAL_COST closely enough; falling back to the last row."
    )
    run = rows[-1]

true_baseline = float(run["true_baseline"])
final_cost = float(run["final_cost"])
noisy_mod = float(run["noisy_modularity"])
approx_ratio = float(run["approx_ratio_rho"])
n_qubits = run["n_qubits"]
shots = run["shots"]
max_layers = run["max_layers"]

print("Selected summary row:", run)

n_total = len(costs)

# ---- 3. Plot ----
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    steps,
    costs,
    lw=1.1,
    color="#2b6cb0",
    alpha=0.85,
    label="cost history (per evaluation)",
)

# Light rolling average to make the trend easier to read through shot noise
window = 15
if len(costs) >= window:
    roll = [
        sum(costs[max(0, i - window + 1) : i + 1])
        / len(costs[max(0, i - window + 1) : i + 1])
        for i in range(len(costs))
    ]
    ax.plot(steps, roll, lw=2.2, color="#1a365d")

# Reference lines from the summary CSV
ax.axhline(
    true_baseline,
    color="gray",
    ls="--",
    lw=1,
    label=f"true_baseline = {true_baseline:.4f}",
)


ax.set_xlabel(
    f"Optimization step (single fixed-depth COBYLA run, n={n_total} evaluations)"
)
ax.set_ylabel("Cost  ⟨H⟩")
ax.set_title(
    f"Noisy VQE convergence — connected_caveman_graph(l=4,k=2)\n"
    f"n_qubits={n_qubits}, shots={shots}, n_layers={max_layers} (fixed depth), "
    f"max_evals={n_total}, optimizer=COBYLA"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)

textstr = (
    f"true_baseline = {true_baseline:.4f}\n"
    f"noisy_modularity = {noisy_mod:.4f}\n"
    f"approx_ratio = {approx_ratio:.4f}"
)
ax.text(
    0.02,
    0.03,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
)

fig.tight_layout()
fig.savefig("vqe_noisy_convergence_k_comm.png", dpi=180)
print("Saved plot")
