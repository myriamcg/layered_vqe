import csv
import matplotlib.pyplot as plt

HISTORY_CSV = (
    "csv_files/lvqe_noisy_k_comm.csv"  # your local file is named lvqe_noisy_k_comm.csv
)
SUMMARY_CSV = "csv_files/lvqe_noisy_results.csv"
TARGET_MAX_ITER_LAYER = 200  # the run you actually care about

LAYER_ITER_CAPS = [200, 200]  # layer 0, layer 1 -- layer 2 gets "whatever's left"

steps, costs = [], []
with open(HISTORY_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        costs.append(-float(row["cost"]))

print(f"Loaded {len(costs)} cost-history points from {HISTORY_CSV}")

with open(SUMMARY_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

matches = [r for r in rows if int(r["max_iter_layer"]) == TARGET_MAX_ITER_LAYER]
if not matches:
    raise ValueError(
        f"No summary row found with max_iter_layer == {TARGET_MAX_ITER_LAYER}"
    )
if len(matches) > 1:
    print(
        f"Warning: {len(matches)} rows match max_iter_layer={TARGET_MAX_ITER_LAYER}; using the last one."
    )
run = matches[-1]

true_baseline = float(run["true_baseline"])
final_cost = float(run["final_cost"])
noisy_mod = float(run["noisy_modularity"])
approx_ratio = float(run["approx_ratio_rho"])
n_qubits = run["n_qubits"]
shots = run["shots"]

print("Selected summary row:", run)

n_total = len(costs)
boundaries = [0]
for cap in LAYER_ITER_CAPS:
    boundaries.append(boundaries[-1] + cap)
boundaries = [min(b, n_total) for b in boundaries]
n_layers_total = len(LAYER_ITER_CAPS) + 1
print(
    f"Intended layer start steps: {boundaries} (layer {len(LAYER_ITER_CAPS)} runs to step {n_total})"
)

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
    -true_baseline,
    color="gray",
    ls="--",
    lw=1,
    label=f"true_baseline = {true_baseline:.4f}",
)

for b in boundaries[1:]:
    if b < n_total:
        ax.axvline(b, color="darkorange", ls="--", lw=1.3, alpha=0.8)

ymin, ymax = ax.get_ylim()
label_y = ymin + 0.93 * (ymax - ymin)
layer_starts = boundaries
layer_ends = boundaries[1:] + [n_total]
for i, (s, e) in enumerate(zip(layer_starts, layer_ends)):
    mid = (s + e) / 2
    ax.text(
        mid,
        label_y,
        f"layer {i}",
        ha="center",
        va="top",
        fontsize=9,
        color="#7a4a00",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7
        ),
    )

ax.set_xlabel("Optimization step (cumulative across all layers)")
ax.set_ylabel("Cost  ⟨H⟩")
ax.set_title(
    f"Noisy L-VQE convergence — connected_caveman_graph(l=4,k=2)\n"
    f"n_qubits={n_qubits}, shots={shots}, max_iter_layer={TARGET_MAX_ITER_LAYER}, optimizer=COBYLA"
)
ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.78))
ax.grid(alpha=0.25)

textstr = (
    f"true_baseline = {true_baseline:.4f}\n"
    # f"final_cost = {final_cost:.4f}\n"
    f"noisy_modularity = {noisy_mod:.4f}\n"
    f"approx_ratio_rho = {approx_ratio:.4f}"
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
fig.savefig("lvqe_noisy_k_comm.png", dpi=180)
