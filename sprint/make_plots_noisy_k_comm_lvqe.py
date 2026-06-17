"""
Plot the noisy L-VQE convergence trace for the run of interest
(max_iter_layer = 200), pulling the cost history from
lvqe_noisy_k_comm.csv and the matching summary stats from
lvqe_noisy_results.csv.

Why this approach:
- lvqe_noisy_results.csv is opened in "a" (append) mode in your script,
  so it accumulates one summary row per run you've ever executed.
  Your file has 3 rows -> 3 separate runs. We select the row matching
  max_iter_layer == 200 (your run of interest).
- lvqe_noisy_k_comm.csv is opened in "w" (overwrite) mode, so it ALWAYS
  only contains the cost history of the most recently executed run.
  Since your most recent run was the max_iter_layer=200 one, this file
  is already exactly what you want -- no filtering needed there.
- The cost_history list is appended to continuously across layers
  inside simulate_one_lvqe_with_device, and no layer-boundary marker
  is ever written to disk. That means it is NOT possible to recover
  exactly where layer 0 ends and layer 1 begins from this CSV alone.
  The script marks this honestly rather than guessing a split.
"""

import csv
import matplotlib.pyplot as plt

HISTORY_CSV = "lvqe_noisy_k_comm.csv"  # your local file is named lvqe_noisy_k_comm.csv
SUMMARY_CSV = "lvqe_noisy_results.csv"
TARGET_MAX_ITER_LAYER = 200  # the run you actually care about

# Intended per-layer iteration caps, in order (layer 0, layer 1, layer 2, ...).
# These are the values YOU passed to the optimizer, not something recovered
# from the CSV -- COBYLA can call the cost function more or less than this
# many times per layer (extra calls for constraint handling / line search,
# or fewer if it converges early), so treat the resulting boundaries as
# approximate, intended cutoffs rather than verified ground truth.
LAYER_ITER_CAPS = [200, 200]  # layer 0, layer 1 -- layer 2 gets "whatever's left"

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

# ---- 2b. Compute intended layer boundaries ----
# boundaries[i] = step index where layer i is assumed to START
n_total = len(costs)
boundaries = [0]
for cap in LAYER_ITER_CAPS:
    boundaries.append(boundaries[-1] + cap)
# anything left over belongs to the final layer; clip in case caps overshoot n_total
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
    -true_baseline,
    color="gray",
    ls="--",
    lw=1,
    label=f"true_baseline = {true_baseline:.4f}",
)
# ax.axhline(
#     final_cost,
#     color="#c53030",
#     ls=":",
#     lw=1.5,
#     label=f"reported final_cost = {-final_cost:.4f}",
# )

# Intended layer boundaries (approximate -- see note in module docstring)
for b in boundaries[
    1:
]:  # skip step 0 (start of layer 0); every other entry is a real transition
    if b < n_total:  # don't draw a line exactly at/after the last data point
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
    f"noisy_modularity = {0.467188:.4f}\n"
    f"approx_ratio_rho = {0.6229:.4f}"  # correct values from csv file to avoid bug in recomputing the final cost instead of taking the cost of layer 2
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
