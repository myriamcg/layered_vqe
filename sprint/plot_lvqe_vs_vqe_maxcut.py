import csv
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_CSV = "csv_files/vqe_vs_lvqe_noisy_maxcut.csv"
N_TRIALS = 10  # trials per method in the run of interest (N_SEEDS in your script)

# ---- 1. Load CSV and select the most recent full run (last N_TRIALS*2 rows) ----
with open(SUMMARY_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} total rows from {SUMMARY_CSV}")

run_rows = rows[-N_TRIALS * 2 :]
if len(run_rows) < N_TRIALS * 2:
    raise ValueError(f"Expected at least {N_TRIALS*2} rows, found only {len(run_rows)}")

lvqe_rows = [r for r in run_rows if r["method"] == "L-VQE"]
vqe_rows = [r for r in run_rows if r["method"] == "VQE"]

if len(lvqe_rows) != N_TRIALS or len(vqe_rows) != N_TRIALS:
    print(
        f"Warning: expected {N_TRIALS} rows per method in the selected block, "
        f"got L-VQE={len(lvqe_rows)}, VQE={len(vqe_rows)}. "
        f"Check N_TRIALS or whether the last run was complete."
    )

# Sort each method's rows by trial index so paired trials line up correctly
lvqe_rows.sort(key=lambda r: int(r["trial"]))
vqe_rows.sort(key=lambda r: int(r["trial"]))

trials = [int(r["trial"]) for r in lvqe_rows]
lvqe_rho = [float(r["approx_ratio"]) for r in lvqe_rows]
vqe_rho = [float(r["approx_ratio"]) for r in vqe_rows]
lvqe_mod = [float(r["modularity"]) for r in lvqe_rows]
vqe_mod = [float(r["modularity"]) for r in vqe_rows]
true_baseline = float(run_rows[0]["true_baseline"])
shots = run_rows[0]["shots"]

lvqe_rho_mean, lvqe_rho_std = np.mean(lvqe_rho), np.std(lvqe_rho)
vqe_rho_mean, vqe_rho_std = np.mean(vqe_rho), np.std(vqe_rho)
lvqe_mod_mean, lvqe_mod_std = np.mean(lvqe_mod), np.std(lvqe_mod)
vqe_mod_mean, vqe_mod_std = np.mean(vqe_mod), np.std(vqe_mod)

print(f"LVQE rho: {lvqe_rho_mean:.4f} +/- {lvqe_rho_std:.4f}")
print(f"VQE  rho: {vqe_rho_mean:.4f} +/- {vqe_rho_std:.4f}")

# ---- 2. Plot: paired per-trial comparison + mean/std summary ----
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.6, 1]}
)

# -- Left panel: per-trial paired approx_ratio (note: approx_ratio is negative
#    by this script's sign convention, since true_baseline < 0 for MaxCut here) --
ax = axes[0]
x = np.arange(N_TRIALS)
width = 0.35
ax.bar(x - width / 2, lvqe_rho, width, color="#1a365d", alpha=0.85, label="L-VQE")
ax.bar(x + width / 2, vqe_rho, width, color="#c53030", alpha=0.85, label="VQE")

ax.axhline(lvqe_rho_mean, color="#1a365d", ls="--", lw=1.2, alpha=0.7)
ax.axhline(vqe_rho_mean, color="#c53030", ls="--", lw=1.2, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in trials])
ax.set_xlabel("Trial")
ax.set_ylabel("Approximation ratio ρ")
ax.set_title(f"Per-trial approx. ratio — MaxCut, 3-regular graph (n=8)\nshots={shots}")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25, axis="y")

# -- Right panel: mean +/- std summary bars --
ax2 = axes[1]
methods = ["L-VQE", "VQE"]
means = [lvqe_rho_mean, vqe_rho_mean]
stds = [lvqe_rho_std, vqe_rho_std]
colors = ["#1a365d", "#c53030"]
ax2.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
ax2.set_ylabel("Mean approximation ratio ρ (± std)")
ax2.set_title(f"Summary over {N_TRIALS} trials")
ax2.grid(alpha=0.25, axis="y")
ax2.set_ylim(min(means) - max(stds) - 0.22, 0.05)

for i, (m, s) in enumerate(zip(means, stds)):
    ax2.text(i, m - s - 0.06, f"{m:.4f}\n±{s:.4f}", ha="center", va="top", fontsize=9)

fig.tight_layout()
fig.savefig("vqe_vs_lvqe_noisy_maxcut_comparison.png", dpi=180)
print("Saved plot")
