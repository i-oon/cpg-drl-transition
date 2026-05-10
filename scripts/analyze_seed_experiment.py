"""
Seed-robustness analysis: baselines + 2×2 ablation (space × dimension).

Aggregates per-window jerk_TRANS across 10 seeds (60 windows each),
produces a summary table and a boxplot.

Usage:
    python scripts/analyze_seed_experiment.py
    python scripts/analyze_seed_experiment.py --out logs/phase2_seed_experiment/results_2x2.png
    python scripts/analyze_seed_experiment.py --mode baselines
    python scripts/analyze_seed_experiment.py --mode ablation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="logs/phase2_seed_experiment")
parser.add_argument("--out", default="logs/phase2_seed_experiment/results_2x2.png")
parser.add_argument("--mode", default="all",
                    choices=["all", "baselines", "ablation"],
                    help="all=all methods, baselines=Discrete/Smoothstep/Residual-α 4D, "
                         "ablation=2×2 factorial only")
args = parser.parse_args()

dt          = 0.02
TRANS_S     = 3.0
PAD_POST    = 25
WIN_STEPS   = int(TRANS_S / dt) + PAD_POST
TRANS_STEPS = int(TRANS_S / dt)

# All methods available. Filtered below by --mode.
ALL_METHODS = {
    "Discrete":       ("discrete",          "#d62728"),
    "Smoothstep":     ("smoothstep",        "#2ca02c"),
    "Res-α 4D":       ("v10",               "#1f77b4"),
    "Res-q 4D":       ("residual_q_4d",     "#aec7e8"),
    "Res-α 12D":      ("residual_alpha_12d","#9467bd"),
    "Res-q 12D":      ("action_space",      "#ff7f0e"),
}

if args.mode == "baselines":
    METHODS = {k: v for k, v in ALL_METHODS.items()
               if k in ("Discrete", "Smoothstep", "Res-α 4D")}
elif args.mode == "ablation":
    METHODS = {k: v for k, v in ALL_METHODS.items()
               if k in ("Res-α 4D", "Res-q 4D", "Res-α 12D", "Res-q 12D")}
else:
    METHODS = ALL_METHODS


def jerk_windows(path):
    df  = pd.read_csv(path)
    ab  = df["alpha_base"].values
    ja  = df[[f"ja{i}" for i in range(12)]].values
    jerk_step = np.sqrt(np.mean((np.diff(ja, axis=0) / dt) ** 2, axis=1))

    starts = []
    for i in range(1, len(ab) - WIN_STEPS):
        jumped = (ab[i] - ab[i - 1]) > 0.5
        ramped = (ab[i - 1] < 0.02) and (ab[i] >= 0.02)
        if jumped or ramped:
            starts.append(i)

    out = []
    for s in starts:
        if s + WIN_STEPS <= len(jerk_step):
            w = jerk_step[s : s + TRANS_STEPS]
            out.append(float(np.sqrt(np.mean(w ** 2))))
    return out


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------
all_windows = {}
for label, (folder, color) in METHODS.items():
    wins = []
    for seed in range(10):
        p = Path(args.dir) / folder / f"playback_s{seed}.csv"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        wins.extend(jerk_windows(str(p)))
    all_windows[label] = np.array(wins)
    print(f"{label:12s}  n={len(wins):3d}  "
          f"mean={wins and np.mean(wins):.0f}  "
          f"std={wins and np.std(wins):.0f}  "
          f"min={wins and np.min(wins):.0f}  "
          f"max={wins and np.max(wins):.0f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n--- Summary ---")
print(f"{'Method':<14} {'N':>4} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
for label, arr in all_windows.items():
    if len(arr):
        print(f"{label:<14} {len(arr):>4} {arr.mean():>7.0f} {arr.std():>7.0f} "
              f"{arr.min():>7.0f} {arr.max():>7.0f}")

# ---------------------------------------------------------------------------
# Plot: boxplot + strip
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

labels  = list(all_windows.keys())
colors  = [METHODS[l][1] for l in labels]
data    = [all_windows[l] for l in labels]

bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2),
                capprops=dict(lw=1.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# strip plot (individual windows)
for i, (arr, color) in enumerate(zip(data, colors), start=1):
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(arr))
    ax.scatter(np.full_like(arr, i) + jitter, arr,
               color=color, alpha=0.35, s=18, zorder=3)

# mean annotations
for i, arr in enumerate(data, start=1):
    ax.text(i, arr.mean(), f"{arr.mean():.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Transition-window jerk RMS [rad/s³]", fontsize=10)
mode_labels = {
    "all":       "All methods",
    "baselines": "Baselines",
    "ablation":  "2×2 ablation: space (α/q) × dimension (4D/12D)",
}
ax.set_title(
    f"{mode_labels[args.mode]} — per-window jerk_TRANS across 10 seeds (60 windows each)\n"
    "Box = IQR, whiskers = 1.5×IQR, dots = individual transition windows",
    fontsize=10,
)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(arr.max() for arr in data if len(arr)) * 1.15)

fig.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
