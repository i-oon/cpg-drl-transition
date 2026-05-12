"""
Per-gait-pair jerk analysis: baselines + 2×2 ablation (space × dimension).

Reads each method's canonical seed=42 playback CSV and computes jerk_TRANS
for each of the 6 directed gait-pair transitions.

NOTE ON EVALUATION SCOPE
─────────────────────────
The seed experiment (playback_s0..s9) was run with 10 environment seeds, but
investigation showed that env_cfg.seed does not vary jerk_TRANS:
  • Transition timing is pinned to 2.0 s in play_b1_phase2.py (_transition_start_s = 2.0).
  • Domain randomization (friction, mass) does not alter gait-phase at switch time.
  • All 10 per-seed CSVs are therefore identical — they represent the same
    canonical deterministic run repeated.

Correct framing: this is a 6-gait-pair analysis (N=6 per method), not a
60-window multi-seed robustness evaluation. The per-gait-pair breakdown
reveals which gait-pair transitions are structurally hard vs easy.

Usage:
    python scripts/analyze_seed_experiment.py
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
parser.add_argument("--out", default="logs/phase2_seed_experiment/results_2x2.png")
parser.add_argument("--mode", default="all",
                    choices=["all", "baselines", "ablation"],
                    help="all=all methods, baselines=Discrete/Smoothstep/Residual-α 4D, "
                         "ablation=2×2 factorial only")
parser.add_argument("--source", default="canonical",
                    choices=["canonical", "seeds"],
                    help="canonical: read playback_seed42.csv per method (N=6, pinned start). "
                         "seeds: read logs/phase2_seed_experiment/{folder}/playback_s{0..9}.csv "
                         "(N up to 60, requires run_seed_experiment.sh to have been run with "
                         "--randomize_start).")
parser.add_argument("--seed_dir", default="logs/phase2_seed_experiment",
                    help="Root dir for seed CSVs (used when --source seeds).")
args = parser.parse_args()

dt          = 0.02
TRANS_S     = 3.0
PAD_POST    = 25
WIN_STEPS   = int(TRANS_S / dt) + PAD_POST
TRANS_STEPS = int(TRANS_S / dt)

# Method definitions: (canonical_csv, seed_experiment_folder, color)
ALL_METHODS_DEF = {
    "Discrete":   ("logs/phase2/baselines/discrete/playback_seed42.csv",        "discrete",          "#d62728"),
    "Smoothstep": ("logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv", "smoothstep",        "#2ca02c"),
    "Res-α 4D":   ("logs/phase2/phase2_v10/playback_seed42.csv",                "v10",               "#1f77b4"),
    "Res-q 4D":   ("logs/phase2/residual_q_4d/playback_seed42.csv",             "residual_q_4d",     "#aec7e8"),
    "Res-α 12D":  ("logs/phase2/residual_alpha_12d/playback_seed42.csv",        "residual_alpha_12d","#9467bd"),
    "Res-q 12D":  ("logs/phase2/residual_q_12d/playback_seed42.csv",            "action_space",      "#ff7f0e"),
}

if args.mode == "baselines":
    METHODS = [k for k in ALL_METHODS_DEF if k in ("Discrete", "Smoothstep", "Res-α 4D")]
elif args.mode == "ablation":
    METHODS = [k for k in ALL_METHODS_DEF if k in ("Res-α 4D", "Res-q 4D", "Res-α 12D", "Res-q 12D")]
else:
    METHODS = list(ALL_METHODS_DEF.keys())


def jerk_windows(path):
    """Return list of per-window jerk_TRANS values from a playback CSV."""
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
# Load: canonical (N=6) or multi-seed (N up to 60)
# ---------------------------------------------------------------------------
all_windows = {}
source_label = "canonical seed=42" if args.source == "canonical" else "10-seed experiment"

for label in METHODS:
    canon_csv, folder, color = ALL_METHODS_DEF[label]

    if args.source == "canonical":
        p = Path(canon_csv)
        if not p.exists():
            print(f"  MISSING: {p}")
            all_windows[label] = np.array([])
            continue
        wins = jerk_windows(str(p))
    else:
        wins = []
        for seed in range(10):
            p = Path(args.seed_dir) / folder / f"playback_s{seed}.csv"
            if not p.exists():
                print(f"  MISSING: {p}")
                continue
            wins.extend(jerk_windows(str(p)))

    all_windows[label] = np.array(wins)
    if len(wins):
        vals_str = "  ".join(f"{v:.0f}" for v in wins[:6])
        suffix = f"  +{len(wins)-6} more" if len(wins) > 6 else ""
        print(f"{label:12s}  n={len(wins):3d}  mean={np.mean(wins):.0f}  "
              f"std={np.std(wins):.0f}  min={np.min(wins):.0f}  max={np.max(wins):.0f}"
              f"  |  {vals_str}{suffix}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n--- jerk_TRANS summary ({source_label}) ---")
print(f"{'Method':<14} {'N':>3} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
for label, arr in all_windows.items():
    if len(arr):
        print(f"{label:<14} {len(arr):>3} {arr.mean():>7.0f} {arr.std():>7.0f} "
              f"{arr.min():>7.0f} {arr.max():>7.0f}")

# ---------------------------------------------------------------------------
# Plot: boxplot + strip
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

labels  = list(all_windows.keys())
colors  = [ALL_METHODS_DEF[l][2] for l in labels]
data    = [all_windows[l] for l in labels]

bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                medianprops=dict(color="black", lw=2),
                whiskerprops=dict(lw=1.2),
                capprops=dict(lw=1.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# strip plot (individual gait-pair values)
for i, (arr, color) in enumerate(zip(data, colors), start=1):
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(arr))
    ax.scatter(np.full_like(arr, i) + jitter, arr,
               color=color, alpha=0.55, s=28, zorder=3)

# mean annotations
for i, arr in enumerate(data, start=1):
    if len(arr):
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
n_total = max((len(d) for d in data if len(d)), default=0)
ax.set_title(
    f"{mode_labels[args.mode]} — jerk_TRANS ({source_label}, N={n_total} per method)\n"
    "Box = IQR, dots = individual transition windows",
    fontsize=10,
)
ax.grid(axis="y", alpha=0.3)
non_empty = [arr for arr in data if len(arr)]
if non_empty:
    ax.set_ylim(0, max(arr.max() for arr in non_empty) * 1.15)

fig.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
