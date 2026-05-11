"""
Aligned transition-window jerk profile.

For every transition window in each method's CSV, time-align to t=0 at
ramp start, then show jerk as a grouped bar chart: divide the 3 s window
into N equal time bins and plot the RMS jerk within each bin.

This shows the SHAPE of each method's transition jerk — discrete's
single-bin spike vs smoothstep's plateau vs v10's lower bars.

Usage:
    python scripts/plot_transition_jerk.py
    python scripts/plot_transition_jerk.py --out logs/phase2/transition_jerk_profile.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="logs/phase2/transition_jerk_profile.png")
parser.add_argument("--nbins", type=int, default=10,
                    help="Number of equal time bins across the 3 s transition window.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
dt           = 0.02
TRANS_S      = 3.0
PAD_POST     = 25
WIN_STEPS    = int(TRANS_S / dt) + PAD_POST   # 175 steps total
TRANS_STEPS  = int(TRANS_S / dt)              # 150 steps = 3 s

RUNS = {
    "Discrete switch":  ("logs/phase2/baselines/discrete/playback_seed42.csv",        "#d62728", 0.85),
    "Linear ramp":      ("logs/phase2/baselines/linear_ramp/playback_seed42.csv",     "#ff7f0e", 0.75),
    "Smoothstep ramp":  ("logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv", "#2ca02c", 0.75),
    "Residual-α 4D":    ("logs/phase2/phase2_v10/playback_seed42.csv",                "#aec7e8", 0.75),
    "Residual-α 12D":   ("logs/phase2/residual_alpha_12d/playback_seed42.csv",        "#1f77b4", 0.90),
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
bin_edges   = np.linspace(0, TRANS_S, args.nbins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width_s = TRANS_S / args.nbins
bin_steps   = TRANS_STEPS // args.nbins

n_methods = len(RUNS)
bar_width  = bin_width_s / (n_methods + 1)
offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                         gridspec_kw={"height_ratios": [3, 1]})
ax_main = axes[0]
ax_rms  = axes[1]

summary_rms = {}

# ---------------------------------------------------------------------------
# Extract windows per method
# ---------------------------------------------------------------------------
for idx, (label, (path, color, alpha)) in enumerate(RUNS.items()):
    if not Path(path).exists():
        print(f"  SKIP (missing): {path}")
        continue

    df  = pd.read_csv(path)
    ab  = df["alpha_base"].values
    ja  = df[[f"ja{i}" for i in range(12)]].values
    jerk_step = np.sqrt(np.mean(
        (np.diff(ja, axis=0) / dt) ** 2, axis=1))   # (T-1,) per-step jerk RMS

    # Find ramp-start indices
    starts = []
    for i in range(1, len(ab) - WIN_STEPS):
        jumped = (ab[i] - ab[i-1]) > 0.5
        ramped = (ab[i-1] < 0.02) and (ab[i] >= 0.02)
        if jumped or ramped:
            starts.append(i)

    if not starts:
        print(f"  WARN: no transition windows found for {label}")
        continue

    windows = [jerk_step[s:s + WIN_STEPS]
               for s in starts if s + WIN_STEPS <= len(jerk_step)]
    if not windows:
        continue

    mat = np.array(windows)   # (n_windows, WIN_STEPS)

    # Transition-window scalar RMS
    jt_rms = float(np.sqrt(np.mean(mat[:, :TRANS_STEPS] ** 2)))
    summary_rms[label] = jt_rms

    # RMS per time bin, averaged across windows
    bin_rms = []
    bin_std = []
    for b in range(args.nbins):
        sl = mat[:, b * bin_steps : (b + 1) * bin_steps]
        per_window_rms = np.sqrt(np.mean(sl ** 2, axis=1))   # (n_windows,)
        bin_rms.append(per_window_rms.mean())
        bin_std.append(per_window_rms.std())

    bin_rms = np.array(bin_rms)
    bin_std = np.array(bin_std)
    x = bin_centers + offsets[idx]

    ax_main.bar(x, bin_rms, width=bar_width * 0.92,
                color=color, alpha=alpha,
                label=f"{label}  (RMS={jt_rms:.0f})", zorder=3)

    print(f"  {label:<22}  n_windows={len(windows)}  jerk_TRANS_RMS={jt_rms:.0f}")

# ---------------------------------------------------------------------------
# Decoration — main panel
# ---------------------------------------------------------------------------
ax_main.axvline(0,       ls=":", c="gray", lw=1.2, zorder=2)
ax_main.axvline(TRANS_S, ls=":", c="gray", lw=1.2, zorder=2)
ax_main.axvspan(0, TRANS_S, alpha=0.04, color="#7B2D8B", zorder=0)

ax_main.set_xlim(-bin_width_s * 0.6, TRANS_S + bin_width_s * 0.6)
ax_main.set_xticks(bin_centers)
ax_main.set_xticklabels([f"{t:.1f}" for t in bin_centers], fontsize=8)
ax_main.set_xlabel("Time since ramp start [s]", fontsize=10)
ax_main.set_ylabel(f"Jerk RMS [rad/s³]\n(per {bin_width_s:.1f} s bin, mean ± std across windows)", fontsize=10)
ax_main.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax_main.grid(axis="y", alpha=0.25, zorder=0)
ax_main.set_title(
    "Transition-window jerk profile — binned RMS, time-aligned to ramp start\n"
    f"Each group = one {bin_width_s:.1f} s bin  |  bars = mean jerk RMS across 6 transition windows",
    fontsize=10,
)

# ---------------------------------------------------------------------------
# Bottom bar chart — scalar jerk_TRANS RMS per method
# ---------------------------------------------------------------------------
if summary_rms:
    labels_sorted = list(summary_rms.keys())
    vals  = [summary_rms[k] for k in labels_sorted]
    cols  = [RUNS[k][1] for k in labels_sorted]
    bars  = ax_rms.barh(labels_sorted, vals, color=cols, alpha=0.85, height=0.55)
    ax_rms.set_xlabel("Transition-window jerk RMS [rad/s³]", fontsize=9)
    ax_rms.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax_rms.text(val + 80, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=8.5)
    ax_rms.set_xlim(0, max(vals) * 1.18)
    if "Residual-α 12D" in summary_rms:
        idx = labels_sorted.index("Residual-α 12D")
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2.0)

fig.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
