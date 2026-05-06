"""
Aligned transition-window jerk profile.

For every transition window in each method's CSV, time-align to t=0 at
ramp start, compute per-step jerk RMS across all 12 joints, then plot
mean ± std across all windows.

This shows the SHAPE of each method's transition jerk — spike vs smooth
envelope — in a way a table cannot.

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
parser.add_argument("--smooth", type=int, default=10,
                    help="Rolling window (steps) for smoothing profiles. 0=off.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
dt        = 0.02          # control step (s)
TRANS_S   = 3.0           # nominal transition duration (s)
PAD_PRE   = 0             # steps before ramp start to include
PAD_POST  = 25            # steps after ramp end to show settling
WIN_STEPS = int(TRANS_S / dt) + PAD_POST   # window length

RUNS = {
    "Discrete switch":   ("logs/phase2/baselines/discrete/playback.csv",      "#d62728", "-",  2.0),
    "Linear ramp":       ("logs/phase2/baselines/linear_ramp/playback.csv",   "#ff7f0e", "--", 1.5),
    "Smoothstep ramp":   ("logs/phase2/baselines/smoothstep_ramp/playback.csv","#2ca02c","--", 1.5),
    "Residual 1-D (abl)":("logs/phase2/residual1d_v1/playback.csv",           "#9467bd", ":",  1.2),
    "Residual v10":      ("logs/phase2/phase2_v10/playback.csv",               "#1f77b4", "-",  2.5),
}

# ---------------------------------------------------------------------------
# Helper: rolling mean
# ---------------------------------------------------------------------------
def _roll(x, w):
    if w <= 1:
        return x
    out = np.convolve(x, np.ones(w) / w, mode="same")
    return out

# ---------------------------------------------------------------------------
# Extract windows per method
# ---------------------------------------------------------------------------
t_axis = np.arange(WIN_STEPS) * dt   # shared x-axis (seconds from ramp start)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
ax_main = axes[0]
ax_rms  = axes[1]   # bottom: single episode-jerk RMS bar per method

summary_rms = {}    # method → scalar jerk_TRANS

for label, (path, color, ls, lw) in RUNS.items():
    if not Path(path).exists():
        print(f"  SKIP (missing): {path}")
        continue

    df  = pd.read_csv(path)
    ab  = df["alpha_base"].values               # (T,)
    ja  = df[[f"ja{i}" for i in range(12)]].values
    jerk_step = np.sqrt(np.mean(
        (np.diff(ja, axis=0) / dt) ** 2, axis=1))   # (T-1,) instantaneous jerk RMS

    # Find ramp-start indices: alpha crosses 0.02 from below (or jumps > 0.5 for discrete)
    starts = []
    for i in range(1, len(ab) - WIN_STEPS):
        jumped = (ab[i] - ab[i-1]) > 0.5                   # discrete: big jump
        ramped = (ab[i-1] < 0.02) and (ab[i] >= 0.02)      # smoothstep/linear: cross threshold
        if jumped or ramped:
            starts.append(i)

    if not starts:
        print(f"  WARN: no transition windows found for {label}")
        continue

    # Extract and store windows
    windows = []
    for s in starts:
        end = s + WIN_STEPS
        if end <= len(jerk_step):
            windows.append(jerk_step[s:end])

    if not windows:
        continue

    mat      = np.array(windows)                             # (n_windows, WIN_STEPS)
    mean_j   = np.mean(mat, axis=0)
    std_j    = np.std(mat,  axis=0)
    mean_j_s = _roll(mean_j, args.smooth)
    std_j_s  = _roll(std_j,  args.smooth)

    # transition-window scalar RMS (first 150 steps = 3 s)
    trans_steps = int(TRANS_S / dt)
    jt_rms = float(np.sqrt(np.mean(mat[:, :trans_steps] ** 2)))
    summary_rms[label] = jt_rms

    # Plot mean line + shaded ±1 std
    ax_main.plot(t_axis, mean_j_s, color=color, ls=ls, lw=lw, label=f"{label}  (RMS={jt_rms:.0f})")
    ax_main.fill_between(t_axis,
                         np.maximum(0, mean_j_s - std_j_s),
                         mean_j_s + std_j_s,
                         color=color, alpha=0.10)

    print(f"  {label:<22}  n_windows={len(windows)}  jerk_TRANS_RMS={jt_rms:.0f}")

# ---------------------------------------------------------------------------
# Decoration — main panel
# ---------------------------------------------------------------------------
# Phase-zone shading on main panel
xform = ax_main.get_xaxis_transform()
ax_main.axvspan(0,       TRANS_S,        alpha=0.06, color="#7B2D8B", zorder=0, label="_nolegend_")
ax_main.axvspan(TRANS_S, t_axis[-1],     alpha=0.06, color="#ED7D31", zorder=0, label="_nolegend_")
ax_main.axvline(0,       ls=":", c="gray", lw=1.0, label="ramp start")
ax_main.axvline(TRANS_S, ls=":", c="gray", lw=1.0, label="ramp end")

ax_main.set_ylabel("Jerk RMS [rad/s³]\n(12-joint mean, per step)", fontsize=10)
ax_main.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax_main.grid(alpha=0.25)
ax_main.set_title(
    "Transition-window jerk profile — time-aligned to ramp start\n"
    "purple = transition window (0–3 s)   orange = settling phase   shading = ±1 std across windows",
    fontsize=10,
)

# ---------------------------------------------------------------------------
# Bottom bar chart — scalar jerk_TRANS per method
# ---------------------------------------------------------------------------
if summary_rms:
    labels_sorted = list(summary_rms.keys())
    vals  = [summary_rms[k] for k in labels_sorted]
    cols  = [RUNS[k][1] for k in labels_sorted]
    bars  = ax_rms.barh(labels_sorted, vals, color=cols, alpha=0.85, height=0.55)
    ax_rms.set_xlabel("Transition-window jerk RMS [rad/s³]", fontsize=9)
    ax_rms.grid(axis="x", alpha=0.3)
    # annotate value on each bar
    for bar, val in zip(bars, vals):
        ax_rms.text(val + 80, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=8.5)
    ax_rms.set_xlim(0, max(vals) * 1.18)
    # Highlight v10 bar
    if "Residual v10" in summary_rms:
        idx = labels_sorted.index("Residual v10")
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2.0)

ax_rms.set_xlabel("Transition-window jerk RMS [rad/s³]", fontsize=9)

fig.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
