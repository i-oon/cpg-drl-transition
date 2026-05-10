"""
Overlay vx and jerk for discrete / smoothstep / v10 on the same transition windows.

Two panels:
  Top:    Full-run vx trace, all 3 methods overlaid (seed 0)
  Bottom: Worst window zoomed in (bound→pace, t=10s), time-aligned to switch

Usage:
    python scripts/plot_seed_overlay.py
    python scripts/plot_seed_overlay.py --out logs/phase2_seed_experiment/overlay.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="logs/phase2_seed_experiment/overlay.png")
args = parser.parse_args()

dt          = 0.02
TRANS_S     = 3.0
PAD_PRE     = 50   # steps before switch to show in zoom
PAD_POST    = 75   # steps after window to show in zoom
WIN_STEPS   = int(TRANS_S / dt) + 25
TRANS_STEPS = int(TRANS_S / dt)

METHODS = {
    "Discrete":   (f"logs/phase2_seed_experiment/discrete/playback_s{args.seed}.csv",   "#d62728", 2.0),
    "Smoothstep": (f"logs/phase2_seed_experiment/smoothstep/playback_s{args.seed}.csv", "#2ca02c", 1.8),
    "Residual-α": (f"logs/phase2_seed_experiment/v10/playback_s{args.seed}.csv",        "#1f77b4", 1.8),
}


def find_starts(ab):
    starts = []
    for i in range(1, len(ab) - WIN_STEPS):
        jumped = (ab[i] - ab[i - 1]) > 0.5
        ramped = (ab[i - 1] < 0.02) and (ab[i] >= 0.02)
        if jumped or ramped:
            starts.append(i)
    return starts


fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 2]})
ax_full, ax_zoom = axes

loaded = {}
for label, (path, color, lw) in METHODS.items():
    df = pd.read_csv(path)
    ja = df[[f"ja{i}" for i in range(12)]].values
    df["jerk"] = np.concatenate([[0], np.sqrt(np.mean((np.diff(ja, axis=0) / dt) ** 2, axis=1))])
    loaded[label] = df

# ---------------------------------------------------------------------------
# Top panel — full run jerk overlay (rolling mean for readability)
# ---------------------------------------------------------------------------
SMOOTH = 15   # steps rolling mean

ab_ref = loaded["Discrete"]["alpha_base"].values
starts = find_starts(ab_ref)

for label, (path, color, lw) in METHODS.items():
    df   = loaded[label]
    t    = df["t"].values
    jerk = pd.Series(df["jerk"].values).rolling(SMOOTH, center=True, min_periods=1).mean().values
    ax_full.plot(t, jerk, color=color, lw=lw, alpha=0.85, label=label)

for s in starts:
    ax_full.axvline(s * dt, ls="--", c="gray", lw=0.8, alpha=0.6)

ax_full.set_ylabel(f"Jerk RMS [rad/s³]\n(rolling mean {SMOOTH} steps)", fontsize=10)
ax_full.set_xlabel("Time [s]", fontsize=10)
ax_full.legend(loc="upper right", fontsize=10)
ax_full.set_title(
    f"Full-run jerk — Discrete vs Smoothstep vs v10  (seed={args.seed})\n"
    "Grey dashed = gait switch commands",
    fontsize=10,
)
ax_full.grid(alpha=0.25)

# ---------------------------------------------------------------------------
# Bottom panel — worst window zoomed (window 2, bound→pace, t≈10s)
# ---------------------------------------------------------------------------
target_win_idx = 1   # 0-indexed: second window (bound→pace at t=10s)

if target_win_idx < len(starts):
    s_ref   = starts[target_win_idx]
    t_start = s_ref * dt

    df_ref = loaded["Discrete"]
    cur = df_ref["current"].iloc[s_ref] if "current" in df_ref.columns else "?"
    tgt = df_ref["target"].iloc[s_ref]  if "target"  in df_ref.columns else "?"
    pair_label = f"{cur}→{tgt}"

    zoom_start = max(0, s_ref - PAD_PRE)
    zoom_end   = min(len(ab_ref), s_ref + TRANS_STEPS + PAD_POST)
    t_z = (np.arange(zoom_start, zoom_end) - s_ref) * dt

    for label, (path, color, lw) in METHODS.items():
        df   = loaded[label]
        jerk = df["jerk"].values
        jerk_s = pd.Series(jerk).rolling(SMOOTH, center=True, min_periods=1).mean().values
        ax_zoom.plot(t_z, jerk_s[zoom_start:zoom_end], color=color, lw=lw + 0.2,
                     alpha=0.9, label=label)
        rms = float(np.sqrt(np.mean(jerk[s_ref:s_ref + TRANS_STEPS] ** 2)))
        ax_zoom.annotate(f"RMS={rms:.0f}", xy=(TRANS_S * 0.5, jerk_s[zoom_start:zoom_end].max() * 0.95),
                         xytext=(0, 0), textcoords="offset points",
                         ha="center", fontsize=0, color=color)  # invisible, just for data

    # RMS annotations in legend-style text
    y_top = max(
        loaded[lb]["jerk"].values[zoom_start:zoom_end].max()
        for lb in METHODS
    ) * 1.05
    for i, (label, (path, color, lw)) in enumerate(METHODS.items()):
        jerk = loaded[label]["jerk"].values
        rms  = float(np.sqrt(np.mean(jerk[s_ref:s_ref + TRANS_STEPS] ** 2)))
        ax_zoom.text(TRANS_S + 0.1, y_top * (0.98 - i * 0.08),
                     f"{label}: RMS={rms:.0f}", color=color, fontsize=9, va="top")

    ax_zoom.axvline(0,       ls="--", c="gray", lw=1.2, label="switch cmd")
    ax_zoom.axvline(TRANS_S, ls=":",  c="gray", lw=1.0)
    ax_zoom.axvspan(0, TRANS_S, alpha=0.06, color="#7B2D8B")

    ax_zoom.set_xlim(t_z[0], t_z[-1])
    ax_zoom.set_xlabel("Time relative to switch [s]", fontsize=10)
    ax_zoom.set_ylabel(f"Jerk RMS [rad/s³]\n(rolling mean {SMOOTH} steps)", fontsize=10)
    ax_zoom.legend(loc="upper left", fontsize=10)
    ax_zoom.set_title(
        f"Zoom — transition window 2 ({pair_label}) at t={t_start:.1f}s  |  "
        f"purple = 3 s blending window",
        fontsize=10,
    )
    ax_zoom.grid(alpha=0.25)

fig.tight_layout()
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
