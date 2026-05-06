"""
Overlay body-acceleration comparison: discrete switching vs residual learning (v10).

Reads two CSVs, computes body forward + vertical acceleration from vx/vz columns
(vx is in CSV; vz is not — we use a rolling derivative of vx as the forward accel signal),
and plots them overlaid on shared axes with phase-zone background shading.

Usage:
    python scripts/plot_body_acc_compare.py
    python scripts/plot_body_acc_compare.py --out logs/phase2/compare_body_acc.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv_v10",      default="logs/phase2/phase2_v10/playback.csv")
parser.add_argument("--csv_discrete", default="logs/phase2/baselines/discrete/playback.csv")
parser.add_argument("--out",          default="logs/phase2/compare_body_acc.png")
parser.add_argument("--smooth",       type=int, default=5,
                    help="Rolling-window half-width for display smoothing (steps). 0=off.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
v10  = pd.read_csv(args.csv_v10)
disc = pd.read_csv(args.csv_discrete)

# Align length in case they differ
n = min(len(v10), len(disc))
v10  = v10.iloc[:n].reset_index(drop=True)
disc = disc.iloc[:n].reset_index(drop=True)

dt  = float(v10["t"].iloc[1] - v10["t"].iloc[0])   # control dt (0.02 s)
t   = v10["t"].values[1:]                           # diff shortens by 1

# ---------------------------------------------------------------------------
# Body forward acceleration = dvx/dt
# ---------------------------------------------------------------------------
ax_v10  = np.diff(v10["vx"].values)  / dt   # m/s²
ax_disc = np.diff(disc["vx"].values) / dt

# Optional smoothing — rolling mean to reduce step-noise while keeping shape
def _smooth(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

w = max(1, args.smooth)
ax_v10_s  = _smooth(ax_v10,  w)
ax_disc_s = _smooth(ax_disc, w)

# Combined acceleration norm (use vx only since vz not in CSV)
norm_v10  = np.abs(ax_v10_s)
norm_disc = np.abs(ax_disc_s)

# ---------------------------------------------------------------------------
# Phase-zone masks from v10's alpha_base (defines transition window)
# ---------------------------------------------------------------------------
ab = v10["alpha_base"].values[1:]   # aligned with diff output
src_mask = ab < 0.05
trs_mask = (ab >= 0.05) & (ab <= 0.95)
tgt_mask = ab > 0.95

# ---------------------------------------------------------------------------
# Rolling RMS for the bottom summary panel
# ---------------------------------------------------------------------------
win = 25   # 25 steps = 0.5 s
def _rolling_rms(x, w):
    out = np.zeros_like(x)
    for i in range(len(x)):
        sl = x[max(0, i - w):i + 1]
        out[i] = np.sqrt(np.mean(sl ** 2))
    return out

rms_v10  = _rolling_rms(ax_v10_s,  win)
rms_disc = _rolling_rms(ax_disc_s, win)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

COLORS = {
    "v10":     ("#1f77b4", "Residual v10"),
    "discrete": ("#d62728", "Discrete switch"),
}

def _shade(ax):
    xform = ax.get_xaxis_transform()
    ax.fill_between(t, 0, 1, where=src_mask, alpha=0.10, color="#4472C4",
                    step="post", transform=xform, zorder=0, label="_nolegend_")
    ax.fill_between(t, 0, 1, where=trs_mask, alpha=0.13, color="#7B2D8B",
                    step="post", transform=xform, zorder=0, label="_nolegend_")
    ax.fill_between(t, 0, 1, where=tgt_mask, alpha=0.10, color="#ED7D31",
                    step="post", transform=xform, zorder=0, label="_nolegend_")

# Top: raw (smoothed) forward acceleration
ax0 = axes[0]
_shade(ax0)
ax0.plot(t, ax_v10_s,  lw=1.0, c=COLORS["v10"][0],      label=COLORS["v10"][1],      alpha=0.9)
ax0.plot(t, ax_disc_s, lw=1.0, c=COLORS["discrete"][0], label=COLORS["discrete"][1], alpha=0.8)
ax0.axhline(0, c="k", lw=0.4)
ax0.set_ylabel("body ax [m/s²]\n(dvx/dt)", fontsize=10)
ax0.legend(loc="upper right", fontsize=9)
ax0.grid(alpha=0.25)

# Bottom: rolling RMS — single interpretable line per method
ax1 = axes[1]
_shade(ax1)
ax1.plot(t, rms_v10,  lw=1.4, c=COLORS["v10"][0],      label=f"v10  RMS = {np.mean(rms_v10):.3f}")
ax1.plot(t, rms_disc, lw=1.4, c=COLORS["discrete"][0], label=f"disc RMS = {np.mean(rms_disc):.3f}")
ax1.set_ylabel(f"|ax| rolling RMS\n({win}-step ≈ {win*dt:.1f} s)", fontsize=10)
ax1.set_xlabel("time [s]", fontsize=10)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

# Gait-switch labels on top axis
unique_pairs = []
prev = None
for i, row in v10.iterrows():
    pair = (row["current"], row["target"])
    if pair != prev:
        unique_pairs.append((row["t"], pair))
        prev = pair
for seg_t, (cur, tgt) in unique_pairs:
    ax0.axvline(seg_t, ls="--", c="k", alpha=0.35, lw=0.8)
    ax1.axvline(seg_t, ls="--", c="k", alpha=0.35, lw=0.8)
    ax0.text(seg_t + 0.1, ax0.get_ylim()[1] * 0.88,
             f"{cur[:3]}→{tgt[:3]}", fontsize=6.5, color="dimgray", rotation=90, va="top")

fig.suptitle(
    "Body forward acceleration — Discrete switching vs Residual learning (v10)\n"
    "blue=src-hold  purple=transition window  orange=tgt-hold  |  "
    f"smoothing window={w} steps",
    fontsize=10,
)
fig.tight_layout()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
