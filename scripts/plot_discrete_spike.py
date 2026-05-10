"""
Single-method transition zoom for Discrete Switch — "naive switching fails" figure.

Three panels (no α panel):
  1. Max |joint velocity| — instant spike at switch
  2. Forward velocity     — velocity dip / reversal
  3. Joint jerk RMS       — motor stress metric

Usage:
    python scripts/plot_discrete_spike.py
    python scripts/plot_discrete_spike.py --window 3
    python scripts/plot_discrete_spike.py --out logs/phase2/discrete_spike.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv",    default="logs/phase2/baselines/discrete/playback.csv")
parser.add_argument("--window", type=int, default=0)
parser.add_argument("--pre_s",  type=float, default=1.5)
parser.add_argument("--post_s", type=float, default=5.0)
parser.add_argument("--out",    default="logs/phase2/discrete_spike.png")
args = parser.parse_args()

DT = 0.02
COLOR = "#d62728"


def rolling_max(arr, w):
    out = np.empty_like(arr)
    half = w // 2
    for i in range(len(arr)):
        out[i] = arr[max(0, i-half):min(len(arr), i+half+1)].max()
    return out


# ── Load ──────────────────────────────────────────────────────────────────────
df     = pd.read_csv(args.csv)
t      = df["t"].values
ab     = df["alpha_base"].values
vx     = df["vx"].values
jv     = df[[f"jv{i}" for i in range(12)]].values
ja     = df[[f"ja{i}" for i in range(12)]].values

jv_max  = np.max(np.abs(jv), axis=1)
jv_rms  = np.sqrt(np.mean(jv ** 2, axis=1))
jerk    = np.sqrt(np.mean((np.diff(ja, axis=0, prepend=ja[:1]) / DT) ** 2, axis=1))

# ── Find transition start ─────────────────────────────────────────────────────
starts = []
for i in range(1, len(ab)):
    jumped = (ab[i] - ab[i-1]) > 0.4
    ramped = (ab[i-1] < 0.02) and (ab[i] > 0.02)
    if jumped or ramped:
        if not starts or (i - starts[-1]) > 100:
            starts.append(i)

win     = min(args.window, len(starts) - 1)
t_start = t[starts[win]]
t_lo    = t_start - args.pre_s
t_hi    = t_start + args.post_s
mask    = (t >= t_lo) & (t <= t_hi)
t_rel   = t[mask] - t_start

# ── Peak values ───────────────────────────────────────────────────────────────
spike_mask = (t >= t_start) & (t <= t_start + 0.3)
jv_max_peak  = jv_max[spike_mask].max()
jerk_peak    = jerk[spike_mask].max()
pre_mask     = (t >= t_lo) & (t < t_start - 0.1)
jv_steady    = jv_max[pre_mask].max()
jerk_steady  = jerk[pre_mask].max()

print(f"Window {args.window}: t_start={t_start:.2f}s")
print(f"  Max |jv| at switch : {jv_max_peak:.2f} rad/s  (steady-state: {jv_steady:.2f})")
print(f"  Jerk peak at switch: {jerk_peak:.0f} rad/s³   (steady-state: {jerk_steady:.0f})")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
fig.subplots_adjust(hspace=0.06, top=0.88, bottom=0.09, left=0.13, right=0.97)

jv_max_plot = rolling_max(jv_max[mask], w=3)

axes[0].plot(t_rel, jv_max_plot,  color=COLOR, lw=2.0)
axes[1].plot(t_rel, vx[mask],     color=COLOR, lw=2.0)
axes[2].plot(t_rel, jerk[mask],   color=COLOR, lw=1.5)

# ── Reference lines ───────────────────────────────────────────────────────────
for ax in axes:
    ax.axvline(0,   color="red",    lw=1.8, ls=":",  alpha=0.85)
    ax.axvline(3.0, color="orange", lw=1.2, ls="--", alpha=0.70)
    ax.axvspan(0, 3, color="gold", alpha=0.07, zorder=0)

# ── Steady-state reference lines ─────────────────────────────────────────────
axes[0].axhline(jv_steady, color="gray", lw=1.0, ls=":", alpha=0.7,
                label=f"steady-state (~{jv_steady:.1f} rad/s)")
axes[2].axhline(jerk_steady, color="gray", lw=1.0, ls=":", alpha=0.7,
                label=f"steady-state (~{jerk_steady:.0f} rad/s³)")

# ── Axis labels & limits ──────────────────────────────────────────────────────
axes[0].set_ylabel("Max |joint velocity|\n[rad/s]", fontsize=10)
axes[0].set_ylim(0, jv_max_peak * 1.25)
axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.85)

axes[1].set_ylabel("Forward velocity\n[m/s]", fontsize=10)
axes[1].axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
axes[1].axhline(0.4, color="gray", lw=0.8, ls=":",  alpha=0.4, label="cmd (0.4 m/s)")
axes[1].legend(loc="lower right", fontsize=8.5, framealpha=0.85)

axes[2].set_ylabel("Joint jerk RMS\n[rad/s³]", fontsize=10)
axes[2].set_ylim(0, jerk_peak * 1.25)
axes[2].legend(loc="upper right", fontsize=8.5, framealpha=0.85)
axes[2].set_xlabel("Time relative to gait switch command [s]", fontsize=10)

# ── Spike annotations ─────────────────────────────────────────────────────────
peak_t_idx = np.argmax(jv_max[spike_mask])
peak_t_rel = t[spike_mask][peak_t_idx] - t_start

axes[0].annotate(
    f"{jv_max_peak:.1f} rad/s",
    xy=(peak_t_rel, jv_max_peak),
    xytext=(0.5, jv_max_peak * 0.88),
    color=COLOR, fontsize=10, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COLOR, lw=1.3),
)

jerk_peak_t_idx = np.argmax(jerk[spike_mask])
jerk_peak_t_rel = t[spike_mask][jerk_peak_t_idx] - t_start
axes[2].annotate(
    f"{jerk_peak:.0f} rad/s³",
    xy=(jerk_peak_t_rel, jerk_peak),
    xytext=(0.5, jerk_peak * 0.85),
    color=COLOR, fontsize=10, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=COLOR, lw=1.3),
)

# ── Legend patches for time markers ──────────────────────────────────────────
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color="red",    alpha=0.8,  label="gait switch (t = 0)"),
    mpatches.Patch(color="orange", alpha=0.6,  label="ramp end (t = 3 s)"),
    mpatches.Patch(color="gold",   alpha=0.35, label="3 s window"),
]
axes[0].legend(
    handles=legend_patches + axes[0].get_legend_handles_labels()[0][:1],
    loc="upper right", fontsize=8.5, framealpha=0.88, ncol=2,
)

# ── Gait pair label ───────────────────────────────────────────────────────────
mask_pair = (df["t"] >= t_lo) & (df["t"] <= t_lo + 0.1)
pair_str  = f"{df.loc[mask_pair,'current'].iloc[0]} → {df.loc[mask_pair,'target'].iloc[0]}" \
    if "current" in df.columns and mask_pair.any() else f"window {args.window}"

fig.suptitle(
    f"Discrete switch — {pair_str}\n"
    f"Instant α = 1 at t = 0: kinematic shock, velocity reversal, and jerk spike",
    fontsize=11, y=0.96,
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
