"""
Transition zoom: trot→bound (seed=42) — Discrete vs Smoothstep.

Three panels showing actual simulation data from playback_seed42.csv:
  1. α       — blending schedule
  2. Joint velocity (RMS across 12 joints)
  3. Joint acceleration (RMS across 12 joints)

Both methods share the same pre-transition data (identical robot state),
so the divergence at t=0 isolates the blending schedule effect.

Usage:
    python scripts/plot_transition_zoom_disc_vs_smth.py
    python scripts/plot_transition_zoom_disc_vs_smth.py --window 1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=0, help="Transition index (0=trot→bound)")
parser.add_argument("--pre_s",  type=float, default=1.5, help="Seconds to show before switch")
parser.add_argument("--post_s", type=float, default=5.0, help="Seconds to show after switch")
parser.add_argument("--out",    default="logs/phase2_v3/transition_zoom_disc_vs_smth.png")
args = parser.parse_args()

BASE  = Path(__file__).parent.parent
DISC  = BASE / "logs/phase2/baselines/discrete/playback_seed42.csv"
SMTH  = BASE / "logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv"

JV_COLS = [f"jv{i}" for i in range(12)]
JA_COLS = [f"ja{i}" for i in range(12)]

GAIT_PAIRS = ["trot→bound", "bound→pace", "pace→trot", "trot→pace", "pace→bound", "bound→trot"]

# ── Load & derive ─────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df["jv_rms"] = np.sqrt((df[JV_COLS] ** 2).mean(axis=1))
    df["ja_rms"] = np.sqrt((df[JA_COLS] ** 2).mean(axis=1))
    # Find transition starts: first step where alpha_base goes from ~0 to >0
    ab = df["alpha_base"].values
    starts = []
    for i in range(1, len(ab)):
        if ab[i - 1] < 0.01 and ab[i] > 0.01:
            starts.append(i)
        elif (ab[i] - ab[i - 1]) > 0.4:   # discrete jump
            starts.append(i)
    return df, starts

disc_df, disc_starts = load(DISC)
smth_df, smth_starts = load(SMTH)

win = min(args.window, len(disc_starts) - 1)
pair_label = GAIT_PAIRS[win] if win < len(GAIT_PAIRS) else f"window {win}"
print(f"Plotting: {pair_label}  (window={win})")

disc_t0 = disc_df["t"].iloc[disc_starts[win]]
smth_t0 = smth_df["t"].iloc[smth_starts[win]]
print(f"  Discrete  switch at t={disc_t0:.2f} s")
print(f"  Smoothstep switch at t={smth_t0:.2f} s")

def clip_window(df, t0):
    lo, hi = t0 - args.pre_s, t0 + args.post_s
    sub = df[(df["t"] >= lo) & (df["t"] <= hi)].copy()
    sub["t_rel"] = sub["t"] - t0
    return sub

d = clip_window(disc_df, disc_t0)
s = clip_window(smth_df, smth_t0)

T_DUR = 3.0   # canonical transition duration

# ── Style ─────────────────────────────────────────────────────────────────────
ORANGE = "#EA580C"
BLUE   = "#2563EB"
GREY   = "#6B7280"
LBLUE  = "#DBEAFE"
LORAN  = "#FFEDD5"

fig = plt.figure(figsize=(8, 9))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.48)

def xdec(ax):
    ax.axvline(0,     color="red",    lw=1.6, ls=":",  alpha=0.85, zorder=3)
    ax.axvline(T_DUR, color=ORANGE,   lw=1.2, ls="--", alpha=0.70, zorder=3)
    ax.axvspan(0, T_DUR, color="gold", alpha=0.07, zorder=0)
    ax.set_xlim(-args.pre_s, args.post_s)
    ax.spines[["top", "right"]].set_visible(False)

# ── Panel 0: α ────────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])

ax0.plot(d["t_rel"], d["alpha_base"], color=ORANGE, lw=2.2, ls="--",
         label="Discrete Switch")
ax0.plot(s["t_rel"], s["alpha_base"], color=BLUE,   lw=2.2,
         label="Smoothstep")

ax0.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.35)
ax0.axhline(1, color=GREY, lw=0.6, ls="--", alpha=0.35)
ax0.text(-args.pre_s + 0.08,  1.06, "bound", fontsize=8.5, color=GREY)
ax0.text(-args.pre_s + 0.08, -0.15, "trot",  fontsize=8.5, color=GREY)
ax0.text(0.06,   -0.15, "switch", fontsize=8, color="red", alpha=0.85)
ax0.text(T_DUR + 0.06, -0.15, f"T={T_DUR:.0f} s", fontsize=8, color=ORANGE)

ax0.set_ylabel(r"$\alpha$  (blending weight)", fontsize=10)
ax0.set_title(
    f"Transition Zoom — {pair_label}  (seed=42)\n"
    f"red: gait switch · dashed orange: ramp end · gold: 3 s window",
    fontsize=11, fontweight="bold",
)
ax0.legend(fontsize=10, loc="center right")
ax0.set_ylim(-0.22, 1.28)
xdec(ax0)

# ── Panel 1: Joint Velocity RMS ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])

# y ceiling: 99th percentile to suppress rare outliers
y_ceil_v = np.percentile(pd.concat([d["jv_rms"], s["jv_rms"]]), 99.5)

ax1.fill_between(s["t_rel"], s["jv_rms"].clip(upper=y_ceil_v),
                 color=LBLUE, alpha=0.30)
ax1.plot(d["t_rel"], d["jv_rms"].clip(upper=y_ceil_v),
         color=ORANGE, lw=2.0, ls="--", label="Discrete Switch")
ax1.plot(s["t_rel"], s["jv_rms"].clip(upper=y_ceil_v),
         color=BLUE,   lw=2.0,          label="Smoothstep")

# Annotate discrete peak
d_pk_idx = d["jv_rms"].idxmax()
d_pk_t   = d.loc[d_pk_idx, "t_rel"]
d_pk_v   = min(d.loc[d_pk_idx, "jv_rms"], y_ceil_v)
s_pk_v   = s["jv_rms"].max()
ratio    = d.loc[d_pk_idx, "jv_rms"] / s_pk_v if s_pk_v > 0 else 0
ax1.annotate(
    f"{d.loc[d_pk_idx, 'jv_rms']:.1f} rad/s  (×{ratio:.1f} vs Smoothstep)",
    xy=(d_pk_t, d_pk_v),
    xytext=(0.6, d_pk_v * 0.78),
    fontsize=9, color=ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
)

ax1.set_ylabel("Joint velocity RMS  (rad/s)", fontsize=10)
ax1.set_ylim(0, y_ceil_v * 1.18)
ax1.legend(fontsize=9, loc="upper right")
xdec(ax1)

# ── Panel 2: Joint Acceleration RMS ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])

y_ceil_a = np.percentile(pd.concat([d["ja_rms"], s["ja_rms"]]), 99.5)

ax2.fill_between(s["t_rel"], s["ja_rms"].clip(upper=y_ceil_a),
                 color=LBLUE, alpha=0.30)
ax2.plot(d["t_rel"], d["ja_rms"].clip(upper=y_ceil_a),
         color=ORANGE, lw=2.0, ls="--", label="Discrete Switch")
ax2.plot(s["t_rel"], s["ja_rms"].clip(upper=y_ceil_a),
         color=BLUE,   lw=2.0,          label="Smoothstep")

# Annotate discrete peak
da_pk_idx = d["ja_rms"].idxmax()
da_pk_t   = d.loc[da_pk_idx, "t_rel"]
da_pk_v   = min(d.loc[da_pk_idx, "ja_rms"], y_ceil_a)
sa_pk_v   = s["ja_rms"].max()
ratio_a   = d.loc[da_pk_idx, "ja_rms"] / sa_pk_v if sa_pk_v > 0 else 0
ax2.annotate(
    f"{d.loc[da_pk_idx, 'ja_rms']:.0f} rad/s²  (×{ratio_a:.1f} vs Smoothstep)",
    xy=(da_pk_t, da_pk_v),
    xytext=(0.5, da_pk_v * 0.80),
    fontsize=9, color=ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
)

ax2.set_ylabel("Joint acceleration RMS  (rad/s²)", fontsize=10)
ax2.set_xlabel("Time relative to gait switch (s)", fontsize=10)
ax2.set_ylim(0, y_ceil_a * 1.18)
ax2.legend(fontsize=9, loc="upper right")
xdec(ax2)

# ─────────────────────────────────────────────────────────────────────────────
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
