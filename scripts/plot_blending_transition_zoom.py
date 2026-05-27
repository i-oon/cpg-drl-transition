"""
Transition zoom: α, α̇, α̈ — Discrete vs Smoothstep overlaid, zoomed at t=0.

Shows the first 40% of the transition window so the singularity (Discrete) vs
bounded onset (Smoothstep) is visually obvious side-by-side on the same axes.

Usage:
    python scripts/plot_blending_transition_zoom.py
    python scripts/plot_blending_transition_zoom.py --T 1.5 --zoom 0.45
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--T",    type=float, default=3.0,  help="Transition duration (s)")
parser.add_argument("--zoom", type=float, default=0.45, help="Fraction of T to show after t=0")
parser.add_argument("--out",  default="logs/phase2_v3/blending_transition_zoom.png")
args = parser.parse_args()

T    = args.T
zoom = args.zoom
dt   = 1e-4
out  = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

pre_t  = T * 0.18          # how much time before t=0 to show
t_end  = T * zoom          # how far past t=0 to show

# ── Smoothstep over full range then slice ─────────────────────────────────────
t_full = np.arange(0.0, T + dt, dt)
x_full = t_full / T

alpha_s_full   = x_full**2 * (3 - 2*x_full)
dalpha_s_full  = np.gradient(alpha_s_full,  dt)
d2alpha_s_full = np.gradient(dalpha_s_full, dt)

# Zoom mask (t=0 to t_end)
mask = t_full <= t_end
t_z       = t_full[mask]
alpha_z   = alpha_s_full[mask]
dalpha_z  = dalpha_s_full[mask]
d2alpha_z = d2alpha_s_full[mask]

# Time axis for the "before" segment (pre)
t_pre = np.linspace(-pre_t, 0.0, int(pre_t / dt))

# Analytical peaks
peak_vel = 1.5 / T
peak_acc = 6.0 / T**2

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREY   = "#6B7280"
LBLUE  = "#DBEAFE"
LORAN  = "#FFEDD5"

fig = plt.figure(figsize=(7, 9))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

xlim = (-pre_t, t_end)

def xdec(ax):
    ax.axvline(0, color=GREY, ls=":", lw=1.2, alpha=0.8, zorder=1)
    ax.set_xlim(*xlim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.015 * t_end, ax.get_ylim()[0],
            "transition\nstart", fontsize=7.5, color=GREY, va="bottom")

# ── Row 0: α ──────────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])

# Smoothstep before t=0 (α=0)
ax0.plot(t_pre, np.zeros_like(t_pre), color=BLUE, lw=2.2, label="Smoothstep")
ax0.plot(t_z,   alpha_z,              color=BLUE, lw=2.2)

# Discrete: 0 before, 1 after
ax0.plot(t_pre,          np.zeros_like(t_pre),    color=ORANGE, lw=2.2, ls="--", label="Discrete")
ax0.plot([0.0, t_end],   [1.0, 1.0],              color=ORANGE, lw=2.2, ls="--")
ax0.plot([0.0, 0.0],     [0.0, 1.0],              color=ORANGE, lw=1.6, ls="--", alpha=0.6)
ax0.scatter([0], [0], color=ORANGE, s=30, zorder=5, facecolors="white", edgecolors=ORANGE)
ax0.scatter([0], [1], color=ORANGE, s=30, zorder=5)

ax0.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.4)
ax0.axhline(1, color=GREY, lw=0.6, ls="--", alpha=0.4)
ax0.set_ylim(-0.14, 1.28)
ax0.set_ylabel(r"$\alpha$  (blending weight)", fontsize=10)
ax0.set_title("Transition Zoom: Discrete vs Smoothstep", fontsize=12, fontweight="bold")
ax0.legend(fontsize=9, loc="upper left", framealpha=0.8)
xdec(ax0)

# ── Row 1: α̇ ──────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1])

# Smoothstep before (α̇=0) and after
ax1.plot(t_pre, np.zeros_like(t_pre), color=BLUE, lw=2.2)
ax1.fill_between(t_z, 0, dalpha_z, color=LBLUE, alpha=0.40)
ax1.plot(t_z,   dalpha_z,            color=BLUE, lw=2.2, label="Smoothstep")

# Discrete: impulse at t=0 (annotated arrow)
ax1.plot(t_pre, np.zeros_like(t_pre), color=ORANGE, lw=2.2, ls="--", label="Discrete")
ax1.plot([dt, t_end], [0.0, 0.0],    color=ORANGE, lw=2.2, ls="--")

arrow_h = peak_vel * 14
ax1.annotate("", xy=(0, arrow_h * 0.80), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax1.plot([-t_end*0.03, t_end*0.03], [arrow_h*0.86]*2, color=ORANGE, lw=1.4, ls="--")
ax1.text(t_end*0.04, arrow_h * 0.58,
         r"$\delta(t)$: $+\infty$", color=ORANGE, fontsize=9, va="center")

# Annotate smoothstep peak
ax1.annotate(
    r"$\dot{\alpha}_{\max}=\frac{3}{2T}=" + f"{peak_vel:.3f}" + r"\,\mathrm{s}^{-1}$" + f"\n(at t=T/2={T/2:.1f}s)",
    xy=(t_z[-1], dalpha_z[-1]),
    xytext=(t_end * 0.50, peak_vel * 1.22),
    fontsize=8.5, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
)

ax1.set_ylim(-arrow_h * 0.12, arrow_h * 1.22)
ax1.set_ylabel(r"$\dot{\alpha}$  (s$^{-1}$)", fontsize=10)
ax1.legend(fontsize=9, loc="upper right", framealpha=0.8)
xdec(ax1)

# ── Row 2: α̈ ──────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])

# Smoothstep before (α̈=0) and after
ax2.plot(t_pre, np.zeros_like(t_pre), color=BLUE, lw=2.2)
ax2.fill_between(t_z, 0, d2alpha_z, where=(d2alpha_z >= 0), color=LBLUE, alpha=0.45)
ax2.fill_between(t_z, 0, d2alpha_z, where=(d2alpha_z <  0), color=LORAN, alpha=0.45)
ax2.plot(t_z,   d2alpha_z,           color=BLUE, lw=2.2, label="Smoothstep")

# Discrete: doublet at t=0 (two opposing arrows)
ax2.plot(t_pre, np.zeros_like(t_pre), color=ORANGE, lw=2.2, ls="--", label="Discrete")
ax2.plot([dt, t_end], [0.0, 0.0],    color=ORANGE, lw=2.2, ls="--")

arrow_h2 = peak_acc * 14
ax2.annotate("", xy=(0,  arrow_h2 * 0.72), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax2.annotate("", xy=(0, -arrow_h2 * 0.72), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax2.plot([-t_end*0.03, t_end*0.03], [ arrow_h2*0.78]*2, color=ORANGE, lw=1.4, ls="--")
ax2.plot([-t_end*0.03, t_end*0.03], [-arrow_h2*0.78]*2, color=ORANGE, lw=1.4, ls="--")
ax2.text(t_end*0.04,  arrow_h2*0.50, r"$+\infty$", color=ORANGE, fontsize=9, va="center")
ax2.text(t_end*0.04, -arrow_h2*0.50, r"$-\infty$", color=ORANGE, fontsize=9, va="center")

# Annotate smoothstep onset
ax2.annotate(
    r"$\ddot{\alpha}_{t=0}=\frac{6}{T^2}=" + f"{peak_acc:.2f}" + r"\,\mathrm{s}^{-2}$",
    xy=(0, peak_acc),
    xytext=(t_end * 0.20, peak_acc * 1.22),
    fontsize=8.5, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
)

ax2.set_ylim(-arrow_h2 * 1.05, arrow_h2 * 1.05)
ax2.set_xlabel("Time (s)", fontsize=10)
ax2.set_ylabel(r"$\ddot{\alpha}$  (s$^{-2}$)", fontsize=10)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.8)
xdec(ax2)

# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.52, -0.01,
         f"Showing t ∈ [{-pre_t:.2f}, {t_end:.2f}] s   (T = {T} s, zoom = first {zoom*100:.0f}% of transition)",
         ha="center", fontsize=8.5, color=GREY)

fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
