"""
Blending Schedule Kinematics: α, α̇, α̈ for Discrete Switch vs Smoothstep.

From:  q_blend = (1−α)·q_src + α·q_tgt
  →  q̇_blend ≈ α̇·Δq
  →  q̈_blend ≈ α̈·Δq

Layout: 3 rows × 2 columns
  Rows: α (position), α̇ (velocity), α̈ (acceleration)
  Cols: Discrete Switch | Smoothstep

Usage:
    python scripts/plot_blending_kinematics.py
    python scripts/plot_blending_kinematics.py --T 1.5 --out logs/phase2_v3/blending_kin_1p5s.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--T",   type=float, default=3.0, help="Transition duration (s)")
parser.add_argument("--dq",  type=float, default=1.0, help="Phase gap Δq (rad) — scales y-axis labels")
parser.add_argument("--out", default="logs/phase2_v3/blending_kinematics.png")
args = parser.parse_args()

T   = args.T
dq  = args.dq
dt  = 1e-4
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

# ── Smoothstep: α and derivatives ────────────────────────────────────────────
t  = np.arange(0.0, T + dt, dt)
x  = t / T

alpha_s   = x**2 * (3 - 2*x)
dalpha_s  = np.gradient(alpha_s,  dt)
d2alpha_s = np.gradient(dalpha_s, dt)

# Analytical peaks
peak_vel = 1.5 / T          # α̇_max at T/2
peak_acc = 6.0 / T**2       # α̈_max at t=0

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREY   = "#6B7280"
LBLUE  = "#DBEAFE"
LORAN  = "#FFEDD5"

fig = plt.figure(figsize=(10, 9))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.60, wspace=0.40)

# helper: shared decorations
def vlines(ax, y0):
    ax.axvline(0, color=GREY, ls=":", lw=1.0, alpha=0.65)
    ax.axvline(T, color=GREY, ls=":", lw=1.0, alpha=0.65)
    ax.text(T*0.02,  y0, "start", fontsize=7.5, color=GREY, va="bottom")
    ax.text(T*0.88,  y0, "end",   fontsize=7.5, color=GREY, va="bottom")

# ═══════════════════════════════════════════════════════════════════════════════
# Column 0 — Discrete Switch
# ═══════════════════════════════════════════════════════════════════════════════

# ── Row 0: α (Heaviside step) ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
pre = np.linspace(-T*0.18, 0, 200)
ax.plot(pre, np.zeros_like(pre), color=ORANGE, lw=2.2)
ax.plot(t,   np.ones_like(t),   color=ORANGE, lw=2.2)
ax.plot([0, 0], [0, 1], color=ORANGE, lw=2.2, ls="--", alpha=0.55)
ax.scatter([0], [0], color=ORANGE, s=28, zorder=5, facecolors="white", edgecolors=ORANGE)
ax.scatter([0], [1], color=ORANGE, s=28, zorder=5)
ax.set_xlim(-T*0.18, T*1.06)
ax.set_ylim(-0.18, 1.32)
ax.set_ylabel(r"$\alpha$  (blending weight)", fontsize=9)
ax.set_title("Discrete Switch", fontsize=11, fontweight="bold", color=ORANGE)
ax.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.4)
ax.axhline(1, color=GREY, lw=0.6, ls="--", alpha=0.4)
vlines(ax, -0.14)
ax.spines[["top", "right"]].set_visible(False)

# ── Row 1: α̇ (Dirac δ impulse) ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
arrow_h = peak_vel * 18
ax.plot([-T*0.18, T], [0, 0], color=GREY, lw=1.2)
ax.annotate("", xy=(0, arrow_h * 0.82), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax.plot([-T*0.02, T*0.02], [arrow_h*0.87]*2, color=ORANGE, lw=1.4, ls="--")
ax.text(T*0.04, arrow_h*0.60,
        r"$\delta(t)$" + "\n" + r"$(+\infty$ at $t\!=\!0)$",
        color=ORANGE, fontsize=9, va="center")
ax.set_xlim(-T*0.18, T*1.06)
ax.set_ylim(-arrow_h*0.14, arrow_h*1.22)
ax.set_ylabel(r"$\dot{\alpha}$  (s$^{-1}$)", fontsize=9)
ax.axvline(0, color=GREY, ls=":", lw=1.0, alpha=0.65)
ax.spines[["top", "right"]].set_visible(False)

# ── Row 2: α̈ (derivative of δ — doublet) ─────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
arrow_h2 = peak_acc * 18
ax.plot([-T*0.18, T], [0, 0], color=GREY, lw=1.2)
ax.annotate("", xy=(0,  arrow_h2*0.72), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax.annotate("", xy=(0, -arrow_h2*0.72), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4, mutation_scale=15))
ax.plot([-T*0.02, T*0.02], [ arrow_h2*0.78]*2, color=ORANGE, lw=1.4, ls="--")
ax.plot([-T*0.02, T*0.02], [-arrow_h2*0.78]*2, color=ORANGE, lw=1.4, ls="--")
ax.text(T*0.04,  arrow_h2*0.50, r"$+\infty$", color=ORANGE, fontsize=9, va="center")
ax.text(T*0.04, -arrow_h2*0.50, r"$-\infty$", color=ORANGE, fontsize=9, va="center")
ax.set_xlim(-T*0.18, T*1.06)
ax.set_ylim(-arrow_h2*1.05, arrow_h2*1.05)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(r"$\ddot{\alpha}$  (s$^{-2}$)", fontsize=9)
ax.axvline(0, color=GREY, ls=":", lw=1.0, alpha=0.65)
ax.spines[["top", "right"]].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# Column 1 — Smoothstep
# ═══════════════════════════════════════════════════════════════════════════════

# ── Row 0: α ──────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.fill_between(t, 0, alpha_s, color=LBLUE, alpha=0.40)
ax.plot(t, alpha_s, color=BLUE, lw=2.2)
ax.axhline(0, color=GREY, lw=0.6, ls="--", alpha=0.4)
ax.axhline(1, color=GREY, lw=0.6, ls="--", alpha=0.4)
ax.set_xlim(-T*0.06, T*1.06)
ax.set_ylim(-0.10, 1.22)
ax.set_title(f"Smoothstep  (T = {T} s)", fontsize=11, fontweight="bold", color=BLUE)
ax.annotate(r"$\alpha=3x^2-2x^3$", xy=(T*0.48, 0.50),
            xytext=(T*0.60, 0.22), fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
vlines(ax, -0.08)
ax.spines[["top", "right"]].set_visible(False)

# ── Row 1: α̇ ──────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.fill_between(t, 0, dalpha_s, color=LBLUE, alpha=0.40)
ax.plot(t, dalpha_s, color=BLUE, lw=2.2)
ax.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5)
ax.set_xlim(-T*0.06, T*1.06)
vlines(ax, dalpha_s.min() * 0.08)
ax.annotate(
    r"$\dot{\alpha}_{\max}=\dfrac{3}{2T}=" + f"{peak_vel:.3f}" + r"\,\mathrm{s}^{-1}$",
    xy=(T/2, peak_vel),
    xytext=(T*0.56, peak_vel * 1.20),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
)
ax.set_ylabel(r"$\dot{\alpha}$  (s$^{-1}$)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── Row 2: α̈ ──────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
ax.fill_between(t, 0, d2alpha_s, where=(d2alpha_s >= 0), color=LBLUE, alpha=0.45)
ax.fill_between(t, 0, d2alpha_s, where=(d2alpha_s <  0), color=LORAN, alpha=0.45)
ax.plot(t, d2alpha_s, color=BLUE, lw=2.2)
ax.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.5)
ax.set_xlim(-T*0.06, T*1.06)
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel(r"$\ddot{\alpha}$  (s$^{-2}$)", fontsize=9)
vlines(ax, -peak_acc * 0.12)
ax.annotate(
    r"$\ddot{\alpha}_{\max}=\dfrac{6}{T^2}=" + f"{peak_acc:.2f}" + r"\,\mathrm{s}^{-2}$",
    xy=(0, peak_acc),
    xytext=(T*0.16, peak_acc * 1.18),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
)
ax.annotate(
    r"$\ddot{\alpha}_{\min}=-\dfrac{6}{T^2}$",
    xy=(T/2, -peak_acc),
    xytext=(T*0.58, -peak_acc * 1.32),
    fontsize=9, color=ORANGE,
    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0),
)
ax.spines[["top", "right"]].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    r"Blending Schedule Kinematics:  $\alpha$,  $\dot{\alpha}$,  $\ddot{\alpha}$"
    + f"\n" + r"$q_{\mathrm{blend}} = (1-\alpha)\,q_{\mathrm{src}} + \alpha\,q_{\mathrm{tgt}}$",
    fontsize=11, y=1.02,
)

fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
