"""
CPG-RBF visualization: why these phase offsets for each gait?

Produces a figure with four panels per gait (walk / trot):
  1. Unit circle — where each leg's phase point sits at t=0
  2. RBF activation heatmap — which neurons fire for each leg over one cycle
  3. Dominant neuron index over time — shows the "phase clock" per leg
  4. Gait diagram — inferred stance (dark) / swing (light) timing

Run from project root:
    conda activate env_isaaclab
    python scripts/visualize_cpg.py

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from networks.cpg_rbf import CPGRBFNetwork, PHASE_OFFSETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEG_LABELS = ["FL", "FR", "RL", "RR"]
LEG_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d"]  # distinct per leg

GAITS = ["walk", "trot", "steer"]
GAIT_TITLES = {
    "walk":  "Walk  (FL=0°, FR=180°, RL=90°, RR=270°) — 4-beat, one leg at a time",
    "trot":  "Trot  (FL=0°, FR=180°, RL=180°, RR=0°)  — 2-beat, diagonal pairs",
    "steer": "Steer (FL=0°, FR=180°, RL=90°, RR=270°)  — same timing as walk, W matrix creates left/right asymmetry",
}

# Steer groups legs by side for the asymmetry panel
LEFT_LEGS  = [0, 2]   # FL, RL indices
RIGHT_LEGS = [1, 3]   # FR, RR indices

# Heuristic: neuron index < 10 → roughly "swing phase" (upper half of circle),
# neuron index ≥ 10 → roughly "stance phase" (lower half).
# RBF_0 = phase 0° (touchdown), RBF_10 = phase 180° (liftoff)
STANCE_THRESHOLD = 10   # neurons 0–9 ≈ stance, 10–19 ≈ swing


def run_cpg_one_cycle(phase_offsets: np.ndarray, n_cycles: int = 2):
    """
    Step CPG for n_cycles and return per-leg RBF activations over time.

    Returns
    -------
    rbf_history : (T, 4, 20)  RBF activations per timestep, leg, neuron
    dominant    : (T, 4)      Index of most-active RBF neuron per leg
    """
    net = CPGRBFNetwork()
    steps_per_cycle = int(1.0 / (net.freq * net.dt))   # ≈ 167
    T = steps_per_cycle * n_cycles

    rbf_history = np.zeros((T, 4, 20))
    dominant = np.zeros((T, 4), dtype=int)

    for t in range(T):
        net._step_oscillator()
        for k, theta in enumerate(phase_offsets):
            x_rot, y_rot = net._rotate_state(float(theta))
            rbf = net._compute_rbf(x_rot, y_rot).numpy()
            rbf_history[t, k] = rbf
            dominant[t, k] = int(np.argmax(rbf))

    return rbf_history, dominant, steps_per_cycle


def get_phase_points(phase_offsets: np.ndarray):
    """
    Return (x, y) on the unit circle for each leg's phase offset at t=0.
    Used for the circle plot.
    """
    xs = np.cos(phase_offsets)
    ys = np.sin(phase_offsets)
    return xs, ys


def infer_stance(dominant: np.ndarray) -> np.ndarray:
    """
    Heuristic stance detection from dominant RBF neuron index.
    Neurons 0–9  → stance  (lower arc: touchdown to mid-stance to liftoff)
    Neurons 10–19 → swing  (upper arc: liftoff to mid-swing to touchdown)

    Returns boolean array (T, 4): True = stance.
    """
    return dominant < STANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_steer_asymmetry(ax, rbf_history: np.ndarray, steps_per_cycle: int):
    """
    Extra panel only for steer: compare mean RBF activation per neuron for
    left legs (FL, RL) vs right legs (FR, RR) over one cycle.

    Both sides have identical activations because the W matrix — not the phase
    offsets — is what introduces left/right asymmetry during turning.
    This panel makes that explicit: the bars are equal, the arrow labels explain
    where the asymmetry actually lives.
    """
    T = steps_per_cycle   # one cycle

    left_mean  = rbf_history[:T, LEFT_LEGS, :].mean(axis=(0, 1))   # (20,)
    right_mean = rbf_history[:T, RIGHT_LEGS, :].mean(axis=(0, 1))  # (20,)

    neuron_idx = np.arange(20)
    bar_w = 0.35

    ax.bar(neuron_idx - bar_w / 2, left_mean,  bar_w,
           color="#e63946", alpha=0.75, label="Left  (FL, RL)")
    ax.bar(neuron_idx + bar_w / 2, right_mean, bar_w,
           color="#457b9d", alpha=0.75, label="Right (FR, RR)")

    ax.set_xlabel("RBF neuron index", fontsize=7)
    ax.set_ylabel("Mean activation", fontsize=7)
    ax.set_title("Left vs Right RBF activations\n(identical → asymmetry lives in W, not phase)", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(axis="both", labelsize=6)

    # Annotation box
    ax.text(0.5, 0.62,
            "Same phase offsets as walk\n"
            "→ left/right see identical RBFs\n\n"
            "Turning comes from W_steer:\n"
            "  inner legs  →  smaller amplitude\n"
            "  outer legs  →  larger amplitude",
            transform=ax.transAxes, fontsize=7,
            va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.85))


def plot_gait(ax_circle, ax_heatmap, ax_clock, ax_gait, gait_name):
    offsets = PHASE_OFFSETS[gait_name]
    rbf_history, dominant, steps_per_cycle = run_cpg_one_cycle(offsets, n_cycles=2)
    T = rbf_history.shape[0]
    time_axis = np.arange(T) / steps_per_cycle   # in cycles

    # ---- 1. Unit circle -----------------------------------------------
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax_circle.plot(np.cos(theta_circle), np.sin(theta_circle),
                   color="lightgray", lw=1.5, zorder=0)

    # RBF centers
    center_angles = 2 * np.pi * np.arange(20) / 20
    ax_circle.scatter(np.cos(center_angles), np.sin(center_angles),
                      s=18, color="lightgray", zorder=1)

    # Label key neurons
    for idx, label in [(0, "RBF_0\n(touchdown)"),
                       (5, "RBF_5\n(stance)"),
                       (10, "RBF_10\n(liftoff)"),
                       (15, "RBF_15\n(swing)")]:
        ax_circle.annotate(label,
                           xy=(np.cos(center_angles[idx]), np.sin(center_angles[idx])),
                           fontsize=5.5, ha="center", va="center",
                           color="gray",
                           xytext=(1.28 * np.cos(center_angles[idx]),
                                   1.28 * np.sin(center_angles[idx])))

    # Leg phase points
    xs, ys = get_phase_points(offsets)
    for k in range(4):
        ax_circle.scatter(xs[k], ys[k], s=120, color=LEG_COLORS[k],
                          zorder=3, edgecolors="white", linewidths=0.8)
        ax_circle.annotate(LEG_LABELS[k],
                           xy=(xs[k], ys[k]),
                           xytext=(xs[k] * 1.18, ys[k] * 1.18),
                           fontsize=8, fontweight="bold",
                           color=LEG_COLORS[k], ha="center", va="center")

    # Phase offset arrows from origin
    for k in range(4):
        ax_circle.annotate("", xy=(xs[k] * 0.82, ys[k] * 0.82),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->",
                                          color=LEG_COLORS[k], lw=1.2))

    ax_circle.set_xlim(-1.55, 1.55)
    ax_circle.set_ylim(-1.55, 1.55)
    ax_circle.set_aspect("equal")
    ax_circle.axis("off")
    ax_circle.set_title("Phase offsets\non unit circle", fontsize=8)

    # ---- 2. RBF activation heatmap (first 2 cycles, all legs stacked) ----
    # Stack legs vertically: FL on top, RR on bottom
    combined = np.concatenate([rbf_history[:, k, :] for k in range(4)], axis=1)
    # Shape: (T, 80) — 4 legs × 20 neurons

    im = ax_heatmap.imshow(combined.T, aspect="auto", origin="lower",
                           cmap="hot", interpolation="nearest",
                           extent=[0, time_axis[-1], 0, 80])
    # Leg boundary lines
    for k in range(1, 4):
        ax_heatmap.axhline(k * 20, color="white", lw=0.8, ls="--")
    # Leg labels on y-axis
    ax_heatmap.set_yticks([10, 30, 50, 70])
    ax_heatmap.set_yticklabels(LEG_LABELS, fontsize=7)
    ax_heatmap.set_xlabel("Cycle", fontsize=7)
    ax_heatmap.set_title("RBF activations per leg\n(hot = most active)", fontsize=8)
    ax_heatmap.tick_params(axis="both", labelsize=7)

    # ---- 3. Dominant neuron index over time ----------------------------
    for k in range(4):
        ax_clock.plot(time_axis, dominant[:, k],
                      color=LEG_COLORS[k], lw=1.2, label=LEG_LABELS[k],
                      alpha=0.85)
    ax_clock.axhline(STANCE_THRESHOLD, color="gray", lw=0.8, ls=":",
                     label=f"stance/swing boundary (RBF {STANCE_THRESHOLD})")
    ax_clock.set_ylim(-1, 21)
    ax_clock.set_yticks([0, 5, 10, 15, 19])
    ax_clock.set_ylabel("Dominant\nRBF neuron", fontsize=7)
    ax_clock.set_xlabel("Cycle", fontsize=7)
    ax_clock.set_title("Phase clock per leg", fontsize=8)
    ax_clock.legend(fontsize=6, ncol=2, loc="upper right")
    ax_clock.tick_params(axis="both", labelsize=7)

    # ---- 4. Gait diagram (inferred stance/swing) ----------------------
    stance = infer_stance(dominant)   # (T, 4) bool

    for k in range(4):
        for t in range(T - 1):
            color = LEG_COLORS[k] if stance[t, k] else "white"
            ax_gait.barh(k, 1 / steps_per_cycle,
                         left=time_axis[t], height=0.6,
                         color=color, edgecolor="none")

    ax_gait.set_yticks(range(4))
    ax_gait.set_yticklabels(LEG_LABELS, fontsize=8)
    ax_gait.set_xlabel("Cycle", fontsize=7)
    ax_gait.set_title("Gait diagram\n(color = stance, white = swing)", fontsize=8)
    ax_gait.set_xlim(0, time_axis[-1])
    ax_gait.tick_params(axis="both", labelsize=7)

    # Add border to make white swing visible
    for spine in ax_gait.spines.values():
        spine.set_edgecolor("lightgray")

    return rbf_history, steps_per_cycle


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "CPG-RBF Phase Offsets: Walk / Trot / Steer\n"
    "Phase offset = rotate oscillator state on unit circle → selects which RBF neurons activate",
    fontsize=11, fontweight="bold", y=0.99
)

# Three rows (one per gait)
# Walk / Trot: 4 cols  (circle | heatmap | clock | gait diagram)
# Steer:       5 cols  (circle | heatmap | clock | gait diagram | asymmetry)
outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50)

ROW_Y = [0.83, 0.52, 0.19]   # approximate vertical centres for row labels

for row, gait in enumerate(GAITS):
    if gait == "steer":
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[row],
            width_ratios=[1, 2, 2, 2, 2], wspace=0.38
        )
        ax_circle  = fig.add_subplot(inner[0])
        ax_heatmap = fig.add_subplot(inner[1])
        ax_clock   = fig.add_subplot(inner[2])
        ax_gait    = fig.add_subplot(inner[3])
        ax_asym    = fig.add_subplot(inner[4])

        rbf_history, spc = plot_gait(ax_circle, ax_heatmap, ax_clock, ax_gait, gait)
        plot_steer_asymmetry(ax_asym, rbf_history, spc)
    else:
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[row],
            width_ratios=[1, 2, 2, 2], wspace=0.35
        )
        ax_circle  = fig.add_subplot(inner[0])
        ax_heatmap = fig.add_subplot(inner[1])
        ax_clock   = fig.add_subplot(inner[2])
        ax_gait    = fig.add_subplot(inner[3])
        plot_gait(ax_circle, ax_heatmap, ax_clock, ax_gait, gait)

    # Row label (rotated, left margin)
    fig.text(
        0.01, ROW_Y[row],
        gait.upper(), fontsize=13, fontweight="bold",
        rotation=90, va="center", color="#333333"
    )

    # Per-row subtitle above the circle
    ax_circle.set_title(
        GAIT_TITLES[gait] + "\n\nPhase offsets\non unit circle",
        fontsize=7.5, loc="left",
        transform=ax_circle.transAxes,
        x=-0.05, y=1.12
    )

out_path = Path("logs") / "cpg_phase_visualization.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
