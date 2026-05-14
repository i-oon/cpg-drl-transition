"""
Base-swap validation figure (Silver et al. style).

Shows that the trained Res-α 12D MLP is genuinely residual to Smoothstep:
when the base is replaced with linear ramp at evaluation time (no retraining),
the MLP's corrections become miscalibrated — it remains active but hurts on
4/6 gait pairs, raising mean jerk_TRANS by +12% above the linear-ramp baseline.

With the correct base (Smoothstep), the MLP reduces mean jerk by −6.5%.
The divergence between SS+MLP and LR+MLP proves the corrections are
Smoothstep-specific and cannot compensate for a mismatched base schedule.

Three panels:
  Top    — effective α(t) for all four cases on bound→trot (window 6)
  Middle — per-gait-pair jerk_TRANS for the 2×2 base × MLP comparison
  Bottom — 2×2 scalar mean table as an annotated heatmap

Usage:
    python scripts/plot_base_swap.py
    python scripts/plot_base_swap.py --out logs/phase2/base_swap_validation.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="logs/phase2/base_swap_validation.png")
args = parser.parse_args()

dt = 0.02
TRANS_S = 3.0
TRANS_STEPS = int(TRANS_S / dt)
WIN_STEPS   = TRANS_STEPS + 25
PRE_S       = 0.5
POST_S      = 1.0

PAIRS = ["trot→bound", "bound→pace", "pace→trot",
         "trot→pace",  "pace→bound", "bound→trot"]

# ── Data paths ────────────────────────────────────────────────────────────────
PATHS = {
    "LR base, Δα=0":  "logs/phase2/baselines/linear_ramp/playback_seed42.csv",
    "LR base + MLP":  "logs/phase2/residual_alpha_12d/playback_seed42_linearbase.csv",
    "SS base, Δα=0":  "logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv",
    "SS base + MLP":  "logs/phase2/residual_alpha_12d/playback_seed42.csv",
}

COLORS = {
    "LR base, Δα=0": "#ff7f0e",
    "LR base + MLP":  "#ffbb78",
    "SS base, Δα=0":  "#2ca02c",
    "SS base + MLP":  "#1f77b4",
}
STYLES = {
    "LR base, Δα=0": ("--", 1.6),
    "LR base + MLP":  (":",  2.2),
    "SS base, Δα=0":  ("--", 1.6),
    "SS base + MLP":  ("-",  2.4),
}

# ── Load helper ───────────────────────────────────────────────────────────────
def get_starts(ab):
    starts = []
    for i in range(1, len(ab) - WIN_STEPS):
        if (ab[i] - ab[i-1]) > 0.5 or ((ab[i-1] < 0.02) and (ab[i] >= 0.02)):
            starts.append(i)
    return starts

def jerk_per_window(path):
    df = pd.read_csv(path)
    ab = df["alpha_base"].values
    ja = df[[f"ja{i}" for i in range(12)]].values
    j  = np.sqrt(np.mean((np.diff(ja, axis=0) / dt) ** 2, axis=1))
    return [float(np.sqrt(np.mean(j[s:s+TRANS_STEPS]**2)))
            for s in get_starts(ab) if s+TRANS_STEPS <= len(j)]

datasets = {}
for label, path in PATHS.items():
    df = pd.read_csv(path)
    ab = df["alpha_base"].values
    t  = np.arange(len(ab)) * dt
    if all(f"d{i}" in df.columns for i in range(12)):
        da = df[[f"d{i}" for i in range(12)]].values.mean(axis=1)
    else:
        da = np.zeros(len(ab))
    alpha_eff = ab + da
    datasets[label] = dict(t=t, ab=ab, alpha_eff=alpha_eff,
                           da=da, starts=get_starts(ab))

jerks = {label: jerk_per_window(path) for label, path in PATHS.items()}

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 11))
gs  = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 1.4],
                       hspace=0.45, wspace=0.32,
                       left=0.09, right=0.97, top=0.93, bottom=0.07)

ax_alpha = fig.add_subplot(gs[0, :])
ax_jerk  = fig.add_subplot(gs[1, :])
ax_heat  = fig.add_subplot(gs[2, :])

# ── Panel 1: α(t) on bound→trot (window 6) — one of 2 pairs where LR+MLP helps ──
WIN_IDX = 5   # bound→trot

for label, d in datasets.items():
    if WIN_IDX >= len(d["starts"]):
        continue
    s  = d["starts"][WIN_IDX]
    ts = d["t"][s]
    lo = max(0, s - int(PRE_S / dt))
    hi = min(len(d["t"]) - 1, s + TRANS_STEPS + int(POST_S / dt))
    mask  = np.arange(lo, hi)
    t_rel = d["t"][mask] - ts
    ls, lw = STYLES[label]

    ax_alpha.plot(t_rel, d["ab"][mask],
                  color=COLORS[label], ls="--", lw=1.0, alpha=0.35)
    ax_alpha.plot(t_rel, d["alpha_eff"][mask],
                  color=COLORS[label], ls=ls, lw=lw, label=label)

ax_alpha.axvline(0,      color="red",    lw=1.5, ls=":", alpha=0.8)
ax_alpha.axvline(TRANS_S, color="orange", lw=1.2, ls="--", alpha=0.6)
ax_alpha.axvspan(0, TRANS_S, color="gold", alpha=0.07, zorder=0)
ax_alpha.set_xlim(-PRE_S, TRANS_S + POST_S)
ax_alpha.set_ylim(-0.05, 1.35)
ax_alpha.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_alpha.set_xlabel("Time relative to switch command [s]", fontsize=10)
ax_alpha.set_ylabel("Effective α\n(base + Δα)", fontsize=10)
ax_alpha.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
ax_alpha.grid(axis="y", alpha=0.25)

# Annotate SS+MLP peak Δα
d_mlp = datasets["SS base + MLP"]
s_ref  = d_mlp["starts"][WIN_IDX]
ts_ref = d_mlp["t"][s_ref]
t_win   = d_mlp["t"][s_ref:s_ref+TRANS_STEPS]
da_win  = d_mlp["da"][s_ref:s_ref+TRANS_STEPS]
peak_i  = np.argmax(da_win)
t_peak  = t_win[peak_i] - ts_ref
da_peak = da_win[peak_i]
aeff_peak = d_mlp["alpha_eff"][s_ref + peak_i]
if da_peak > 0.01:
    ax_alpha.annotate(
        f"SS+MLP: Δα = +{da_peak:.3f}",
        xy=(t_peak, aeff_peak),
        xytext=(t_peak + 0.4, aeff_peak + 0.12),
        fontsize=8.5, color=COLORS["SS base + MLP"],
        arrowprops=dict(arrowstyle="->", color=COLORS["SS base + MLP"], lw=1.1),
    )

# Annotate LR+MLP peak Δα — MLP is active but weaker
d_lr_mlp = datasets["LR base + MLP"]
if WIN_IDX < len(d_lr_mlp["starts"]):
    s_lr   = d_lr_mlp["starts"][WIN_IDX]
    ts_lr  = d_lr_mlp["t"][s_lr]
    t_win_lr  = d_lr_mlp["t"][s_lr:s_lr+TRANS_STEPS]
    da_win_lr = d_lr_mlp["da"][s_lr:s_lr+TRANS_STEPS]
    pk_lr     = np.argmax(da_win_lr)
    t_pk_lr   = t_win_lr[pk_lr] - ts_lr
    da_pk_lr  = da_win_lr[pk_lr]
    aeff_pk_lr = d_lr_mlp["alpha_eff"][s_lr + pk_lr]
    if da_pk_lr > 0.005:
        ax_alpha.annotate(
            f"LR+MLP: Δα = +{da_pk_lr:.3f}\n(weaker, miscalibrated)",
            xy=(t_pk_lr, aeff_pk_lr),
            xytext=(t_pk_lr - 1.1, aeff_pk_lr + 0.10),
            fontsize=8.0, color=COLORS["LR base + MLP"],
            arrowprops=dict(arrowstyle="->", color=COLORS["LR base + MLP"], lw=1.0),
        )

ax_alpha.set_title(
    f"Panel A — Effective α trajectory: bound→trot (window {WIN_IDX+1}/6)\n"
    "Dashed thin = base schedule  |  solid/styled = base + Δα\n"
    "LR+MLP is active (Δα>0) but weaker than SS+MLP — bound→trot is one of only 2/6 pairs where LR+MLP helps",
    fontsize=9, loc="left"
)

# ── Panel 2: per-gait-pair jerk_TRANS grouped bars ────────────────────────────
labels_order = ["LR base, Δα=0", "LR base + MLP", "SS base, Δα=0", "SS base + MLP"]
n_methods = len(labels_order)
x = np.arange(len(PAIRS))
bar_w = 0.18
offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * bar_w

for mi, label in enumerate(labels_order):
    vals = jerks[label]
    ax_jerk.bar(x + offsets[mi], vals, width=bar_w * 0.92,
                color=COLORS[label], alpha=0.85, label=label, zorder=3)

# Shade pairs where LR+MLP helps (pace→bound idx=4, bound→trot idx=5)
for good_idx in [4, 5]:
    ax_jerk.axvspan(good_idx - 0.45, good_idx + 0.45,
                    color="lightcyan", zorder=0, alpha=0.5)
ax_jerk.text(4, 1000, "LR+MLP\nhelps", ha="center", va="bottom", fontsize=7, color="steelblue")
ax_jerk.text(5, 1000, "LR+MLP\nhelps", ha="center", va="bottom", fontsize=7, color="steelblue")

ax_jerk.set_xticks(x)
ax_jerk.set_xticklabels(PAIRS, fontsize=9)
ax_jerk.set_ylabel("Transition-window jerk RMS [rad/s³]", fontsize=10)
ax_jerk.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
ax_jerk.grid(axis="y", alpha=0.25, zorder=0)
ax_jerk.set_title(
    "Panel B — Per-gait-pair jerk_TRANS: 2×2 (base schedule) × (MLP on/off)\n"
    "LR+MLP hurts on 4/6 pairs (bars taller) — MLP is active but Smoothstep-calibrated corrections misfire on linear-ramp base\n"
    "Highlighted pairs (pace→bound, bound→trot): the only 2/6 where LR+MLP incidentally helps",
    fontsize=9, loc="left"
)

# ── Panel 3: 2×2 scalar heatmap ───────────────────────────────────────────────
means = np.array([
    [np.mean(jerks["LR base, Δα=0"]),  np.mean(jerks["SS base, Δα=0"])],
    [np.mean(jerks["LR base + MLP"]),  np.mean(jerks["SS base + MLP"])],
])
im = ax_heat.imshow(means, cmap="RdYlGn_r", aspect="auto",
                    vmin=6500, vmax=9500)

# Compute percentage changes vs no-MLP row
lr_no_mlp = means[0, 0]
ss_no_mlp = means[0, 1]
pct = [[0.0, 0.0],
       [(means[1,0] - lr_no_mlp) / lr_no_mlp * 100,
        (means[1,1] - ss_no_mlp) / ss_no_mlp * 100]]

for r in range(2):
    for c in range(2):
        val = means[r, c]
        bg  = plt.cm.RdYlGn_r((val - 6500) / 3000)
        lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
        txt_color = "white" if lum < 0.5 else "black"
        sign = f"+{pct[r][c]:.1f}%" if pct[r][c] > 0 else f"{pct[r][c]:.1f}%"
        label_txt = f"{val:.0f}"
        if r == 1:
            label_txt += f"\n({sign})"
        ax_heat.text(c, r, label_txt, ha="center", va="center",
                     fontsize=14, fontweight="bold", color=txt_color)

ax_heat.set_xticks([0, 1])
ax_heat.set_xticklabels(["Linear ramp base", "Smoothstep base"], fontsize=11)
ax_heat.set_yticks([0, 1])
ax_heat.set_yticklabels(["Δα = 0\n(no MLP)", "Δα = MLP output"], fontsize=11)
ax_heat.set_title(
    "Panel C — Mean jerk_TRANS (N=6)  |  lower = better\n"
    "MLP with correct base (SS): −6.5%  |  MLP with wrong base (LR): +12.2% WORSE — corrections actively misfire",
    fontsize=9, loc="left"
)
plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02,
             label="jerk_TRANS [rad/s³]")

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Base-swap validation: the trained Res-α 12D MLP is Smoothstep-specific\n"
    "With correct base (SS): −6.5% jerk  |  With wrong base (LR, no retraining): +12.2% jerk — corrections misfire on 4/6 pairs",
    fontsize=10, y=0.975
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
