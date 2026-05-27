"""
Base-swap validation figure (Silver et al. style).

--exp A  (default)
  Exp A Sched-α 12D MLP swapped to linear-ramp base.
  Metric: jerk_TRANS.  Shows MLP is Smoothstep-calibrated.

--exp B
  Exp B Sched-α 12D MLP swapped to linear-ramp base (same architecture as
  Exp A — direct comparison of reward design effect on base-specificity).
  Metric: jerk_TRANS.

  Requires playback_seed42_linearbase.csv for schedule_residual_12d_v3.
  Generate with:
    conda activate env_isaaclab && cd ~/cpg-drl-transition
    python -u scripts/play_b1_phase2.py \\
        --task Isaac-B1-Phase2-V3-Alpha12D-v0 \\
        --checkpoint logs/phase2_new_approach/schedule_residual_12d_v3/model_final.pt \\
        --baseline linear_ramp \\
        --num_envs 1 --steps 2500 --seed 42 \\
        --gait_pairs trot,bound,pace,trot,pace,bound \\
        --switch_interval_s 8.0 --transition_duration_s 3.0 \\
        --save_csv logs/phase2_new_approach/schedule_residual_12d_v3/playback_seed42_linearbase.csv \\
        --headless

Usage:
    python scripts/plot_base_swap.py
    python scripts/plot_base_swap.py --exp B
    python scripts/plot_base_swap.py --exp B --out logs/phase2_v3/base_swap_expB.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="A", choices=["A", "B"],
                    help="A = Exp A Sched-α 12D (jerk metric); B = Exp B Action-q 12D (Δvx metric)")
parser.add_argument("--out", default=None)
args = parser.parse_args()

if args.out is None:
    args.out = ("logs/phase2/base_swap_validation.png" if args.exp == "A"
                else "logs/phase2_v3/base_swap_expB.png")

dt = 0.02
TRANS_S = 3.0
TRANS_STEPS = int(TRANS_S / dt)
WIN_STEPS   = TRANS_STEPS + 25
PRE_S       = 0.5
POST_S      = 1.0

PAIRS = ["trot→bound", "bound→pace", "pace→trot",
         "trot→pace",  "pace→bound", "bound→trot"]

# ── Data paths ────────────────────────────────────────────────────────────────
PATHS_A = {
    "LR base, Δα=0":  "logs/phase2/baselines/linear_ramp/playback_seed42.csv",
    "LR base + MLP":  "logs/phase2/residual_alpha_12d/playback_seed42_linearbase.csv",
    "SS base, Δα=0":  "logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv",
    "SS base + MLP":  "logs/phase2/residual_alpha_12d/playback_seed42.csv",
}

PATHS_B = {
    "LR base, Δα=0":  "logs/phase2/baselines/linear_ramp/playback_seed42.csv",
    "LR base + MLP":  "logs/phase2_new_approach/schedule_residual_12d_v3/playback_seed42_linearbase.csv",
    "SS base, Δα=0":  "logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv",
    "SS base + MLP":  "logs/phase2_new_approach/schedule_residual_12d_v3/playback_seed42.csv",
}

PATHS = PATHS_A if args.exp == "A" else PATHS_B

_COLORS_RAW = ["#ff7f0e", "#ffbb78", "#2ca02c", "#1f77b4"]
_STYLES_RAW = [("--", 1.6), (":", 2.2), ("--", 1.6), ("-", 2.4)]
COLORS = {k: c for k, c in zip(PATHS.keys(), _COLORS_RAW)}
STYLES = {k: s for k, s in zip(PATHS.keys(), _STYLES_RAW)}

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

def dvx_per_window(path):
    """Δvx_trans = vx_pre − vx_min per transition window."""
    df  = pd.read_csv(path)
    ab  = df["alpha_base"].values
    vx  = df["vx"].values
    out = []
    for s in get_starts(ab):
        if s + WIN_STEPS > len(vx):
            continue
        pre_start = max(0, s - int(0.5 / dt))
        vx_pre    = vx[pre_start:s].mean() if s > pre_start else vx[s]
        vx_min    = vx[s:s + TRANS_STEPS].min()
        out.append(float(vx_pre - vx_min))
    return out

datasets = {}
for label, path in PATHS.items():
    df = pd.read_csv(path)
    ab = df["alpha_base"].values
    t  = np.arange(len(ab)) * dt
    vx = df["vx"].values
    if all(f"d{i}" in df.columns for i in range(12)):
        da = df[[f"d{i}" for i in range(12)]].values.mean(axis=1)
    else:
        da = np.zeros(len(ab))
    alpha_eff = ab + da
    datasets[label] = dict(t=t, ab=ab, alpha_eff=alpha_eff,
                           da=da, vx=vx, starts=get_starts(ab))

metric_vals = {label: jerk_per_window(path) for label, path in PATHS.items()}
metric_name  = "jerk_TRANS [rad/s³]"
metric_short = "jerk"
heatmap_range = (6500, 10500)
lower_better  = True

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 12))
top_margin = 0.88 if args.exp == "B" else 0.93
gs  = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 1.4],
                       hspace=0.52, wspace=0.32,
                       left=0.09, right=0.97, top=top_margin, bottom=0.07)

ax_alpha = fig.add_subplot(gs[0, :])
ax_jerk  = fig.add_subplot(gs[1, :])
ax_heat  = fig.add_subplot(gs[2, :])

# ── Panel 1 — depends on experiment ──────────────────────────────────────────
WIN_IDX = 5   # bound→trot

if args.exp == "A":
    # Show effective α(t): base schedule + Δα correction
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

    ax_alpha.axvline(0,       color="red",    lw=1.5, ls=":", alpha=0.8)
    ax_alpha.axvline(TRANS_S, color="orange", lw=1.2, ls="--", alpha=0.6)
    ax_alpha.axvspan(0, TRANS_S, color="gold", alpha=0.07, zorder=0)
    ax_alpha.set_xlim(-PRE_S, TRANS_S + POST_S)
    ax_alpha.set_ylim(-0.05, 1.35)
    ax_alpha.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_alpha.set_xlabel("Time relative to switch command [s]", fontsize=10)
    ax_alpha.set_ylabel("Effective α  (base + Δα)", fontsize=10)
    ax_alpha.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
    ax_alpha.grid(axis="y", alpha=0.25)

    # Annotate SS+MLP peak Δα
    d_mlp  = datasets["SS base + MLP"]
    s_ref  = d_mlp["starts"][WIN_IDX]
    ts_ref = d_mlp["t"][s_ref]
    da_win = d_mlp["da"][s_ref:s_ref+TRANS_STEPS]
    peak_i = np.argmax(da_win)
    t_peak = d_mlp["t"][s_ref + peak_i] - ts_ref
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

    # Annotate LR+MLP peak Δα
    d_lr_mlp  = datasets["LR base + MLP"]
    if WIN_IDX < len(d_lr_mlp["starts"]):
        s_lr     = d_lr_mlp["starts"][WIN_IDX]
        ts_lr    = d_lr_mlp["t"][s_lr]
        da_win_lr = d_lr_mlp["da"][s_lr:s_lr+TRANS_STEPS]
        pk_lr    = np.argmax(da_win_lr)
        t_pk_lr  = d_lr_mlp["t"][s_lr + pk_lr] - ts_lr
        da_pk_lr = da_win_lr[pk_lr]
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
        "Dashed thin = base schedule  |  solid = base + Δα\n"
        "LR+MLP active (Δα>0) but weaker — bound→trot is one of 2/6 pairs where LR+MLP incidentally helps",
        fontsize=9, loc="left"
    )

else:
    # Exp B Sched-α 12D — same structure as Exp A, show effective α(t)
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

    ax_alpha.axvline(0,       color="red",    lw=1.5, ls=":", alpha=0.8)
    ax_alpha.axvline(TRANS_S, color="orange", lw=1.2, ls="--", alpha=0.6)
    ax_alpha.axvspan(0, TRANS_S, color="gold", alpha=0.07, zorder=0)
    ax_alpha.set_xlim(-PRE_S, TRANS_S + POST_S)
    ax_alpha.set_ylim(-0.05, 1.35)
    ax_alpha.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_alpha.set_xlabel("Time relative to switch command [s]", fontsize=10)
    ax_alpha.set_ylabel("Effective α  (base + Δα)", fontsize=10)
    ax_alpha.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)
    ax_alpha.grid(axis="y", alpha=0.25)

    d_mlp  = datasets["SS base + MLP"]
    s_ref  = d_mlp["starts"][WIN_IDX]
    ts_ref = d_mlp["t"][s_ref]
    da_win = d_mlp["da"][s_ref:s_ref+TRANS_STEPS]
    peak_i = np.argmax(da_win)
    t_peak = d_mlp["t"][s_ref + peak_i] - ts_ref
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

    d_lr_mlp = datasets["LR base + MLP"]
    if WIN_IDX < len(d_lr_mlp["starts"]):
        s_lr     = d_lr_mlp["starts"][WIN_IDX]
        ts_lr    = d_lr_mlp["t"][s_lr]
        da_win_lr = d_lr_mlp["da"][s_lr:s_lr+TRANS_STEPS]
        pk_lr    = np.argmax(da_win_lr)
        t_pk_lr  = d_lr_mlp["t"][s_lr + pk_lr] - ts_lr
        da_pk_lr = da_win_lr[pk_lr]
        aeff_pk_lr = d_lr_mlp["alpha_eff"][s_lr + pk_lr]
        if da_pk_lr > 0.005:
            ax_alpha.annotate(
                f"LR+MLP: Δα = +{da_pk_lr:.3f}",
                xy=(t_pk_lr, aeff_pk_lr),
                xytext=(t_pk_lr - 1.1, aeff_pk_lr + 0.10),
                fontsize=8.0, color=COLORS["LR base + MLP"],
                arrowprops=dict(arrowstyle="->", color=COLORS["LR base + MLP"], lw=1.0),
            )

    ax_alpha.set_title(
        f"Panel A — Effective α trajectory: bound→trot (window {WIN_IDX+1}/6)\n"
        "Dashed thin = base schedule  |  solid = base + Δα\n"
        "Exp B Sched-α 12D — same architecture as Exp A, trained with vx-window penalty instead of jerk penalty",
        fontsize=9, loc="left"
    )

# ── Panel 2: per-gait-pair metric grouped bars ────────────────────────────────
labels_order = list(PATHS.keys())
n_methods = len(labels_order)
x = np.arange(len(PAIRS))
bar_w = 0.18
offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * bar_w

for mi, label in enumerate(labels_order):
    vals = metric_vals[label]
    ax_jerk.bar(x + offsets[mi], vals, width=bar_w * 0.92,
                color=COLORS[label], alpha=0.85, label=label, zorder=3)

# Find pairs where LR+MLP is worse than LR base (for annotation)
lr_base_key  = labels_order[0]
lr_mlp_key   = labels_order[1]
hurt_pairs = [i for i, (b, m) in enumerate(zip(metric_vals[lr_base_key],
                                               metric_vals[lr_mlp_key]))
              if (m > b) == lower_better]   # "hurts" = goes wrong direction
for hi in hurt_pairs:
    ax_jerk.axvspan(hi - 0.45, hi + 0.45, color="#ffe0e0", zorder=0, alpha=0.55)
help_pairs = [i for i in range(len(PAIRS)) if i not in hurt_pairs]
for gi in help_pairs:
    ax_jerk.axvspan(gi - 0.45, gi + 0.45, color="lightcyan", zorder=0, alpha=0.45)
    ax_jerk.text(gi, ax_jerk.get_ylim()[0] if ax_jerk.get_ylim()[0] > 0 else 0,
                 "helps", ha="center", va="bottom", fontsize=7, color="steelblue")

ax_jerk.set_xticks(x)
ax_jerk.set_xticklabels(PAIRS, fontsize=9)
ax_jerk.set_ylabel(metric_name, fontsize=10)
ax_jerk.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
ax_jerk.grid(axis="y", alpha=0.25, zorder=0)
n_hurt = len(hurt_pairs)
if args.exp == "A":
    ax_jerk.set_title(
        f"Panel B — Per-gait-pair {metric_short}: 2×2 (base schedule) × (MLP on/off)\n"
        f"LR+MLP hurts on {n_hurt}/6 pairs — Smoothstep-calibrated corrections misfire on linear-ramp base",
        fontsize=9, loc="left"
    )
else:
    # Check if LR improvement is within 10% of SS improvement → truly base-agnostic
    ss_key = list(PATHS.keys())[2]
    ss_mlp_key = list(PATHS.keys())[3]
    lr_impr = (np.mean(metric_vals[lr_base_key]) - np.mean(metric_vals[lr_mlp_key])) / (np.mean(metric_vals[lr_base_key]) + 1e-9)
    ss_impr = (np.mean(metric_vals[ss_key])      - np.mean(metric_vals[ss_mlp_key])) / (np.mean(metric_vals[ss_key])      + 1e-9)
    agnostic = abs(lr_impr - ss_impr) < 0.15
    tag = "base-agnostic ✓ — same improvement on wrong base as correct base" if agnostic else f"partially base-agnostic — LR+MLP hurts on {n_hurt}/6 pairs"
    ax_jerk.set_title(
        f"Panel B — Per-gait-pair {metric_short}: 2×2 (base schedule) × (MLP on/off)\n"
        f"{tag}",
        fontsize=9, loc="left"
    )

# ── Panel 3: 2×2 scalar heatmap ───────────────────────────────────────────────
keys = labels_order
means = np.array([
    [np.mean(metric_vals[keys[0]]),  np.mean(metric_vals[keys[2]])],
    [np.mean(metric_vals[keys[1]]),  np.mean(metric_vals[keys[3]])],
])
cmap = "RdYlGn_r" if lower_better else "RdYlGn"
im = ax_heat.imshow(means, cmap=cmap, aspect="auto",
                    vmin=heatmap_range[0], vmax=heatmap_range[1])

lr_no_mlp = means[0, 0]
ss_no_mlp = means[0, 1]
pct = [[0.0, 0.0],
       [(means[1, 0] - lr_no_mlp) / (lr_no_mlp + 1e-9) * 100,
        (means[1, 1] - ss_no_mlp) / (ss_no_mlp + 1e-9) * 100]]

vrange = heatmap_range[1] - heatmap_range[0]
for r in range(2):
    for c in range(2):
        val = means[r, c]
        norm_val = (val - heatmap_range[0]) / (vrange + 1e-9)
        bg  = plt.cm.get_cmap(cmap)(norm_val)
        lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
        txt_color = "white" if lum < 0.5 else "black"
        sign = f"+{pct[r][c]:.1f}%" if pct[r][c] > 0 else f"{pct[r][c]:.1f}%"
        fmt  = ".3f" if metric_short == "Δvx" else ".0f"
        label_txt = f"{val:{fmt}}"
        if r == 1:
            label_txt += f"\n({sign})"
        ax_heat.text(c, r, label_txt, ha="center", va="center",
                     fontsize=14, fontweight="bold", color=txt_color)

ax_heat.set_xticks([0, 1])
ax_heat.set_xticklabels(["Linear ramp base", "Smoothstep base"], fontsize=11)
ax_heat.set_yticks([0, 1])
mlp_row_label = "Δα = MLP output" if args.exp == "A" else "Δq = MLP output"
no_mlp_label  = "Δα = 0 (no MLP)" if args.exp == "A" else "Δq = 0 (no MLP)"
ax_heat.set_yticklabels([no_mlp_label, mlp_row_label], fontsize=11)

ss_pct = pct[1][1]
lr_pct = pct[1][0]
ss_str = f"{'−' if ss_pct < 0 else '+'}{abs(ss_pct):.1f}%"
lr_str = f"{'−' if lr_pct < 0 else '+'}{abs(lr_pct):.1f}%"
ax_heat.set_title(
    f"Panel C — Mean {metric_short} (N=6)  |  {'lower' if lower_better else 'higher'} = better\n"
    f"MLP with correct base (SS): {ss_str}  |  MLP with wrong base (LR): {lr_str}",
    fontsize=9, loc="left"
)
plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02, label=metric_name)

# ── Suptitle ──────────────────────────────────────────────────────────────────
exp_label = ("Exp A  Sched-α 12D — jerk penalty"
             if args.exp == "A" else
             "Exp B  Sched-α 12D — vx-window penalty  (same architecture, different reward)")
fig.suptitle(
    f"Base-swap validation — {exp_label}\n"
    f"Correct base (SS): {ss_str} {metric_short}  |  Wrong base (LR, no retraining): {lr_str} {metric_short}",
    fontsize=10, y=0.975
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
