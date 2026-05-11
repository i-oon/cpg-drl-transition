"""
Transition-window zoom figure.

Modes
-----
baselines  — Discrete / Linear ramp / Smoothstep / Residual-α 4D (Ours)
             (pass --no_e2e to drop E2E; default omits it)
ablation   — 2×2: Res-α 4D / Res-q 4D / Res-α 12D / Res-q 12D

Four panels per figure:
  1. α(t)                 — blending schedule shape
  2. Max |joint velocity| — instant spike for discrete (clearest signal)
  3. Joint velocity RMS   — energy across all 12 joints
  4. Forward velocity     — physical consequence

Usage:
    python scripts/plot_transition_zoom.py --mode baselines
    python scripts/plot_transition_zoom.py --mode ablation
    python scripts/plot_transition_zoom.py --mode baselines --window 3
    python scripts/plot_transition_zoom.py --mode baselines --out logs/phase2/zoom_baselines.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="baselines",
                    choices=["baselines", "ablation"],
                    help="baselines = discrete/ramp/smoothstep/ours; "
                         "ablation = 2×2 residual variants")
parser.add_argument("--include_e2e", action="store_true",
                    help="Add E2E baseline to baselines plot")
parser.add_argument("--window", type=int, default=0,
                    help="Which transition window to zoom (0=first, 1=second …)")
parser.add_argument("--pre_s",  type=float, default=1.2)
parser.add_argument("--post_s", type=float, default=4.5)
parser.add_argument("--out", default=None,
                    help="Output path (default: logs/phase2/transition_zoom_<mode>.png)")
args = parser.parse_args()

if args.out is None:
    args.out = f"logs/phase2/transition_zoom_{args.mode}.png"

DT = 0.02

# (label, path, color, linestyle, linewidth)
BASELINES = [
    ("Discrete Switch",  "logs/phase2/baselines/discrete/playback_seed42.csv",       "#d62728", "--", 2.0),
    ("Linear ramp",      "logs/phase2/baselines/linear_ramp/playback_seed42.csv",    "#8c564b", "-.", 1.8),
    ("Smoothstep",       "logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv","#2ca02c", "-",  1.8),
    ("Residual-α 12D",   "logs/phase2/residual_alpha_12d/playback_seed42.csv",       "#1f77b4", "-",  2.2),
]
E2E_ENTRY = \
    ("E2E",             "logs/phase2/baselines/e2e/playback_seed42.csv",             "#9467bd", ":",  1.8)

# Ablation uses canonical seed=42 playback CSVs
ABLATION = [
    ("Res-α 4D",  "logs/phase2/phase2_v10/playback_seed42.csv",         "#1f77b4", "-",  1.8),
    ("Res-α 12D", "logs/phase2/residual_alpha_12d/playback_seed42.csv", "#1f77b4", "--", 2.2),
    ("Res-q 4D",  "logs/phase2/residual_q_4d/playback_seed42.csv",      "#ff7f0e", "-",  1.8),
    ("Res-q 12D", "logs/phase2/residual_q_12d/playback_seed42.csv",     "#ff7f0e", "--", 1.6),
]

if args.mode == "baselines":
    METHODS = list(BASELINES)
    if args.include_e2e:
        METHODS.insert(3, E2E_ENTRY)   # insert before Residual
else:
    METHODS = ABLATION


def rolling_max(arr, w):
    out = np.empty_like(arr)
    half = w // 2
    for i in range(len(arr)):
        out[i] = arr[max(0, i-half):min(len(arr), i+half+1)].max()
    return out


def load(path):
    df     = pd.read_csv(path)
    t      = df["t"].values
    # Guard: if t is clearly broken (max ≪ expected episode length), reconstruct
    if t.max() < 1.0:
        print(f"  WARNING: corrupt t column in {path} — reconstructing from row index")
        t = np.arange(len(df)) * DT
    ab     = df["alpha_base"].values
    vx     = df["vx"].values
    jv     = df[[f"jv{i}" for i in range(12)]].values
    jv_max = np.max(np.abs(jv), axis=1)          # worst single joint per step
    jv_rms = np.sqrt(np.mean(jv ** 2, axis=1))   # RMS across 12 joints per step
    return t, ab, vx, jv_max, jv_rms


def find_transition_starts(ab, min_gap=100):
    starts = []
    for i in range(1, len(ab)):
        jumped = (ab[i] - ab[i-1]) > 0.4
        ramped = (ab[i-1] < 0.02) and (ab[i] > 0.02)
        if jumped or ramped:
            if not starts or (i - starts[-1]) > min_gap:
                starts.append(i)
    return starts


# ── Load ──────────────────────────────────────────────────────────────────────
datasets = {}
for label, path, color, ls, lw in METHODS:
    t, ab, vx, jv_max, jv_rms = load(path)
    starts = find_transition_starts(ab)
    win    = min(args.window, len(starts) - 1)
    datasets[label] = dict(t=t, ab=ab, vx=vx, jv_max=jv_max, jv_rms=jv_rms,
                           color=color, ls=ls, lw=lw,
                           t_start=t[starts[win]])

# ── Reference t_start = discrete (or first method) for the title ──────────────
ref_label = METHODS[0][0]
t_start_ref = datasets[ref_label]["t_start"]
print(f"Mode: {args.mode}  |  Window {args.window}")
for label, d in datasets.items():
    print(f"  {label:26s}: own t_start={d['t_start']:.2f}s")

# ── Spike peaks ───────────────────────────────────────────────────────────────
def _peak(label, key, post=0.3):
    d  = datasets[label]
    ts = d["t_start"]
    mask = (d["t"] >= ts) & (d["t"] <= ts + post)
    return d[key][mask].max() if mask.any() else 0.0

peaks = {label: _peak(label, "jv_max") for label, *_ in METHODS}
print("\nMax |jv| at switch (first 0.3 s):")
for label, p in peaks.items():
    print(f"  {label:26s}: {p:.2f} rad/s")

# y-axis ceiling: use the highest-spike method (typically discrete)
jv_max_ceil = max(peaks.values()) * 1.20

# steady-state reference (pre-switch, first method)
pre_mask = (datasets[ref_label]["t"] >= t_start_ref - args.pre_s) & \
           (datasets[ref_label]["t"] <  t_start_ref - 0.1)
jv_steady = datasets[ref_label]["jv_max"][pre_mask].max()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
fig.subplots_adjust(hspace=0.06, top=0.90, bottom=0.07, left=0.12, right=0.97)

for label, d in datasets.items():
    ts    = d["t_start"]
    lo    = ts - args.pre_s
    hi    = ts + args.post_s
    mask  = (d["t"] >= lo) & (d["t"] <= hi)
    t_rel = d["t"][mask] - ts

    jv_max_plot = rolling_max(d["jv_max"][mask], w=3)

    axes[0].plot(t_rel, d["ab"][mask],     color=d["color"], ls=d["ls"], lw=d["lw"], label=label)
    axes[1].plot(t_rel, jv_max_plot,       color=d["color"], ls=d["ls"], lw=d["lw"])
    axes[2].plot(t_rel, d["jv_rms"][mask], color=d["color"], ls=d["ls"], lw=d["lw"])
    axes[3].plot(t_rel, d["vx"][mask],     color=d["color"], ls=d["ls"], lw=d["lw"])

# ── Reference lines ───────────────────────────────────────────────────────────
for ax in axes:
    ax.axvline(0,   color="red",    lw=1.8, ls=":", alpha=0.85)
    ax.axvline(3.0, color="orange", lw=1.2, ls="--", alpha=0.7)
    ax.axvspan(0, 3, color="gold", alpha=0.07, zorder=0)

# ── Panel labels & limits ─────────────────────────────────────────────────────
axes[0].set_ylabel("α (blend weight)", fontsize=10)
axes[0].set_ylim(-0.05, 1.25)
axes[0].set_yticks([0, 0.5, 1.0])
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.92)

axes[1].set_ylabel("Max |joint velocity|\n[rad/s]", fontsize=10)
axes[1].set_ylim(0, jv_max_ceil)
axes[1].axhline(jv_steady, color="gray", lw=0.9, ls=":",
                alpha=0.55, label=f"steady-state (~{jv_steady:.1f} rad/s)")
axes[1].legend(loc="upper right", fontsize=8, framealpha=0.85)

axes[2].set_ylabel("Joint velocity\nRMS [rad/s]", fontsize=10)
axes[2].set_ylim(0, None)

axes[3].set_ylabel("Forward velocity\n[m/s]", fontsize=10)
axes[3].axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
axes[3].axhline(0.4, color="gray", lw=0.8, ls=":",  alpha=0.4)
axes[3].set_xlabel("Time relative to gait switch command [s]", fontsize=10)

# ── Spike annotation (baselines mode only — annotate the discrete spike) ──────
if args.mode == "baselines" and "Discrete Switch" in datasets:
    disc = datasets["Discrete Switch"]
    ts   = disc["t_start"]
    mask_n = (disc["t"] >= ts) & (disc["t"] <= ts + 0.3)
    peak_val = disc["jv_max"][mask_n].max()
    peak_t_rel = disc["t"][mask_n][np.argmax(disc["jv_max"][mask_n])] - ts

    ours_label = "Residual-α 12D"
    ours_peak  = peaks.get(ours_label, 1.0)
    if ours_peak > 0:
        ratio = peak_val / ours_peak
        axes[1].annotate(
            f"{peak_val:.1f} rad/s  (×{ratio:.0f} vs ours)",
            xy=(peak_t_rel, peak_val),
            xytext=(0.6, peak_val * 0.82),
            color="#d62728", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        )

# ── Legend for markers ────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="red",    alpha=0.8,  label="gait switch (t = 0)"),
    mpatches.Patch(color="orange", alpha=0.6,  label="ramp end (t = 3 s)"),
    mpatches.Patch(color="gold",   alpha=0.35, label="3 s transition window"),
]
axes[3].legend(handles=legend_patches, loc="lower right", fontsize=8.5,
               framealpha=0.88, ncol=3)

# ── Ablation legend hint ──────────────────────────────────────────────────────
if args.mode == "ablation":
    axes[0].text(0.01, 0.05,
                 "Blue = α-space  |  Orange = q-space\n"
                 "Solid = 4D  |  Dashed = 12D  (best: Res-α 12D)",
                 transform=axes[0].transAxes, fontsize=8,
                 va="bottom", color="0.35",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

# ── Title ─────────────────────────────────────────────────────────────────────
# Infer gait pair from smoothstep or first available file
for label, path, *_ in METHODS:
    try:
        df_ref = pd.read_csv(path)
        if "current" in df_ref.columns:
            ts_ref  = datasets[label]["t_start"]
            mp      = (df_ref["t"] >= ts_ref - 0.1) & (df_ref["t"] <= ts_ref + 0.1)
            pair_str = f"{df_ref.loc[mp,'current'].iloc[0]} → {df_ref.loc[mp,'target'].iloc[0]}" \
                if mp.any() else f"window {args.window}"
            break
    except Exception:
        pair_str = f"window {args.window}"

mode_title = {
    "baselines": "Baselines comparison",
    "ablation":  "2×2 ablation — output space × action dimension",
}
fig.suptitle(
    f"{mode_title[args.mode]} — {pair_str}\n"
    f"t = 0: gait switch command   |   shaded: 3 s blending window",
    fontsize=11, y=0.97,
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
