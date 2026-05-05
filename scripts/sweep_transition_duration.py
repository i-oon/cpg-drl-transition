"""Sweep transition_duration_s — playback-only experiment.

Tests whether v10's smoothness advantage over Smoothstep_Ramp grows as
transition windows get shorter (harder). v10 is NOT retrained — same
checkpoint at every duration. Smoothstep is just the no-MLP baseline
running with the same shortened ramp.

No retraining means the v10 MLP sees out-of-distribution alpha_baseline
rates (it was trained at duration=3.0). If its smoothness gain holds at
duration=0.5 — strong claim. If it collapses — confirms the architectural
ceiling is duration-bounded too.

Run names: {method}_dur{0.5,1.0,2.0,3.0,5.0}
Stored under: logs/phase2/duration_sweep/

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/sweep_transition_duration.py --headless          # run all playbacks
    python scripts/sweep_transition_duration.py --plot              # produce comparison figure
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

DURATIONS = [0.5, 1.0, 2.0, 3.0, 5.0]

METHODS = [
    # name, extra args to play_b1_phase2.py
    ("v10",
     ["--checkpoint", "logs/phase2/phase2_v10/model_final.pt"]),
    ("smoothstep",
     ["--baseline", "smoothstep_ramp"]),
]

OUT_DIR = REPO / "logs" / "phase2" / "duration_sweep"

parser = argparse.ArgumentParser(description="Transition-duration playback sweep")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--skip", type=str, default="",
                    help="Comma-separated run keys to skip, format `method_dur0.5`")
args = parser.parse_args()
skip = set(args.skip.split(",")) if args.skip else set()


def edit_duration(duration_s: float) -> None:
    """Mutate transition_duration_s in env cfg in-place."""
    cfg = REPO / "envs" / "b1_phase2_env_cfg.py"
    txt = cfg.read_text()
    out_lines = []
    found = False
    for line in txt.splitlines():
        s = line.lstrip()
        if (s.startswith("transition_duration_s: float =")
                and "# SWEEP" not in line and "MISSING" not in line):
            indent = len(line) - len(s)
            out_lines.append(
                " " * indent
                + f"transition_duration_s: float = {duration_s}        # SWEEP")
            found = True
        else:
            out_lines.append(line)
    if not found:
        # Allow line that already has SWEEP marker (idempotent re-run)
        out_lines = []
        for line in txt.splitlines():
            s = line.lstrip()
            if s.startswith("transition_duration_s: float ="):
                indent = len(line) - len(s)
                out_lines.append(
                    " " * indent
                    + f"transition_duration_s: float = {duration_s}        # SWEEP")
                found = True
            else:
                out_lines.append(line)
    if not found:
        sys.exit("ERROR: transition_duration_s line not found in env cfg")
    cfg.write_text("\n".join(out_lines) + "\n")


def run_one(method_name: str, method_args: list[str], duration: float) -> None:
    key = f"{method_name}_dur{duration}"
    if key in skip:
        print(f"\n[skip {key}]"); return
    out = OUT_DIR / key
    csv = out / "playback.csv"
    if csv.exists():
        print(f"\n[exists {key}/playback.csv] skipping")
        return
    print(f"\n========== PLAYBACK {key} (transition_duration_s={duration}) ==========")
    cmd = [
        "python", "scripts/play_b1_phase2.py",
        "--num_envs", "4", "--steps", "2000", "--seed", "42",
        "--gait_pairs", "trot,bound,pace", "--switch_interval_s", "8.0",
        "--save_plots", str(out / "diag"),
        "--save_csv", str(csv),
    ] + method_args
    if args.headless:
        cmd.append("--headless")
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        sys.exit(f"ERROR: playback {key} failed (exit {res.returncode})")


def run_all() -> None:
    cfg_path = REPO / "envs" / "b1_phase2_env_cfg.py"
    original_cfg = cfg_path.read_text()
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for duration in DURATIONS:
            edit_duration(duration)
            for method_name, method_args in METHODS:
                run_one(method_name, method_args, duration)
    finally:
        cfg_path.write_text(original_cfg)
        print(f"\n  [cfg restored to original]")


def metrics_for(csv_path: Path) -> dict:
    import csv as _csv
    import numpy as np
    DT = 0.02
    with csv_path.open() as f:
        rows = list(_csv.DictReader(f))
    vx = np.array([float(r["vx"]) for r in rows])
    h  = np.array([float(r["h"]) for r in rows])
    tilt = np.array([float(r["tilt"]) for r in rows])
    ja = np.stack([[float(r[f"ja{i}"]) for i in range(12)] for r in rows])
    tq = np.stack([[float(r[f"tq{i}"]) for i in range(12)] for r in rows])
    jv = np.stack([[float(r[f"jv{i}"]) for i in range(12)] for r in rows])
    x_w = np.array([float(r["x_w"]) for r in rows])
    jerk = np.diff(ja, axis=0) / DT
    energy = float(np.sum(np.abs(tq * jv)) * DT)
    distance = max(float(x_w[-1] - x_w[0]), 1e-6)
    return {
        "vx_mean": float(vx.mean()),
        "vx_min":  float(vx.min()),
        "vx_std":  float(vx.std()),
        "tilt_max": float(tilt.max()),
        "jerk_RMS": float(np.sqrt(np.mean(jerk ** 2))),
        "jerk_max": float(np.max(np.abs(jerk))),
        "CoT": energy / (50.0 * 9.81 * distance),
    }


def plot_sweep() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {m_name: [] for m_name, _ in METHODS}
    for d in DURATIONS:
        for m_name, _ in METHODS:
            csv = OUT_DIR / f"{m_name}_dur{d}" / "playback.csv"
            if csv.exists():
                m = metrics_for(csv)
                rows[m_name].append((d, m))
            else:
                print(f"[missing {m_name}_dur{d}]")

    print()
    print(f"{'duration':>8} | {'method':<12} | {'vx_min':>7} {'vx_std':>7} | "
          f"{'jerk_RMS':>9} {'jerk_max':>9} | {'CoT':>5}")
    print("-" * 80)
    for d in DURATIONS:
        for m_name, _ in METHODS:
            csv = OUT_DIR / f"{m_name}_dur{d}" / "playback.csv"
            if not csv.exists():
                continue
            m = metrics_for(csv)
            print(f"{d:>8.1f} | {m_name:<12} | {m['vx_min']:+7.3f} {m['vx_std']:7.3f} | "
                  f"{m['jerk_RMS']:9.0f} {m['jerk_max']:9.0f} | {m['CoT']:5.2f}")

    # 2x2 panel: jerk_RMS, jerk_max, vx_min, CoT vs duration
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    panels = [
        ("jerk_RMS", "jerk_RMS [rad/s³]  (lower = smoother)", axes[0, 0], False),
        ("jerk_max", "jerk_max [rad/s³]  (peak motor stress)", axes[0, 1], False),
        ("vx_min", "vx_min [m/s]  (stability — closer to cmd is better)", axes[1, 0], False),
        ("CoT",  "Cost of Transport  (lower = more efficient)", axes[1, 1], False),
    ]
    colors = {"v10": "C2", "smoothstep": "C1"}
    markers = {"v10": "o", "smoothstep": "s"}
    for key, label, ax, ylog in panels:
        for m_name in rows:
            xs = [d for d, _ in rows[m_name]]
            ys = [m[key] for _, m in rows[m_name]]
            ax.plot(xs, ys, marker=markers[m_name], color=colors[m_name],
                    label=m_name, lw=1.6, markersize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        if ylog:
            ax.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("transition_duration_s [s]  (smaller = harder)")
    fig.suptitle("Transition-duration sweep — v10 vs Smoothstep (no retraining)\n"
                 "Tests whether MLP smoothness gain grows with task difficulty",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "duration_sweep.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nFigure saved → {out}")


if __name__ == "__main__":
    if args.plot:
        plot_sweep()
    else:
        run_all()
