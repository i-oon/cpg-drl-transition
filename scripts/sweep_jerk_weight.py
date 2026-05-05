"""Sweep `rew_joint_jerk` with v10's architecture (sigmoid clamp [0, 0.3]).

Trains 5 models, each with a different jerk-penalty weight, holding all other
config fixed. Result: a CoT-vs-jerk_RMS scatter showing the smoothness↔energy
tradeoff under the v10 architecture.

Run names:
    sweep_w0      — rew_joint_jerk = 0          (no penalty, low CoT, high jerk)
    sweep_w_low   — rew_joint_jerk = -2e-11
    sweep_w_med   — rew_joint_jerk = -1e-10     (= current v10)
    sweep_w_hi    — rew_joint_jerk = -5e-10     (over-strong, expect compressed-jump)
    sweep_w_xhi   — rew_joint_jerk = -1e-9      (extreme — may collapse)

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/sweep_jerk_weight.py --headless --num_envs 2048 \
        --max_iterations 2000

After all 5 train, run playback for each via:
    python scripts/sweep_jerk_weight.py --playback

Then plot:
    python scripts/sweep_jerk_weight.py --plot
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

# (run_name, rew_joint_jerk weight)
SWEEP = [
    ("sweep_w0",     0.0),
    ("sweep_w_low",  -2e-11),
    ("sweep_w_med",  -1e-10),
    ("sweep_w_hi",   -5e-10),
    ("sweep_w_xhi",  -1e-9),
]

# Where each run's logs land
RUN_DIR = REPO / "logs" / "phase2"

parser = argparse.ArgumentParser(description="Jerk-weight sweep for v10 architecture")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--playback", action="store_true",
                    help="After training, run a 2000-step playback for each run.")
parser.add_argument("--plot", action="store_true",
                    help="Plot CoT vs jerk_RMS scatter (after all playbacks done).")
parser.add_argument("--skip", type=str, default="",
                    help="Comma-separated run names to skip (already trained).")
args = parser.parse_args()

skip = set(args.skip.split(",")) if args.skip else set()


def edit_jerk_weight(weight: float) -> None:
    """Mutate envs/b1_phase2_env_cfg.py to set rew_joint_jerk to the given weight.

    Replaces the line `rew_joint_jerk: float = ...` (one line, deterministic).
    """
    cfg = REPO / "envs" / "b1_phase2_env_cfg.py"
    txt = cfg.read_text()
    out_lines = []
    found = False
    for line in txt.splitlines():
        if line.lstrip().startswith("rew_joint_jerk: float ="):
            indent = len(line) - len(line.lstrip())
            out_lines.append(" " * indent + f"rew_joint_jerk: float = {weight:g}  # SWEEP")
            found = True
        else:
            out_lines.append(line)
    if not found:
        sys.exit("ERROR: rew_joint_jerk line not found in env cfg.")
    cfg.write_text("\n".join(out_lines) + "\n")
    print(f"  → set rew_joint_jerk = {weight:g}")


def run(cmd: list[str]) -> None:
    print(f"\n  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        sys.exit(f"ERROR: command failed (exit {res.returncode})")


def train_all() -> None:
    cfg_path = REPO / "envs" / "b1_phase2_env_cfg.py"
    original_cfg = cfg_path.read_text()
    try:
        for run_name, weight in SWEEP:
            if run_name in skip:
                print(f"\n[skip {run_name}]")
                continue
            if (RUN_DIR / run_name / "model_final.pt").exists():
                print(f"\n[exists {run_name}] skipping training (delete to redo)")
                continue
            print(f"\n========== TRAIN {run_name}  weight={weight:g} ==========")
            edit_jerk_weight(weight)
            cmd = [
                "python", "scripts/train_b1_phase2.py",
                "--num_envs", str(args.num_envs),
                "--max_iterations", str(args.max_iterations),
                "--run_name", run_name,
                "--seed", str(args.seed),
            ]
            if args.headless:
                cmd.append("--headless")
            run(cmd)
    finally:
        # Restore the env cfg to v10's original setting so subsequent normal
        # training runs (not via this sweep script) get the right weight.
        cfg_path.write_text(original_cfg)
        print(f"\n  [cfg restored to original — {cfg_path}]")


def playback_all() -> None:
    for run_name, _ in SWEEP:
        ckpt = RUN_DIR / run_name / "model_final.pt"
        csv = RUN_DIR / run_name / "playback.csv"
        if not ckpt.exists():
            print(f"[missing {run_name}] no model — train first")
            continue
        if csv.exists():
            print(f"[exists {run_name}/playback.csv] skipping (delete to redo)")
            continue
        print(f"\n========== PLAYBACK {run_name} ==========")
        cmd = [
            "python", "scripts/play_b1_phase2.py",
            "--checkpoint", str(ckpt),
            "--num_envs", "4", "--steps", "2000", "--seed", str(args.seed),
            "--gait_pairs", "trot,bound,pace", "--switch_interval_s", "8.0",
            "--save_plots", str(RUN_DIR / run_name / "diag"),
            "--save_csv",   str(csv),
        ]
        if args.headless:
            cmd.append("--headless")
        run(cmd)


def plot_pareto() -> None:
    import csv as _csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DT = 0.02
    points = []  # list of (run_name, weight, jerk_RMS, CoT, vx_mean)
    for run_name, weight in SWEEP:
        path = RUN_DIR / run_name / "playback.csv"
        if not path.exists():
            print(f"[missing {run_name}/playback.csv] skipping")
            continue
        with path.open() as f:
            rows = list(_csv.DictReader(f))
        ja = np.stack([[float(r[f"ja{i}"]) for i in range(12)] for r in rows])
        tq = np.stack([[float(r[f"tq{i}"]) for i in range(12)] for r in rows])
        jv = np.stack([[float(r[f"jv{i}"]) for i in range(12)] for r in rows])
        vx = np.array([float(r["vx"]) for r in rows])
        x_w = np.array([float(r["x_w"]) for r in rows])
        jerk = np.diff(ja, axis=0) / DT
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
        power = np.sum(np.abs(tq * jv), axis=1)
        energy = float(np.sum(power) * DT)
        distance = max(float(x_w[-1] - x_w[0]), 1e-6)
        cot = energy / (50.0 * 9.81 * distance)
        points.append((run_name, weight, jerk_rms, cot, float(vx.mean())))
        print(f"  {run_name:<14} weight={weight:>9g}  jerk_RMS={jerk_rms:6.0f}  "
              f"CoT={cot:5.2f}  vx={vx.mean():+.3f}")

    if not points:
        sys.exit("no data to plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, w, jerk, cot, vx in points:
        ax.scatter(jerk, cot, s=80, label=f"{name} (w={w:g}, vx={vx:+.2f})")
        ax.annotate(name, (jerk, cot), xytext=(6, 6), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("jerk_RMS [rad/s³]  (smoother → left)")
    ax.set_ylabel("Cost of Transport  (more efficient → down)")
    ax.set_title("Smoothness ↔ Energy tradeoff — v10 architecture, jerk-weight sweep")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out = RUN_DIR / "sweep_jerk_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nPareto plot saved → {out}")


if __name__ == "__main__":
    if args.plot:
        plot_pareto()
    elif args.playback:
        playback_all()
    else:
        train_all()
