"""
Play a trained gait with visualization.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/play_gait.py --gait walk               # with GUI
    python scripts/play_gait.py --gait walk --headless     # headless

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained CPG gait")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--gait", type=str, default="walk",
                    choices=["walk", "trot", "pace", "bound", "steer"])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000,
                    help="Steps to run (1000 = 20s at 50Hz)")
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- Safe to import now ----
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from envs.unitree_b1_env import UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASE_OFFSETS = {
    "walk":  [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
    "trot":  [0.0, math.pi, math.pi, 0.0],
    "pace":  [0.0, math.pi, 0.0, math.pi],
    "bound": [0.0, 0.0, math.pi, math.pi],
    "steer": [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
}

WEIGHTS_PATH = {
    "walk":  "weights/W_walk.npy",
    "trot":  "weights/W_trot.npy",
    "pace":  "weights/W_pace.npy",
    "bound": "weights/W_bound.npy",
    "steer": "weights/W_steer.npy",
}

# ---------------------------------------------------------------------------
# Build env and load weights
# ---------------------------------------------------------------------------

weights_file = Path(WEIGHTS_PATH[args.gait])
if not weights_file.exists():
    print(f"[ERROR] Weights not found: {weights_file}")
    print(f"  Train first: python scripts/train_phase1_{args.gait}.py --headless")
    sim_app.close()
    sys.exit(1)

cfg = UnitreeB1EnvCfg()
cfg.gait_name = args.gait
cfg.phase_offsets = PHASE_OFFSETS[args.gait]
cfg.scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=True)

env = UnitreeB1Env(cfg)
W = np.load(weights_file)
env.set_weights(W)

print(f"\n  Gait    : {args.gait}")
print(f"  Weights : {weights_file}  (norm={np.linalg.norm(W):.3f})")
print(f"  Envs    : {args.num_envs}")
print(f"  Steps   : {args.steps}  ({args.steps * 0.02:.1f}s)")
print(f"  Press Ctrl-C to stop early.\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

env.reset()
dummy_actions = torch.zeros(env.num_envs, 12, device=env.device)

total_reward = torch.zeros(env.num_envs, device=env.device)

# Accumulators for summary stats
vx_history = []
height_history = []
tilt_history = []
vz_history = []
contact_history = []    # (steps, 4) boolean — foot contact state per step (env 0)
phase_history = []      # CPG phi per step

LEG_LABELS = ["FL", "FR", "RL", "RR"]

# Header
print(f"  {'step':>6} | {'vx':>7} {'vz':>6} | {'h':>5} {'tilt':>6} | "
      f"{'gait':>4} {'match':>5} | {'r_vx':>5} {'r_gait':>6} {'r_ori':>6} {'r_ht':>6} {'r_vz':>6} | "
      f"{'R_tot':>8}")
print("  " + "-" * 105)

prev_total = 0.0

# Reward component accumulators (per 50-step window)
reward_components = {"vx": 0, "gait": 0, "orient": 0, "height": 0, "vz": 0, "yaw": 0, "action": 0}

try:
    for step in range(args.steps):
        obs, reward, terminated, timed_out, info = env.step(dummy_actions)
        total_reward += reward

        # Record foot contact state for env 0 (every step)
        if env._contact_sensor is not None and env._foot_ids is not None:
            net_forces = env._contact_sensor.data.net_forces_w_history
            foot_forces = torch.norm(net_forces[:, 0, :, :], dim=-1)
            in_contact = (foot_forces[0, env._foot_ids] > 1.0).cpu().numpy()
            contact_history.append(in_contact)
            phase_history.append(env._phi % (2 * np.pi))

        # Compute reward breakdown for env 0
        d = env._robot.data
        _vx = d.root_lin_vel_b[0, 0].item()
        _h = d.root_pos_w[0, 2].item()
        _tilt = torch.sum(torch.square(d.projected_gravity_b[0, :2])).item()
        _vz2 = d.root_lin_vel_b[0, 2].item() ** 2
        _yaw2 = d.root_ang_vel_b[0, 2].item() ** 2

        # Gait match for env 0
        _gait_match = 0.0
        if hasattr(env, '_leg_offsets'):
            leg_phases = env._phi + env._leg_offsets
            expected = (torch.cos(leg_phases) > 0).cpu().numpy()
            if contact_history:
                actual = contact_history[-1]
                _gait_match = float(np.mean(actual == expected))

        reward_components["vx"] += _vx
        reward_components["gait"] += _gait_match
        reward_components["orient"] += _tilt
        reward_components["height"] += abs(_h - 0.42)
        reward_components["vz"] += _vz2

        if (step + 1) % 50 == 0:
            vx_avg = reward_components["vx"] / 50
            gait_avg = reward_components["gait"] / 50
            orient_avg = reward_components["orient"] / 50
            height_avg = reward_components["height"] / 50
            vz_avg = reward_components["vz"] / 50

            h_now = d.root_pos_w[:, 0].mean().item()  # not used, just for display
            h_z = d.root_pos_w[0, 2].item()
            tilt_now = torch.sum(torch.square(d.projected_gravity_b[0, :2])).item()

            r_total = total_reward.mean().item()

            # Gait ASCII
            gait_str = "".join("█" if c else "·" for c in contact_history[-1]) if contact_history else "????"

            vx_history.append(vx_avg)
            height_history.append(h_z)
            tilt_history.append(tilt_now)
            vz_history.append(vz_avg)

            # Reward breakdown per 50 steps (weighted by config values)
            r_vx_w = vx_avg * 1.0      # approximate with w1=1
            r_gait_w = gait_avg * 1.0   # w_gait=1
            r_ori_w = -orient_avg * 2.0
            r_ht_w = -height_avg * 8.0
            r_vz_w = -vz_avg * 2.0

            print(f"  {step+1:6d} | {vx_avg:+7.3f} {vz_avg:6.3f} | "
                  f"{h_z:5.3f} {tilt_now:6.4f} | "
                  f"{gait_str:>4} {gait_avg:5.2f} | "
                  f"{r_vx_w:+5.2f} {r_gait_w:+6.2f} {r_ori_w:+6.2f} {r_ht_w:+6.2f} {r_vz_w:+6.2f} | "
                  f"{r_total:8.2f}")

            # Reset accumulators
            for k in reward_components:
                reward_components[k] = 0

except KeyboardInterrupt:
    print("\n  Stopped by user.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 60
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — {args.gait.upper()}")
print(sep)
print(f"  Total reward  : {total_reward.mean().item():.2f}")
print(f"  Steps         : {args.steps}")

if vx_history:
    vx_arr = np.array(vx_history)
    h_arr  = np.array(height_history)
    tilt_arr = np.array(tilt_history)
    vz_arr = np.array(vz_history)

    print(f"\n  Forward velocity (vx):")
    print(f"    mean={vx_arr.mean():+.3f}  std={vx_arr.std():.3f}  "
          f"min={vx_arr.min():+.3f}  max={vx_arr.max():+.3f}  target=0.4 m/s")

    print(f"  Height:")
    print(f"    mean={h_arr.mean():.3f}  std={h_arr.std():.4f}  "
          f"min={h_arr.min():.3f}  max={h_arr.max():.3f}  target=0.42 m")

    print(f"  Tilt (gravity_xy²):")
    print(f"    mean={tilt_arr.mean():.4f}  max={tilt_arr.max():.4f}  "
          f"(0=perfect upright)")

    print(f"  Vertical velocity (vz):")
    print(f"    mean={abs(vz_arr).mean():.3f}  max={abs(vz_arr).max():.3f}  "
          f"(0=no bounce)")

    # W matrix info
    print(f"\n  W matrix ({W.shape[0]}×{W.shape[1]}):")
    print(f"    total norm={np.linalg.norm(W):.3f}")
    if W.shape[1] == 3:
        # Indirect encoding: shared W for all legs
        for j, jn in enumerate(["hip", "thigh", "calf"]):
            print(f"    {jn}: max={np.degrees(np.abs(W[:, j]).max()):.1f}°  "
                  f"norm={np.linalg.norm(W[:, j]):.3f}")
    else:
        # Direct encoding: per-leg columns
        for k, leg in enumerate(["FL", "FR", "RL", "RR"]):
            h_max = np.degrees(np.abs(W[:, k*3]).max())
            t_max = np.degrees(np.abs(W[:, k*3+1]).max())
            c_max = np.degrees(np.abs(W[:, k*3+2]).max())
            print(f"    {leg}: hip={h_max:.1f}°  thigh={t_max:.1f}°  calf={c_max:.1f}°")

print(sep)

# ---------------------------------------------------------------------------
# Gait diagram (save PNG) + duty factor analysis
# ---------------------------------------------------------------------------

if contact_history:
    contact_arr = np.array(contact_history)   # (steps, 4) bool
    n_steps = contact_arr.shape[0]
    dt = 0.02  # control step seconds
    time_axis = np.arange(n_steps) * dt

    # Duty factor per leg = fraction of time in stance
    duty = contact_arr.mean(axis=0)
    print(f"\n  Duty factor (fraction of time on ground):")
    for i, lab in enumerate(LEG_LABELS):
        print(f"    {lab}: {duty[i]*100:.1f}%")

    # Detect first contact times for each leg (for phase offset analysis)
    print(f"\n  Footfall pattern (first few stance-start times, seconds):")
    for i, lab in enumerate(LEG_LABELS):
        # Find rising edges (False → True)
        col = contact_arr[:, i].astype(int)
        edges = np.where(np.diff(col) == 1)[0] + 1
        times = edges * dt
        first_few = ", ".join(f"{t:.2f}" for t in times[:5])
        print(f"    {lab}: {first_few}")

    # Try matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive (for headless)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d"]

        for i, (lab, color) in enumerate(zip(LEG_LABELS, colors)):
            # Draw horizontal bars where foot is in stance
            stance = contact_arr[:, i]
            for start in np.where(np.diff(np.concatenate([[0], stance.astype(int), [0]])) == 1)[0]:
                end = start + np.argmin(stance[start:] | ~stance[start:]) if start < n_steps else start
                # Actually find where it turns False
                tail = np.where(~stance[start:])[0]
                end = start + (tail[0] if len(tail) else n_steps - start)
                ax.barh(i, (end - start) * dt, left=start * dt,
                        height=0.7, color=color, edgecolor="none", alpha=0.85)

        ax.set_yticks(range(4))
        ax.set_yticklabels(LEG_LABELS, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_title(f"Gait Diagram — {args.gait.upper()}\n"
                     f"(solid = stance/foot on ground, empty = swing/foot in air)",
                     fontsize=11)
        ax.set_xlim(0, n_steps * dt)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        # Annotation: duty factors
        for i, lab in enumerate(LEG_LABELS):
            ax.text(-0.02 * n_steps * dt, i, f"{duty[i]*100:.0f}%",
                    va="center", ha="right", fontsize=9, color="#555")

        out_path = Path("logs") / f"gait_diagram_{args.gait}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Gait diagram saved → {out_path}")
    except Exception as e:
        print(f"\n  (matplotlib gait plot skipped: {e})")

sim_app.close()
