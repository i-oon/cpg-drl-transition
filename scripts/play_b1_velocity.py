"""
Play a PPO-trained B1 velocity-tracking policy.

Follows Isaac Lab's canonical RSL-RL play pattern:
  parse_env_cfg → gym.make(cfg=env_cfg) → RslRlVecEnvWrapper
  → OnPolicyRunner.load → runner.get_inference_policy

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/play_b1_velocity.py --checkpoint logs/ppo_b1/<run>/model_final.pt
    python scripts/play_b1_velocity.py --checkpoint <path> --num_envs 4 --steps 1000

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play B1 velocity-tracking policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str,
                    default="Isaac-Velocity-Flat-Unitree-B1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", type=str, default=None,
                    help="If set, record video to this directory.")
parser.add_argument("--video_length", type=int, default=None,
                    help="Number of steps to record. Default = --steps.")
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import envs.b1_velocity_env_cfg  # noqa: F401
from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

agent_cfg = B1FlatPPORunnerCfg()

env_cfg = parse_env_cfg(
    args.task,
    device=agent_cfg.device,
    num_envs=args.num_envs,
)

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

if args.video:
    video_dir = Path(args.video)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_length = args.video_length if args.video_length is not None else args.steps
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=video_length,
        disable_logger=True,
        name_prefix="play",
    )
    print(f"  [video] recording {video_length} steps → {video_dir}/")

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------

runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=None,
    device=agent_cfg.device,
)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=agent_cfg.device)

print(f"\n  Task          : {args.task}")
print(f"  Checkpoint    : {args.checkpoint}")
print(f"  Envs          : {env.num_envs}")
print(f"  Steps         : {args.steps} ({args.steps * 0.02:.1f}s)\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

obs, _ = env.get_observations()
total_reward = torch.zeros(env.num_envs, device=env.device)

print(f"  {'step':>5} | {'vx':>6} {'vy':>6} {'vz':>6} | {'h':>5} {'tilt':>6} | "
      f"{'gait':>4} | {'R_tot':>8}")
print("  " + "-" * 70)

vx_hist, vy_hist, h_hist, tilt_hist = [], [], [], []
contact_history = []   # (steps, 4) bool — env 0's foot contacts for diagram
foot_z_history = []    # (steps, 4) — foot z (world-frame)
hip_q_history = []     # (steps, 4) — hip joint position (rad)
thigh_q_history = []   # (steps, 4)
calf_q_history = []    # (steps, 4)
reset_count = 0

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene["contact_forces"]

# Articulation foot body indices in [FL, FR, RL, RR] order — used for foot-z lookup.
_all_body_names = robot.body_names
foot_body_ids_art_ord = [_all_body_names.index(n)
                         for n in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]

# Joint indices for hip / thigh / calf joints in [FL, FR, RL, RR] order
_joint_names = robot.joint_names
hip_joint_ids = [_joint_names.index(n) for n in
                  ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]]
thigh_joint_ids = [_joint_names.index(n) for n in
                    ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]]
calf_joint_ids = [_joint_names.index(n) for n in
                   ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]]

# Find foot body indices on the contact sensor (B1 uses *_foot links).
# Order: FL, FR, RL, RR (we'll re-order to that).
foot_ids, foot_names = contact_sensor.find_bodies(".*_foot$")
# Build a permutation so contact_history columns are FL, FR, RL, RR
desired = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
perm = [foot_names.index(n) for n in desired if n in foot_names]
foot_ids = [foot_ids[i] for i in perm]

for step in range(args.steps):
    with torch.no_grad():
        actions = policy(obs)
    obs, reward, dones, extras = env.step(actions)
    total_reward += reward
    reset_count += int(dones.sum().item())

    # Foot contact (env 0) — use Isaac Lab's internal current_contact_time
    # which is the SAME source training rewards (excessive_air_time,
    # feet_air_time) read from. Single-frame net_forces snapshots miss
    # brief 1-2 step contacts in fast walking gaits.
    contact_time = contact_sensor.data.current_contact_time[0, foot_ids]  # (4,)
    in_contact = (contact_time > 0.0).cpu().numpy()                       # (4,) bool
    contact_history.append(in_contact)

    # Foot world-frame z (height above ground) — env 0 only
    foot_z = robot.data.body_pos_w[0, foot_body_ids_art_ord, 2].cpu().numpy()
    foot_z_history.append(foot_z)

    # Joint positions (env 0) per leg in [FL, FR, RL, RR] order
    jp = robot.data.joint_pos[0].cpu().numpy()
    hip_q_history.append(jp[hip_joint_ids])
    thigh_q_history.append(jp[thigh_joint_ids])
    calf_q_history.append(jp[calf_joint_ids])

    d = robot.data
    vx = d.root_lin_vel_b[:, 0].mean().item()
    vy = d.root_lin_vel_b[:, 1].mean().item()
    vz = d.root_lin_vel_b[:, 2].mean().item()
    h  = d.root_pos_w[:, 2].mean().item()
    tilt = torch.sum(torch.square(d.projected_gravity_b[:, :2]), dim=1).mean().item()

    vx_hist.append(vx); vy_hist.append(vy); h_hist.append(h); tilt_hist.append(tilt)

    if (step + 1) % 50 == 0:
        gait_str = "".join("█" if c else "·" for c in in_contact)
        print(f"  {step+1:5d} | {vx:+6.3f} {vy:+6.3f} {vz:+6.3f} | "
              f"{h:5.3f} {tilt:6.4f} | {gait_str:>4} | "
              f"{total_reward.mean().item():8.2f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 60
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — B1 velocity tracking")
print(sep)

vx_a = np.array(vx_hist); h_a = np.array(h_hist); tilt_a = np.array(tilt_hist)

print(f"  Steps         : {args.steps}")
print(f"  Total reward  : {total_reward.mean().item():.2f}")
print(f"  Env resets    : {reset_count}  (falls / timeouts)")
print(f"\n  vx  : mean={vx_a.mean():+.3f}  std={vx_a.std():.3f}  "
      f"min={vx_a.min():+.3f}  max={vx_a.max():+.3f}")
print(f"  Height: mean={h_a.mean():.3f}  std={h_a.std():.3f}  target=0.42")
print(f"  Tilt  : mean={tilt_a.mean():.4f}  max={tilt_a.max():.4f}  (0=upright)")
print(sep)

# ---------------------------------------------------------------------------
# Gait analysis (env 0 only): duty factors + footfall PNG
# ---------------------------------------------------------------------------

if contact_history:
    contact_arr = np.array(contact_history)   # (steps, 4) bool, FL/FR/RL/RR
    duty = contact_arr.mean(axis=0) * 100

    print(f"\n  Duty factor (% time on ground):")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        print(f"    {lab}: {duty[i]:5.1f}%")

    # Stance-start times for each leg (rising-edge detection)
    print(f"\n  Footfall pattern (first 5 stance starts, seconds):")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        starts = np.where(np.diff(contact_arr[:, i].astype(int)) == 1)[0] + 1
        starts_s = starts * 0.02
        print(f"    {lab}: " + ", ".join(f"{t:.2f}" for t in starts_s[:5]))

    # Per-leg swing-amplitude diagnostic
    if foot_z_history:
        z_arr = np.array(foot_z_history)  # (steps, 4)
        print(f"\n  Foot height during swing (apex above floor, cm):")
        for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
            stance_mask = contact_arr[:, i]
            swing_mask = ~stance_mask
            if stance_mask.any() and swing_mask.any():
                floor_z = z_arr[stance_mask, i].mean()
                swing_apex = z_arr[swing_mask, i].max() - floor_z
                swing_mean = z_arr[swing_mask, i].mean() - floor_z
                print(f"    {lab}: apex={swing_apex*100:5.1f}  mean_lift={swing_mean*100:5.2f}")

    # Per-joint angle range — directly answers the "is FL hip stuck?" question
    if hip_q_history:
        hip_q = np.array(hip_q_history)        # (steps, 4)
        thigh_q = np.array(thigh_q_history)
        calf_q = np.array(calf_q_history)
        print(f"\n  Joint angle range during episode (deg) — peak-to-peak swing:")
        print(f"    {'leg':<5} {'hip_p2p':>9} {'thigh_p2p':>10} {'calf_p2p':>9}")
        for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
            hip_range_deg = np.degrees(hip_q[:, i].max() - hip_q[:, i].min())
            thigh_range_deg = np.degrees(thigh_q[:, i].max() - thigh_q[:, i].min())
            calf_range_deg = np.degrees(calf_q[:, i].max() - calf_q[:, i].min())
            print(f"    {lab:<5} {hip_range_deg:9.2f} {thigh_range_deg:10.2f} {calf_range_deg:9.2f}")

    # PNG diagram
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3))
        for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
            stance = contact_arr[:, i]
            t = np.arange(len(stance)) * 0.02
            ax.fill_between(t, i + 0.1, i + 0.9, where=stance,
                            color="C0", alpha=0.85, step="post")
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(["FL", "FR", "RL", "RR"])
        ax.set_xlabel("time (s)")
        ax.set_title(f"Gait diagram — {args.checkpoint}")
        ax.set_xlim(0, args.steps * 0.02)
        ax.invert_yaxis()
        out_png = "logs/gait_diagram_ppo_b1.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=110)
        plt.close()
        print(f"\n  Gait diagram saved → {out_png}")
    except ImportError:
        print("\n  (matplotlib not installed — skipping PNG)")

env.close()
sim_app.close()
