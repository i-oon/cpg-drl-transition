"""
Play a PPO-trained CPG policy.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/play_ppo.py --checkpoint logs/phase1/ppo_walk/trained_policy.pt

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play PPO + CPG policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--gait", type=str, default="walk",
                    choices=["walk", "trot", "steer"])
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from envs.unitree_b1_env import UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg
from rsl_rl.modules import ActorCritic

PHASE_OFFSETS = {
    "walk":  [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
    "trot":  [0.0, math.pi, math.pi, 0.0],
    "steer": [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
}

# Build env
cfg = UnitreeB1EnvCfg()
cfg.gait_name = args.gait
cfg.phase_offsets = PHASE_OFFSETS[args.gait]
cfg.scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=True)
env = UnitreeB1Env(cfg)

# Load policy
ckpt = torch.load(args.checkpoint, map_location=env.device)
policy = ActorCritic(
    num_actor_obs=cfg.observation_space,
    num_critic_obs=cfg.observation_space,
    num_actions=cfg.action_space,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
).to(env.device)
policy.load_state_dict(ckpt["model_state_dict"])
policy.eval()

print(f"\n  Gait: {args.gait}")
print(f"  Checkpoint: {args.checkpoint}")
print(f"  Steps: {args.steps} ({args.steps * 0.02:.1f}s)\n")

# Run
env.reset()
obs_dict = env._get_observations()
obs = obs_dict["policy"]

total_reward = torch.zeros(env.num_envs, device=env.device)

print(f"  {'step':>5} | {'vx':>6} {'vy':>6} {'vz':>6} | {'h':>5} {'tilt':>6} | "
      f"{'yaw_r':>6} | {'gait':>4} | {'W_max':>6} | {'resets':>6} | {'R_tot':>8}")
print("  " + "-" * 100)

# Accumulators for final summary
vx_hist, vy_hist, vz_hist, h_hist, tilt_hist = [], [], [], [], []
w_max_hist = []
contact_hist = []
reset_count = 0
prev_ep_len = torch.zeros(env.num_envs, device=env.device)

for step in range(args.steps):
    with torch.no_grad():
        actions = policy.act_inference(obs)
    obs_dict, reward, terminated, timed_out, info = env.step(actions)
    obs = obs_dict["policy"]
    total_reward += reward

    # Count resets: episode_length_buf drops when env resets
    cur_ep = env.episode_length_buf if hasattr(env, 'episode_length_buf') else None
    if cur_ep is not None:
        reset_count += int((cur_ep < prev_ep_len).sum().item())
        prev_ep_len = cur_ep.clone()

    # Foot contact for gait diagram
    if env._contact_sensor is not None and env._foot_ids is not None:
        nf = env._contact_sensor.data.net_forces_w_history
        ff = torch.norm(nf[:, 0, :, :], dim=-1)
        in_contact = (ff[0, env._foot_ids] > 1.0).cpu().numpy()
        contact_hist.append(in_contact)

    d = env._robot.data
    vx = d.root_lin_vel_b[:, 0].mean().item()
    vy = d.root_lin_vel_b[:, 1].mean().item()
    vz = d.root_lin_vel_b[:, 2].mean().item()
    h = d.root_pos_w[:, 2].mean().item()
    tilt = torch.sum(torch.square(d.projected_gravity_b[:, :2]), dim=1).mean().item()
    yaw_r = d.root_ang_vel_b[:, 2].mean().item()
    w_max = actions.abs().max().item()

    vx_hist.append(vx); vy_hist.append(vy); vz_hist.append(vz)
    h_hist.append(h); tilt_hist.append(tilt); w_max_hist.append(w_max)

    if (step + 1) % 20 == 0:
        gait_str = "".join("█" if c else "·" for c in contact_hist[-1]) if contact_hist else "????"
        print(f"  {step+1:5d} | {vx:+6.3f} {vy:+6.3f} {vz:+6.3f} | "
              f"{h:5.3f} {tilt:6.4f} | {yaw_r:+6.2f} | "
              f"{gait_str:>4} | {w_max:6.2f} | {reset_count:6d} | "
              f"{total_reward.mean().item():8.2f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 60
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — PPO {args.gait.upper()}")
print(sep)

vx_a = np.array(vx_hist); h_a = np.array(h_hist); tilt_a = np.array(tilt_hist)
vz_a = np.array(vz_hist); w_a = np.array(w_max_hist)

print(f"  Steps:         {args.steps}")
print(f"  Total reward:  {total_reward.mean().item():.2f}")
print(f"  Env resets:    {reset_count}  (falls/terminations)")
print(f"\n  vx:    mean={vx_a.mean():+.3f}  std={vx_a.std():.3f}  min={vx_a.min():+.3f}  max={vx_a.max():+.3f}")
print(f"  Height: mean={h_a.mean():.3f}  std={h_a.std():.3f}  min={h_a.min():.3f}  max={h_a.max():.3f}  (target 0.42)")
print(f"  Tilt:   mean={tilt_a.mean():.4f}  max={tilt_a.max():.4f}  (0 = upright)")
print(f"  vz:     mean={abs(vz_a).mean():.3f}  max={abs(vz_a).max():.3f}  (0 = no bounce)")
print(f"  W_max:  mean={w_a.mean():.3f}  max={w_a.max():.3f}  (clamp at ±2.0)")

if contact_hist:
    contact_arr = np.array(contact_hist)
    duty = contact_arr.mean(axis=0) * 100
    print(f"\n  Duty factors (% stance):")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        print(f"    {lab}: {duty[i]:.1f}%")

print(sep)

sim_app.close()
