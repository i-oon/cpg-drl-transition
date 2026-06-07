"""
Sanity-check the B1 velocity env with the current config.

Does NOT need a checkpoint — just verifies spawn pose, joint defaults,
observation shape, and that the env steps without error.

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/test_env.py --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps",    type=int, default=50)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import envs.b1_velocity_env_cfg  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg
from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg

SEP = "-" * 60

def ok(msg):   print(f"  [PASS] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print("  Building Isaac-Velocity-Flat-Unitree-B1-v0")
print(SEP)

agent_cfg = B1FlatPPORunnerCfg()
env_cfg = parse_env_cfg(
    "Isaac-Velocity-Flat-Unitree-B1-v0",
    device=agent_cfg.device,
    num_envs=args.num_envs,
)

env = gym.make("Isaac-Velocity-Flat-Unitree-B1-v0", cfg=env_cfg)
ok(f"Env created  num_envs={args.num_envs}  device={agent_cfg.device}")

# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------

obs, _ = env.reset()
obs_tensor = obs["policy"]
print(f"\n  Obs shape: {tuple(obs_tensor.shape)}")
if obs_tensor.shape[1] == 48:
    ok("Obs dim = 48 (no height scan — correct for blind policy)")
else:
    warn(f"Unexpected obs dim: {obs_tensor.shape[1]}")

if torch.isfinite(obs_tensor).all():
    ok("Obs is finite")
else:
    fail(f"Obs has {(~torch.isfinite(obs_tensor)).sum()} non-finite values")

# ---------------------------------------------------------------------------
# Spawn pose check
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print("  Spawn pose (env 0) — should match real home pose")
print(SEP)

robot = env.unwrapped.scene["robot"]
jp = robot.data.joint_pos[0].cpu()
jnames = robot.joint_names

print(f"\n  {'Joint':<25} {'actual':>8}  {'default':>8}  {'target':>8}")
print(f"  {'-'*55}")

# Real home pose in robot/joint space
real_home = {
    "FL_hip_joint":    0.03,  "FR_hip_joint":   -0.03,
    "RL_hip_joint":    0.03,  "RR_hip_joint":   -0.03,
    "FL_thigh_joint":  1.08,  "FR_thigh_joint":  1.08,
    "RL_thigh_joint":  1.08,  "RR_thigh_joint":  1.08,
    "FL_calf_joint":  -1.94,  "FR_calf_joint":  -1.94,
    "RL_calf_joint":  -1.94,  "RR_calf_joint":  -1.94,
}

default_jp = robot.data.default_joint_pos[0].cpu()
all_ok = True
for i, name in enumerate(jnames):
    actual  = jp[i].item()
    default = default_jp[i].item()
    target  = real_home.get(name, float("nan"))
    match = abs(default - target) < 0.02 if target == target else True
    tag = "✓" if match else "✗"
    if not match:
        all_ok = False
    print(f"  {name:<25} {actual:+8.4f}  {default:+8.4f}  {target:+8.4f}  {tag}")

if all_ok:
    ok("Default joint positions match real hardware home pose")
else:
    fail("Some defaults do not match — check UNITREE_B1_CFG.init_state.joint_pos")

height = robot.data.root_pos_w[0, 2].item()
print(f"\n  Spawn height: {height:.4f} m  (base_height_l2 target = 0.42 m)")
if abs(height - 0.42) < 0.05:
    ok(f"Height {height:.3f} m close to 0.42 m target")
elif height > 0.45:
    warn(f"Height {height:.3f} m above target — base_height_l2 will need to pull down")
else:
    warn(f"Height {height:.3f} m below target — base_height_l2 will push up")

# ---------------------------------------------------------------------------
# Short rollout
# ---------------------------------------------------------------------------

print(f"\n{SEP}")
print(f"  Rolling out {args.steps} steps with zero actions")
print(SEP)

actions = torch.zeros(args.num_envs, 12, device=agent_cfg.device)
rewards = []
for _ in range(args.steps):
    obs, reward, terminated, truncated, info = env.step(actions)
    rewards.append(reward.mean().item())

import numpy as np
print(f"\n  Reward mean: {np.mean(rewards):.4f}")
print(f"  Reward std:  {np.std(rewards):.4f}")
print(f"  Falls (base_contact): {terminated.sum().item()}")

if terminated.sum().item() == 0:
    ok("No falls in zero-action rollout")
else:
    warn(f"{terminated.sum().item()} envs fell with zero actions — check spawn height / joint defaults")

final_h = robot.data.root_pos_w[:, 2].mean().item()
print(f"  Final height (mean): {final_h:.4f} m")

print(f"\n{SEP}")
ok("Test complete")
print(SEP + "\n")

env.close()
sim_app.close()
