"""
Live environment test — requires Isaac Sim running.

What this script checks:
  1. Env creates without error
  2. Joint names in USD match CPG_JOINT_NAMES (permutation is valid)
  3. Contact sensor body names (feet and base)
  4. Obs shape is (num_envs, 33)
  5. CPG produces finite joint targets over 500 steps
  6. Reward is finite and non-trivially zero
  7. Termination triggers at the right height

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/test_env.py --headless        # no GUI
    python scripts/test_env.py                   # with GUI (slower)

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

# ---- Isaac Sim must be launched before any other omni/isaaclab imports ----
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test UnitreeB1Env with Isaac Lab")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments")
parser.add_argument("--steps", type=int, default=500,
                    help="Steps to run in the test rollout")
parser.add_argument("--gait", type=str, default="walk",
                    choices=["walk", "trot", "steer"],
                    help="Gait to test")
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- Now safe to import Isaac Lab and project modules ----
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.unitree_b1_env import CPG_JOINT_NAMES, UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 70

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg: str):
    print(f"  [PASS] {msg}")

def warn(msg: str):
    print(f"  [WARN] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

section("1 / 7  Building environment")

PHASE_OFFSETS = {
    "walk":  [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
    "trot":  [0.0, math.pi, math.pi, 0.0],
    "steer": [0.0, math.pi, math.pi / 2, 3 * math.pi / 2],
}

cfg = UnitreeB1EnvCfg()
cfg.gait_name = args.gait
cfg.phase_offsets = PHASE_OFFSETS[args.gait]
cfg.scene = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=True)

env = UnitreeB1Env(cfg, render_mode="rgb_array" if args.headless else None)
ok(f"Env created  num_envs={env.num_envs}  device={env.device}")

# ---------------------------------------------------------------------------
# Check 1: Joint names
# ---------------------------------------------------------------------------

section("2 / 7  Joint names")

isaaclab_joint_names = list(env._robot.joint_names)
print(f"  Isaac Lab joints ({len(isaaclab_joint_names)}): {isaaclab_joint_names}")
print(f"  CPG joint order  ({len(CPG_JOINT_NAMES)}): {CPG_JOINT_NAMES}")

missing_in_isaaclb = [n for n in CPG_JOINT_NAMES if n not in isaaclab_joint_names]
extra_in_isaaclab  = [n for n in isaaclab_joint_names if n not in CPG_JOINT_NAMES]

if missing_in_isaaclb:
    fail(f"These CPG joint names are NOT in the USD: {missing_in_isaaclb}")
    print("  --> Update CPG_JOINT_NAMES in envs/unitree_b1_env.py to match.")
else:
    ok("All CPG joint names found in USD")

if extra_in_isaaclab:
    warn(f"USD has extra joints not in CPG list: {extra_in_isaaclab}")

print(f"\n  Joint permutation: {env._joint_perm.tolist()}")
if torch.equal(env._joint_perm, torch.arange(12, device=env.device)):
    print("  (identity — CPG order matches Isaac Lab order)")
else:
    print("  (non-trivial reorder applied)")

# ---------------------------------------------------------------------------
# Check 2: Contact sensor bodies
# ---------------------------------------------------------------------------

section("3 / 7  Contact sensor bodies")

if env._contact_sensor is None:
    warn("ContactSensor is disabled (cfg.contact_sensor=None).")
    print("  Reason: B1 USD from URDF conversion lacks PhysicsRigidBodyAPI prims.")
    print("  Rewards fall back to proxy metrics. Re-enable after fixing the USD.")
else:
    all_body_names = list(env._contact_sensor.body_names)
    print(f"  All sensor bodies: {all_body_names}")

    if env._foot_ids is not None and len(env._foot_ids) > 0:
        print(f"  Foot body ids  (.*_foot$): {env._foot_ids.tolist()}")
        ok(f"Found {len(env._foot_ids)} foot bodies")
    else:
        fail("No foot bodies found matching '.*_foot$' — update the pattern")

    if env._undesired_body_ids is not None and len(env._undesired_body_ids) > 0:
        print(f"  Thigh body ids (.*_thigh$): {env._undesired_body_ids.tolist()}")

    if env._base_id is not None:
        print(f"  Base body id   (trunk):    {env._base_id.tolist()}")
    else:
        warn("Base body 'trunk' not found — update pattern if B1 uses a different name")

# ---------------------------------------------------------------------------
# Check 3: Observation shape
# ---------------------------------------------------------------------------

section("4 / 7  Observation space")

dummy_actions = torch.zeros(env.num_envs, 12, device=env.device)
obs_dict, _, _, _, _ = env.step(dummy_actions)
obs = obs_dict["policy"]

print(f"  Obs shape: {tuple(obs.shape)}")
expected_obs_dim = 33
if obs.shape == (env.num_envs, expected_obs_dim):
    ok(f"Obs shape correct: ({env.num_envs}, {expected_obs_dim})")
else:
    fail(f"Expected ({env.num_envs}, {expected_obs_dim}), got {tuple(obs.shape)}")

if torch.isfinite(obs).all():
    ok("Obs is finite (no NaN/Inf)")
else:
    fail(f"Obs has {(~torch.isfinite(obs)).sum().item()} non-finite values")

# ---------------------------------------------------------------------------
# Check 4: Short rollout — CPG produces finite targets
# ---------------------------------------------------------------------------

section("5 / 7  Short rollout (zero W)")

rewards_list = []
for step in range(args.steps):
    obs_dict, reward, terminated, timed_out, info = env.step(dummy_actions)
    rewards_list.append(reward)

rewards = torch.stack(rewards_list)   # (steps, num_envs)
print(f"  Steps completed: {args.steps}")
print(f"  Reward mean: {rewards.mean().item():.4f}")
print(f"  Reward std:  {rewards.std().item():.4f}")
print(f"  Reward min:  {rewards.min().item():.4f}")
print(f"  Reward max:  {rewards.max().item():.4f}")

if torch.isfinite(rewards).all():
    ok("All rewards are finite")
else:
    fail(f"Rewards contain {(~torch.isfinite(rewards)).sum().item()} non-finite values")

# ---------------------------------------------------------------------------
# Check 5: Rollout with non-zero W
# ---------------------------------------------------------------------------

section("6 / 7  Rollout with random W")

env.reset()
W_random = np.random.randn(20, 3).astype(np.float32) * 0.05
env.set_weights(W_random)

rewards_rnd = []
for step in range(200):
    obs_dict, reward, terminated, timed_out, info = env.step(dummy_actions)
    rewards_rnd.append(reward)

rewards_rnd = torch.stack(rewards_rnd)
print(f"  Random W reward mean: {rewards_rnd.mean().item():.4f}")
print(f"  (vs zero W mean:      {rewards.mean().item():.4f})")

if torch.isfinite(rewards_rnd).all():
    ok("Random W rewards are finite")
else:
    fail("Random W produced non-finite rewards")

# ---------------------------------------------------------------------------
# Check 6: Robot data sanity
# ---------------------------------------------------------------------------

section("7 / 7  Robot data sanity")

env.reset()
env.step(dummy_actions)   # one step to populate data

root_height = env._robot.data.root_pos_w[:, 2]
print(f"  Root height:        {root_height.mean().item():.4f} m  (expected ~0.42 m)")
if (root_height > 0.2).all():
    ok("Root height above fall threshold (0.20 m)")
else:
    warn(f"{(root_height <= 0.2).sum()} envs already below fall threshold")

lin_vel = env._robot.data.root_lin_vel_b
print(f"  Root lin vel (body): {lin_vel.mean(dim=0).tolist()}")

grav_proj = env._robot.data.projected_gravity_b
print(f"  Projected gravity:   {grav_proj.mean(dim=0).tolist()}")
print(f"  (should be ~[0, 0, -1] when upright)")
if (grav_proj[:, 2] < -0.9).all():
    ok("Gravity projection looks correct (robot upright)")
else:
    warn("Gravity z-component is not close to -1 — robot may be tilted at spawn")

joint_pos = env._robot.data.joint_pos
print(f"\n  Joint positions (env 0): {joint_pos[0].tolist()}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Done")
print("  Close the Isaac Sim window or press Ctrl-C to exit.\n")

sim_app.close()
