"""
Diagnose what the trained CPG is actually doing.

Prints: default joint positions, CPG targets, final commanded positions,
and joint limits — to understand why the robot does push-ups.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/diagnose_cpg.py --headless

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose CPG targets")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from envs.unitree_b1_env import CPG_JOINT_NAMES, UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg

# Build env
cfg = UnitreeB1EnvCfg()
cfg.gait_name = "walk"
cfg.phase_offsets = [0.0, math.pi, math.pi / 2, 3 * math.pi / 2]
cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
env = UnitreeB1Env(cfg)

# Load trained weights
W = np.load("weights/W_walk.npy")
env.set_weights(W)

print("\n" + "=" * 70)
print("  CPG DIAGNOSIS")
print("=" * 70)

# 1. Default joint positions
default_pos = env._robot.data.default_joint_pos[0]
print(f"\n  Default joint positions (Isaac Lab order):")
for i, name in enumerate(env._robot.joint_names):
    print(f"    {name:25s}  {default_pos[i].item():+.4f} rad  ({math.degrees(default_pos[i].item()):+.1f}°)")

# 2. Joint limits
lower = env._robot.data.soft_joint_pos_limits[0, :, 0]
upper = env._robot.data.soft_joint_pos_limits[0, :, 1]
print(f"\n  Joint limits (Isaac Lab order):")
for i, name in enumerate(env._robot.joint_names):
    print(f"    {name:25s}  [{lower[i].item():+.3f}, {upper[i].item():+.3f}] rad")

# 3. W matrix
print(f"\n  W matrix (20×3) — norm = {np.linalg.norm(W):.3f}")
print(f"    Column norms: hip={np.linalg.norm(W[:, 0]):.3f}  "
      f"thigh={np.linalg.norm(W[:, 1]):.3f}  calf={np.linalg.norm(W[:, 2]):.3f}")
print(f"    W range: [{W.min():.3f}, {W.max():.3f}]")

# 4. Step CPG for 20 steps and show targets
print(f"\n  CPG targets over 20 steps (env 0, CPG order: FL_hip FL_thigh FL_calf ...):")
env.reset()
dummy = torch.zeros(1, 12, device=env.device)

for step in range(20):
    cpg_targets = env._step_cpg_batch()  # (1, 12) in Isaac Lab order
    # Convert back to CPG order for readability
    inv_perm = torch.argsort(env._joint_perm)
    cpg_order = cpg_targets[0, inv_perm]

    commanded = default_pos + cpg_targets[0]

    if step % 5 == 0:
        print(f"\n  Step {step:3d}:")
        print(f"    CPG offsets (CPG order):  {[f'{v:+.3f}' for v in cpg_order.tolist()]}")
        print(f"    Commanded (IsaacLab):     {[f'{v:+.3f}' for v in commanded.tolist()]}")

        # Check if any commanded positions are outside joint limits
        below = commanded < lower
        above = commanded > upper
        if below.any() or above.any():
            violations = []
            for i in range(12):
                if below[i]:
                    violations.append(f"{env._robot.joint_names[i]}: {commanded[i].item():.3f} < {lower[i].item():.3f}")
                if above[i]:
                    violations.append(f"{env._robot.joint_names[i]}: {commanded[i].item():.3f} > {upper[i].item():.3f}")
            print(f"    ⚠ LIMIT VIOLATIONS: {violations}")

# 5. Actual robot state after a few steps
env.reset()
for _ in range(50):
    env.step(dummy)

actual_pos = env._robot.data.joint_pos[0]
print(f"\n  Actual joint positions after 50 steps:")
for i, name in enumerate(env._robot.joint_names):
    print(f"    {name:25s}  actual={actual_pos[i].item():+.4f}  "
          f"default={default_pos[i].item():+.4f}  "
          f"diff={actual_pos[i].item() - default_pos[i].item():+.4f}")

root_h = env._robot.data.root_pos_w[0, 2].item()
vx = env._robot.data.root_lin_vel_b[0, 0].item()
print(f"\n  Root height: {root_h:.3f} m    vx: {vx:+.3f} m/s")

print("\n" + "=" * 70)
sim_app.close()
