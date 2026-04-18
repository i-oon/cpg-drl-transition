"""
Quick test: does the B1 stand and walk with unified thigh defaults (0.9 rad)?

Test 1: Standing stability — robot spawns with all thighs at 0.9, no CPG, 200 steps
  PASS: height stays within ±5cm of spawn, tilt < 0.05
  FAIL: robot falls or tilts excessively → asymmetry is structural

Test 2: Walk with trained W — load W_walk, run 500 steps
  Compare vx, height, duty factors against the original 0.8/1.0 results

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/test_unified_default.py --headless

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test unified thigh defaults")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from envs.unitree_b1_env import UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg

# Monkey-patch the B1 init_state to use unified thigh = 0.9
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG
from isaaclab.assets.articulation import ArticulationCfg

UNITREE_B1_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.42),
    joint_pos={
        ".*L_hip_joint": 0.1,
        ".*R_hip_joint": -0.1,
        ".*_thigh_joint": 0.9,       # ← UNIFIED (was F=0.8, R=1.0)
        ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)

print("\n" + "=" * 60)
print("  TEST: Unified thigh defaults (all = 0.9 rad)")
print("=" * 60)

# Build env
cfg = UnitreeB1EnvCfg()
cfg.gait_name = "walk"
cfg.phase_offsets = [0.0, math.pi, math.pi / 2, 3 * math.pi / 2]
cfg.scene = InteractiveSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=True)
env = UnitreeB1Env(cfg)

# =====================================================================
# TEST 1: Standing stability (no CPG, W=0)
# =====================================================================
print("\n--- TEST 1: Standing stability (W=0, 200 steps) ---")

env.reset()
dummy = torch.zeros(4, 12, device=env.device)

heights = []
tilts = []
for step in range(200):
    env.step(dummy)
    h = env._robot.data.root_pos_w[:, 2].mean().item()
    t = torch.sum(torch.square(env._robot.data.projected_gravity_b[:, :2]), dim=1).mean().item()
    heights.append(h)
    tilts.append(t)

h_arr = np.array(heights)
t_arr = np.array(tilts)

print(f"  Height: mean={h_arr.mean():.3f}  std={h_arr.std():.4f}  "
      f"min={h_arr.min():.3f}  max={h_arr.max():.3f}")
print(f"  Tilt:   mean={t_arr.mean():.4f}  max={t_arr.max():.4f}")

# Check defaults
default_pos = env._robot.data.default_joint_pos[0]
print(f"\n  Default joint positions (verify unified):")
for i, name in enumerate(env._robot.joint_names):
    if "thigh" in name:
        print(f"    {name:25s}  {default_pos[i].item():+.4f} rad")

standing_pass = h_arr.min() > 0.35 and t_arr.max() < 0.05
print(f"\n  STANDING TEST: {'PASS' if standing_pass else 'FAIL'}")

if not standing_pass:
    print("  → Asymmetry is structural. Revert to 0.8/1.0.")
    sim_app.close()
    sys.exit(0)

# =====================================================================
# TEST 2: Walk with trained W
# =====================================================================
print("\n--- TEST 2: Walk playback (W_walk, 500 steps) ---")

weights_path = Path("weights/W_walk.npy")
if not weights_path.exists():
    print("  W_walk.npy not found — skipping walk test")
    sim_app.close()
    sys.exit(0)

W = np.load(weights_path)
env.set_weights(W)
env.reset()

vx_list = []
h_list = []
contact_list = []

for step in range(500):
    env.step(dummy)
    vx_list.append(env._robot.data.root_lin_vel_b[:, 0].mean().item())
    h_list.append(env._robot.data.root_pos_w[:, 2].mean().item())

    if env._contact_sensor is not None and env._foot_ids is not None:
        net_forces = env._contact_sensor.data.net_forces_w_history
        foot_forces = torch.norm(net_forces[:, 0, :, :], dim=-1)
        in_contact = (foot_forces[0, env._foot_ids] > 1.0).cpu().numpy()
        contact_list.append(in_contact)

vx_arr = np.array(vx_list)
h_arr = np.array(h_list)

print(f"  Forward velocity: mean={vx_arr.mean():+.3f}  std={vx_arr.std():.3f}")
print(f"  Height:           mean={h_arr.mean():.3f}  std={h_arr.std():.4f}")

if contact_list:
    contact_arr = np.array(contact_list)
    duty = contact_arr.mean(axis=0) * 100
    print(f"\n  Duty factors:")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        print(f"    {lab}: {duty[i]:.1f}%")

    front_mean = duty[:2].mean()
    rear_mean = duty[2:].mean()
    gap = abs(rear_mean - front_mean)
    print(f"\n  Front avg: {front_mean:.1f}%   Rear avg: {rear_mean:.1f}%   Gap: {gap:.1f}%")

    # Compare against original results (indirect encoding, 0.8/1.0 defaults)
    print(f"\n  Original (0.8/1.0 defaults): FL=45% FR=56% RL=80% RR=84%  Gap=31.6%")
    print(f"  Unified  (0.9/0.9 defaults): FL={duty[0]:.0f}% FR={duty[1]:.0f}% "
          f"RL={duty[2]:.0f}% RR={duty[3]:.0f}%  Gap={gap:.1f}%")

    if gap < 20:
        print(f"\n  WALK TEST: IMPROVED (front/rear gap reduced from 31.6% to {gap:.1f}%)")
    else:
        print(f"\n  WALK TEST: NO IMPROVEMENT (gap still {gap:.1f}%)")

print("\n" + "=" * 60)
sim_app.close()
