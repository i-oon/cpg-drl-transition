"""
Train Phase 2 E2E rate-based α policy.

MLP outputs dα/dt via sigmoid; α is integrated monotonically from 0 each
episode. The MLP controls HOW FAST the transition happens rather than the
raw α value. This prevents the instant-jump collapse seen in E2E-sigmoid
while keeping the transition shape fully learned (no smoothstep assumption).

At max rate (sigmoid=1): α reaches 1 in exactly transition_duration_s (3 s).
At min rate (sigmoid=0): α stays at 0 (degenerate: stay at source gait).

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_b1_phase2_e2e_rate.py --headless --num_envs 2048 \
        --max_iterations 1500 --run_name e2e_rate_v1

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train B1 Phase 2 E2E rate-based α")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-B1-Phase2-E2E-Rate-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import envs.b1_phase2_env  # noqa: F401
from envs.b1_phase2_env_cfg import B1Phase2E2ERateEnvCfg
from envs.b1_velocity_ppo_cfg import Phase2E2ERatePPORunnerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

agent_cfg = Phase2E2ERatePPORunnerCfg()
agent_cfg.seed = args.seed
agent_cfg.max_iterations = args.max_iterations

env_cfg = B1Phase2E2ERateEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = agent_cfg.device
env_cfg.seed = args.seed

for path in env_cfg.base_policy_paths:
    p = Path(__file__).parent.parent / path
    if not p.exists():
        print(f"ERROR: Base policy not found: {path}")
        sys.exit(1)
    print(f"  Found base policy: {p}")

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"\n  Task       : {args.task}")
print(f"  Num envs   : {env.num_envs}")
print(f"  Obs dim    : {env.num_obs}")
print(f"  Action dim : {env.num_actions}  (dα/dt rate, integrated each step)")
print(f"  Episode len: {env.max_episode_length} steps")
print(f"  Max iters  : {agent_cfg.max_iterations}")

run_name = args.run_name or ("e2e_rate_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
log_dir = Path(__file__).parent.parent / "logs" / "phase2" / run_name
log_dir.mkdir(parents=True, exist_ok=True)
print(f"  Log dir    : {log_dir.resolve()}\n")

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

final_path = log_dir / "model_final.pt"
runner.save(str(final_path))
print(f"\n  Saved final policy → {final_path}")

env.close()
sim_app.close()
