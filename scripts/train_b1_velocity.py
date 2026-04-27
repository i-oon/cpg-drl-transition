"""
Train a velocity-tracking PPO policy for Unitree B1 on flat terrain.

Task:       Isaac-Velocity-Flat-Unitree-B1-v0
Registered: envs/b1_velocity_env_cfg.py (via module import side-effect)
Config:     envs/b1_velocity_ppo_cfg.py  (RSL-RL PPO hyperparameters)

Follows Isaac Lab's canonical (non-Hydra) RSL-RL training pattern:
  parse_env_cfg → gym.make(cfg=env_cfg) → RslRlVecEnvWrapper → OnPolicyRunner

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_b1_velocity.py --headless --num_envs 4096
    python scripts/train_b1_velocity.py --headless --max_iterations 1500
    python scripts/train_b1_velocity.py --headless --resume path/to/model.pt

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train B1 velocity-tracking PPO")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-B1-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override max_iterations from the PPO config.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a .pt checkpoint to resume from.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Subdir name under logs/ppo_b1/; default = timestamp.")
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# Project root on sys.path so `envs.*` is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import torch

# isaaclab_tasks registers all stock MDP term entry points used by
# LocomotionVelocityRoughEnvCfg (our parent class).
import isaaclab_tasks  # noqa: F401

# Side-effect: registers Isaac-Velocity-Flat-Unitree-B1-v0 / -Play-v0
import envs.b1_velocity_env_cfg  # noqa: F401
from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

agent_cfg = B1FlatPPORunnerCfg()
agent_cfg.seed = args.seed
if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations

env_cfg = parse_env_cfg(
    args.task,
    device=agent_cfg.device,
    num_envs=args.num_envs,
)
env_cfg.seed = args.seed

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"\n  Task          : {args.task}")
print(f"  Num envs      : {env.num_envs}")
print(f"  Obs dim       : {env.num_obs}")
print(f"  Act dim       : {env.num_actions}")
print(f"  Episode len   : {env.max_episode_length} steps")
print(f"  Max iters     : {agent_cfg.max_iterations}")
print(f"  Device        : {agent_cfg.device}")

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------

run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(__file__).parent.parent / "logs" / "ppo_b1" / run_name
log_dir.mkdir(parents=True, exist_ok=True)
print(f"  Log dir       : {log_dir.resolve()}\n")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

runner = OnPolicyRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=str(log_dir),
    device=agent_cfg.device,
)

if args.resume is not None:
    print(f"  Resuming from : {args.resume}")
    runner.load(args.resume)

runner.learn(
    num_learning_iterations=agent_cfg.max_iterations,
    init_at_random_ep_len=True,
)

# Save final
final_path = log_dir / "model_final.pt"
runner.save(str(final_path))
print(f"\n  Saved final policy → {final_path}")

env.close()
sim_app.close()
