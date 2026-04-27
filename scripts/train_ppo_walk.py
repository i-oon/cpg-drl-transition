"""
Phase 1 – Train walk gait using PPO (RSL-RL) with CPG-RBF structure.

PPO policy outputs W (60D) → CPG generates joint targets → robot walks.

Run:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_ppo_walk.py --headless --num_envs 4096

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train walk via PPO + CPG")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=300)
args = parser.parse_args()
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- Safe to import now ----
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from envs.unitree_b1_env import UnitreeB1Env, UnitreeB1EnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

cfg = UnitreeB1EnvCfg()
cfg.gait_name = "walk"
cfg.phase_offsets = [0.0, math.pi, math.pi / 2, 3 * math.pi / 2]
cfg.episode_length_s = 20.0
cfg.scene = InteractiveSceneCfg(
    num_envs=args.num_envs, env_spacing=2.5, replicate_physics=True
)

env = UnitreeB1Env(cfg)
env = RslRlVecEnvWrapper(env)

print(f"\n  Env: {env.num_envs} envs, obs={env.num_obs}, act={env.num_actions}")
print(f"  Max episode length: {env.max_episode_length}")

# ---------------------------------------------------------------------------
# PPO config (Isaac Lab RSL-RL format)
# ---------------------------------------------------------------------------

from isaaclab.utils import configclass

@configclass
class CpgPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = args.max_iterations
    save_interval = 50
    experiment_name = "cpg_ppo_walk"
    empirical_normalization = False
    seed = 42
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

train_cfg = CpgPpoRunnerCfg()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

log_dir = Path("logs/phase1/ppo_walk")
log_dir.mkdir(parents=True, exist_ok=True)

runner = OnPolicyRunner(env, train_cfg.to_dict(), log_dir=str(log_dir), device=env.device)

print(f"\n  PPO training — {args.max_iterations} iterations")
print(f"  Policy: MLP {train_cfg.policy.actor_hidden_dims}")
print(f"  Log dir: {log_dir}\n")

runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

# Save the trained policy
policy_path = log_dir / "trained_policy.pt"
runner.save(str(policy_path))
print(f"\n  Policy saved → {policy_path}")

env.close()
sim_app.close()
