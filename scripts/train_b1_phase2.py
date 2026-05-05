"""
Train Phase 2 residual transition policy.

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/train_b1_phase2.py --headless --num_envs 1024
    python scripts/train_b1_phase2.py --headless --max_iterations 2000
    python scripts/train_b1_phase2.py --headless --resume <path>

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train B1 Phase 2 residual transition policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-B1-Phase2-Transition-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override max_iterations from the PPO config.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--warmstart", type=str, default=None,
                    help="Path to a checkpoint trained with a smaller obs dim. "
                         "The first linear layer of actor and critic is extended "
                         "by appending zero-initialised columns for new obs dims. "
                         "Noise std is reset to the configured init_noise_std. "
                         "Use this to warm-start v11 (46-D obs) from v10 (45-D obs).")
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import envs.b1_phase2_env  # noqa: F401  -- registers gym ID
from envs.b1_velocity_ppo_cfg import Phase2PPORunnerCfg, Phase2V11PPORunnerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

def _load_warmstart(runner: OnPolicyRunner, path: str,
                    old_obs_dim: int, new_obs_dim: int,
                    init_noise_std: float) -> None:
    """Load a checkpoint trained with old_obs_dim and adapt it to new_obs_dim.

    The first linear layer of actor and critic is extended: weights for the
    original dims are preserved; weights for new dims are zero-initialised
    (so the policy initially ignores the new features and learns them from
    scratch on top of the already-competent base policy).

    Noise std is reset to init_noise_std rather than inheriting the
    checkpoint's converged (potentially very small) value.

    The optimizer is NOT loaded — training restarts with a fresh Adam state,
    which avoids momentum from the old training trajectory interfering with
    the new curriculum.
    """
    loaded = torch.load(path, map_location=runner.device, weights_only=False)
    state = loaded["model_state_dict"]

    if old_obs_dim != new_obs_dim:
        for key in ("actor.0.weight", "critic.0.weight"):
            if key not in state:
                print(f"  Warning: key '{key}' not found in checkpoint — skipping dim adapt")
                continue
            old_w = state[key]                                  # (hidden_dim, old_obs_dim)
            new_w = torch.zeros(old_w.shape[0], new_obs_dim,
                                dtype=old_w.dtype, device=old_w.device)
            new_w[:, :old_obs_dim] = old_w
            state[key] = new_w

    # Reset noise std — handle both 'std' (linear) and 'log_std' (log-space) conventions
    for std_key in ("std", "log_std"):
        if std_key in state:
            if std_key == "log_std":
                import math
                state[std_key] = torch.full_like(state[std_key], math.log(init_noise_std))
            else:
                state[std_key] = torch.full_like(state[std_key], init_noise_std)
            break

    runner.alg.actor_critic.load_state_dict(state)
    print(f"  Warm-started from : {path}")
    print(f"  Obs dim adapted   : {old_obs_dim} → {new_obs_dim}")
    print(f"  Noise std reset to: {init_noise_std}")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

# Select PPO config based on task so v11's hyperparameters (entropy=0, tight
# noise, longer budget) are used automatically without needing a separate script.
agent_cfg = Phase2V11PPORunnerCfg() if "V11" in args.task else Phase2PPORunnerCfg()
agent_cfg.seed = args.seed
if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations

# Phase 2 env cfg has its own scene defined in the cfg class — instantiate directly
from envs.b1_phase2_env_cfg import B1Phase2EnvCfg
env_cfg = B1Phase2EnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = agent_cfg.device
env_cfg.seed = args.seed

# Verify all four base policy checkpoints exist
for path in env_cfg.base_policy_paths:
    p = Path(__file__).parent.parent / path
    if not p.exists():
        print(f"ERROR: Base policy checkpoint not found: {path}")
        print("       Run Phase 1 training first and copy checkpoints to logs/phase1_final/")
        sys.exit(1)
    print(f"  Found base policy: {p}")

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

print(f"\n  Task          : {args.task}")
print(f"  Num envs      : {env.num_envs}")
print(f"  Obs dim       : {env.num_obs}")
print(f"  Action dim    : {env.num_actions}  (per-leg Δα)")
print(f"  Episode len   : {env.max_episode_length} steps")
print(f"  Max iters     : {agent_cfg.max_iterations}")
print(f"  Device        : {agent_cfg.device}")

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------

run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(__file__).parent.parent / "logs" / "phase2" / run_name
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

if args.warmstart is not None:
    _load_warmstart(
        runner,
        path=args.warmstart,
        old_obs_dim=45,
        new_obs_dim=env_cfg.observation_space,
        init_noise_std=agent_cfg.policy.init_noise_std,
    )

runner.learn(
    num_learning_iterations=agent_cfg.max_iterations,
    init_at_random_ep_len=True,
)

final_path = log_dir / "model_final.pt"
runner.save(str(final_path))
print(f"\n  Saved final policy → {final_path}")

env.close()
sim_app.close()
