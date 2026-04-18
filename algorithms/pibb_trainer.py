"""
PIBB (Policy Improvement with Black-Box gradients) optimizer for Phase 1.

Optimises the CPG weight matrix W ∈ ℝ^(20×3) for a single gait using
parallel rollouts in Isaac Lab. At each iteration:
  1. Sample num_envs perturbations ε_i ~ N(0, σI).
  2. Set W_i = W + ε_i for each parallel env.
  3. Run one full episode and collect cumulative reward R_i per env.
  4. Update W using reward-weighted average of perturbations.
  5. Decay σ by noise_decay_rate.

Update rule (normalised REINFORCE / NES):
  h_i   = (R_i - mean(R)) / (std(R) + 1e-8)
  ΔW    = (1/n) * Σ_i [ h_i * ε_i ]
  W_new = W + ΔW

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# Leg labels for per-leg W-norm logging
_LEG_LABELS = ["FL", "FR", "RL", "RR"]


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds as  Xh Ym Zs  or  Ym Zs  or  Zs."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class PIBBTrainer:
    """
    PIBB optimizer for CPG Phase 1 training.

    Args:
        env:     A UnitreeB1Env instance (already constructed with the right gait).
        config:  Dict loaded from a phase1_*.yaml file (full config dict).
        log_dir: Root directory for TensorBoard logs and weight checkpoints.
    """

    def __init__(self, env, config: dict, log_dir: str | Path):
        self.env = env
        self.cfg = config

        pibb = config["pibb"]
        self.max_iterations: int    = pibb["max_iterations"]
        self.noise_init: float      = pibb["exploration_noise"]
        self.noise_decay: float     = pibb["noise_decay_rate"]
        self.convergence_thr: float = pibb["convergence_threshold"]
        self.pibb_h: float          = pibb.get("temperature", 10.0)
        self.init_var_boost: float  = pibb.get("init_var_boost", 2.0)

        self.num_rbf: int      = config.get("rbf", {}).get("num_neurons", 20)
        self.episode_steps: int = config["env"]["episode_length"]

        # Direct encoding: W shape (20, 12) — 20 RBF neurons × 12 joints
        self.num_joints: int = 12

        # Current best W — random init (LocoNets uses uniform(-0.1, 0.1))
        # Small random values give PIBB initial leg movements to explore from
        self.W = np.random.uniform(-0.1, 0.1,
            (self.num_rbf, self.num_joints)).astype(np.float32)

        # Running noise std
        self.sigma: float = self.noise_init

        # ---- Logging setup ----
        log_cfg = config.get("logging", {})
        log_root = Path(log_dir)
        run_name = log_cfg.get("run_name", "pibb")
        self.log_interval: int  = log_cfg.get("log_interval", 10)
        self.save_interval: int = log_cfg.get("save_interval", 100)

        # Each run gets a timestamped subfolder so TensorBoard shows all runs
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir = log_root / run_name / timestamp
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Write a scalar immediately so TensorBoard creates the event file now —
        # confirms the writer opened successfully before the long training loop.
        self.writer.add_text(
            "config/gait",
            f"gait={config['gait']['name']}  "
            f"num_envs={env.num_envs}  "
            f"σ_init={self.noise_init}  "
            f"decay={self.noise_decay}  "
            f"max_iter={self.max_iterations}",
            global_step=0,
        )
        self.writer.flush()
        logger.info("TensorBoard log dir: %s", tb_dir.resolve())

        # ---- Weight output ----
        self.weights_path = Path(config["output"]["weights_path"])
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PIBBTrainer — gait=%s  num_envs=%d  σ=%.4f  max_iter=%d",
            config["gait"]["name"],
            env.num_envs,
            self.sigma,
            self.max_iterations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> np.ndarray:
        """
        Run the full PIBB training loop.

        Returns:
            W: (num_slots, num_rbf, 3) optimised weight matrix.
        """
        best_reward = -np.inf
        reward_history: list[float] = []
        iter_times: list[float] = []

        self._print_header()

        train_start = time.time()

        for iteration in range(1, self.max_iterations + 1):
            t0 = time.time()

            # 1. Sample perturbations and set per-env weights
            perturbations = self._sample_perturbations(iteration)  # (num_envs, 20, 12)
            W_batch = self.W[None] + perturbations           # (num_envs, 20, 12)
            self.env.set_weights_batch(W_batch)

            # 2. Run one episode, collect cumulative rewards
            rewards = self._run_episode()                    # (num_envs,)

            # 3. PIBB weight update
            self._update_W(rewards, perturbations)

            # 4. Decay noise
            self.sigma *= self.noise_decay

            # 5. Bookkeeping
            mean_r = float(rewards.mean())
            std_r  = float(rewards.std())
            reward_history.append(mean_r)

            elapsed = time.time() - t0
            iter_times.append(elapsed)

            new_best = mean_r > best_reward
            if new_best:
                best_reward = mean_r
                np.save(self.weights_path, self.W)

            # --- Terminal log every iteration ---
            eta_s = self._eta(iter_times, iteration, self.max_iterations)
            self._print_iter(iteration, mean_r, std_r, best_reward, self.sigma, elapsed, eta_s, new_best)

            # --- TensorBoard every log_interval ---
            if iteration % self.log_interval == 0:
                self._log_tensorboard(iteration, rewards, mean_r, std_r, best_reward)

            # --- Checkpoint every save_interval ---
            if iteration % self.save_interval == 0:
                ckpt = self.weights_path.with_stem(
                    self.weights_path.stem + f"_iter{iteration}"
                )
                np.save(ckpt, self.W)
                logger.info("  >> Checkpoint saved → %s", ckpt)

            # --- Convergence check ---
            if self._has_converged(reward_history):
                logger.info(
                    "\n  Converged at iteration %d  (reward variance < %.4f)\n",
                    iteration, self.convergence_thr,
                )
                break

        total_time = time.time() - train_start
        self._print_footer(best_reward, total_time)

        np.save(self.weights_path, self.W)
        self.writer.flush()
        self.writer.close()
        return self.W

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Per-joint noise scaling for direct encoding (20×12).
    # 12 columns = [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    #               RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]
    # Hip columns (0,3,6,9) get 10× less noise.
    _JOINT_NOISE_SCALE = np.array(
        [0.1, 1.0, 0.8] * 4, dtype=np.float32   # repeated for 4 legs
    )  # shape (12,)

    def _sample_perturbations(self, iteration: int) -> np.ndarray:
        """Sample (num_envs, 20, 12) Gaussian noise with per-joint scaling.

        Hip columns get 10× less noise. First iteration uses init_var_boost.
        """
        scale = self.sigma * (self.init_var_boost if iteration == 1 else 1.0)
        noise = np.random.randn(
            self.env.num_envs, self.num_rbf, self.num_joints
        ).astype(np.float32)
        noise *= scale * self._JOINT_NOISE_SCALE[None, None, :]  # (1, 1, 12) broadcast
        return noise

    def _run_episode(self) -> np.ndarray:
        """
        Reset all envs and step for episode_steps steps.

        Returns:
            cumulative_rewards: (num_envs,) numpy array.
        """
        self.env.reset()
        cumulative = np.zeros(self.env.num_envs, dtype=np.float32)
        dummy_actions = torch.zeros(
            self.env.num_envs, self.env.cfg.action_space,
            device=self.env.device,
        )
        for _ in range(self.episode_steps):
            _, reward, _, _, _ = self.env.step(dummy_actions)
            cumulative += reward.cpu().numpy()

        return cumulative

    def _update_W(self, rewards: np.ndarray, perturbations: np.ndarray):
        """
        PI^BB cost-weighted averaging (Thor et al.):
          s_i = exp(h * (R_i - R_min) / (R_max - R_min + 1e-8))
          p_i = s_i / Σ s_i                    (softmax probability)
          W  += Σ_i [ p_i * ε_i ]              (weighted sum of perturbations)

        Temperature h controls exploitation: high h → focus on best rollouts.
        """
        r_min = rewards.min()
        r_max = rewards.max()
        r_range = r_max - r_min + 1e-8

        # Softmax with temperature h over normalised fitness
        s = np.exp(self.pibb_h * (rewards - r_min) / r_range)     # (num_envs,)
        p = s / s.sum()                                            # (num_envs,)

        # Weighted sum of perturbations
        # perturbations: (num_envs, 20, 12), p: (num_envs,)
        delta_W = (p[:, None, None] * perturbations).sum(axis=0)   # (20, 12)
        self.W += delta_W

    def _has_converged(self, reward_history: list[float], window: int = 50) -> bool:
        """Return True when reward variance over the last `window` iters is tiny."""
        if len(reward_history) < window:
            return False
        return float(np.array(reward_history[-window:]).var()) < self.convergence_thr

    # ------------------------------------------------------------------
    # Terminal formatting
    # ------------------------------------------------------------------

    def _print_header(self):
        gait = self.cfg["gait"]["name"]
        sep = "=" * 72
        print(sep)
        print(f"  PIBB Training  —  gait: {gait.upper()}")
        print(f"  num_envs : {self.env.num_envs}  (parallel rollouts per iteration)")
        print(f"  episode  : {self.episode_steps} steps  "
              f"({self.episode_steps * self.cfg['cpg']['dt']:.1f} s)")
        print(f"  σ_init   : {self.noise_init:.4f}   decay: {self.noise_decay:.4f}/iter   "
              f"h={self.pibb_h:.1f}   boost={self.init_var_boost:.1f}×")
        print(f"  max_iter : {self.max_iterations}")
        print(f"  output   : {self.weights_path}")
        print(sep)
        print(f"  {'iter':>6} | {'R_mean':>8} {'R_std':>8} {'R_best':>8} | "
              f"{'sigma':>8} | {'W_norm':>7} | {'iter_t':>7} | ETA")
        print("  " + "-" * 70, flush=True)

    def _print_iter(
        self,
        iteration: int,
        mean_r: float,
        std_r: float,
        best_r: float,
        sigma: float,
        elapsed: float,
        eta_s: float,
        new_best: bool,
    ):
        marker = " *" if new_best else "  "
        print(
            f"{marker}{iteration:6d} | {mean_r:8.3f} {std_r:8.3f} {best_r:8.3f} | "
            f"{sigma:8.5f} | {np.linalg.norm(self.W):7.3f} | "
            f"{elapsed:5.1f}s  | {_fmt_time(eta_s)}",
            flush=True,
        )

    def _print_footer(self, best_reward: float, total_time: float):
        sep = "=" * 72
        print("  " + "-" * 70)
        print("  Training complete")
        print(f"  Best reward : {best_reward:.4f}")
        print(f"  Total time  : {_fmt_time(total_time)}")
        print(f"  Saved       : {self.weights_path}")
        print(sep, flush=True)

    @staticmethod
    def _eta(iter_times: list[float], current: int, total: int) -> float:
        """Estimate remaining time from mean of last 20 iteration times."""
        recent = iter_times[-20:] if len(iter_times) >= 20 else iter_times
        mean_t = sum(recent) / len(recent)
        return mean_t * (total - current)

    def _log_tensorboard(
        self,
        iteration: int,
        rewards: np.ndarray,
        mean_r: float,
        std_r: float,
        best_r: float,
    ):
        """Log scalars, histograms, and per-leg W norms to TensorBoard."""
        # --- Reward stats ---
        self.writer.add_scalar("reward/mean",  mean_r,  iteration)
        self.writer.add_scalar("reward/std",   std_r,   iteration)
        self.writer.add_scalar("reward/best",  best_r,  iteration)
        self.writer.add_scalar("reward/min",   float(rewards.min()), iteration)
        self.writer.add_scalar("reward/max",   float(rewards.max()), iteration)

        # --- Exploration noise ---
        self.writer.add_scalar("noise/sigma", self.sigma, iteration)

        # --- Weight matrix (direct: 20×12) ---
        self.writer.add_scalar("weights/total_norm", float(np.linalg.norm(self.W)), iteration)
        leg_names = ["FL", "FR", "RL", "RR"]
        joint_names = ["hip", "thigh", "calf"]
        for k, leg in enumerate(leg_names):
            for j, jn in enumerate(joint_names):
                col = k * 3 + j
                self.writer.add_scalar(
                    f"weights/{leg}_{jn}", float(np.linalg.norm(self.W[:, col])), iteration
                )
        self.writer.add_histogram("weights/W_flat", self.W.flatten(), iteration)

        # --- Reward distribution across envs ---
        self.writer.add_histogram("reward/env_distribution", rewards, iteration)

        self.writer.flush()
