"""
RSL-RL PPO hyperparameters for B1 velocity-tracking training.

Mirrors Isaac Lab's stock UnitreeGo2FlatPPORunnerCfg but with deeper nets
(B1 is ~50 kg vs Go2's ~15 kg — more capacity to discover stable gaits)
and a higher iteration budget.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class B1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "b1_flat"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # entropy_coef reduced from stock 0.01 to 0.005 as a prophylactic
        # against bang-bang noise-std runaway on heavy robots. Combined with
        # action_rate_l2 = -0.025 in the env cfg, the analytic noise-std
        # equilibrium σ² = entropy/(4·w·dt) ≈ 2.5 → σ ≈ 1.6. If training
        # still shows Policy/mean_noise_std climbing past 2 by iter ~300,
        # drop this to 0.001 (external project confirmed it caps σ ≈ 0.7).
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Phase2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2 residual MLP — small network for 4-D action."""
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "b1_phase2_transition"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,                     # smaller than Phase 1's 1.0 — residuals are small corrections
        actor_hidden_dims=[128, 128],           # matches CLAUDE.md spec
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,                   # smaller LR for residual fine-tuning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Phase2E2EPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """E2E PPO baseline — learns full blending scalar α from scratch.

    Same backbone as Phase2PPORunnerCfg ([128, 128]) but action_space=1,
    no residual structure, no hand-designed ramp. Trains from scratch.
    """
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "b1_phase2_e2e"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                     # from scratch → higher init noise
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,                   # standard LR — training from scratch
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Phase2Residual1DPPORunnerCfg(Phase2PPORunnerCfg):
    """Ablation: scalar residual (1-D Δα broadcast to all legs).

    Identical to Phase2PPORunnerCfg except experiment_name. Same LR, same
    network size, same smoothstep baseline — only the action dimension differs.
    """
    experiment_name = "b1_phase2_residual1d"


@configclass
class Phase2E2ERatePPORunnerCfg(Phase2E2EPPORunnerCfg):
    """E2E rate-based α — MLP outputs dα/dt, α integrated from 0.

    Same backbone and LR as Phase2E2EPPORunnerCfg. Training from scratch
    since the rate-integration structure is a new action semantics.
    """
    experiment_name = "b1_phase2_e2e_rate"
