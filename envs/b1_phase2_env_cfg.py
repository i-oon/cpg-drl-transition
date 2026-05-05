"""
Phase 2 environment config — per-leg residual transition learning.

Loads four frozen base policies (trot, bound, pace, steer) from
logs/phase1_final/. The Phase 2 policy outputs a 4-D per-leg Δα residual
correction. The env internally:
  1. samples (current_gait, target_gait) at episode start
  2. ramps α_baseline from 0 → 1 over 3 s
  3. queries both base policies for joint-offset actions
  4. blends per leg: α_k = clip(α_base + Δα_k, 0, 1)
  5. applies blended joint targets to the robot

Defines a DirectRLEnv (not manager-based) because manager-based actions
are applied directly to actuators; we need a custom step to interpose
the per-leg blending logic between policy output and actuators.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Reuse the Phase 1 local B1 cfg (deep-copied with stiffness=400 etc.)
from envs.b1_velocity_env_cfg import UNITREE_B1_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class B1Phase2SceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with B1 + foot contact sensor."""

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    robot: ArticulationCfg = UNITREE_B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@configclass
class B1Phase2EnvCfg(DirectRLEnvCfg):
    """Phase 2 — per-leg residual transition learning.

    The 4-D action is (Δα_FL, Δα_FR, Δα_RL, Δα_RR), each ∈ [-0.2, +0.2]
    (bounded by tanh × 0.2 inside the env).
    """

    # --- DirectRLEnvCfg required fields ---
    decimation: int = 4                      # 50 Hz control (sim 200 Hz)
    episode_length_s: float = 10.0           # 500 control steps per episode

    # 4-D residual correction action (per-leg Δα)
    action_space: int = 4

    # Observation:
    #   base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
    #   joint_pos_rel(12) + joint_vel(12) + last_action(4) +
    #   gait_current_onehot(3) + gait_target_onehot(3) +
    #   alpha_baseline(1) + cycles_elapsed(1) = 45 D
    # (3-gait portfolio: trot/bound/pace — steer dropped because its
    # training velocity command range doesn't match Phase 2's fixed cmd.)
    observation_space: int = 45

    # No state-dependent state space (single agent)
    state_space: int = 0

    # --- Sim ---
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # --- Scene ---
    scene: B1Phase2SceneCfg = B1Phase2SceneCfg(num_envs=1024, env_spacing=3.5)

    # --- Phase 2 specific ---
    # Paths to frozen Phase 1 base policy checkpoints (3 coordinations only).
    # Steer dropped: it was trained with yaw=(0.4, 1.0) command range —
    # Phase 2's fixed yaw=0 puts it out-of-distribution → garbage joint
    # targets → blend fails. The three retained gaits (trot/bound/pace)
    # all share the forward-only training distribution and produce
    # coherent outputs under the same Phase 2 velocity command.
    base_policy_paths: tuple = (
        "logs/phase1_final/trot.pt",
        "logs/phase1_final/bound.pt",
        "logs/phase1_final/pace.pt",
    )

    # Order of gaits (used for one-hot encoding) — 3 entries
    gait_names: tuple = ("trot", "bound", "pace")

    # Transition timing
    # transition_start_s is sampled UNIFORMLY in [transition_start_min_s, transition_start_max_s]
    # at episode reset — gives policy robustness to different starting moments
    transition_start_min_s: float = 1.5      # min current-gait stance hold
    transition_start_max_s: float = 3.5      # max current-gait stance hold
    # α_baseline ramp duration. When min == max (default), every episode uses
    # the same fixed duration. Set min < max to sample uniformly each episode
    # (curriculum training — see B1Phase2V11EnvCfg).
    transition_duration_min_s: float = 3.0
    transition_duration_max_s: float = 3.0
    # v7: α_baseline schedule — "linear" (constant dα/dt) or "smoothstep"
    # (Hermite 3x²−2x³, derivative=0 at both endpoints → smoother kinematic
    # blend, especially near α=1 where the linear ramp produced tilt spikes).
    # "e2e": no hand-designed baseline; MLP outputs full α directly (used by
    # B1Phase2E2EEnvCfg for the E2E PPO baseline comparison).
    alpha_schedule: str = "smoothstep"       # "linear" | "smoothstep" | "e2e"

    # Δα bounding.
    # v1 (tanh, max=0.2): too tight, MLP saturated and stood still.
    # v2 (tanh, max=0.8): wider, working transitions but allowed delay-rush exploit.
    # v3-v9 (tanh, max=0.8): kept 0.8 with various reward tweaks.
    # v10 (sigmoid, max=0.3): asymmetric clamp — Δα ∈ [0, 0.3] only.
    #   Sigmoid removes negative-Δα capability (no more delay below smoothstep).
    #   max=0.3 prevents E2E-style aggressive ramp compression.
    #   Net constraint: α_baseline ≤ α_per_joint ≤ α_baseline + 0.3.
    delta_alpha_max: float = 0.3

    # Time-gating of residual — Δα is FORCED to zero outside the transition
    # window so the source/target gaits run untouched during steady-state
    # holds. Window is [transition_start_s - pad, transition_start_s +
    # transition_duration_s + pad]. Outside this, Δα = 0 regardless of MLP
    # output → source gait visible at full purity during the hold window.
    residual_window_padding_s: float = 0.05

    # Joint position action scale (matches Phase 1 base policies)
    action_scale: float = 0.25

    # Velocity command for base policies (we hold a fixed velocity during
    # transition so the base policies have a sensible setpoint)
    velocity_cmd_x: float = 0.4
    velocity_cmd_y: float = 0.0
    velocity_cmd_yaw: float = 0.0

    # Reward weights
    rew_track_lin_vel: float = 1.5
    rew_track_ang_vel: float = 0.75
    rew_orientation: float = -2.0
    # v6 polish: orientation penalty multiplier applied DURING the transition
    # window only (so the MLP sees a strong gradient to suppress mid-window
    # tilt spikes; steady-state orientation pressure stays at 1×).
    # Effective in-window weight = rew_orientation × (1 + transition_orientation_boost).
    transition_orientation_boost: float = 3.0
    rew_height: float = -50.0
    # v5 polish: tightened from -0.05 → -0.15 to suppress Δα jitter at switch
    # instants (was the source of v4's tilt spikes).
    rew_action_rate: float = -0.15           # per-leg Δα step-to-step smoothness
    # v5 polish: joint-acceleration L2 — penalizes acceleration MAGNITUDE
    # (despite the old comment claiming it targets jerk, it does not — a
    # constant-high-acc trajectory has zero jerk yet large jacc²). Kept
    # as a defensive PD-overshoot penalty.
    rew_joint_acc: float = -2.5e-7
    # v8: Joint JERK L2 — the actual smoothness metric (rad/s³).
    # jerk = (q̈_t − q̈_{t-1}) / control_dt; penalises rapid acceleration
    # reversals which are what stress motor housings/gears.
    #
    # Tuning history:
    #   v8 (-5e-10): too strong → MLP found "compressed-jump" strategy
    #     (Δα→±0.4 to spike α to ≈1 at window start, minimising mid-α time)
    #     Result: jerk 10291, vx 0.398 (lost all v7 tracking benefit)
    #   v9 (-1e-10): 5× lighter pressure — keeps v7's per-leg coordination
    #     corrections, pays a token jerk cost.  Aim: vx ≈ 0.42, jerk ≈ 9500.
    rew_joint_jerk: float = -1.0e-10
    rew_alive: float = 0.5                   # bonus per step alive
    # |Δα|² penalty — bumped -0.5 → -3.0 to keep the residual near zero
    # even WITHIN the transition window unless it's actively earning its
    # keep via tracking/orientation rewards. Combined with hard time-gating,
    # this yields the explainability story: Δα ≈ 0 in steady-state, small
    # but nonzero during transitions only when it stabilizes the body.
    rew_residual_sparsity: float = -3.0
    target_height: float = 0.42

    # Termination
    base_contact_threshold_n: float = 50.0
    bad_orientation_limit: float = 1.0       # |projected_gravity_xy|² limit


@configclass
class B1Phase2E2EEnvCfg(B1Phase2EnvCfg):
    """E2E PPO baseline — MLP learns the full blending scalar α from scratch.

    Action space is 1-D (single α scalar, same for all 4 legs).
    No hand-designed baseline ramp; the MLP must infer transition timing
    from cycles_elapsed and gait one-hot alone.

    Obs: same 45-D as residual, but alpha_baseline is always 0
         (no baseline signal — MLP must discover the ramp shape).
    """

    action_space: int = 1
    alpha_schedule: str = "e2e"
    rew_residual_sparsity: float = 0.0
    rew_action_rate: float = -0.15


@configclass
class B1Phase2Residual1DEnvCfg(B1Phase2EnvCfg):
    """Ablation: scalar residual — 1-D Δα broadcast uniformly to all 4 legs.

    Same smoothstep baseline, same time-gating, same sparsity penalty as
    residual-4D (v7). The only difference is action_space=1: the MLP outputs
    a single correction that is applied identically to FL, FR, RL, RR.

    Comparing residual-1D vs residual-4D isolates whether the per-leg
    asymmetry in Δα is necessary, with all other design choices held fixed.
    """

    action_space: int = 1
    # alpha_schedule, delta_alpha_max, time-gating all inherited from B1Phase2EnvCfg


@configclass
class B1Phase2V11EnvCfg(B1Phase2EnvCfg):
    """v11 — curriculum training over transition durations.

    Samples transition_duration uniformly from [1.5, 5.0] s at each episode
    reset. The normalized duration (duration / 5.0) is appended to the
    observation so the MLP can condition on ramp speed explicitly.

    Architecture is otherwise identical to v10 (smoothstep baseline,
    sigmoid × 0.3 asymmetric clamp, same reward weights). Training on a
    range of durations tests whether the per-leg residual gain at d=3 s
    generalises beyond the training distribution.
    """

    # Curriculum range — uniform sample each episode reset
    transition_duration_min_s: float = 1.5
    transition_duration_max_s: float = 5.0

    # +1 obs dim: normalised transition duration appended to the 45-D base obs
    observation_space: int = 46


@configclass
class B1Phase2E2ERateEnvCfg(B1Phase2EnvCfg):
    """E2E rate-based α — MLP outputs dα/dt, α integrated monotonically from 0.

    Action space is 1-D. At each control step:
        rate   = sigmoid(action) / transition_duration_s   ∈ [0, 1/3 per s]
        α_t    = clamp(α_{t-1} + rate × dt, 0, 1)

    This forces α to start at 0 at episode start and rise at a speed the
    MLP controls. Unlike E2E-sigmoid (which can jump α=1 immediately), the
    integrated design structurally requires genuine multi-step interpolation.
    The MLP must discover the transition shape (could rediscover smoothstep,
    or find something better) rather than collapsing to always-target.

    No time-gating (the integral is its own implicit timing structure).
    No sparsity penalty (MLP needs freedom to control the rate).
    """

    action_space: int = 1
    alpha_schedule: str = "e2e_rate"
    rew_residual_sparsity: float = 0.0
    rew_action_rate: float = -0.15
