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

    # --- Residual mode ---
    # "alpha": MLP corrects blending weights (when each leg transitions)
    # "joint": MLP corrects joint positions directly (Silver et al. RPL style)
    residual_mode: str = "alpha"

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
    # V2 (bidirectional_alpha=True, tanh, max=0.3): restores symmetric range
    #   Δα ∈ [−0.3, +0.3] so the MLP can DELAY a leg (wait for phase alignment)
    #   as well as advance it. This is the architecturally correct range for
    #   correcting phase mismatch between frozen policies.
    bidirectional_alpha: bool = False
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
    # Velocity-stability penalty during transition window only.
    # Computes (vx − cmd_vx)² × in_window × weight. Must be NEGATIVE to penalise
    # velocity error (positive weight would reward deviation — a bug).
    # Default 0 → disabled. V2 configs use -2.0.
    rew_vx_window: float = 0.0
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


@configclass
class B1Phase2ActionSpaceEnvCfg(B1Phase2EnvCfg):
    """Residual-q 12D — joint-position correction, per-joint (Silver et al. RPL).

    MLP outputs 12-D Δq added directly to the smoothstep-blended joint output.
    residual_mode="joint", action_space=12.

    Part of the 2×2 ablation (space × dimension):
      Residual-α 4D  |  Residual-q 4D
      Residual-α 12D |  Residual-q 12D  ← this config
    """

    residual_mode: str = "joint"
    action_space: int = 12
    delta_action_max: float = 0.25
    rew_residual_sparsity: float = -3.0


@configclass
class B1Phase2Joint4DEnvCfg(B1Phase2EnvCfg):
    """Residual-q 4D — joint-position correction, per-leg scalar.

    MLP outputs 4-D Δq (one scalar per leg), broadcast uniformly to all 3
    joints of that leg. This matches Residual-α 4D's dimension exactly while
    changing the output space from α (blending weight) to q (joint position).

    Part of the 2×2 ablation (space × dimension):
      Residual-α 4D  |  Residual-q 4D  ← this config
      Residual-α 12D |  Residual-q 12D
    """

    residual_mode: str = "joint"
    action_space: int = 4
    delta_action_max: float = 0.25
    rew_residual_sparsity: float = -3.0


@configclass
class B1Phase2Alpha12DEnvCfg(B1Phase2EnvCfg):
    """Residual-α 12D — blending-weight correction, per-joint.

    MLP outputs 12-D Δα (one per joint), clamped via sigmoid to [0, 0.3]
    (advance-only). V2 subclass overrides bidirectional_alpha=True → tanh
    → Δα ∈ [−0.3, +0.3]. Every joint gets its own independent α correction.
    This matches Residual-q 12D's dimension exactly while keeping
    the output space as α (blending weight).

    Part of the 2×2 ablation (space × dimension):
      Residual-α 4D  |  Residual-q 4D
      Residual-α 12D |  Residual-q 12D  ← this config
                ↑ this config
    """

    residual_mode: str = "alpha"
    action_space: int = 12
    # delta_alpha_max inherited (0.3) — base uses sigmoid (advance-only); V2 subclass overrides to tanh (bidirectional)


@configclass
class B1Phase2Alpha12DPhaseAwareEnvCfg(B1Phase2Alpha12DEnvCfg):
    """Residual-α 12D with foot-contact phase observation.

    Adds 4D binary foot contact (FL/FR/RL/RR) to observation so the MLP has
    an explicit gait-phase proxy at switch time — the root cause of N=6/N=60
    velocity-safety disagreement is the MLP having no phase observation.

    Observation: 45D base + 4D foot contact = 49D.
    Reward: no jerk/jacc shaping — smoothness emerges naturally from velocity
    tracking + sparsity + phase-aware corrections (AllGaits philosophy).
    """

    observation_space: int = 49
    rew_residual_sparsity: float = -0.5
    rew_joint_acc: float = 0.0    # dropped: let policy find smooth behavior naturally
    rew_joint_jerk: float = 0.0   # dropped: let policy find low-jerk naturally


# ---------------------------------------------------------------------------
# Phase-aware variants of the full 2×2 ablation
# All share: obs=49D (45D base + 4D foot contact), sparsity=-0.5,
# rew_joint_acc=0, rew_joint_jerk=0.  Only output space and dimension vary.
# ---------------------------------------------------------------------------

@configclass
class B1Phase2Alpha4DPhaseAwareEnvCfg(B1Phase2EnvCfg):
    """Residual-α 4D with foot-contact phase observation.

    2×2 ablation: α-space × 4D (per-leg blending weight).
    Same reward recipe as Alpha12DPhaseAware: no jerk/jacc shaping.
    """
    observation_space: int = 49
    rew_residual_sparsity: float = -0.5
    rew_joint_acc: float = 0.0
    rew_joint_jerk: float = 0.0


@configclass
class B1Phase2Joint4DPhaseAwareEnvCfg(B1Phase2Joint4DEnvCfg):
    """Residual-q 4D with foot-contact phase observation.

    2×2 ablation: q-space × 4D (per-leg joint-position correction).
    Same reward recipe as Alpha12DPhaseAware: no jerk/jacc shaping.
    """
    observation_space: int = 49
    rew_residual_sparsity: float = -0.5
    rew_joint_acc: float = 0.0
    rew_joint_jerk: float = 0.0


@configclass
class B1Phase2ActionSpacePhaseAwareEnvCfg(B1Phase2ActionSpaceEnvCfg):
    """Residual-q 12D with foot-contact phase observation.

    2×2 ablation: q-space × 12D (per-joint position correction).
    Same reward recipe as Alpha12DPhaseAware: no jerk/jacc shaping.
    """
    observation_space: int = 49
    rew_residual_sparsity: float = -0.5
    rew_joint_acc: float = 0.0
    rew_joint_jerk: float = 0.0


# ---------------------------------------------------------------------------
# V2 configs — policy-output phase observation + randomised duration
#
# Observation (70D):
#   base(45) + norm_duration(1) + π_current(12) + π_target(12) = 70
#
# The frozen-policy outputs π_current and π_target are direct gait-phase
# signals: the MLP can see what each frozen policy is commanding at every
# step, giving it the phase relationship needed to time its corrections.
# This is richer than the binary foot-contact proxy (obs_space=49).
#
# Duration is sampled ∈ [1.5, 5.0]s each episode so the MLP learns a
# general correction strategy rather than a 3s-specific one. norm_duration
# is included in the observation so the MLP can condition on ramp speed.
#
# Jerk penalty removed (rew_joint_jerk=0, rew_joint_acc=0): the policy
# has one task — velocity tracking — and should discover low-jerk behaviour
# as a consequence of not stumbling, not because jerk is explicitly penalised.
# A direct vx_window penalty (weight=2.0) during the transition window
# provides the safety signal jerk_penalty was trying to proxy.
# ---------------------------------------------------------------------------

@configclass
class B1Phase2V2Alpha4DEnvCfg(B1Phase2EnvCfg):
    """V2: Residual-α 4D with policy-phase observation + randomised duration.

    Correction space: α (blending weight), action_space=4 (per-leg).
    Bidirectional: Δα ∈ [−0.3, +0.3] — MLP can delay or advance per leg.
    """
    observation_space: int = 70
    transition_duration_min_s: float = 1.5
    transition_duration_max_s: float = 5.0
    rew_joint_jerk: float = 0.0
    rew_joint_acc: float = 0.0
    rew_residual_sparsity: float = -0.5
    rew_vx_window: float = -2.0
    rew_action_rate: float = -0.5
    bidirectional_alpha: bool = True


@configclass
class B1Phase2V2Alpha12DEnvCfg(B1Phase2Alpha12DEnvCfg):
    """V2: Residual-α 12D with policy-phase observation + randomised duration.

    Correction space: α (blending weight), action_space=12 (per-joint).
    Bidirectional: Δα ∈ [−0.3, +0.3] — MLP can delay or advance per joint.
    Obs 78D: 53D base (with full 12D last_action) + 1 norm_duration + 12 π_current + 12 π_target.
    Full 12D last_action gives the MLP per-joint correction memory → smoother corrections.
    """
    observation_space: int = 78
    transition_duration_min_s: float = 1.5
    transition_duration_max_s: float = 5.0
    rew_joint_jerk: float = 0.0
    rew_joint_acc: float = 0.0
    rew_residual_sparsity: float = -0.167  # -0.5 * 4/12 — normalised per-dim vs 4D variant
    rew_vx_window: float = -2.0
    rew_action_rate: float = -0.5          # stronger smoothness pressure for 12D corrections
    bidirectional_alpha: bool = True


@configclass
class B1Phase2V2Joint4DEnvCfg(B1Phase2Joint4DEnvCfg):
    """V2: Residual-q 4D with policy-phase observation + randomised duration.

    Correction space: q (joint position), action_space=4 (per-leg scalar).
    """
    observation_space: int = 70
    transition_duration_min_s: float = 1.5
    transition_duration_max_s: float = 5.0
    rew_joint_jerk: float = 0.0
    rew_joint_acc: float = 0.0
    rew_residual_sparsity: float = -0.5
    rew_vx_window: float = -2.0
    rew_action_rate: float = -0.5


@configclass
class B1Phase2V2Joint12DEnvCfg(B1Phase2ActionSpaceEnvCfg):
    """V2: Residual-q 12D with policy-phase observation + randomised duration.

    Correction space: q (joint position), action_space=12 (per-joint).
    Obs 78D: 53D base (with full 12D last_action) + 1 norm_duration + 12 π_current + 12 π_target.
    Full 12D last_action gives the MLP per-joint correction memory → smoother corrections.
    """
    observation_space: int = 78
    transition_duration_min_s: float = 1.5
    transition_duration_max_s: float = 5.0
    rew_joint_jerk: float = 0.0
    rew_joint_acc: float = 0.0
    rew_residual_sparsity: float = -0.167  # -0.5 * 4/12 — normalised per-dim vs 4D variant
    rew_vx_window: float = -2.0
    rew_action_rate: float = -0.5          # stronger smoothness pressure for 12D corrections


# V3 configs — V2 + explicit foot contact (4D)
#
# Observation:
#   4D variants:  74D = base(45) + norm_duration(1) + π_current(12) + π_target(12) + foot_contact(4)
#   12D variants: 82D = base(53) + norm_duration(1) + π_current(12) + π_target(12) + foot_contact(4)
#
# Foot contact resolves the stance/swing ambiguity in π_current/π_target:
# the same joint angle occurs at two points per gait cycle, but contact state
# is unambiguous. The MLP can now detect phase mismatch between source and
# target policies and time its corrections accordingly.
# ---------------------------------------------------------------------------

@configclass
class B1Phase2V3Alpha4DEnvCfg(B1Phase2V2Alpha4DEnvCfg):
    """V3: Residual-α 4D — V2 + foot contact (4D). Obs = 74D."""
    observation_space: int = 74


@configclass
class B1Phase2V3Alpha12DEnvCfg(B1Phase2V2Alpha12DEnvCfg):
    """V3: Residual-α 12D — V2 + foot contact (4D). Obs = 82D."""
    observation_space: int = 82


@configclass
class B1Phase2V3Joint4DEnvCfg(B1Phase2V2Joint4DEnvCfg):
    """V3: Residual-q 4D — V2 + foot contact (4D). Obs = 74D."""
    observation_space: int = 74


@configclass
class B1Phase2V3Joint12DEnvCfg(B1Phase2V2Joint12DEnvCfg):
    """V3: Residual-q 12D — V2 + foot contact (4D). Obs = 82D."""
    observation_space: int = 82
