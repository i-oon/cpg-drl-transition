"""
Manager-based velocity-tracking environment for Unitree B1.

Built as an alternative Phase-1 pipeline to the CPG-RBF + PIBB path in
`envs/unitree_b1_env.py`. Policy outputs 12-D joint position offsets
(standard legged-gym convention), NOT W weights. Avoids the bang-bang
W-space failure mode documented as Encoding Experiment #8 in CLAUDE.md.

Inherits Isaac Lab's stock `LocomotionVelocityRoughEnvCfg` and:
  - Swaps in a PROJECT-LOCAL copy of `UNITREE_B1_CFG` with raised stiffness
    (external project measured that 200 N·m/rad sags 9 cm under B1's body
    weight; 400 brings sag under 5 cm, 600 fully supports). Deep-copied so
    we never mutate the shared isaaclab_assets cfg.
  - Flattens the terrain (no rough/curriculum for initial gait learning).
  - Points foot-contact sensors at `.*_foot$` (B1 convention, anchored).
  - Raises base_contact threshold to 50 N so the PhysX settling transient
    (feet spawn ~7.7 cm under ground at default joint angles) doesn't
    trigger immediate terminations.
  - Removes the `undesired_contacts` reward (Go2 referenced Head_* which B1
    has no link for; we don't enforce leg-body contact avoidance yet).

Registers gym IDs on import:
  - Isaac-Velocity-Flat-Unitree-B1-v0          (train)
  - Isaac-Velocity-Flat-Unitree-B1-Play-v0     (play / few envs, no noise)
"""

import copy

import gymnasium as gym

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG as _UNITREE_B1_CFG_SHARED
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from envs.b1_velocity_mdp import (
    air_time_variance_penalty,
    bound_coordination_reward,
    duty_factor_target_penalty,
    excessive_air_time,
    excessive_contact_time,
    gait_phase_match_reward,
    joint_lr_symmetry_penalty,
    must_move_penalty,
    pace_coordination_reward,
    short_swing_penalty,
    true_bound_reward,
    true_pace_reward,
    true_walk_reward,
)


# ---------------------------------------------------------------------------
# Project-local B1 articulation cfg — deep-copied from the shared
# isaaclab_assets cfg so we can raise stiffness without affecting any other
# project (cpg-drl-transition's CPG env, other repos, etc.) that imports the
# shared config. Debug-log item #5: mutating shared assets is a trap.
# ---------------------------------------------------------------------------

UNITREE_B1_CFG = copy.deepcopy(_UNITREE_B1_CFG_SHARED)

# Raise PD gains: 200 N·m/rad sags B1's 50-60 kg body ~9 cm below commanded
# stance height → rear-leg-over-extension → nose-up tilt → backward drift.
# 400 is the middle of the "partial fix" range (partial → stable across
# external measurements 200/400/600). Start at 400; bump to 500-600 only if
# body still sags in play mode.
UNITREE_B1_CFG.actuators["base_legs"].stiffness = 400.0
# Damping scales linearly with stiffness at ~0.025 ratio (matches Go2).
UNITREE_B1_CFG.actuators["base_legs"].damping = 10.0

# Raise spawn by 8 cm so feet start above (not 7.7 cm below) the ground at
# default joint angles, avoiding the settling-transient contact spike.
_default_pos = UNITREE_B1_CFG.init_state.pos
UNITREE_B1_CFG.init_state.pos = (_default_pos[0], _default_pos[1],
                                 _default_pos[2] + 0.08)  # 0.42 → 0.50


@configclass
class B1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = UNITREE_B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # B1 footprint is ~1.7× Go2's (110×52 cm vs 65×28 cm)
        self.scene.env_spacing = 3.5

        # --- Flat terrain (no curriculum for Phase 1) ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # --- Actions: joint-offset (NOT W) ---
        # default_joint_pos + 0.25 * tanh(π(obs)) — same convention as
        # Isaac-Velocity-Flat-Unitree-Go2-v0.
        self.actions.joint_pos.scale = 0.25

        # --- B1 body naming ---
        # B1's USD keeps the URDF `trunk` link name (Go2 renames it to `base`).
        # Every body_names="base" inherited from LocomotionVelocityRoughEnvCfg
        # must be overridden to "trunk" or the managers silently find 0 bodies.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"

        # Raise base_contact termination threshold from 1 N to 50 N.
        # Debug-log item #2: even with feet above ground at spawn, PhysX
        # settling produces contact transients in the 20-40 N range on a
        # 50-60 kg body. Real falls peak at 300+ N, so 50 N is a safe floor.
        self.terminations.base_contact.params["threshold"] = 50.0

        # --- Domain randomization scaled for B1's 50 kg mass ---
        # Stock (-5, 5) kg is ±33% for Go2's 15 kg; for B1 the equivalent is ±10 kg.
        self.events.add_base_mass.params["mass_distribution_params"] = (-10.0, 10.0)

        # Stock reset gives the spawn ±0.5 m/s linear + ±0.5 rad/s angular velocity.
        # For a 50 kg body at 0.42 m height that's enough to face-plant at t=0.
        # unitree_rl_lab zeroes these for the same reason.
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        # Start near default pose (stock (0.5, 1.5) randomizes ±50%, which
        # spawns B1 in weird configurations and triggers immediate falls).
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # --- Feet contact reference (B1 uses *_foot links, anchored $) ---
        # Anchor with $ to avoid accidentally matching a *_foot_variant link
        # (debug-log item #1 — we only hit this on Anymal, but the habit is
        # cheap and prevents silent mis-indexing on future USDs).
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot$"
        self.rewards.feet_air_time.weight = 0.25

        # Lower air-time threshold from stock 0.5 s to 0.3 s. B1's heavy legs
        # + stiffness 400 can't sustain 0.5 s swings in early training; the
        # reward stayed negative (no foot reached threshold) for 200+ iters
        # because the reachable bar was too high. 0.3 s is realistic for a
        # slow B1 walk and matches the shorter swing duration of a heavy
        # quadruped at walking speed (cf. CLAUDE.md CPG period = 1.0 s at
        # 1 Hz → 0.3-0.5 s swing per leg).
        self.rewards.feet_air_time.params["threshold"] = 0.3

        # --- Velocity-tracking reward weights ---
        # Stock LocomotionVelocityRoughEnvCfg ships 1.0 / 0.5 (calibrated for
        # 15 kg Go2 where moving is cheap). On B1 those values collapse to a
        # standstill policy (feet never lift). Isaac Lab's Go2 *rough* and
        # unitree_rl_lab's Go2 both use 1.5 / 0.75 — the standard
        # strong-track weights. Matches here.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # --- Joint-acceleration penalty ---
        # Stock -2.5e-7 is tuned for light Go2 joints. Smoke tests showed
        # dof_acc_l2 at -223 cumulative (24% of penalty budget) — enough to
        # discourage the fast leg-lift accelerations needed to swing a
        # heavy B1 leg. Halving gives a per-step ceiling around -0.1 instead
        # of -0.24, letting the policy actuate freely enough to walk while
        # still discouraging torque-saturating thrashes.
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # --- Velocity command ranges ---
        # Stock (-1.0, 1.0) is too aggressive for early B1 learning. The
        # policy can't physically achieve 1 m/s with a still-forming gait,
        # so the track_lin_vel_xy_exp reward saturates at a low ceiling and
        # gradient signal is weak. Narrow initial ranges so "matching the
        # command" is achievable from day 1; the policy can later be
        # fine-tuned at wider ranges if needed.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # No "stand still" commands — we want a walking policy, not a
        # mixed walk/stand policy. Stock CommandsCfg samples zero-velocity
        # commands for 2% of envs which makes playback look like the
        # robot is "just standing around" some of the time.
        self.commands.base_velocity.rel_standing_envs = 0.0

        # --- Posture rewards (NEW — fix the v1 crawling exploit) ---
        # First-pass training (3000 iters) tracked velocity perfectly
        # (error_vel_xy = 0.10) but exhibited two cosmetic failure modes
        # in playback: (a) body sagged from 0.42 m down to 0.18 m =
        # crawling, (b) policy adopted a body-low flat-shuffle posture.
        # Stock LocomotionVelocityRoughEnvCfg has no height or pose
        # constraints — training reward saturated by tracking velocity
        # alone, no incentive to maintain stance height.
        # base_height_l2 weight raised -10 → -50 after the v2 policy stood
        # 6 cm tall on extended legs to enable single-leg balance. Stronger
        # height enforcement removes that degree of freedom.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-50.0,
            params={"target_height": 0.42},
        )
        # Penalize hip ABDUCTION drift only (not thigh/calf which must
        # swing for the gait). Hip default is ±0.1 rad — walking shouldn't
        # change it, so any deviation is the policy adopting a wide / low
        # crawling stance to cheat the velocity reward.
        # Weight tuned down from -0.5 → -0.2 after 200-iter smoke showed
        # -0.5 contributed -0.215/step (14% of max velocity reward),
        # crushing learning headroom. -0.2 gives ~ -0.086/step at the
        # same hip deviation, mild enough to coexist with velocity tracking.
        self.rewards.hip_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # --- Anti-pathology rewards (NEW — fix the 2-leg-trot exploit) ---
        # First posture-fixed policy (logs/ppo_b1/20260424_100603) converged
        # to a degenerate gait under pure-forward command: FR planted 87%,
        # FL airborne 99.8%, RL cycling, RR airborne 98%. Robot achieved
        # perfect velocity tracking by hopping on the FR+RL diagonal pair
        # while permanently lifting FL and RR. None of our existing rewards
        # punished this because:
        #  - height/orientation/hip-deviation: all satisfied by 2 legs
        #  - velocity tracking: saturated
        #  - feet_air_time: per-leg basis, lifted legs technically meet it
        # Two new terms close these loopholes:

        # 1) feet_slide — penalizes foot lateral velocity DURING contact.
        # The "permanently planted" FR foot is being dragged forward as the
        # body moves at 0.4 m/s → big slide accumulation → strong negative
        # gradient pushing the policy to lift FR periodically.
        self.rewards.feet_slide = RewTerm(
            func=feet_slide,
            weight=-0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot$"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # 2) excessive_air_time — penalizes any foot held airborne longer
        # than 0.5 s (tightened from 1.0 s after v2 policy still left FL/RL
        # airborne for the entire episode under deterministic forward command).
        # 0.5 s ceiling is roughly 2× a normal trot swing, so it allows real
        # walking but kills "leg permanently raised" behavior.
        self.rewards.excessive_air_time = RewTerm(
            func=excessive_air_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "max_air_time": 0.5,
            },
        )

        # 3) excessive_contact_time — symmetric counterpart that punishes a
        # foot for being planted longer than 0.5 s. The v2 policy used RR
        # at 70% duty (RR planted for 14 s out of 20) for single-leg
        # support; with this term, that pattern would generate ~13/step
        # penalty — incompatible with the velocity reward. Together with
        # excessive_air_time it forces every foot into < 0.5+0.5 = 1.0 s
        # full cycles → ≥ 1 Hz gait, all four legs participating.
        self.rewards.excessive_contact_time = RewTerm(
            func=excessive_contact_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "max_contact_time": 0.5,
            },
        )

        # 4) short_swing_penalty — kills the "rapid tap-tap-tap" exploit.
        # v3 policy still let RR support the body via 5 Hz tapping (each
        # touch <0.5 s so excessive_contact_time was silent). This term
        # fires only at touchdown if the prior air phase was <0.3 s, so
        # tapping incurs ~2.5/sec penalty per offending foot while real
        # walking (swing 0.3-0.5 s) sees zero cost.
        self.rewards.short_swing_penalty = RewTerm(
            func=short_swing_penalty,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "min_swing_time": 0.3,
            },
        )

        # 5) air_time_variance_penalty — kills the "3+1 asymmetric trot"
        # exploit. v4 policy converged to FR/RL/RR cycling at ~0.4 s swings
        # while FL cycled at ~0.75 s (half the rate). Per-foot threshold
        # rewards don't notice this because each foot stays within bounds
        # individually. Variance of last_air_time across feet directly
        # measures the asymmetry.
        #
        # Weight initially set to -5.0; smoke showed that produced
        # -0.40/step (36% of penalty budget) during exploration and
        # collapsed velocity tracking + all other metrics. Reduced to -1.0
        # so converged-symmetric ~0, converged-v4-asymmetric ~-0.023/step
        # — gentle tie-breaker rather than structural shockwave.
        self.rewards.air_time_variance_penalty = RewTerm(
            func=air_time_variance_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # 6) joint_lr_symmetry_penalty — penalizes left/right per-leg motion
        # asymmetry within fronts (FL vs FR) and rears (RL vs RR). Catches
        # the per-leg specialization PPO converges to on B1 (e.g. trot
        # baseline had FL hip 4°, FR hip 9° — bilaterally asymmetric even
        # though duty was symmetric). Sum of velocity-magnitude squared
        # differences across paired legs.
        #
        # Front/rear DIFFERENCES are not penalized (B1's 0.8/1.0 thigh
        # default asymmetry makes those physically required). Steer
        # config overrides weight to 0 since L/R asymmetry is required
        # for turning.
        self.rewards.joint_lr_symmetry_penalty = RewTerm(
            func=joint_lr_symmetry_penalty,
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # --- Reward weights scaled for B1's actuator strength ---
        # Orientation penalty (stock default 0.0, enabling is standard for legged).
        self.rewards.flat_orientation_l2.weight = -2.5

        # Torque penalty: B1's peak motor torque is 280 N·m vs Go2's ~23 N·m
        # (~12× bigger, ~148× in the squared penalty). Go2's -2e-4 would
        # completely dominate the reward on B1 (penalty ≈ 50+ vs track
        # reward ≈ 1.5). Scale down by ~200× to match relative impact.
        self.rewards.dof_torques_l2.weight = -1.0e-6

        # Action-rate penalty: balance between noise-std suppression and
        # letting the policy actuate enough to walk.
        #
        # Stock value is -0.01. With our entropy_coef=0.005, analytic σ
        # equilibrium is σ² = entropy/(4·w·dt) = 0.005/(4·0.01·0.02) = 6.25
        # → σ=2.5. That sounds risky, but first smoke test at w=-0.025
        # landed at observed σ=0.96 (well below equilibrium), proving
        # OTHER PPO gradients push σ down independently — we don't need
        # the action_rate term to do heavy lifting on noise containment.
        #
        # Meanwhile w=-0.025 dominated the reward (~42% of the penalty
        # budget), crushed gradient signal, and the policy couldn't learn
        # to swing its legs (feet_air_time was negative → 50% falls).
        # Stock -0.01 restores learning headroom.
        self.rewards.action_rate_l2.weight = -0.01

        # Remove undesired_contacts (stock pattern `.*THIGH` is Anymal-style
        # uppercase; B1 uses `.*_thigh` but Phase 1 doesn't need it).
        self.rewards.undesired_contacts = None

        # --- Sim settings: larger bodies need more contact patches ---
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15


@configclass
class B1FlatEnvCfg_PLAY(B1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for visual playback
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        # Deterministic observations during play
        self.observations.policy.enable_corruption = False

        # No external disturbances during play
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Force a deterministic pure-forward command so the gait diagram
        # is clean — eliminates per-env command-sampling variance and lets
        # us see whether the policy itself produces a symmetric gait.
        # To restore the random range, comment these out.
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


# ===========================================================================
# Phase-2 specialized gait configs
#
# Three narrow-regime configs that fine-tune the generalist B1FlatEnvCfg
# into distinct gait artifacts for Phase 2 transition learning. Each is
# trained via --resume from the generalist checkpoint so velocity tracking
# transfers; only the gait specialization is relearned.
#
# Gait characteristics targeted:
#   Walk  — slow forward, longer stance (high duty ~70%), ~1 Hz cycle
#   Trot  — medium forward, diagonal pairs in phase, ~2.5 Hz cycle (this
#           is what the generalist naturally converges to)
#   Steer — forward + strong yaw, medium cycle, asymmetric inner/outer
# ===========================================================================


@configclass
class B1FlatWalkEnvCfg(B1FlatEnvCfg):
    """True 1-3-2-4 lateral-sequence walk.

    Same approach as bound: explicit reference-pattern matching on a walk
    schedule (each leg at unique phase, 75% duty) PLUS a positive reward
    for the walk signature (exactly 3 feet in contact at any instant).
    Trot's diagonal coordination scores 0 on true_walk_reward (only 2
    feet in stance). Bound also scores 0. Walk gets full +1 per step.
    """
    def __post_init__(self):
        super().__post_init__()

        # Slow-only commands → walk regime
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # 1. Phase matching — walk schedule with lateral-sequence offsets.
        # Period 50 steps = 1.0 s cycle (1 Hz, walking pace). Stance fraction
        # 0.75 = each leg planted 37 of 50 steps. Offsets [0, 25, 12, 37]
        # = FL leads, RL @ +0.24 cycle, FR @ +0.50, RR @ +0.74 (1-3-2-4
        # lateral footfall sequence). Max +4/step (all 4 legs match target).
        # Weight 2.0 → up to +8/step.
        self.rewards.gait_phase_match_reward = RewTerm(
            func=gait_phase_match_reward,
            weight=2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "leg_phase_offsets": [0, 25, 12, 37],
                "stance_fraction": 0.75,
                "period_steps": 50,
            },
        )

        # 2. True-walk reward: +1 when exactly 3 feet in contact, 0 else.
        # Trot has 2 feet planted (diagonal), bound has 2 (fore-aft) — both
        # score 0 on this term. Walk's 1-leg-swing pattern is the only
        # coordination that scores. Weight reduced 3.0 → 1.5 after v3
        # showed the policy gamed this by standing still and flicking
        # one leg (3-stance + 1-air without moving). Lower weight makes
        # static-flick less profitable than actually walking.
        self.rewards.true_walk_reward = RewTerm(
            func=true_walk_reward,
            weight=1.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # Walk uses small velocities (0.1-0.25 m/s); the stock
        # track_lin_vel_xy_exp at std=0.5 gives ~88% reward even at
        # vx=0 → near-zero gradient toward moving. Tighten std to 0.25
        # (sharper exp) and bump weight 1.5 → 3.0 so standing pays
        # ~61% × 3 = 1.83/step vs perfect tracking at 3.0/step.
        import math as _math
        self.rewards.track_lin_vel_xy_exp.params["std"] = _math.sqrt(0.0625)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0

        # NEW — must_move_penalty: explicit -1/step when commanded to
        # move but standing still. Closes the "stand and flick" exploit
        # at its source. Weight -3.0 → -3/step penalty when caught,
        # exceeds the +1.5/step true_walk_reward → standing always loses.
        self.rewards.must_move_penalty = RewTerm(
            func=must_move_penalty,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "cmd_threshold": 0.05,
                "actual_threshold": 0.05,
            },
        )

        # 3. Long-stance, deliberate-swing timings.
        # min_swing_time bumped 0.15 → 0.30 after v5 showed 6 Hz mincing
        # (each swing only 0.04 s). 0.30 forces realistic walk-speed
        # swings. short_swing_penalty weight also bumped (in below code)
        # to make the penalty bite.
        self.rewards.excessive_contact_time.params["max_contact_time"] = 2.0
        self.rewards.excessive_air_time.params["max_air_time"] = 0.5
        self.rewards.short_swing_penalty.params["min_swing_time"] = 0.30
        self.rewards.short_swing_penalty.weight = -8.0  # was -2.0 from base
        self.rewards.feet_air_time.params["threshold"] = 0.3
        # Also tighten base_height for walk (-50 → -200) — v5 squatted 8.6 cm
        # below target. Stronger height penalty forces upright stance.
        self.rewards.base_height_l2.weight = -200.0

        # Walk has staggered phases by design — air-time variance is
        # naturally non-zero even when healthy. Relax.
        self.rewards.air_time_variance_penalty.weight = -0.3

        # 4. PER-LEG duty target — closes the v4 "FR drags" exploit.
        # Walk doesn't inherit this term from the base config, so we
        # ADD it here (bound/pace inherit/add it from their own configs).
        # true_walk_reward only counts how many feet are in stance, not
        # WHICH ones. The v4 policy planted FR at 98% + cycled the other
        # 3. Per-leg duty target with 0.75 forces EACH leg to spend 75%
        # planted — uniform across legs. Weight -20 makes FR-pivot
        # exploit cost ~-3/step (from per-leg deviations²).
        self.rewards.duty_factor_target_penalty = RewTerm(
            func=duty_factor_target_penalty,
            weight=-20.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "target": 0.75,
            },
        )


@configclass
class B1FlatWalkEnvCfg_PLAY(B1FlatWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Deterministic slow forward
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class B1FlatTrotEnvCfg(B1FlatEnvCfg):
    """Medium-forward trot: matches generalist's natural convergence."""
    def __post_init__(self):
        super().__post_init__()

        # Medium-speed forward — the generalist's sweet spot
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # Keep generalist timing constraints (already trot-shaped)
        # — no overrides needed


@configclass
class B1FlatTrotEnvCfg_PLAY(B1FlatTrotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Deterministic medium forward
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class B1FlatSteerEnvCfg(B1FlatEnvCfg):
    """Forward + strong yaw turning."""
    def __post_init__(self):
        super().__post_init__()

        # Forward-biased with significant yaw — turning-while-moving
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (0.4, 1.0)

        # Asymmetric gait is expected here (inner legs shorter stride);
        # relax the symmetry variance penalty so policy can specialize
        self.rewards.air_time_variance_penalty.weight = -0.3
        # Steer needs L/R joint motion asymmetry (inner legs work harder
        # than outer during turn). Disable the L/R symmetry penalty.
        self.rewards.joint_lr_symmetry_penalty.weight = 0.0


@configclass
class B1FlatSteerEnvCfg_PLAY(B1FlatSteerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Deterministic forward + moderate left turn
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.6, 0.6)


@configclass
class B1FlatBoundEnvCfg(B1FlatEnvCfg):
    """Bound gait via reference-pattern matching + explicit anti-trot reward.

    Previous bound runs converged to diagonal-trot variants because:
      (a) phase_match alone allows trot to score +2/step on bound target
      (b) stability rewards (lin_vel_z_l2, ang_vel_xy_l2, flat_orientation_l2)
          actively penalize bound's natural fore-aft body pitch
      (c) joint_lr_symmetry rewards don't bias against trot

    This config:
      1. ADDs `true_bound_reward` — rewards FL+FR/RL+RR pair sync AND
         penalizes FL+RR/FR+RL diagonal sync. Pure trot scores -2/step,
         pure bound scores +2/step.
      2. BUMPS phase_match weight 1.0 → 2.0 to dominate trot's appeal.
      3. RELAXES stability penalties that fight bound's body pitch.
    """
    def __post_init__(self):
        super().__post_init__()

        # Bound is a fast gait
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # 1. Phase matching — bumped weight 1.0 → 2.0 (+8/step max).
        self.rewards.gait_phase_match_reward = RewTerm(
            func=gait_phase_match_reward,
            weight=2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "leg_phase_offsets": [0, 0, 15, 15],
                "stance_fraction": 0.5,
                "period_steps": 30,
            },
        )

        # 2. NEW — explicit anti-trot, pro-bound reward. Score ∈ [-2, +2]
        # per step. With weight 3.0, pure trot pays -6/step (catastrophic),
        # pure bound earns +6/step. Eliminates the trot local optimum.
        self.rewards.true_bound_reward = RewTerm(
            func=true_bound_reward,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # 3. Relax stability penalties that fight bound's fore-aft pitch.
        # During leap, body lifts ~5-10 cm and pitches 5-15° fore/aft —
        # these penalties were quietly punishing exactly that motion.
        self.rewards.lin_vel_z_l2.weight = -0.5         # was -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.02       # was -0.05
        self.rewards.flat_orientation_l2.weight = -0.5  # was -2.5

        # 4. Tighten base_height_l2 (-50 → -150). v3 bound succeeded but
        # the policy used the relaxed orientation/vz penalties to also
        # adopt a sustained 6.5 cm low stance (h=0.355 vs target 0.42).
        # Body-height penalty is the only place that *targets* posture
        # height directly; bumping it eliminates the squat without
        # interfering with transient leap dynamics.
        self.rewards.base_height_l2.weight = -150.0

        # Disable the old instantaneous-coordination term (superseded
        # by phase_match + true_bound).
        self.rewards.bound_coordination_reward = RewTerm(
            func=bound_coordination_reward,
            weight=0.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # Phase matching naturally enforces 50% duty. Keep small backups.
        self.rewards.air_time_variance_penalty.weight = -0.1
        self.rewards.duty_factor_target_penalty = RewTerm(
            func=duty_factor_target_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "target": 0.5,
            },
        )


@configclass
class B1FlatBoundEnvCfg_PLAY(B1FlatBoundEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Deterministic medium forward
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class B1FlatPaceEnvCfg(B1FlatEnvCfg):
    """Pace gait — applies the FULL bound-recipe lessons from start.

    Pace is structurally identical to bound (alternating pair sync) but
    with LATERAL pairs (FL+RL, FR+RR) instead of fore-aft (FL+FR, RL+RR).
    Reusing the proven bound recipe, mirrored for lateral pairs:
      1. true_pace_reward: rewards lateral-pair sync, penalizes diagonal
         (trot) sync — closes the "trot pretending to be pace" exploit
         that bombed every previous pace attempt.
      2. gait_phase_match_reward with pace schedule [0, 15, 0, 15].
      3. duty_factor_target_penalty (target 0.5) — closes lock-pair
         exploit (right side planted forever, left side flicked).
      4. Relaxed stability penalties (lin_vel_z, ang_vel_xy,
         flat_orientation) — pace's body roll motion would otherwise
         be punished.
      5. base_height_l2 bumped (-50 → -150) to prevent squatting.
      6. CRITICAL: joint_lr_symmetry_penalty DISABLED — pace has inherent
         instantaneous L/R asymmetry (left legs swing while right legs
         stance and vice versa). The penalty would constantly fight the
         gait. Bilateral motion symmetry holds over time, just not
         instantaneously.
      7. Old pace_coordination_reward (weight 0) — superseded.
    """
    def __post_init__(self):
        super().__post_init__()

        # Pace medium speed
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # 1. Phase matching with pace schedule. Period 30 (1.5 Hz),
        # offsets [FL=0, FR=15, RL=0, RR=15] → lateral pairs in stance
        # alternate every half cycle. Weight 2.0 → max +8/step.
        self.rewards.gait_phase_match_reward = RewTerm(
            func=gait_phase_match_reward,
            weight=2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "leg_phase_offsets": [0, 15, 0, 15],
                "stance_fraction": 0.5,
                "period_steps": 30,
            },
        )

        # 2. Anti-trot, pro-pace coordination reward.
        # Score [-2, +2]: pure pace +2, pure trot -2, pure bound 0.
        # Weight 3.0 → trot pays -6/step, pace +6/step.
        self.rewards.true_pace_reward = RewTerm(
            func=true_pace_reward,
            weight=3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # 3. Relax stability penalties — pace has natural body roll.
        # lin_vel_z bumped back toward stock (-0.5 → -1.5) after pace_v2
        # showed extreme foot lifts (25-30 cm apex) driven by body bob;
        # tighter vz penalty discourages the bobbing without preventing
        # the lateral roll. ang_vel_xy and flat_orientation stay relaxed
        # because pace's body roll IS valid lateral motion.
        self.rewards.lin_vel_z_l2.weight = -1.5         # was -0.5 (originally -2.0)
        self.rewards.ang_vel_xy_l2.weight = -0.02       # was -0.05
        self.rewards.flat_orientation_l2.weight = -0.5  # was -2.5

        # 4. Tighten body height — prevents squatting.
        self.rewards.base_height_l2.weight = -150.0

        # 5. CRITICAL: pace has inherent instantaneous L/R asymmetry
        # (left side legs swing while right side stance). Disable the
        # bilateral L/R symmetry penalty inherited from base — it would
        # constantly fight the gait pattern.
        self.rewards.joint_lr_symmetry_penalty.weight = 0.0

        # 6. Disable old (gameable) pace_coordination_reward.
        self.rewards.pace_coordination_reward = RewTerm(
            func=pace_coordination_reward,
            weight=0.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # Air-time variance relaxed (pace has half-cycle offset between pairs).
        self.rewards.air_time_variance_penalty.weight = -0.1

        # 7. Per-leg duty target = 0.5 — closes lock-pair exploit.
        self.rewards.duty_factor_target_penalty = RewTerm(
            func=duty_factor_target_penalty,
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "target": 0.5,
            },
        )


@configclass
class B1FlatPaceEnvCfg_PLAY(B1FlatPaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


##
# Gym registration — executed on import.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-B1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:B1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:B1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-B1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:B1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:B1FlatPPORunnerCfg",
    },
)

# --- Phase-2 specialized gaits ---------------------------------------------

for _gait, _cfg_name in [("Walk", "B1FlatWalkEnvCfg"),
                          ("Trot", "B1FlatTrotEnvCfg"),
                          ("Steer", "B1FlatSteerEnvCfg"),
                          ("Bound", "B1FlatBoundEnvCfg"),
                          ("Pace", "B1FlatPaceEnvCfg")]:
    gym.register(
        id=f"Isaac-Velocity-Flat-Unitree-B1-{_gait}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg_name}",
            "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:B1FlatPPORunnerCfg",
        },
    )
    gym.register(
        id=f"Isaac-Velocity-Flat-Unitree-B1-{_gait}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:{_cfg_name}_PLAY",
            "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:B1FlatPPORunnerCfg",
        },
    )
