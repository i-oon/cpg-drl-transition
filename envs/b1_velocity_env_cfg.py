"""
Velocity-tracking environment for Unitree B1.

Based on Go2 reference config (known-good baseline). Only changes from Go2:
  - B1 URDF + trunk body name
  - Spawn height +8 cm (feet below ground at default angles)
  - Base contact threshold 50 N (PhysX settling transient)
  - Mass randomization ±10 kg (B1 is 4× heavier than Go2)
  - Zero initial velocity (50 kg face-plants at ±0.5 m/s)
  - kp=300, kd=5 (hardware-validated)
  - dof_torques_l2 ~200× lighter (B1 motors 12× stronger)
  - dof_acc_l2 halved (heavy legs need swing freedom)
  - action_rate_l2 10× lighter (found to crush B1 learning at Go2 value)
  - feet_air_time threshold 0.5→0.3 (heavy legs can't sustain 0.5 s early training)
  - Gain DR ±25%, joint friction DR (sim-to-real)
  - undesired_contacts removed (Go2 references Head_* links B1 does not have)
  - Command ranges widened for omnidirectional movement

Exploit-fix reward terms are intentionally absent. Add only if a specific exploit
is observed during training.
"""

import copy
from pathlib import Path

import gymnasium as gym

from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.unitree import UNITREE_B1_CFG as _UNITREE_B1_CFG_SHARED
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from envs.b1_actuator_net import B1ActuatorNet, B1ActuatorNetCfg
from envs.b1_velocity_mdp import (
    excessive_air_time,
    excessive_contact_time,
    duty_factor_target_penalty,
    feet_in_contact,
    yaw_stability_penalty,
)

# ---------------------------------------------------------------------------
# Actuator switch — flip this to True once walking bags with /B1/joint_target
# are re-recorded and the net is retrained with q_error as an input channel.
# False = ideal PD (kp=300, kd=5)   True = LSTM actuator net
# ---------------------------------------------------------------------------
USE_ACTUATOR_NET = False
_ACTUATOR_NET_DIR = Path(__file__).parent.parent / "actuator_net_model"


# ---------------------------------------------------------------------------
# Project-local B1 config — deep-copied so we never mutate the shared asset.
# ---------------------------------------------------------------------------

UNITREE_B1_CFG = copy.deepcopy(_UNITREE_B1_CFG_SHARED)
UNITREE_B1_CFG.actuators["base_legs"].stiffness = 300.0  # hardware-validated standing ref
UNITREE_B1_CFG.actuators["base_legs"].damping = 5.0      # matches real B1 standing ref

# Override default joint positions to match real B1 hardware home pose.
# HOME_POSE = [0.17, 0.23, -0.38] * 4  (control space) converts to robot/URDF space as:
#   hip   = direction × 0.17 + offset_hip  → FR/RR: -0.03 rad,  FL/RL: +0.03 rad
#   thigh = 1 × 0.23 + 0.85               → 1.08 rad  (SYMMETRIC — URDF 0.8/1.0 was wrong)
#   calf  = 1 × -0.38 + -1.56             → -1.94 rad
#
# Rear thigh reduced 1.08 → 0.9 rad: at the same joint angle, the rear hip sits 0.69 m
# behind the front hip, so an equal positive (backward) thigh angle places the rear foot
# far further behind the body than the front foot. 0.9 rad is a modest reduction to bring
# the rear leg geometry closer to the front without deviating from the hardware too much.
UNITREE_B1_CFG.init_state.joint_pos = {
    "FL_hip_joint":    0.03,
    "FR_hip_joint":   -0.03,
    "RL_hip_joint":    0.03,
    "RR_hip_joint":   -0.03,
    "FL_thigh_joint":  1.08,
    "FR_thigh_joint":  1.08,
    "RL_thigh_joint":  0.9,
    "RR_thigh_joint":  0.9,
    ".*_calf_joint":  -1.94,
}

_default_pos = UNITREE_B1_CFG.init_state.pos
UNITREE_B1_CFG.init_state.pos = (_default_pos[0], _default_pos[1],
                                 _default_pos[2] + 0.08)  # feet spawn below ground at defaults


@configclass
class B1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot / Actuator ---
        if USE_ACTUATOR_NET:
            # LSTM actuator net — only enable after re-recording walking bags with
            # /B1/joint_target logged and retraining with q_error as input channel.
            # Without q_error the net predicts ~0 torque for standing poses and the
            # robot collapses immediately. See ACTUATOR_NET.md §14 point 4.
            UNITREE_B1_CFG.actuators["base_legs"] = B1ActuatorNetCfg(
                class_type=B1ActuatorNet,
                joint_names_expr=[".*"],
                network_file=str(_ACTUATOR_NET_DIR / "lstm_scripted.pt"),
                scalers_file=str(_ACTUATOR_NET_DIR / "scalers.pkl"),
                effort_limit=80.0,
                velocity_limit=15.7,
                stiffness=300.0,
                damping=5.0,
                saturation_effort=80.0,
                window=30,
                dt=0.02,
            )
        # else: ideal PD (kp=300, kd=5) already set on UNITREE_B1_CFG above
        self.scene.robot = UNITREE_B1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 3.5

        # --- Flat terrain ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # --- Observations (sim-to-real) ---
        # Foot contact: zero sim-to-real gap (/B1/foot_contact available on hardware).
        # Small noise simulates brief false readings during foot impact transitions.
        self.observations.policy.foot_contact = ObsTerm(
            func=feet_in_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # base_lin_vel: parent adds ±0.1 noise. Increase to ±0.2 to reflect kinematic
        # estimator errors on hardware (estimation degrades during fast direction changes).
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.2, n_max=0.2)

        # --- Actions ---
        self.actions.joint_pos.scale = 0.25

        # --- B1 body naming (USD uses "trunk", not Go2's "base") ---
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"

        # 50 N threshold: PhysX settling produces 20-40 N transients on a ~50 kg body
        self.terminations.base_contact.params["threshold"] = 50.0

        # --- Domain randomization ---
        self.events.add_base_mass.params["mass_distribution_params"] = (-10.0, 10.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)

        # Actuator gain DR: randomize kp/kd ±25% to cover real hardware gain uncertainty.
        # This is the primary sim-to-real bridge when using ideal PD (no actuator net).
        # Disabled automatically when USE_ACTUATOR_NET=True (net captures gain variability).
        if not USE_ACTUATOR_NET:
            from isaaclab.managers import EventTermCfg as EventTerm
            self.events.randomize_actuator_gains = EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "stiffness_distribution_params": (0.75, 1.25),
                    "damping_distribution_params": (0.75, 1.25),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )

        # --- Yaw stability: penalise body rotation when not commanded to turn ---
        # During strafe/fwd/bwd the policy rotates its body to face the velocity
        # vector rather than pure-strafing. Weight -2.0 gives ~1.38/step penalty
        # at wz=0.83 rad/s with cwz=0, exceeding the ~1.4 angular-tracking loss
        # and making rotation clearly unprofitable when no turn is commanded.
        # The exp(-cwz²/0.09) gate fades the penalty to ~5% at |cwz|=0.8 rad/s
        # so turning gaits are unaffected.
        self.rewards.yaw_stability = RewTerm(
            func=yaw_stability_penalty,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )

        # --- Velocity tracking ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # Raised from 0.75 → 1.5 to fix turning. Keep at 1.5 — reducing it
        # to 1.0 degraded turn tracking when combined with yaw_stability.
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # --- Orientation (Go2 value) ---
        self.rewards.flat_orientation_l2.weight = -2.5

        # --- Feet (B1 uses *_foot links) ---
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot$"
        self.rewards.feet_air_time.weight = 0.3          # increased: robot never cleared 0.20s threshold
        self.rewards.feet_air_time.params["threshold"] = 0.1  # lowered: robot gets 0.06-0.12s, make it reachable

        self.rewards.feet_slide = RewTerm(
            func=feet_slide,
            weight=-0.1,  # Go2 value
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot$"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
            },
        )

        # --- Hip lateral splay (observed: all hips swinging ±8-11° vs hardware default ±1.72°) ---
        # Real home pose: hip ±0.03 rad = ±1.72°. Hip does lateral splay, not stride.
        # Thigh/calf drive all forward propulsion — hip should stay near zero.
        self.rewards.hip_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # --- Thigh posture ---
        # Front thighs swing too wide (59-70° P2P) making them look nearly horizontal.
        # Rear thighs barely move (12-13° P2P) — no penalty there, let them stride freely.
        # Separate weights: front anchored tighter, rear unpunished.
        self.rewards.thigh_deviation_l1 = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(FL|FR)_thigh_joint")},
        )

        # --- Height ---
        # Real B1 mid-trunk in home pose: 0.55-0.58m. Sim trunk-origin at default joints:
        # ~0.53m (URDF origin sits below the visual geometric center). Target 0.53m matches
        # the sim's natural default pose — no crouching needed, rear calves free to stride
        # rather than permanently bent to compensate for a below-default height target.
        # History: started at 0.42m, raised to 0.46, 0.50, now 0.53 as stance improved.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-50.0,
            params={"target_height": 0.53},
        )


        # --- FL 3-leg exploit fix (observed: FL duty 24.9% vs others 61-69%) ---
        # joint_lr_symmetry_penalty removed: penalises instantaneous L/R velocity
        # differences which are ALWAYS large during trot (FL/FR in opposite phases).
        # excessive_air_time + excessive_contact_time bound duty cycles instead.

        # Bound each leg to cycle: can't be airborne >0.5s (catches RL exploit)
        # and can't be planted >0.5s (catches FL/FR high-duty exploit).
        # Together they force all four legs into active gait cycles.
        self.rewards.excessive_air_time = RewTerm(
            func=excessive_air_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "max_air_time": 0.5,
            },
        )
        self.rewards.excessive_contact_time = RewTerm(
            func=excessive_contact_time,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "max_contact_time": 0.5,
            },
        )

        # Lower feet_air_time threshold: 0.3s is unachievable at natural gait frequency
        # (~0.08s air phases), causing every touchdown to be penalised and incentivising
        # feet to stay planted. 0.1s is within reach and turns the reward positive.
        self.rewards.feet_air_time.params["threshold"] = 0.1

        # --- Duty factor target (replaces tight excessive_air_time threshold approach) ---
        # 0.15s excessive_air_time threshold created a 0.05s conflict window with the 0.1s
        # feet_air_time threshold — policy escaped by tapping rapidly (<0.1s) causing
        # feet_slide. duty_factor_target_penalty directly penalises deviation from 50% duty
        # using last completed stance/air cycle times, no threshold conflict.
        # At 25% duty: (0.25-0.5)^2 × 4 feet = 0.25/step → ~250/episode penalty.
        self.rewards.duty_factor_target = RewTerm(
            func=duty_factor_target_penalty,
            weight=-1.0,
            # Reduced from -2.0: turning requires asymmetric left/right duty factors.
            # At -2.0 the duty penalty outweighed the angular tracking reward, causing
            # the policy to never turn. -1.0 still penalises 3-leg dragging exploits
            # while leaving room for the wz reward to drive turning gaits.
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot$"),
                "target": 0.5,
            },
        )

        # --- Weight adjustments forced by B1 hardware ---
        self.rewards.dof_acc_l2.weight = -1.25e-7    # halved: heavy legs need swing freedom
        self.rewards.dof_torques_l2.weight = -1.0e-6  # ~200× lighter: B1 motors 12× stronger
        self.rewards.action_rate_l2.weight = -0.01    # Go2 uses -0.1; crushes B1 learning

        # --- Remove Go2-specific term (Head_* links don't exist on B1) ---
        self.rewards.undesired_contacts = None

        # --- Command ranges (omnidirectional) ---
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.05

        # --- Sim ---
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15


@configclass
class B1FlatEnvCfg_PLAY(B1FlatEnvCfg):
    """Play config — deterministic observations, no disturbances.

    Default: random commands across full omni range (stress-test all directions).
    Pin specific directions by overriding command ranges at the CLI or subclassing:

        Forward:  lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        Lateral:  lin_vel_x=(0.0, 0.0), lin_vel_y=(0.4, 0.4), ang_vel_z=(0.0, 0.0)
        Turn:     lin_vel_x=(0.3, 0.3), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.8, 0.8)
        Backward: lin_vel_x=(-0.4,-0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    """
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


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
