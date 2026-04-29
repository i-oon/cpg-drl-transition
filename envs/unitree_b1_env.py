"""
Unitree B1 environment for CPG-RBF Phase 1 (PIBB) training.

Uses Isaac Lab DirectRLEnv (v0.36.3). The CPG generates joint position targets
internally; PIBB perturbs W externally via set_weights() / set_weights_batch().

Key design choices:
  - Batched CPG: one oscillator state per env (num_envs, 2), one W per env
    (num_envs, 20, 3), all ops vectorised with torch.
  - PIBB API: set_weights(W) sets the same W for all envs;
    set_weights_batch(W_batch) sets a different W per env.
  - Gait dispatch: reward function is chosen by cfg.gait_name at runtime.

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_B1_CFG

# Project-local B1 config with correct actuator tuning for 50 kg body.
# deepcopy prevents mutating the shared isaaclab_assets cfg.
_UNITREE_B1_CFG = copy.deepcopy(UNITREE_B1_CFG)
_UNITREE_B1_CFG.actuators["base_legs"].stiffness = 400.0   # 200 sags 9 cm under body weight
_UNITREE_B1_CFG.actuators["base_legs"].damping   = 10.0    # proportional to stiffness
_UNITREE_B1_CFG.init_state.pos = (0.0, 0.0, 0.50)         # feet were 7.7 cm underground at 0.42


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CPG leg order throughout this project: FL, FR, RL, RR
_CPG_LEG_ORDER = ["FL", "FR", "RL", "RR"]
_CPG_JOINT_ORDER = ["hip_joint", "thigh_joint", "calf_joint"]

# CPG joint names in flat order (12,): FL_hip, FL_thigh, FL_calf, FR_hip, ...
CPG_JOINT_NAMES: list[str] = [
    f"{leg}_{jt}" for leg in _CPG_LEG_ORDER for jt in _CPG_JOINT_ORDER
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class UnitreeB1EnvCfg(DirectRLEnvCfg):
    """Configuration for the Unitree B1 CPG-RBF environment."""

    # --- Episode ---
    episode_length_s: float = 10.0    # 500 steps at 50 Hz
    decimation: int = 4               # Physics 200 Hz → control 50 Hz

    # --- Spaces ---
    # Obs: root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3)
    #      + joint_pos_rel(12) + joint_vel(12) = 33
    observation_space: int = 33
    action_space: int = 12            # Unused by PIBB (CPG generates targets internally)
    state_space: int = 0

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- Terrain ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=2.5, replicate_physics=True
    )

    # --- Robot ---
    robot: ArticulationCfg = _UNITREE_B1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- Contact sensor ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # --- CPG parameters ---
    cpg_alpha: float = 1.01    # SO(2) self-excitation
    cpg_freq: float = 0.3      # Hz (fixed during Phase 1)
    cpg_sigma2: float = 0.04   # RBF variance
    cpg_num_rbf: int = 20

    # --- Gait ---
    # Set by make_env_from_config() — do not set manually.
    gait_name: str = "walk"
    phase_offsets: list = None  # [FL, FR, RL, RR] in radians

    # --- Reward weights (populated from YAML by make_env_from_config) ---
    reward_w1: float = 1.0           # Forward velocity tracking
    reward_w2: float = 0.5           # Orientation (tilt / flat body)
    reward_w3: float = 0.2           # Height error
    reward_w4: float = 0.3           # Vertical bounce
    reward_w_energy: float = 0.01    # Joint velocity penalty
    reward_w_air_time: float = 0.5   # Air-time variance (prevents 1-leg shuffle)
    reward_w_action_rate: float = 0.01  # Action smoothness penalty
    reward_height_nominal: float = 0.42   # B1 standing height (m)
    reward_target_velocity: float = 0.4   # m/s — target forward speed

    # --- Action scaling ---
    action_scale: float = 0.25           # CPG output × scale → joint offset (matches Isaac Lab convention)

    # --- Termination ---
    termination_height: float = 0.20     # Fall if body drops below this (m)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class UnitreeB1Env(DirectRLEnv):
    """
    Unitree B1 DirectRLEnv with CPG-RBF locomotion.

    Designed for Phase 1 (PIBB): the environment steps its own CPG internally
    and ignores the action tensor from the policy. PIBB sets W externally.

    For Phase 2 (PPO), this env is subclassed/extended with α-blending logic.
    """

    cfg: UnitreeB1EnvCfg

    def __init__(self, cfg: UnitreeB1EnvCfg, render_mode: str | None = None, **kwargs):
        # Placeholders — populated after super().__init__() starts PhysX
        self._foot_ids: torch.Tensor | None = None
        self._undesired_body_ids: torch.Tensor | None = None
        self._base_id: torch.Tensor | None = None

        super().__init__(cfg, render_mode, **kwargs)
        # PhysX is now running — safe to query joint names and body indices

        # ---- Joint permutation (requires PhysX-initialized articulation) ----
        self._build_joint_permutation()

        # ---- Contact sensor body indices ----
        if hasattr(self, "_contact_sensor") and self._contact_sensor is not None:
            foot_ids, _ = self._contact_sensor.find_bodies(".*_foot$")
            thigh_ids, _ = self._contact_sensor.find_bodies(".*_thigh$")
            base_ids, _ = self._contact_sensor.find_bodies("trunk")
            self._foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
            self._undesired_body_ids = torch.tensor(thigh_ids, dtype=torch.long, device=self.device)
            self._base_id = torch.tensor(base_ids, dtype=torch.long, device=self.device)

        # ---- CPG state (batched across envs) ----
        self._init_cpg()

        # ---- Previous joint targets for action smoothness penalty ----
        self._prev_joint_targets = torch.zeros(self.num_envs, 12, device=self.device)

        # ---- Logged reward components ----
        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in ["forward_vel", "orientation", "contact_penalty", "extra"]
        }

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        if self.cfg.contact_sensor is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        else:
            self._contact_sensor = None

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_joint_permutation(self):
        """
        Compute the index permutation from CPG joint order to Isaac Lab joint order.

        CPG order: FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf
        Isaac Lab order: whatever the USD articulation uses (typically alphabetical)

        self._joint_perm[i] = index in CPG order for Isaac Lab joint i.
        Usage: isaaclab_targets = cpg_targets[:, self._joint_perm]

        Falls back to identity if _robot is not yet initialised (unit-test mocks).
        """
        if not hasattr(self, "_robot"):
            self._joint_perm = torch.arange(12, dtype=torch.long, device=self.device)
            return

        isaaclab_names = list(self._robot.joint_names)
        try:
            perm = [CPG_JOINT_NAMES.index(name) for name in isaaclab_names]
        except ValueError as e:
            raise ValueError(
                f"Joint name mismatch between CPG and Isaac Lab articulation.\n"
                f"CPG names: {CPG_JOINT_NAMES}\n"
                f"Isaac Lab names: {isaaclab_names}\n"
                f"Error: {e}"
            )
        self._joint_perm = torch.tensor(perm, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # CPG (batched)
    # ------------------------------------------------------------------

    def _init_cpg(self):
        """Initialise batched CPG state for all envs (LocoNets pre-compute approach).

        Pre-computes one full period of the SO(2) oscillator trajectory ONCE,
        samples RBF centers from the actual trajectory, and pre-computes the
        full RBF activation lookup table (KENNE).
        At runtime: KENNE[phase_index] gives RBF activations directly.
        """
        H = self.cfg.cpg_num_rbf
        alpha = self.cfg.cpg_alpha
        # LocoNets uses phi=0.06π (0.1885 rad/step) — controls period length
        phi = 2.0 * math.pi * self.cfg.cpg_freq * self.cfg.sim.dt * self.cfg.decimation

        # 1. Pre-compute one full period of the oscillator trajectory
        w11 = alpha * math.cos(phi)
        w12 = alpha * math.sin(phi)
        w21 = -w12
        w22 = w11
        x = [-0.197]   # LocoNets initial state
        y = [0.0]
        period = 0
        # Run until y wraps back (one full cycle)
        while y[period] >= y[0]:
            period += 1
            x.append(math.tanh(w11 * x[period-1] + w12 * y[period-1]))
            y.append(math.tanh(w22 * y[period-1] + w21 * x[period-1]))
        while y[period] <= y[0]:
            period += 1
            x.append(math.tanh(w11 * x[period-1] + w12 * y[period-1]))
            y.append(math.tanh(w22 * y[period-1] + w21 * x[period-1]))
        self._period = period
        x_arr = np.array(x, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)

        # 2. Sample RBF centers FROM the trajectory at evenly-spaced time indices
        ci = np.linspace(1, period, H + 1, dtype=int)[:-1]
        cx = x_arr[ci]
        cy = y_arr[ci]

        # 3. Pre-compute full RBF activation table — KENNE shape: (period+1, H)
        var = self.cfg.cpg_sigma2
        kenne = np.zeros((len(x_arr), H), dtype=np.float32)
        for i in range(H):
            rx = x_arr - cx[i]
            ry = y_arr - cy[i]
            kenne[:, i] = np.exp(-(rx**2 + ry**2) / var)
        self._KENNE = torch.from_numpy(kenne).to(self.device)   # (period+1, H)

        # 4. Phase offsets per leg, converted to integer step offsets
        offsets = self.cfg.phase_offsets or [0.0, math.pi, math.pi/2, 3*math.pi/2]
        # Convert phase (radians) → step offset (integer)
        # phase_rad / (2π) gives fraction of cycle, × period → steps
        leg_step_offsets = [int(round((off / (2.0 * math.pi)) * period)) % period
                            for off in offsets]
        self._leg_step_offsets = torch.tensor(leg_step_offsets,
                                              dtype=torch.long, device=self.device)

        # 5. Phase counter (integer, scalar — same for all envs)
        self._phase_idx: int = 0
        # Continuous phi tracked separately for rewards / visualization
        self._phi: float = 0.0
        self._delta_phi = phi

        # Indirect encoding: shared W shape (num_envs, 20, 3) — same W for all 4 legs
        self._W = torch.zeros(self.num_envs, H, 3, device=self.device)

    def _reset_cpg(self, env_ids: torch.Tensor):
        """Reset CPG phase index for selected envs."""
        # Phase is shared across all envs — reset only when ALL envs reset
        if env_ids is None or len(env_ids) == self.num_envs:
            self._phase_idx = 0
            self._phi = 0.0

    def _step_cpg_batch(self) -> torch.Tensor:
        """
        Pre-computed CPG (LocoNets approach) with per-leg phase offsets.
        Indirect encoding: all 4 legs share the SAME W (20, 3).
        Per-leg timing comes entirely from integer phase-step offsets.
        """
        # Per-leg phase indices: (4,) long in [0, period)
        leg_idx = (self._phase_idx + self._leg_step_offsets) % self._period
        rbf_legs = self._KENNE[leg_idx]          # (4, H)

        # Shared W: (E, H, 3) — applied identically to each leg
        # rbf_legs @ W → (E, 4, 3), then flatten to (E, 12)
        raw = torch.einsum("kn,enj->ekj", rbf_legs, self._W)   # (E, 4, 3)
        out = raw.reshape(self._W.shape[0], 12)
        cpg_flat = torch.tanh(out)

        # Advance integer phase
        self._phase_idx = (self._phase_idx + 1) % self._period
        self._phi += self._delta_phi

        return cpg_flat[:, self._joint_perm]

    # ------------------------------------------------------------------
    # Public PIBB API
    # ------------------------------------------------------------------

    def set_weights(self, W: np.ndarray):
        """
        Set the same W matrix for all parallel environments.

        Args:
            W: (20, 3) indirect-encoding weight matrix (hip/thigh/calf).
        """
        H = self.cfg.cpg_num_rbf
        assert W.shape == (H, 3), f"Expected W shape ({H}, 3), got {W.shape}"
        W_t = torch.tensor(W, dtype=torch.float32, device=self.device)
        self._W[:] = W_t.unsqueeze(0)

    def set_weights_batch(self, W_batch: np.ndarray):
        """
        Args:
            W_batch: (num_envs, 20, 3) numpy array.
        """
        H = self.cfg.cpg_num_rbf
        N = self.num_envs
        assert W_batch.shape == (N, H, 3), (
            f"Expected W_batch shape ({N}, {H}, 3), got {W_batch.shape}"
        )
        self._W = torch.tensor(W_batch, dtype=torch.float32, device=self.device)

    def get_weights(self) -> np.ndarray:
        return self._W[0].cpu().numpy()

    # ------------------------------------------------------------------
    # DirectRLEnv interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Compute CPG joint targets. External actions ignored for Phase 1 PIBB."""
        self._prev_joint_targets = self._joint_targets.clone() if hasattr(self, '_joint_targets') else torch.zeros(self.num_envs, 12, device=self.device)
        self._joint_targets = self._step_cpg_batch()  # (num_envs, 12)

    def _apply_action(self):
        """Apply scaled CPG offsets on top of the default standing pose."""
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos + self.cfg.action_scale * self._joint_targets
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,           # (N, 3) forward/lateral/up in body frame
                self._robot.data.root_ang_vel_b,           # (N, 3) roll/pitch/yaw rates
                self._robot.data.projected_gravity_b,      # (N, 3) tilt signal: [0,0,-1] when upright
                self._robot.data.joint_pos                  # (N, 12) joint positions
                - self._robot.data.default_joint_pos,       #         relative to default
                self._robot.data.joint_vel,                # (N, 12) joint velocities
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        gait = self.cfg.gait_name
        if gait == "walk":
            reward = self._reward_walk()
        elif gait == "trot":
            reward = self._reward_trot()
        elif gait == "pace":
            reward = self._reward_pace()
        elif gait == "bound":
            reward = self._reward_bound()
        elif gait == "steer":
            reward = self._reward_steer()
        else:
            raise ValueError(f"Unknown gait_name: {gait!r}")

        # Accumulate for logging
        self._episode_sums["forward_vel"] += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Fall: body too low
        body_height = self._robot.data.root_pos_w[:, 2]
        fallen = body_height < self.cfg.termination_height

        # Fall: base link makes contact (body slam) — only when sensor is active
        if self._contact_sensor is not None and self._base_id is not None:
            net_forces = self._contact_sensor.data.net_forces_w_history
            base_contact = (
                torch.max(
                    torch.norm(net_forces[:, :, self._base_id], dim=-1), dim=1
                )[0] > 1.0
            ).any(dim=1)
            fallen = fallen | base_contact

        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset robot to default pose + env origin offset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset CPG oscillator state
        self._reset_cpg(env_ids)

        # Reset episode sums for logging
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

    # ------------------------------------------------------------------
    # Reward functions (one per gait)
    # ------------------------------------------------------------------

    def _reward_simple(self) -> torch.Tensor:
        """
        Outcome-only reward: velocity + stability + cycling regularity.
        No gait-specific phase target — coordination emerges from the CPG structure.
        """
        vx = self._robot.data.root_lin_vel_b[:, 0]
        target = self.cfg.reward_target_velocity
        overshoot = torch.clamp(vx - target, min=0.0)
        vel_reward = vx - 2.0 * overshoot   # linear up to target, -1 slope above

        flat_orient = torch.sum(torch.square(
            self._robot.data.projected_gravity_b[:, :2]
        ), dim=1)

        height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.abs(height - self.cfg.reward_height_nominal)

        lin_vel_z = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        action_rate = torch.sum(torch.square(
            self._joint_targets - self._prev_joint_targets
        ), dim=1)

        energy = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)

        # Air-time variance — penalise 1-leg shuffle (all 4 feet should cycle at similar rates)
        air_var = torch.zeros(self.num_envs, device=self.device)
        if self._contact_sensor is not None and self._foot_ids is not None:
            foot_ids = self._foot_ids.cpu().tolist() if isinstance(self._foot_ids, torch.Tensor) else list(self._foot_ids)
            air_times = self._contact_sensor.data.last_air_time[:, foot_ids].float()
            if air_times.shape[1] >= 2:
                air_var = air_times.var(dim=1, correction=0)

        return (self.cfg.reward_w1    * vel_reward
                - self.cfg.reward_w2  * flat_orient
                - self.cfg.reward_w3  * height_error
                - self.cfg.reward_w4  * lin_vel_z
                - self.cfg.reward_w_action_rate * action_rate
                - self.cfg.reward_w_energy      * energy
                - self.cfg.reward_w_air_time    * air_var)

    def _reward_walk(self) -> torch.Tensor:
        return self._reward_simple()


    def _reward_trot(self) -> torch.Tensor:
        return self._reward_simple()

    def _reward_pace(self) -> torch.Tensor:
        return self._reward_simple()

    def _reward_bound(self) -> torch.Tensor:
        return self._reward_simple()

    def _reward_steer(self) -> torch.Tensor:
        return self._reward_simple()

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _compute_gait_reward(self) -> torch.Tensor:
        """
        Phase-locked gait reward: feet should match expected stance/swing
        based on the CPG oscillator phase.

        Uses cos(φ + θ_k) to determine expected contact:
          cos > 0 → stance (foot on ground)
          cos < 0 → swing  (foot in air)

        Returns (num_envs,) reward in [0, 1]: fraction of legs matching.
        """
        if self._foot_ids is None:
            return torch.zeros(self.num_envs, device=self.device)

        # Expected contact state from CPG phase: (4,) bool
        leg_phases = self._phi + self._leg_offsets   # (4,)
        expected_contact = torch.cos(leg_phases) > 0  # (4,) stance=True

        # Actual contact from sensor: (num_envs, 4)
        net_forces = self._contact_sensor.data.net_forces_w_history
        foot_forces = torch.norm(net_forces[:, 0, :, :], dim=-1)  # (N, B)
        actual_contact = foot_forces[:, self._foot_ids] > 1.0     # (N, 4)

        # Reward: fraction of legs matching expected phase
        match = (actual_contact == expected_contact.unsqueeze(0)).float()  # (N, 4)
        return match.mean(dim=1)   # (N,)

    def _compute_slippage(self) -> torch.Tensor:
        """
        Foot slippage: norm of foot contact force × foot speed proxy.

        Uses lateral + vertical root velocity as a rough foot-speed proxy
        (exact foot velocity requires forward kinematics — avoid for now).

        Returns (num_envs,) slippage penalty.
        """
        if self._foot_ids is None:
            return torch.zeros(self.num_envs, device=self.device)

        net_forces = self._contact_sensor.data.net_forces_w_history   # (N, H, B, 3)
        foot_forces = torch.norm(net_forces[:, 0, :, :], dim=-1)      # (N, B)
        in_contact = foot_forces[:, self._foot_ids] > 1.0              # (N, 4)

        # Proxy: lateral (y) velocity when in contact indicates slipping
        # Use only vy (index 1), NOT vz which includes physics settling noise
        lat_vel = torch.abs(self._robot.data.root_lin_vel_b[:, 1])    # (N,)
        slippage = in_contact.float().sum(dim=1) * lat_vel             # (N,)
        return slippage

    def _compute_trot_phase_error(self) -> torch.Tensor:
        """
        Phase error for trot: penalise asymmetry between diagonal pairs.

        Measures difference in last air time between FL-RR (pair A)
        and FR-RL (pair B). Perfect trot has pairs alternating equally.
        """
        if self._foot_ids is None or len(self._foot_ids) < 4:
            return torch.zeros(self.num_envs, device=self.device)

        air_times = self._contact_sensor.data.last_air_time[:, self._foot_ids]  # (N, 4)
        # foot order matches B1 calf order: FL, FR, RL, RR (verify with find_bodies)
        # Pair A: FL(0), RR(3); Pair B: FR(1), RL(2)
        pair_a_diff = torch.abs(air_times[:, 0] - air_times[:, 3])
        pair_b_diff = torch.abs(air_times[:, 1] - air_times[:, 2])
        return pair_a_diff + pair_b_diff

    def _compute_bound_phase_error(self) -> torch.Tensor:
        """
        Phase error for bound: penalise asymmetry within front pair and rear pair.

        Front pair: FL(0) and FR(1) should have matching air times.
        Rear pair: RL(2) and RR(3) should have matching air times.
        """
        if self._foot_ids is None or len(self._foot_ids) < 4:
            return torch.zeros(self.num_envs, device=self.device)

        air_times = self._contact_sensor.data.last_air_time[:, self._foot_ids]
        front_diff = torch.abs(air_times[:, 0] - air_times[:, 1])  # FL vs FR
        rear_diff  = torch.abs(air_times[:, 2] - air_times[:, 3])  # RL vs RR
        return front_diff + rear_diff

    def _compute_pace_phase_error(self) -> torch.Tensor:
        """
        Phase error for pace: penalise asymmetry between lateral pairs.

        Measures difference in last air time between FL-RL (left pair)
        and FR-RR (right pair). Perfect pace has same-side pairs in sync.
        """
        if self._foot_ids is None or len(self._foot_ids) < 4:
            return torch.zeros(self.num_envs, device=self.device)

        air_times = self._contact_sensor.data.last_air_time[:, self._foot_ids]  # (N, 4)
        # Left pair: FL(0), RL(2); Right pair: FR(1), RR(3)
        left_diff  = torch.abs(air_times[:, 0] - air_times[:, 2])
        right_diff = torch.abs(air_times[:, 1] - air_times[:, 3])
        return left_diff + right_diff

    def _compute_air_time_bonus(self) -> torch.Tensor:
        """
        Reward feet for spending appropriate time in the air.

        Target air time = 0.35 × cycle_period (roughly 35% swing duty factor).
        At 1.0 Hz: target = 0.35s.  At 0.3 Hz: target = 1.17s.
        """
        if self._foot_ids is None:
            return torch.zeros(self.num_envs, device=self.device)

        target_air = 0.35 / self.cfg.cpg_freq   # swing fraction × cycle time

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)
        foot_first_contact = first_contact[:, self._foot_ids]               # (N, 4)
        last_air = self._contact_sensor.data.last_air_time[:, self._foot_ids]  # (N, 4)
        bonus = torch.sum((last_air - target_air) * foot_first_contact, dim=1)  # (N,)
        return bonus.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env_from_config(config_path: str, num_envs: int | None = None) -> UnitreeB1Env:
    """
    Create a UnitreeB1Env from a Phase 1 YAML config file.

    Args:
        config_path: Path to a phase1_*.yaml config (relative to project root).
        num_envs:    Override num_envs from the YAML (optional).

    Returns:
        Configured UnitreeB1Env instance.
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    env_cfg = UnitreeB1EnvCfg()

    # Gait
    env_cfg.gait_name = cfg_dict["gait"]["name"]
    env_cfg.phase_offsets = cfg_dict["gait"]["phase_offsets"]

    # CPG
    cpg = cfg_dict.get("cpg", {})
    env_cfg.cpg_alpha = cpg.get("alpha", 1.01)
    env_cfg.cpg_freq = cpg.get("freq_train", 0.3)
    env_cfg.cpg_sigma2 = cfg_dict.get("rbf", {}).get("variance", 0.04)

    # Reward
    rwd = cfg_dict.get("reward", {})
    env_cfg.reward_w1 = float(rwd.get("w1_distance", rwd.get("w1_velocity", 1.0)))
    env_cfg.reward_w2 = float(rwd.get("w2_instability", 0.5))
    env_cfg.reward_w3 = float(rwd.get("w3_height_error", rwd.get("w3_collision", 0.2)))
    env_cfg.reward_w4 = float(rwd.get("w4_slippage", 0.3))
    env_cfg.reward_w_energy = float(rwd.get("w_energy", 0.01))
    env_cfg.reward_w_air_time = float(rwd.get("w_air_time", 0.5))
    env_cfg.reward_w_action_rate = float(rwd.get("w_action_rate", 0.01))
    env_cfg.reward_height_nominal = float(rwd.get("height_nominal", 0.42))
    env_cfg.reward_target_velocity = float(rwd.get("target_velocity", 0.4))

    # Episode
    env_cfg.episode_length_s = cfg_dict.get("env", {}).get("episode_length", 500) * env_cfg.sim.dt * env_cfg.decimation

    # Num envs
    n = num_envs or cfg_dict.get("env", {}).get("num_envs", 64)
    env_cfg.scene = InteractiveSceneCfg(num_envs=n, env_spacing=2.5, replicate_physics=True)

    return UnitreeB1Env(env_cfg)
