"""
B1 LSTM Actuator Net — Isaac Lab custom actuator.

Replaces the ideal PD actuator in sim with a data-driven LSTM model trained
on real Unitree B1 hardware recordings (see ACTUATOR_NET.md, primary model:
lstm_accel w=30, R²=0.9553, RMSE=3.72 N·m).

Model spec (must match training):
  Input  : (n_envs × 12, W=30, 16)  = 4 sensor channels + 12 joint one-hot
  Channels: [q (rad), q̇ (rad/s), q̈ (rad/s², causal backward diff), foot (raw SDK counts)]
  Output : (n_envs × 12, 1)  scaled torque → inverse-transform → N·m

Joint order:
  Training used SDK order : FR FL RR RL × hip/thigh/calf (indices 0–11)
  Isaac Lab uses           : FL FR RL RR × hip/thigh/calf (indices 0–11)
  Remapped internally via _IL_TO_SDK.

Foot contact:
  Sim contact forces (Newtons) are converted to approximate SDK integer counts
  using body-weight normalization: count = clamp(F_N / 309 × 684, 0, 684).
  Wire up by setting actuator.contact_sensor = env.scene["contact_forces"]
  in the training script after gym.make().
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators.actuator_cfg import DCMotorCfg
from isaaclab.actuators.actuator_pd import DCMotor
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

# Isaac Lab order: FL(0-2) FR(3-5) RL(6-8) RR(9-11) × hip/thigh/calf
# SDK / model order: FR(0-2) FL(3-5) RR(6-8) RL(9-11) × hip/thigh/calf
# _IL_TO_SDK[il_idx] = sdk_idx
_IL_TO_SDK: list[int] = [
    3, 4, 5,    # IL 0-2  = FL  → SDK 3-5
    0, 1, 2,    # IL 3-5  = FR  → SDK 0-2
    9, 10, 11,  # IL 6-8  = RL  → SDK 9-11
    6, 7, 8,    # IL 9-11 = RR  → SDK 6-8
]

# Foot contact normalization: sim Newtons → approximate SDK integer counts.
# B1 body weight = 63 kg × 9.81 m/s² ≈ 618 N → 154 N per foot at rest.
# Peak impact force ≈ 2× static → 309 N per foot.
# Max SDK count observed in rosbag walking data = 684.
# Mapping: count = clamp(F_N / _MAX_FORCE_N × _MAX_SDK_COUNT, 0, _MAX_SDK_COUNT)
_MAX_FORCE_N: float = 63.0 * 9.81 * 2.0 / 4.0   # ≈ 309 N per foot
_MAX_SDK_COUNT: float = 684.0

# SDK leg order used to rearrange contact forces: FR=0, FL=1, RR=2, RL=3
_SDK_FOOT_NAMES: list[str] = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


@configclass
class B1ActuatorNetCfg(DCMotorCfg):
    """Configuration for the B1 LSTM actuator net."""

    class_type: type = None  # patched after class definition below

    network_file: str = MISSING
    """Absolute path to lstm_scripted.pt."""

    scalers_file: str = MISSING
    """Absolute path to scalers.pkl (dict with keys 'x' and 'y', StandardScaler)."""

    window: int = 30
    """Input window W — must match training (default 30)."""

    dt: float = 0.02
    """Sim timestep in seconds (50 Hz → 0.02 s)."""


class B1ActuatorNet(DCMotor):
    """LSTM actuator net replacing the ideal PD for B1 sim-to-real training."""

    cfg: B1ActuatorNetCfg

    def __init__(self, cfg: B1ActuatorNetCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        W = cfg.window
        n = self._num_envs
        J = self.num_joints  # 12

        # --- TorchScript model ---
        # Re-exported with torch.jit.script (not trace) so hidden states are
        # device-dynamic — no CPU tensors baked in, runs cleanly on CUDA.
        self.network = torch.jit.load(cfg.network_file, map_location=self._device).eval()
        # Compact LSTM weights into a single contiguous chunk for cuDNN efficiency.
        for m in self.network.modules():
            if hasattr(m, "flatten_parameters"):
                m.flatten_parameters()

        # --- Scalers → device tensors ---
        # sklearn is not installed in env_isaaclab. Load the StandardScaler
        # objects by patching the unpickler to return a plain object that still
        # carries the numpy array attributes (mean_, scale_) set during fit.
        import pickle as _pkl

        class _FakeScaler:
            pass

        class _PatchedUnpickler(_pkl.Unpickler):
            def find_class(self, module, name):
                if name == "StandardScaler":
                    return _FakeScaler
                return super().find_class(module, name)

        with open(cfg.scalers_file, "rb") as f:
            sc = _PatchedUnpickler(f).load()
        xs, ys = sc["x"], sc["y"]
        self._x_mean = torch.tensor(xs.mean_, dtype=torch.float32, device=self._device)
        self._x_std  = torch.tensor(xs.scale_, dtype=torch.float32, device=self._device)
        self._y_mean = float(ys.mean_[0])
        self._y_std  = float(ys.scale_[0])

        # --- Joint one-hot in SDK order ---
        il_to_sdk = torch.tensor(_IL_TO_SDK, dtype=torch.long)
        onehot = torch.zeros(J, J, dtype=torch.float32, device=self._device)
        onehot[torch.arange(J), il_to_sdk] = 1.0
        self._joint_onehot = onehot.unsqueeze(0).unsqueeze(0)  # (1, 1, J, J)

        # --- SDK leg index per IL joint (for foot lookup) ---
        # _sdk_leg[il_joint] = SDK leg index (FR=0, FL=1, RR=2, RL=3)
        self._sdk_leg = torch.tensor(
            [_IL_TO_SDK[j] // 3 for j in range(J)],
            dtype=torch.long, device=self._device,
        )

        # --- Sensor ring buffer and velocity history ---
        self._sensor_buf = torch.zeros(n, W, J, 4, dtype=torch.float32, device=self._device)
        self._prev_q_dot = torch.zeros(n, J, dtype=torch.float32, device=self._device)
        self._is_warm = torch.zeros(n, dtype=torch.bool, device=self._device)

        self._W = W
        self._dt = cfg.dt

        # --- Contact sensor (set from training script after gym.make()) ---
        # Wire up: actuator.contact_sensor = env.unwrapped.scene["contact_forces"]
        # Shape of net_forces_w: (n_envs, all_bodies, 3)
        self.contact_sensor = None
        # Lazy-initialized on first compute() after contact_sensor is set.
        # Shape: (4,) — indices into net_forces_w body dimension, in SDK order.
        self._contact_body_ids: torch.Tensor | None = None

    def _init_contact_mapping(self) -> None:
        """Build foot body index mapping by name — called once on first compute()."""
        ids, names = self.contact_sensor.find_bodies(".*_foot$")
        print(f"[B1ActuatorNet] Contact sensor foot bodies found: {dict(zip(names, ids))}")
        try:
            ordered_ids = [ids[names.index(foot)] for foot in _SDK_FOOT_NAMES]
        except ValueError as e:
            raise RuntimeError(
                f"[B1ActuatorNet] Could not find expected foot body in contact sensor.\n"
                f"  Expected: {_SDK_FOOT_NAMES}\n"
                f"  Found:    {names}\n"
                f"  Error: {e}"
            )
        self._contact_body_ids = torch.tensor(
            ordered_ids, dtype=torch.long, device=self._device
        )
        print(f"[B1ActuatorNet] Foot body ids (SDK order FR/FL/RR/RL): {ordered_ids}")

    def reset(self, env_ids: Sequence[int]):
        self._sensor_buf[env_ids] = 0.0
        self._prev_q_dot[env_ids] = 0.0
        self._is_warm[env_ids] = False

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # Net already encodes speed-torque curve from data — only clamp hardware limit.
        return torch.clamp(effort, -self.effort_limit, self.effort_limit)

    def _get_foot_counts(self, n: int, J: int) -> torch.Tensor:
        """Return foot contact in approximate SDK counts, shape (n, J)."""
        if self.contact_sensor is None:
            # No sensor wired up — return zeros (OOD but safe fallback)
            return torch.zeros(n, J, dtype=torch.float32, device=self._device)

        if self._contact_body_ids is None:
            self._init_contact_mapping()

        # net_forces_w: (n_envs, all_bodies, 3) — extract foot bodies in SDK order
        forces_n = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._contact_body_ids, :],
            dim=-1,
        )  # (n, 4) in SDK order [FR, FL, RR, RL]

        # Normalize to approximate SDK count range
        foot_counts = (forces_n / _MAX_FORCE_N * _MAX_SDK_COUNT).clamp(0.0, _MAX_SDK_COUNT)

        # Broadcast per joint: each joint uses its leg's foot count
        return foot_counts[:, self._sdk_leg]  # (n, J)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        n, J, W = self._num_envs, self.num_joints, self._W

        # 1. Causal backward-difference acceleration
        q_ddot = (joint_vel - self._prev_q_dot) / self._dt
        self._prev_q_dot.copy_(joint_vel)
        # Clamp: real rosbag max |q̈| = 439.5 rad/s² → 500 passes all training data,
        # clips only physics instability outliers.
        q_ddot = torch.clamp(q_ddot, -500.0, 500.0)

        # 2. Foot contact in approximate SDK counts
        foot_j = self._get_foot_counts(n, J)

        # 3. Build current frame [q, q̇, q̈, foot_j]
        frame = torch.stack([joint_pos, joint_vel, q_ddot, foot_j], dim=-1)  # (n, J, 4)

        # 4. Warm-start: fill entire buffer with first frame after reset
        cold = ~self._is_warm
        if cold.any():
            self._sensor_buf[cold] = frame[cold].unsqueeze(1).expand(-1, W, -1, -1)
            self._is_warm[cold] = True

        # 5. Roll window and insert new frame
        self._sensor_buf = self._sensor_buf.roll(-1, dims=1)
        self._sensor_buf[:, -1] = frame

        # 6. Scale sensor channels (channel-wise StandardScaler)
        buf_scaled = (self._sensor_buf - self._x_mean) / self._x_std  # (n, W, J, 4)

        # 7. Append joint one-hot → (n, W, J, 16)
        onehot_exp = self._joint_onehot.expand(n, W, J, J)
        x = torch.cat([buf_scaled, onehot_exp], dim=-1)

        # 8. Reshape to (n*J, W, 16) — batch over envs × joints
        x = x.permute(0, 2, 1, 3).reshape(n * J, W, 16)

        # 9. Inference
        with torch.inference_mode():
            tau_scaled = self.network(x).squeeze(-1)  # (n*J,)

        # Guard against NaN/Inf from LSTM (cuDNN on some GPUs or OOD inputs)
        tau_scaled = torch.nan_to_num(tau_scaled, nan=0.0, posinf=5.0, neginf=-5.0)

        # 10. Inverse-transform → physical N·m
        self.computed_effort = (tau_scaled * self._y_std + self._y_mean).view(n, J)

        # 11. Clip to hardware limit and return
        self._joint_vel[:] = joint_vel
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


B1ActuatorNetCfg.class_type = B1ActuatorNet
