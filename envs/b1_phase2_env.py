"""
Phase 2 environment — DirectRLEnv with frozen base policies + per-leg blending.

Step flow:
    1. Receive 4-D action (per-leg Δα residual) from Phase 2 policy.
    2. v10: Compute Δα = sigmoid(action) × delta_alpha_max  (∈ [0, 0.3]).
       Asymmetric so α_per_joint can only ride at or above smoothstep.
    3. Compute α_baseline from elapsed time in episode.
    4. Build base-policy observations (matching the 48-D Phase 1 obs format).
    5. Query π_current(obs), π_target(obs) → 12-D joint offset actions.
    6. Per-leg blend:  α_k = clip(α_base + Δα_k, 0, 1)
                       blended[3k:3k+3] = (1-α_k)·current + α_k·target
    7. joint_target = default_joint_pos + action_scale × blended.
    8. Step physics, compute reward, return (obs, reward, dones, info).

Episode setup:
    On reset, sample random (current, target) gait pair where current ≠ target.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from rsl_rl.runners import OnPolicyRunner

from envs.b1_phase2_env_cfg import (
    B1Phase2EnvCfg,
    B1Phase2Residual1DEnvCfg,
    B1Phase2E2ERateEnvCfg,
)
from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg


class B1Phase2Env(DirectRLEnv):
    cfg: B1Phase2EnvCfg

    def __init__(self, cfg: B1Phase2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Frozen base policies ----
        # Loaded as RSL-RL OnPolicyRunner instances (option 1 from design discussion).
        # Each runner is created with a tiny dummy env so we can call .load() and
        # .get_inference_policy() to extract the actor MLP.
        self._base_policies = []
        agent_cfg = B1FlatPPORunnerCfg()

        # Use the env wrapper interface — we'll dispatch via a wrapper-like adapter
        # that exposes num_envs / device / num_obs / num_actions for the runner.
        for path in cfg.base_policy_paths:
            adapter = _RunnerAdapter(self.num_envs, self.device, num_obs=48, num_actions=12)
            runner = OnPolicyRunner(adapter, agent_cfg.to_dict(), log_dir=None,
                                     device=str(self.device))
            runner.load(path)
            policy = runner.get_inference_policy(device=str(self.device))
            self._base_policies.append(policy)

        # ---- Articulation references ----
        self._robot: Articulation = self.scene["robot"]
        self._contact_sensor: ContactSensor = self.scene["contact_forces"]

        # ---- Persistent buffers ----
        self._gait_current = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gait_target = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._last_residual = torch.zeros(self.num_envs, 4, device=self.device)
        # Integrated α for e2e_rate mode — starts at 0, rises monotonically each step
        self._alpha_integrated = torch.zeros(self.num_envs, device=self.device)
        # Per-env transition start time, sampled at reset for robustness
        self._transition_start_s = torch.full(
            (self.num_envs,), float(cfg.transition_start_min_s), device=self.device,
        )

        # Per-env transition duration — fixed at min_s when min==max (v10 behaviour),
        # or sampled uniformly from [min_s, max_s] each reset (v11 curriculum).
        self._transition_duration_env = torch.full(
            (self.num_envs,), float(cfg.transition_duration_min_s), device=self.device,
        )

        # CRITICAL: per-policy last_action buffer.
        # Phase 1 base policies were trained with last_action = their own
        # previous 12-D actor output. Feeding zeros (as v2/v3 did) made
        # policies collapse to a stationary "no-op" output during steady-
        # state windows. Track each base policy's last action separately
        # and use it as the last_action component of that policy's obs
        # next step.
        self._base_last_actions = torch.zeros(
            len(self._base_policies), self.num_envs, 12, device=self.device,
        )

        # Trunk / feet body indices — resolved against the CONTACT SENSOR's body
        # ordering, not the robot articulation's body list. The two orderings can
        # differ (sensor resolves from USD prim paths; articulation from joint tree).
        # Using robot.body_names indices here was the root cause of empty gait
        # diagrams in play_b1_phase2.py.
        foot_ids_raw, foot_names = self._contact_sensor.find_bodies(".*_foot$")
        desired_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        perm = [foot_names.index(n) for n in desired_feet]
        self._foot_ids = [foot_ids_raw[i] for i in perm]

        trunk_ids, _ = self._contact_sensor.find_bodies("trunk")
        self._trunk_idx = trunk_ids[0]

        # Default joint position (cached for action assembly)
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()

        # Velocity command tensor used in base-policy observations
        self._cmd = torch.tensor(
            [cfg.velocity_cmd_x, cfg.velocity_cmd_y, cfg.velocity_cmd_yaw],
            device=self.device,
        ).expand(self.num_envs, 3).clone()

    # ------------------------------------------------------------------
    # DirectRLEnv interface
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """DirectRLEnv hook — clone scene per env (the scene is constructed
        from cfg.scene by the parent, no extra setup needed here)."""
        # Articulation is auto-spawned via cfg.scene.robot
        # Contact sensor likewise via cfg.scene.contact_forces
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions between envs
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called once per control step (before decimated physics steps).
        Build base-policy obs, query both policies, blend per leg, apply."""
        t = self.episode_length_buf.float() * (self.cfg.sim.dt * self.cfg.decimation)
        ramp_progress = (t - self._transition_start_s) / self._transition_duration_env

        control_dt = self.cfg.sim.dt * self.cfg.decimation

        if self.cfg.alpha_schedule == "e2e":
            # E2E baseline: MLP outputs a single α scalar, applied uniformly.
            # No hand-designed ramp; no time-gating; no residual structure.
            alpha_scalar = torch.sigmoid(actions[:, 0])           # (E,) ∈ (0, 1)
            alpha_baseline = alpha_scalar
            delta_alpha = torch.zeros(self.num_envs, 4, device=self.device)
            self._last_residual = delta_alpha.detach()
        elif self.cfg.alpha_schedule == "e2e_rate":
            # E2E rate-based: MLP outputs dα/dt via sigmoid, α integrates from 0.
            # rate ∈ [0, 1/transition_duration_s] so at max rate α reaches 1 in
            # exactly transition_duration_s seconds — same horizon as the residual.
            # Structurally prevents instant α=1 collapse (α must accumulate step by step).
            rate = torch.sigmoid(actions[:, 0]) / self._transition_duration_env
            self._alpha_integrated = (self._alpha_integrated + rate * control_dt).clamp(0.0, 1.0)
            alpha_baseline = self._alpha_integrated
            delta_alpha = torch.zeros(self.num_envs, 4, device=self.device)
            self._last_residual = delta_alpha.detach()
        else:
            # --- Δα residual ---
            # v10: Δα is squashed via sigmoid (NOT tanh) so Δα ∈ [0, delta_alpha_max].
            # Combined with the reduced cap (delta_alpha_max=0.3 in v10), this
            # gives Δα ∈ [0, 0.3]: the MLP can ADVANCE α above smoothstep but
            # cannot DELAY it below smoothstep.
            #
            # Why: v9 diagnosis showed the MLP exploiting the symmetric tanh
            # band [−0.8, +0.8] to delay the first half of the ramp (Δα < 0)
            # and rush the second half. The asymmetric sigmoid removes this
            # exploit by construction — α_per_joint ≥ α_baseline always.
            #
            # Earlier versions used tanh(action) × 0.8 = Δα ∈ [−0.8, +0.8].
            # Supports both 4-D per-leg (action_space=4) and 1-D scalar broadcast
            # (Residual-1D ablation). The 1-D case applies the same Δα to all
            # four legs for a direct comparison with Residual-4D.
            if actions.shape[1] == 1:
                delta_alpha = torch.sigmoid(actions[:, 0:1]) * self.cfg.delta_alpha_max
                delta_alpha = delta_alpha.expand(self.num_envs, 4)   # (E, 4) broadcast
            else:
                delta_alpha = torch.sigmoid(actions) * self.cfg.delta_alpha_max  # (E, 4) ∈ [0, max]

            # --- Hard time-gate the residual ---
            # Δα is forced to zero OUTSIDE the transition window. This guarantees
            # the source gait runs untouched during the hold phase (α_baseline=0)
            # and the target gait runs untouched after the ramp (α_baseline=1).
            # The MLP can only intervene during the ramp itself.
            pad = self.cfg.residual_window_padding_s / self._transition_duration_env
            in_window = ((ramp_progress > -pad) & (ramp_progress < 1.0 + pad)).float()
            delta_alpha = delta_alpha * in_window.unsqueeze(1)  # (E, 4) — zeroed outside window
            self._last_residual = delta_alpha.detach()

            # --- α_baseline schedule ---
            # "linear":      α = clamp(ramp_progress, 0, 1)             constant dα/dt
            # "smoothstep":  α = 3x²−2x³                                dα/dt=0 at endpoints
            x = ramp_progress.clamp(0.0, 1.0)
            if self.cfg.alpha_schedule == "smoothstep":
                alpha_baseline = x * x * (3.0 - 2.0 * x)  # (E,)
            else:
                alpha_baseline = x  # linear

        # --- Query each base policy with ITS OWN previous action ---
        # Each policy was trained with last_action = its own previous output.
        # Pass each its proper history so it doesn't collapse to a no-op pose.
        policy_outputs = []
        for i, policy in enumerate(self._base_policies):
            base_obs_i = self._compute_base_policy_obs(self._base_last_actions[i])
            with torch.no_grad():
                out = policy(base_obs_i)   # (E, 12)
            self._base_last_actions[i] = out.detach()
            policy_outputs.append(out)
        # Stack: (3, E, 12)
        policy_outputs = torch.stack(policy_outputs, dim=0)

        # Index per-env: π_current and π_target
        env_idx = torch.arange(self.num_envs, device=self.device)
        action_current = policy_outputs[self._gait_current, env_idx]  # (E, 12)
        action_target = policy_outputs[self._gait_target, env_idx]   # (E, 12)

        # --- Per-leg blending ---
        # delta_alpha shape (E, 4) → expand to per-joint (E, 12) by repeat-3
        delta_per_joint = delta_alpha.repeat_interleave(3, dim=1)        # (E, 12)
        alpha_per_joint = (alpha_baseline.unsqueeze(1) + delta_per_joint).clamp(0.0, 1.0)
        alpha_baseline_per_joint = alpha_baseline.unsqueeze(1).expand_as(alpha_per_joint).clamp(0.0, 1.0)
        blended = (1 - alpha_per_joint) * action_current + alpha_per_joint * action_target

        # Cache intermediates for play-script inspection
        self._last_action_current = action_current.detach()      # (E, 12) π_current raw output
        self._last_action_target  = action_target.detach()       # (E, 12) π_target raw output
        self._last_alpha_per_joint = alpha_per_joint.detach()    # (E, 12) actual α used (with Δα)
        self._last_alpha_baseline_per_joint = alpha_baseline_per_joint.detach()  # (E, 12) schedule-only α
        self._last_blended = blended.detach()                    # (E, 12) final blended action

        # --- Joint targets ---
        joint_target = self._default_joint_pos + self.cfg.action_scale * blended
        self._joint_target = joint_target  # cached for _apply_action

    def _apply_action(self) -> None:
        """Called once per physics step (decimation × times per control step)."""
        self._robot.set_joint_position_target(self._joint_target)

    def _get_observations(self) -> dict:
        """Phase 2 policy observations (47-D)."""
        d = self._robot.data

        # State features
        base_lin_vel = d.root_lin_vel_b                           # (E, 3)
        base_ang_vel = d.root_ang_vel_b                           # (E, 3)
        projected_gravity = d.projected_gravity_b                 # (E, 3)
        joint_pos_rel = d.joint_pos - self._default_joint_pos     # (E, 12)
        joint_vel = d.joint_vel                                   # (E, 12)

        # Phase 2 specific
        last_action = self._last_residual                         # (E, 4)
        n_gaits = len(self.cfg.gait_names)
        gait_current_oh = torch.nn.functional.one_hot(self._gait_current, num_classes=n_gaits).float()
        gait_target_oh = torch.nn.functional.one_hot(self._gait_target, num_classes=n_gaits).float()

        # α_baseline + cycles_elapsed scalars (must match _apply_action's schedule)
        t = self.episode_length_buf.float() * (self.cfg.sim.dt * self.cfg.decimation)
        ramp_progress = (t - self._transition_start_s) / self._transition_duration_env
        x = ramp_progress.clamp(0.0, 1.0)
        if self.cfg.alpha_schedule == "e2e":
            # No hand-designed baseline visible to E2E MLP — it must infer timing
            alpha_baseline = torch.zeros(self.num_envs, 1, device=self.device)
        elif self.cfg.alpha_schedule == "smoothstep":
            alpha_baseline = (x * x * (3.0 - 2.0 * x)).unsqueeze(1)
        else:
            alpha_baseline = x.unsqueeze(1)
        cycles_elapsed = (t / 1.0).unsqueeze(1)                   # 1 Hz CPG-equivalent

        obs_parts = [
            base_lin_vel, base_ang_vel, projected_gravity,         # 9
            joint_pos_rel, joint_vel,                               # 24
            last_action,                                            # 4
            gait_current_oh, gait_target_oh,                        # 8
            alpha_baseline, cycles_elapsed,                         # 2
        ]
        if self.cfg.observation_space == 46:
            # v11: append normalised transition duration so the MLP can
            # condition on ramp speed. Normalised by 5.0 (max of curriculum
            # range) so the value stays in [0.3, 1.0] during training.
            norm_duration = (self._transition_duration_env / 5.0).unsqueeze(1)  # (E, 1)
            obs_parts.append(norm_duration)
        obs = torch.cat(obs_parts, dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self._robot.data
        cfg = self.cfg

        # Velocity tracking (use the fixed self._cmd — same throughout episode)
        lin_vel_error = torch.sum((self._cmd[:, :2] - d.root_lin_vel_b[:, :2]) ** 2, dim=1)
        ang_vel_error = (self._cmd[:, 2] - d.root_ang_vel_b[:, 2]) ** 2
        track_lin = torch.exp(-lin_vel_error / 0.25) * cfg.rew_track_lin_vel
        track_ang = torch.exp(-ang_vel_error / 0.25) * cfg.rew_track_ang_vel

        # Orientation (penalize tilt) — v6: in-window multiplier so the MLP
        # gets stronger gradient to spend Δα on tilt suppression DURING the
        # transition window, where the gait kinematics fight body stability.
        t_now = self.episode_length_buf.float() * (self.cfg.sim.dt * self.cfg.decimation)
        ramp_progress_r = (t_now - self._transition_start_s) / self._transition_duration_env
        in_window_r = ((ramp_progress_r > 0.0) & (ramp_progress_r < 1.0)).float()
        orient_weight = cfg.rew_orientation * (1.0 + cfg.transition_orientation_boost * in_window_r)
        orient = torch.sum(d.projected_gravity_b[:, :2] ** 2, dim=1) * orient_weight

        # Height (penalize deviation from target)
        height_err = (d.root_pos_w[:, 2] - cfg.target_height) ** 2 * cfg.rew_height

        # Action rate (smoothness on Δα itself — penalizes residual jitter)
        action_rate = torch.sum((self._last_residual - getattr(self, "_prev_residual",
                                                                 self._last_residual)) ** 2, dim=1) * cfg.rew_action_rate
        self._prev_residual = self._last_residual

        # Joint-acceleration L2 — penalizes acc magnitude (defensive against
        # PD overshoot). Kept from v7 but is NOT a smoothness metric on its
        # own (a constant high-acc trajectory has zero jerk yet large jacc²).
        joint_acc = torch.sum(d.joint_acc ** 2, dim=1) * cfg.rew_joint_acc

        # v8: Joint JERK L2 — penalizes the rate of change of joint acceleration.
        # Jerk (rad/s³) is the motor-relevant smoothness signal: high jerk =
        # rapid acc reversals = mechanical stress on housing/gears. Computed
        # via finite difference; first step uses jerk=0 by initialising prev
        # to the current value the first time round.
        control_dt = self.cfg.sim.dt * self.cfg.decimation
        prev_jacc = getattr(self, "_prev_joint_acc", d.joint_acc)
        jerk = (d.joint_acc - prev_jacc) / control_dt          # (E, 12) rad/s³
        joint_jerk = torch.sum(jerk ** 2, dim=1) * cfg.rew_joint_jerk
        self._prev_joint_acc = d.joint_acc.detach()

        # Alive bonus
        alive = torch.full_like(track_lin, cfg.rew_alive)

        # Sparsity on |Δα|² — encourages MLP to output near-zero except when
        # correction is genuinely needed (during the transition window).
        # At Δα=0 (steady-state), this term is 0. Large Δα during steady-state
        # is penalized; large Δα during transitions is justified by the
        # tracking/orientation rewards it enables.
        sparsity = torch.sum(self._last_residual ** 2, dim=1) * cfg.rew_residual_sparsity

        return (track_lin + track_ang + orient + height_err + action_rate
                + joint_acc + joint_jerk + alive + sparsity)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, truncated)."""
        d = self._robot.data

        # Trunk contact termination
        nf = self._contact_sensor.data.net_forces_w_history
        trunk_force = torch.norm(nf[:, :, self._trunk_idx, :], dim=-1).max(dim=1)[0]
        terminated_trunk = trunk_force > self.cfg.base_contact_threshold_n

        # Bad orientation termination
        gravity_xy_sq = torch.sum(d.projected_gravity_b[:, :2] ** 2, dim=1)
        terminated_orient = gravity_xy_sq > self.cfg.bad_orientation_limit

        terminated = terminated_trunk | terminated_orient

        # Time-out (truncation)
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset env(s): re-spawn robot, re-sample (current, target) gait pair."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset articulation to default state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_state_to_sim(root_state, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample (current, target) gait pair where current ≠ target.
        # 3 gaits → randint(0, 3) for cur; randint(0, 2) for tgt offset.
        n = len(env_ids)
        cur = torch.randint(0, 3, (n,), device=self.device)
        tgt = torch.randint(0, 2, (n,), device=self.device)
        tgt = torch.where(tgt >= cur, tgt + 1, tgt)  # offset to skip equal index
        self._gait_current[env_ids] = cur
        self._gait_target[env_ids] = tgt

        # Reset residual and integrated-α buffers
        self._last_residual[env_ids] = 0.0
        self._alpha_integrated[env_ids] = 0.0

        # Reset prev_joint_acc so v8's jerk penalty doesn't spike across resets
        # (joint_acc is non-zero post-reset; first-step jerk would otherwise be
        # huge).  Using a fresh zero baseline + the current-fallback in
        # _get_rewards lets jerk = 0 on the first post-reset step.
        if hasattr(self, "_prev_joint_acc"):
            self._prev_joint_acc[env_ids] = 0.0

        # Reset per-policy last-action history for these envs.
        # New episode → no prior policy output → start fresh at zero.
        # (Subsequent steps will populate this with each policy's actual output.)
        self._base_last_actions[:, env_ids, :] = 0.0

        # Sample per-env transition start time ∈ [min_s, max_s]
        # Random transition timing makes the policy robust to different
        # mid-episode start moments rather than overfitting to t=2.0.
        rng = torch.rand(n, device=self.device)
        self._transition_start_s[env_ids] = (
            self.cfg.transition_start_min_s
            + rng * (self.cfg.transition_start_max_s - self.cfg.transition_start_min_s)
        )

        # Sample per-env transition duration.
        # Fixed when min == max (v10 behaviour); uniform curriculum when min < max (v11).
        dur_range = self.cfg.transition_duration_max_s - self.cfg.transition_duration_min_s
        if dur_range > 0.0:
            rng_dur = torch.rand(n, device=self.device)
            self._transition_duration_env[env_ids] = (
                self.cfg.transition_duration_min_s + rng_dur * dur_range
            )
        else:
            self._transition_duration_env[env_ids] = self.cfg.transition_duration_min_s

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_base_policy_obs(self, last_action: torch.Tensor) -> torch.Tensor:
        """Build the 48-D observation expected by Phase 1 base policies.

        Phase 1 obs order (from LocomotionVelocityRoughEnvCfg PolicyCfg):
          base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
          + velocity_commands(3) + joint_pos(12) + joint_vel(12)
          + actions(12) = 48
        """
        d = self._robot.data
        return torch.cat([
            d.root_lin_vel_b,                            # (E, 3)
            d.root_ang_vel_b,                            # (E, 3)
            d.projected_gravity_b,                       # (E, 3)
            self._cmd,                                   # (E, 3)
            d.joint_pos - self._default_joint_pos,       # (E, 12)
            d.joint_vel,                                 # (E, 12)
            last_action,                                 # (E, 12) — policy's OWN previous output
        ], dim=1)


# ---------------------------------------------------------------------------
# Adapter for OnPolicyRunner so it can be constructed without a real env
# ---------------------------------------------------------------------------

class _RunnerAdapter:
    """Minimal interface to satisfy OnPolicyRunner during checkpoint loading.

    RSL-RL's OnPolicyRunner.__init__ inspects extras["observations"]
    to determine if the env exposes a privileged "critic" obs group;
    return an empty dict so it skips that branch.
    """
    def __init__(self, num_envs, device, num_obs, num_actions):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_privileged_obs = 0
        self.max_episode_length = 1000
        self.device = device

    def get_observations(self):
        obs = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        return obs, {"observations": {}}

    def reset(self):
        return self.get_observations()

    def step(self, actions):
        obs = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        rew = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device)
        return obs, rew, dones, {"observations": {}}


# ---------------------------------------------------------------------------
# Gym registration
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-B1-Phase2-Transition-v0",
    entry_point="envs.b1_phase2_env:B1Phase2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.b1_phase2_env_cfg:B1Phase2EnvCfg",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:Phase2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-B1-Phase2-E2E-v0",
    entry_point="envs.b1_phase2_env:B1Phase2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.b1_phase2_env_cfg:B1Phase2E2EEnvCfg",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:Phase2E2EPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-B1-Phase2-Residual1D-v0",
    entry_point="envs.b1_phase2_env:B1Phase2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.b1_phase2_env_cfg:B1Phase2Residual1DEnvCfg",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:Phase2Residual1DPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-B1-Phase2-E2E-Rate-v0",
    entry_point="envs.b1_phase2_env:B1Phase2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "envs.b1_phase2_env_cfg:B1Phase2E2ERateEnvCfg",
        "rsl_rl_cfg_entry_point": "envs.b1_velocity_ppo_cfg:Phase2E2ERatePPORunnerCfg",
    },
)

