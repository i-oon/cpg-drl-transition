"""
Play Phase 2 residual transition policy.

Cycles through scripted (current → target) gait pairs every N seconds and
visualizes the per-leg residual α corrections + transition smoothness.

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/play_b1_phase2.py --checkpoint logs/phase2/<run>/model_final.pt
    python scripts/play_b1_phase2.py --checkpoint <path> --num_envs 4 --steps 2000

Author: Disthorn Suttawet
Course: FRA 503 Deep Reinforcement Learning
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play B1 Phase 2 transition policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-B1-Phase2-Transition-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint path. Not required when --baseline is set.")
parser.add_argument("--baseline", type=str, default=None,
                    choices=["linear_ramp", "smoothstep_ramp", "discrete"],
                    help="Run a no-training baseline instead of a learned policy.\n"
                         "  linear_ramp    — pure hand-designed linear α ramp, Δα≡0\n"
                         "  smoothstep_ramp — same with smoothstep α schedule (v7 schedule)\n"
                         "  discrete       — instant α=1 at switch time, no ramp")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--gait_pairs", type=str, default="trot,bound,pace",
                    help="Comma-separated gait sequence to cycle through.")
parser.add_argument("--switch_interval_s", type=float, default=8.0,
                    help="Seconds between gait switches.")
parser.add_argument("--robot_mass_kg", type=float, default=50.0,
                    help="B1 mass for Cost-of-Transport calc.")
parser.add_argument("--save_plots", type=str, default=None,
                    help="If set, save diagnostic plots to this dir (e.g. logs/phase2/<run>/diag).")
parser.add_argument("--save_csv", type=str, default=None,
                    help="If set, dump per-step time-series to this CSV path.")
parser.add_argument("--legacy_4gait", action="store_true",
                    help="Replay v1/v2 checkpoints (47-D obs, 4 gaits including steer). "
                         "Note: rollout uses CURRENT env logic (time-gating + last_action fix), "
                         "so behavior may differ from original v1/v2 training-time runs.")
parser.add_argument("--video", type=str, default=None,
                    help="If set, record a video to this directory (e.g. logs/phase2/<run>/video).")
parser.add_argument("--video_length", type=int, default=None,
                    help="Number of control steps to record. Default = full --steps.")
parser.add_argument("--cam_eye", type=str, default="2.5,2.5,1.5",
                    help="Camera position relative to tracked robot (x,y,z) [m].")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.0,0.5",
                    help="Camera lookat offset from tracked robot (x,y,z) [m].")
parser.add_argument("--seed", type=int, default=42,
                    help="RNG seed for env domain randomization. Fix across baselines for fair comparison.")
args = parser.parse_args()

# Video implies headless cameras
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch

torch.manual_seed(args.seed)
np.random.seed(args.seed)

import isaaclab_tasks  # noqa: F401
import envs.b1_phase2_env  # noqa: F401
from envs.b1_phase2_env_cfg import (
    B1Phase2EnvCfg, B1Phase2E2EEnvCfg,
    B1Phase2Residual1DEnvCfg, B1Phase2E2ERateEnvCfg,
)
from envs.b1_velocity_ppo_cfg import (
    Phase2PPORunnerCfg, Phase2E2EPPORunnerCfg,
    Phase2Residual1DPPORunnerCfg, Phase2E2ERatePPORunnerCfg,
)
from isaaclab.envs.common import ViewerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

_task_cfg_map = {
    "Isaac-B1-Phase2-E2E-v0":        (Phase2E2EPPORunnerCfg,      B1Phase2E2EEnvCfg),
    "Isaac-B1-Phase2-Residual1D-v0": (Phase2Residual1DPPORunnerCfg, B1Phase2Residual1DEnvCfg),
    "Isaac-B1-Phase2-E2E-Rate-v0":   (Phase2E2ERatePPORunnerCfg,   B1Phase2E2ERateEnvCfg),
}
_runner_cls, _env_cls = _task_cfg_map.get(
    args.task, (Phase2PPORunnerCfg, B1Phase2EnvCfg)
)
agent_cfg = _runner_cls()
env_cfg = _env_cls()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = agent_cfg.device
env_cfg.seed = args.seed

# Legacy 4-gait config (v1, v2 — when steer was still in the portfolio)
if args.legacy_4gait:
    env_cfg.gait_names = ("trot", "bound", "pace", "steer")
    env_cfg.observation_space = 47
    env_cfg.base_policy_paths = (
        "logs/phase1_final/trot.pt",
        "logs/phase1_final/bound.pt",
        "logs/phase1_final/pace.pt",
        "logs/phase1_final/steer.pt",
    )
    print("  [legacy_4gait] obs=47, gaits=trot/bound/pace/steer")

# Validate args
if args.baseline is None and args.checkpoint is None:
    print("ERROR: either --checkpoint or --baseline must be specified.")
    sys.exit(1)

# Baseline mode: override alpha_schedule
if args.baseline == "linear_ramp":
    env_cfg.alpha_schedule = "linear"
elif args.baseline == "smoothstep_ramp":
    env_cfg.alpha_schedule = "smoothstep"
elif args.baseline == "discrete":
    env_cfg.alpha_schedule = "linear"   # discrete overrides α at switch time in the loop

# Camera follows env 0's robot
cam_eye = tuple(float(x) for x in args.cam_eye.split(","))
cam_lookat = tuple(float(x) for x in args.cam_lookat.split(","))
env_cfg.viewer = ViewerCfg(
    origin_type="asset_root",
    asset_name="robot",
    env_index=0,
    eye=cam_eye,
    lookat=cam_lookat,
    resolution=(1280, 720),
)

env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

# Video recording — wrap raw gym env BEFORE RslRl wrapper
if args.video:
    video_dir = Path(args.video)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_length = args.video_length if args.video_length is not None else args.steps
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=video_length,
        disable_logger=True,
        name_prefix="phase2_play",
    )
    print(f"  [video] recording {video_length} steps → {video_dir}/")

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ---------------------------------------------------------------------------
# Load policy  (skipped for no-training baselines)
# ---------------------------------------------------------------------------

if args.baseline is not None:
    # No-training baseline: send zeros as actions → Δα=0 (tanh(0)=0)
    _n_actions = env.num_actions
    _device    = agent_cfg.device
    policy = lambda obs: torch.zeros(obs.shape[0], _n_actions, device=_device)
    print(f"  [baseline={args.baseline}] No checkpoint loaded — using zero actions (Δα≡0).")
else:
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

# ---------------------------------------------------------------------------
# Build gait switch schedule
# ---------------------------------------------------------------------------

gait_names = list(env.unwrapped.cfg.gait_names)
# When --legacy_4gait is on, expand the default sequence to include steer
if args.legacy_4gait and args.gait_pairs == "trot,bound,pace":
    args.gait_pairs = "trot,bound,pace,steer"
gait_seq = args.gait_pairs.split(",")
for g in gait_seq:
    if g not in gait_names:
        print(f"ERROR: gait '{g}' not in {gait_names}")
        sys.exit(1)

# Cycle: gait_seq[0] → gait_seq[1] → ... → gait_seq[-1] → gait_seq[0] ...
control_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
switch_steps = int(args.switch_interval_s / control_dt)

method_label = args.baseline if args.baseline else f"residual({Path(args.checkpoint).parent.name})"
print(f"\n  Task          : {args.task}")
print(f"  Method        : {method_label}")
print(f"  Checkpoint    : {args.checkpoint}")
print(f"  Envs          : {env.num_envs}")
print(f"  Steps         : {args.steps}  ({args.steps * control_dt:.1f}s)")
print(f"  Gait sequence : {' → '.join(gait_seq)}  (switch every {args.switch_interval_s}s)")
print()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

obs, _ = env.get_observations()
total_reward = torch.zeros(env.num_envs, device=env.device)

print(f"  {'step':>5} | {'gait':>14} | {'vx':>6} {'vz':>6} | {'h':>5} {'tilt':>6} | "
      f"{'Δα_FL':>6} {'Δα_FR':>6} {'Δα_RL':>6} {'Δα_RR':>6}")
print("  " + "-" * 100)

# History (env 0)
vx_hist, h_hist, tilt_hist, vz_hist = [], [], [], []
delta_hist = []
gait_hist = []
joint_pos_hist = []          # (T, 12)
joint_vel_hist = []           # (T, 12)
joint_acc_hist = []           # (T, 12)
torque_hist = []              # (T, 12)
power_inst_hist = []          # (T,) Σ|τ·q̇|
x_pos_hist = []               # (T,) world x for distance integration
switch_step_marks = []        # control-step indices when a switch fired
contact_hist = []             # (T, 4) bool — FL FR RL RR stance/swing
action_current_hist = []     # (T, 12) π_current raw output
action_target_hist = []      # (T, 12) π_target raw output
alpha_hist = []               # (T, 12) actual per-joint α (with Δα)
alpha_baseline_hist = []     # (T, 12) schedule-only α (without Δα)
blended_hist = []             # (T, 12) final blended action sent to robot
ramp_progress_hist = []      # (T,) ramp_progress scalar for env 0
termination_hist = []        # (T,) bool — True if env0 terminated after this state

robot = env.unwrapped._robot
contact_sensor = env.unwrapped._contact_sensor
foot_ids = env.unwrapped._foot_ids    # [FL, FR, RL, RR] indices into sensor
e0_idx = 0  # env 0 for verbose printing

# Override env's gait sampling — schedule deterministic transitions
seq_idx = 0
def _set_gait_pair(env, current_name, target_name):
    """Force all envs to (current, target) gait pair."""
    cur = gait_names.index(current_name)
    tgt = gait_names.index(target_name)
    env.unwrapped._gait_current[:] = cur
    env.unwrapped._gait_target[:] = tgt

current_name = gait_seq[0]
target_name = gait_seq[1] if len(gait_seq) > 1 else gait_seq[0]
_set_gait_pair(env, current_name, target_name)
# Initialise α clock consistently so segment 1 also has a 2 s source hold.
# _reset_idx() at env init randomises _transition_start_s to [1.5, 3.5];
# overwrite here so all three methods start from the same baseline.
env.unwrapped.episode_length_buf[:] = 0
env.unwrapped._transition_start_s[:] = 2.0

for step in range(args.steps):
    # Switch gait pair every switch_interval_s
    if step > 0 and step % switch_steps == 0:
        seq_idx = (seq_idx + 1) % len(gait_seq)
        current_name = target_name
        target_name = gait_seq[(seq_idx + 1) % len(gait_seq)]
        _set_gait_pair(env, current_name, target_name)
        # Reset α clock and pin transition_start_s to a fixed 2 s so the
        # source gait is always visible for 2 s before the ramp begins,
        # regardless of what _reset_idx may have randomised it to.
        env.unwrapped.episode_length_buf[:] = 0
        env.unwrapped._transition_start_s[:] = 2.0
        switch_step_marks.append(step)

    # Discrete baseline: hold source for 2 s per segment, then instant α jump.
    # Controlled via _transition_start_s (not episode_length_buf) because
    # _pre_physics_step runs BEFORE episode_length_buf is incremented (line 335
    # vs 360 in direct_rl_env.py), so buf manipulation gets seen one step late.
    # Setting _transition_start_s to ±1e6 forces ramp_progress << 0 (α=0) or
    # >> 1 (α=1) regardless of episode_length_buf value.
    if args.baseline == "discrete":
        _transition_start_steps = int(2.0 / control_dt)   # 100 steps = 2.0 s
        steps_since_switch = (step if len(switch_step_marks) == 0
                              else step - switch_step_marks[-1])
        if steps_since_switch >= _transition_start_steps:
            env.unwrapped._transition_start_s[:] = -1e6  # ramp >> 1 → α=1
        else:
            env.unwrapped._transition_start_s[:] = 1e6   # ramp << 0 → α=0

    with torch.no_grad():
        actions = policy(obs)
    obs, reward, dones, extras = env.step(actions)
    total_reward += reward

    d = robot.data
    vx = d.root_lin_vel_b[e0_idx, 0].item()
    vz = d.root_lin_vel_b[e0_idx, 2].item()
    h = d.root_pos_w[e0_idx, 2].item()
    tilt = torch.sum(d.projected_gravity_b[e0_idx, :2] ** 2).item()

    delta = env.unwrapped._last_residual[e0_idx].cpu().numpy()  # (4,)

    jp = d.joint_pos[e0_idx].cpu().numpy()                      # (12,)
    jv = d.joint_vel[e0_idx].cpu().numpy()                      # (12,)
    ja = d.joint_acc[e0_idx].cpu().numpy()                      # (12,)
    tq = d.applied_torque[e0_idx].cpu().numpy()                 # (12,)
    p_inst = float(np.sum(np.abs(tq * jv)))                     # Σ|τ·q̇|  [W]
    x_w = d.root_pos_w[e0_idx, 0].item()

    ct = contact_sensor.data.current_contact_time[e0_idx, foot_ids]
    contact_hist.append((ct > 0.0).cpu().numpy())   # (4,) bool FL FR RL RR

    eu = env.unwrapped
    action_current_hist.append(eu._last_action_current[e0_idx].cpu().numpy())
    action_target_hist.append(eu._last_action_target[e0_idx].cpu().numpy())
    alpha_hist.append(eu._last_alpha_per_joint[e0_idx].cpu().numpy())
    alpha_baseline_hist.append(eu._last_alpha_baseline_per_joint[e0_idx].cpu().numpy())
    blended_hist.append(eu._last_blended[e0_idx].cpu().numpy())
    t_ep = eu.episode_length_buf[e0_idx].float().item() * control_dt
    ramp_prog = (t_ep - eu._transition_start_s[e0_idx].item()) / eu._transition_duration_env[e0_idx].item()
    ramp_progress_hist.append(ramp_prog)

    vx_hist.append(vx); h_hist.append(h); tilt_hist.append(tilt); vz_hist.append(vz)
    delta_hist.append(delta)
    gait_hist.append((current_name, target_name))
    joint_pos_hist.append(jp)
    joint_vel_hist.append(jv)
    joint_acc_hist.append(ja)
    torque_hist.append(tq)
    power_inst_hist.append(p_inst)
    x_pos_hist.append(x_w)

    # Record whether env0 terminated this step (time-out or body fall)
    termination_hist.append(dones[e0_idx].item())

    # Re-assert scripted gait pair + segment-aligned α clock after any auto-reset.
    # _reset_idx() re-randomizes _gait_current, _gait_target, _transition_start_s,
    # and sets episode_length_buf=0, which would restart the source-hold timer
    # from scratch mid-segment — making ramp timing inconsistent across methods.
    # Restoring episode_length_buf=steps_since_switch keeps the α clock aligned
    # with the switch boundary, so all methods are directly comparable.
    if dones.any():
        _steps_since_switch = (step - switch_step_marks[-1] if switch_step_marks
                               else step)
        env.unwrapped._gait_current[:] = gait_names.index(current_name)
        env.unwrapped._gait_target[:] = gait_names.index(target_name)
        env.unwrapped._transition_start_s[:] = 2.0
        env.unwrapped.episode_length_buf[:] = _steps_since_switch

    if (step + 1) % 50 == 0:
        gait_str = f"{current_name[:5]}→{target_name[:5]}"
        alpha_b = eu._last_alpha_baseline_per_joint[e0_idx, 0].item()
        gc_env = eu._gait_current[e0_idx].item()
        print(f"  {step+1:5d} | {gait_str:>14} | gc={gc_env} α={alpha_b:.3f} | "
              f"{vx:+6.3f} {h:5.3f} {tilt:6.4f} | "
              f"Δα={delta[0]:+.3f} {delta[1]:+.3f} {delta[2]:+.3f} {delta[3]:+.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 70
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — Phase 2 transition learning")
print(sep)

vx_a = np.array(vx_hist); h_a = np.array(h_hist); tilt_a = np.array(tilt_hist)
vz_a = np.array(vz_hist)
delta_a = np.array(delta_hist)
jp_a = np.array(joint_pos_hist)         # (T, 12)
jv_a = np.array(joint_vel_hist)
ja_a = np.array(joint_acc_hist)
tq_a = np.array(torque_hist)
p_a  = np.array(power_inst_hist)
x_a  = np.array(x_pos_hist)
ac_a  = np.array(action_current_hist)   # (T, 12) π_current raw output
at_a  = np.array(action_target_hist)    # (T, 12) π_target raw output
al_a  = np.array(alpha_hist)            # (T, 12) actual α per joint
alb_a = np.array(alpha_baseline_hist)   # (T, 12) schedule-only α
bl_a  = np.array(blended_hist)          # (T, 12) final blended action
rp_a  = np.array(ramp_progress_hist)    # (T,) ramp_progress for env 0
alpha_base_1d = alb_a[:, 0]            # (T,) scalar — same value across all 12 joints
# Residual joint contribution: how much Δα shifts each joint target
# = (α_actual − α_baseline) × (π_target − π_current), scaled by action_scale
residual_joint_a = (al_a - alb_a) * (at_a - ac_a)  # (T, 12), in action units
residual_joint_scaled_a = residual_joint_a * env_cfg.action_scale  # in radians

# Detect ramp-start (ramp_progress crosses 0 from below) and
# ramp-end (ramp_progress crosses 1 from below) for vertical line markers.
ramp_start_times, ramp_end_times = [], []
for i in range(1, len(rp_a)):
    if rp_a[i - 1] < 0 <= rp_a[i]:
        ramp_start_times.append(i * control_dt)
    if rp_a[i - 1] < 1 <= rp_a[i]:
        ramp_end_times.append(i * control_dt)

# Body acceleration (forward + vertical) — used for body_acc plot
body_acc_x    = np.diff(vx_a) / control_dt   # (T-1,) m/s²
body_acc_z    = np.diff(vz_a) / control_dt   # (T-1,) m/s²
body_acc_norm = np.sqrt(body_acc_x**2 + body_acc_z**2)
body_acc_rms  = float(np.sqrt(np.mean(body_acc_norm**2)))
body_acc_max  = float(body_acc_norm.max())

# Termination times — steps where env0 terminated (pre-reset state was logged)
term_a = np.array(termination_hist)
termination_times = np.where(term_a)[0] * control_dt

# Print termination count
n_terms = int(term_a.sum())
print(f"  Terminations  : {n_terms}  (env 0 resets during playback)")

# Cost of Transport over the run
distance = max(x_a[-1] - x_a[0], 1e-6)              # m
energy   = float(np.sum(p_a) * control_dt)          # J
cot      = energy / (args.robot_mass_kg * 9.81 * distance)

# Joint accel: RMS across all 12 joints, mean over time
joint_acc_rms = float(np.sqrt(np.mean(ja_a ** 2)))

# Joint JERK = d(joint_acc)/dt — the actual smoothness metric (rad/s³).
# `jacc_RMS` measures acceleration magnitude, NOT smoothness — a constant-
# high-acc trajectory has zero jerk yet large jacc². Use jerk for any
# "smooth motion" claim.
jerk_a = np.diff(ja_a, axis=0) / control_dt          # (T-1, 12)  rad/s³
jerk_rms = float(np.sqrt(np.mean(jerk_a ** 2)))
jerk_max = float(np.max(np.abs(jerk_a)))

print(f"  Steps         : {args.steps}")
print(f"  Total reward  : {total_reward.mean().item():.2f}")
print(f"\n  vx     : mean={vx_a.mean():+.3f}  std={vx_a.std():.3f}  "
      f"min={vx_a.min():+.3f}  max={vx_a.max():+.3f}")
print(f"  Height : mean={h_a.mean():.3f}  std={h_a.std():.3f}  target=0.42")
print(f"  Tilt   : mean={tilt_a.mean():.4f}  max={tilt_a.max():.4f}")
print(f"\n  Distance       : {distance:.2f} m  (over {args.steps * control_dt:.1f}s)")
print(f"  Energy used    : {energy:.1f} J")
print(f"  Cost of Transp : {cot:.3f}      (lower = better; biological quadrupeds ~0.2)")
print(f"  Joint-acc RMS  : {joint_acc_rms:.2f} rad/s²")
print(f"  Joint-JERK RMS : {jerk_rms:8.0f} rad/s³")
print(f"  Joint-JERK max : {jerk_max:8.0f} rad/s³")
print(f"  Body-acc RMS   : {body_acc_rms:.3f} m/s²  (forward+vertical combined)")
print(f"  Body-acc max   : {body_acc_max:.3f} m/s²")
print(f"\n  Per-leg Δα (residual correction) statistics:")
print(f"    {'leg':<5} {'mean':>8} {'std':>8} {'|Δα|max':>9}")
for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
    m = delta_a[:, i].mean()
    s = delta_a[:, i].std()
    amax = np.abs(delta_a[:, i]).max()
    print(f"    {lab:<5} {m:+8.3f} {s:8.3f} {amax:9.3f}")
print(sep)

# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

if args.save_plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.save_plots)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_axis = np.arange(args.steps) * control_dt

    # B1 joint order from the asset cfg:
    #   FL_hip, FR_hip, RL_hip, RR_hip,
    #   FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    #   FL_calf, FR_calf, RL_calf, RR_calf
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    contact_a = np.array(contact_hist)   # (T, 4) bool

    # Segment boundaries and labels for annotation
    seg_boundaries = [0.0] + [s * control_dt for s in switch_step_marks] + [t_axis[-1]]
    seg_labels = []
    for j in range(len(seg_boundaries) - 1):
        seg_idx = j % len(gait_seq)
        tgt_idx = (seg_idx + 1) % len(gait_seq)
        seg_labels.append(f"{gait_seq[seg_idx]}→{gait_seq[tgt_idx]}")

    def _draw_phase_bands(ax, label_segments=False):
        """3-zone background shading + switch/termination lines.

        Blue   = source-gait hold  (α_base < 0.05)
        Purple = transition window  (0.05 ≤ α_base ≤ 0.95)
        Orange = target-gait hold  (α_base > 0.95)
        Red dashed = gait-pair swap | Magenta dotted = episode termination
        """
        xform = ax.get_xaxis_transform()   # x in data coords, y in axes (0-1) coords
        src_mask = alpha_base_1d < 0.05
        trs_mask = (alpha_base_1d >= 0.05) & (alpha_base_1d <= 0.95)
        tgt_mask = alpha_base_1d > 0.95
        ax.fill_between(t_axis, 0, 1, where=src_mask, alpha=0.10, color="#4472C4",
                        step="post", transform=xform, zorder=0)
        ax.fill_between(t_axis, 0, 1, where=trs_mask, alpha=0.12, color="#7B2D8B",
                        step="post", transform=xform, zorder=0)
        ax.fill_between(t_axis, 0, 1, where=tgt_mask, alpha=0.10, color="#ED7D31",
                        step="post", transform=xform, zorder=0)
        for s in switch_step_marks:
            ax.axvline(s * control_dt, ls="--", c="r", alpha=0.7, lw=1.2)
        for t in termination_times:
            ax.axvline(t, ls=":", c="magenta", alpha=0.9, lw=1.5)
        if label_segments:
            for j, label in enumerate(seg_labels):
                mid = (seg_boundaries[j] + seg_boundaries[j + 1]) / 2
                ax.text(mid, 1.01, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7.5, color="darkred",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # 0. Gait diagram — foot contact timing (same style as Phase 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_gait = axes[0]
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        stance = contact_a[:, i]
        ax_gait.fill_between(t_axis, i + 0.1, i + 0.9, where=stance,
                             color="C0", alpha=0.85, step="post")
    _draw_phase_bands(ax_gait, label_segments=True)
    ax_gait.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax_gait.set_yticklabels(["FL", "FR", "RL", "RR"])
    ax_gait.invert_yaxis()
    ax_gait.set_ylabel("foot")
    ax_gait.set_title(f"Gait diagram — {method_label}  "
                      f"(blue=src-hold  purple=transition  orange=tgt-hold  red=swap)")
    ax_gait.grid(alpha=0.2)
    # Bottom panel: vx trace
    ax_vx = axes[1]
    ax_vx.plot(t_axis, vx_a, lw=0.8, c="C1")
    ax_vx.axhline(env_cfg.velocity_cmd_x, ls=":", c="g", lw=1.0, label="cmd vx")
    _draw_phase_bands(ax_vx, label_segments=True)
    ax_vx.set_ylabel("vx [m/s]")
    ax_vx.set_xlabel("time [s]")
    ax_vx.legend(loc="upper right", fontsize=8)
    ax_vx.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "gait_diagram.png", dpi=120)
    plt.close(fig)

    # 1. Joint positions per leg
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (leg, k) in zip(axes, leg_idx.items()):
        ax.plot(t_axis, jp_a[:, k],      label=f"{leg}_hip",   lw=1.0)
        ax.plot(t_axis, jp_a[:, 4 + k],  label=f"{leg}_thigh", lw=1.0)
        ax.plot(t_axis, jp_a[:, 8 + k],  label=f"{leg}_calf",  lw=1.0)
        _draw_phase_bands(ax, label_segments=True)
        ax.set_ylabel(f"{leg} [rad]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint positions per leg  (blue=src-hold  purple=transition  orange=tgt-hold)")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_positions.png", dpi=120)
    plt.close(fig)

    # 1b. Joint acceleration per leg — spikes visible at transition boundaries
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (leg, k) in zip(axes, leg_idx.items()):
        ax.plot(t_axis, ja_a[:, k],      label=f"{leg}_hip",   lw=0.8, alpha=0.85)
        ax.plot(t_axis, ja_a[:, 4 + k],  label=f"{leg}_thigh", lw=0.8, alpha=0.85)
        ax.plot(t_axis, ja_a[:, 8 + k],  label=f"{leg}_calf",  lw=0.8, alpha=0.85)
        ax.axhline(0, c="k", lw=0.5)
        _draw_phase_bands(ax, label_segments=True)
        ax.set_ylabel(f"{leg} [rad/s²]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint acceleration per leg  (spikes = transition shocks)  "
                 "(blue=src-hold  purple=transition  orange=tgt-hold)")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_acc.png", dpi=120)
    plt.close(fig)

    # 1c. Body acceleration — forward (ax) and vertical (az) components.
    # More interpretable than joint jerk: shows the "jolt" the trunk experiences.
    # body_acc_x = dvx/dt, body_acc_z = dvz/dt, both in m/s².
    # Spikes during the purple transition zone = blend-induced trunk disturbance.
    t_bacc = t_axis[1:]   # np.diff is one element shorter
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.5]})
    axes[0].plot(t_bacc, body_acc_x, lw=0.9, c="C0", label="ax = dvx/dt")
    axes[0].axhline(0, c="k", lw=0.4)
    _draw_phase_bands(axes[0], label_segments=True)
    axes[0].set_ylabel("body ax\n[m/s²]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_bacc, body_acc_z, lw=0.9, c="C2", label="az = dvz/dt")
    axes[1].axhline(0, c="k", lw=0.4)
    _draw_phase_bands(axes[1])
    axes[1].set_ylabel("body az\n[m/s²]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    # Bottom: combined norm with episode RMS reference line
    axes[2].plot(t_bacc, body_acc_norm, lw=0.9, c="C3", label="‖a‖ = √(ax²+az²)")
    axes[2].axhline(body_acc_rms, c="k", ls=":", lw=1.0,
                    label=f"episode RMS = {body_acc_rms:.3f}")
    _draw_phase_bands(axes[2])
    axes[2].set_ylabel("‖body acc‖\n[m/s²]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Body acceleration — {method_label}  "
                 f"(RMS={body_acc_rms:.3f} m/s²,  max={body_acc_max:.3f} m/s²)\n"
                 f"blue=src-hold  purple=transition  orange=tgt-hold  "
                 f"— spikes in purple = blend-induced trunk jolt")
    fig.tight_layout()
    fig.savefig(out_dir / "body_acc.png", dpi=120)
    plt.close(fig)

    # 2. Δα(t) per leg + residual joint contribution
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        axes[0].plot(t_axis, delta_a[:, i], label=f"Δα_{lab}", lw=1.0)
    _draw_phase_bands(axes[0], label_segments=True)
    axes[0].set_ylabel("Δα (per leg)"); axes[0].axhline(0, c="k", lw=0.5)
    axes[0].set_title("Residual correction  (blue=src-hold  purple=transition  orange=tgt-hold)")
    axes[0].legend(loc="upper right", fontsize=8); axes[0].grid(alpha=0.3)
    colors = ["C0", "C1", "C2", "C3"]
    joint_types = ["hip", "thigh", "calf"]
    for k, (leg, col) in enumerate(zip(["FL", "FR", "RL", "RR"], colors)):
        for j, jt in enumerate(joint_types):
            jidx = k + j * 4
            ls = ["-", "--", ":"][j]
            axes[1].plot(t_axis, residual_joint_scaled_a[:, jidx],
                         color=col, ls=ls, lw=0.9, label=f"{leg}_{jt}" if k == 0 else None)
    _draw_phase_bands(axes[1], label_segments=True)
    axes[1].axhline(0, c="k", lw=0.5)
    axes[1].set_ylabel("Δjoint [rad]  (residual only)")
    axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_alpha.png", dpi=120)
    plt.close(fig)

    # 2a-extra. α_baseline over time — definitive proof of transition timing
    # This is the single most important diagnostic: if the source hold is working,
    # α_baseline must be 0 for the first 2s of each segment, then ramp/jump to 1.
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(t_axis, alb_a[:, 0], lw=1.5, c="C3", label="α_baseline (FL_hip)")
    ax.axhline(0, c="k", lw=0.5, ls=":")
    ax.axhline(1, c="k", lw=0.5, ls=":")
    _draw_phase_bands(ax, label_segments=True)
    ax.set_ylabel("α_baseline")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"α baseline schedule — {method_label}  "
                 f"(blue=src-hold  purple=transition  orange=tgt-hold)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_baseline.png", dpi=120)
    plt.close(fig)

    # 2b. Source vs target vs blended joint commands
    # Layout: for EACH joint type, one plot with:
    #   Row 0: α_baseline(t) — unambiguous timing proof (must be 0 during source hold)
    #   Row 1-4: per-leg joint offset.  Each leg panel has:
    #     - background shading: light-blue = source-hold (α<0.05),
    #                           light-orange = target-hold (α>0.95)
    #     - blue: π_current action,  orange: π_target action,
    #       green: blended action,   purple dashed: (π_current - π_target) × asc
    #       (the purple line is the most honest indicator — zero means both
    #        policies agree, non-zero means blending is doing real work)
    asc = env_cfg.action_scale   # 0.25
    jt_fullname = {"hip": "hip (abduction)", "thigh": "thigh (flexion)", "calf": "calf (knee)"}
    for j, jt in enumerate(["hip", "thigh", "calf"]):
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2, 2, 2, 2]})
        # Top panel: α_baseline over time
        ax_a = axes[0]
        ax_a.plot(t_axis, alpha_base_1d, lw=1.8, c="C3", label="α_baseline")
        ax_a.fill_between(t_axis, 0, alpha_base_1d, alpha=0.15, color="C1")
        ax_a.set_ylim(-0.05, 1.15)
        ax_a.axhline(0, c="k", lw=0.5, ls=":")
        ax_a.axhline(1, c="k", lw=0.5, ls=":")
        _draw_phase_bands(ax_a, label_segments=True)
        ax_a.set_ylabel("α_baseline\n(0=src, 1=tgt)")
        ax_a.legend(loc="upper right", fontsize=7)
        ax_a.grid(alpha=0.3)
        # Per-leg panels
        for k, (leg, ax) in enumerate(zip(["FL", "FR", "RL", "RR"], axes[1:])):
            jidx = k + j * 4
            src  = ac_a[:, jidx] * asc
            tgt  = at_a[:, jidx] * asc
            bld  = bl_a[:, jidx] * asc
            diff = (src - tgt)            # policy disagreement; zero → no difference

            # Phase bands drawn by _draw_phase_bands below
            ax.plot(t_axis, src,  label="π_src",  lw=0.8, alpha=0.7, c="C0")
            ax.plot(t_axis, tgt,  label="π_tgt",  lw=0.8, alpha=0.7, c="C1")
            ax.plot(t_axis, bld,  label="blended", lw=1.2, c="C2")
            ax.plot(t_axis, diff, label="src−tgt", lw=0.7, ls="--", c="C5", alpha=0.8)
            ax.axhline(0, c="k", lw=0.3)
            _draw_phase_bands(ax, label_segments=(k == 0))
            ax.set_ylabel(f"{leg}_{jt} [rad]")
            ax.legend(loc="upper right", fontsize=6, ncol=2)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{jt_fullname[jt]} — joint offset [rad]  |  {method_label}\n"
                     f"src−tgt≠0 → policies differ (blending does real work)  "
                     f"blue=src-hold  purple=transition  orange=tgt-hold")
        fig.tight_layout()
        fig.savefig(out_dir / f"blend_{jt}.png", dpi=120)
        plt.close(fig)

    # 3. Body-state overview
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t_axis, vx_a, lw=1.0); axes[0].axhline(env_cfg.velocity_cmd_x,
                                                         c="g", ls=":", label="cmd")
    axes[0].set_ylabel("vx [m/s]"); axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)
    axes[1].plot(t_axis, h_a, lw=1.0); axes[1].axhline(env_cfg.target_height,
                                                        c="g", ls=":", label="target")
    axes[1].set_ylabel("height [m]"); axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
    axes[2].plot(t_axis, tilt_a, lw=1.0)
    axes[2].set_ylabel("|grav_xy|²"); axes[2].set_xlabel("time [s]"); axes[2].grid(alpha=0.3)
    for ax in axes:
        _draw_phase_bands(ax, label_segments=True)
    fig.suptitle("Body state  (blue=src-hold  purple=transition  orange=tgt-hold)")
    fig.tight_layout()
    fig.savefig(out_dir / "body_state.png", dpi=120)
    plt.close(fig)

    print(f"\n  Plots saved to: {out_dir}/")

# ---------------------------------------------------------------------------
# CSV dump
# ---------------------------------------------------------------------------

if args.save_csv:
    import csv
    csv_path = Path(args.save_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = (["t", "current", "target", "vx", "h", "tilt", "alpha_base",
                   "dFL", "dFR", "dRL", "dRR",
                   "power_W", "x_w"]
                  + [f"jp{i}" for i in range(12)]
                  + [f"jv{i}" for i in range(12)]
                  + [f"ja{i}" for i in range(12)]
                  + [f"tq{i}" for i in range(12)])
        w.writerow(header)
        for k in range(args.steps):
            row = [k * control_dt, gait_hist[k][0], gait_hist[k][1],
                   vx_a[k], h_a[k], tilt_a[k], alpha_base_1d[k],
                   *delta_a[k].tolist(), p_a[k], x_a[k],
                   *jp_a[k].tolist(), *jv_a[k].tolist(),
                   *ja_a[k].tolist(), *tq_a[k].tolist()]
            w.writerow(row)
    print(f"  CSV saved to:   {csv_path}")

env.close()
sim_app.close()
