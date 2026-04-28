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
vx_hist, h_hist, tilt_hist = [], [], []
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

for step in range(args.steps):
    # Switch gait pair every switch_interval_s
    if step > 0 and step % switch_steps == 0:
        seq_idx = (seq_idx + 1) % len(gait_seq)
        current_name = target_name
        target_name = gait_seq[(seq_idx + 1) % len(gait_seq)]
        _set_gait_pair(env, current_name, target_name)
        # Reset env episode_length_buf so α_baseline ramps from 0 again
        env.unwrapped.episode_length_buf[:] = 0
        switch_step_marks.append(step)

    # Discrete baseline: force α=1 immediately at/after each switch
    if args.baseline == "discrete" and step in switch_step_marks:
        _past_ramp = int(
            (env.unwrapped.cfg.transition_start_max_s
             + env.unwrapped.cfg.transition_duration_s + 0.5)
            / control_dt
        )
        env.unwrapped.episode_length_buf[:] = _past_ramp

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
    # ramp_progress for this env: used to mark ramp-start and ramp-end
    t_ep = eu.episode_length_buf[e0_idx].float().item() * control_dt
    ramp_prog = (t_ep - eu._transition_start_s[e0_idx].item()) / eu.cfg.transition_duration_s
    ramp_progress_hist.append(ramp_prog)

    vx_hist.append(vx); h_hist.append(h); tilt_hist.append(tilt)
    delta_hist.append(delta)
    gait_hist.append((current_name, target_name))
    joint_pos_hist.append(jp)
    joint_vel_hist.append(jv)
    joint_acc_hist.append(ja)
    torque_hist.append(tq)
    power_inst_hist.append(p_inst)
    x_pos_hist.append(x_w)

    if (step + 1) % 50 == 0:
        gait_str = f"{current_name[:5]}→{target_name[:5]}"
        print(f"  {step+1:5d} | {gait_str:>14} | "
              f"{vx:+6.3f} {vz:+6.3f} | {h:5.3f} {tilt:6.4f} | "
              f"{delta[0]:+6.3f} {delta[1]:+6.3f} {delta[2]:+6.3f} {delta[3]:+6.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 70
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — Phase 2 transition learning")
print(sep)

vx_a = np.array(vx_hist); h_a = np.array(h_hist); tilt_a = np.array(tilt_hist)
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

# Cost of Transport over the run
distance = max(x_a[-1] - x_a[0], 1e-6)              # m
energy   = float(np.sum(p_a) * control_dt)          # J
cot      = energy / (args.robot_mass_kg * 9.81 * distance)

# Joint accel: RMS across all 12 joints, mean over time (lower = smoother)
joint_acc_rms = float(np.sqrt(np.mean(ja_a ** 2)))

print(f"  Steps         : {args.steps}")
print(f"  Total reward  : {total_reward.mean().item():.2f}")
print(f"\n  vx     : mean={vx_a.mean():+.3f}  std={vx_a.std():.3f}  "
      f"min={vx_a.min():+.3f}  max={vx_a.max():+.3f}")
print(f"  Height : mean={h_a.mean():.3f}  std={h_a.std():.3f}  target=0.42")
print(f"  Tilt   : mean={tilt_a.mean():.4f}  max={tilt_a.max():.4f}")
print(f"\n  Distance       : {distance:.2f} m  (over {args.steps * control_dt:.1f}s)")
print(f"  Energy used    : {energy:.1f} J")
print(f"  Cost of Transp : {cot:.3f}      (lower = better; biological quadrupeds ~0.2)")
print(f"  Joint-acc RMS  : {joint_acc_rms:.2f} rad/s²  (lower = smoother transitions)")
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

    def _draw_phase_lines(ax, label_segments=False):
        """Vertical lines: red=pair-swap, green=ramp-start, orange=ramp-end.
        label_segments=True adds gait-pair text at top of each segment."""
        for s in switch_step_marks:
            ax.axvline(s * control_dt, ls="--", c="r",  alpha=0.7, lw=1.2, label="_nolegend_")
        for t in ramp_start_times:
            ax.axvline(t, ls="--", c="g",  alpha=0.8, lw=1.0, label="_nolegend_")
        for t in ramp_end_times:
            ax.axvline(t, ls="--", c="orange", alpha=0.8, lw=1.0, label="_nolegend_")
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
    _draw_phase_lines(ax_gait, label_segments=True)
    ax_gait.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax_gait.set_yticklabels(["FL", "FR", "RL", "RR"])
    ax_gait.invert_yaxis()
    ax_gait.set_ylabel("foot")
    ax_gait.set_title(f"Gait diagram — {method_label}  "
                      f"(red=pair-swap  green=ramp-start  orange=ramp-end)")
    ax_gait.grid(alpha=0.2)
    # Bottom panel: vx trace
    ax_vx = axes[1]
    ax_vx.plot(t_axis, vx_a, lw=0.8, c="C1")
    ax_vx.axhline(env_cfg.velocity_cmd_x, ls=":", c="g", lw=1.0, label="cmd vx")
    _draw_phase_lines(ax_vx, label_segments=True)
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
        _draw_phase_lines(ax, label_segments=True)
        ax.set_ylabel(f"{leg} [rad]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Joint positions per leg  (red=swap  green=ramp-start  orange=ramp-end)")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_positions.png", dpi=120)
    plt.close(fig)

    # 2. Δα(t) per leg + residual joint contribution
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        axes[0].plot(t_axis, delta_a[:, i], label=f"Δα_{lab}", lw=1.0)
    _draw_phase_lines(axes[0], label_segments=True)
    axes[0].set_ylabel("Δα (per leg)"); axes[0].axhline(0, c="k", lw=0.5)
    axes[0].set_title("Residual correction  (red=swap  green=ramp-start  orange=ramp-end)")
    axes[0].legend(loc="upper right", fontsize=8); axes[0].grid(alpha=0.3)
    colors = ["C0", "C1", "C2", "C3"]
    joint_types = ["hip", "thigh", "calf"]
    for k, (leg, col) in enumerate(zip(["FL", "FR", "RL", "RR"], colors)):
        for j, jt in enumerate(joint_types):
            jidx = k + j * 4
            ls = ["-", "--", ":"][j]
            axes[1].plot(t_axis, residual_joint_scaled_a[:, jidx],
                         color=col, ls=ls, lw=0.9, label=f"{leg}_{jt}" if k == 0 else None)
    _draw_phase_lines(axes[1], label_segments=True)
    axes[1].axhline(0, c="k", lw=0.5)
    axes[1].set_ylabel("Δjoint [rad]  (residual only)")
    axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_alpha.png", dpi=120)
    plt.close(fig)

    # 2b. Source vs target vs blended joint commands
    # Scale by action_scale so values are in radians (joint offset from default pose)
    asc = env_cfg.action_scale   # 0.25
    jt_fullname = {"hip": "hip (abduction)", "thigh": "thigh (flexion)", "calf": "calf (knee)"}
    for j, jt in enumerate(["hip", "thigh", "calf"]):
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for k, (leg, ax) in enumerate(zip(["FL", "FR", "RL", "RR"], axes)):
            jidx = k + j * 4
            ax.plot(t_axis, ac_a[:, jidx] * asc, label="π_current", lw=0.8, alpha=0.7, c="C0")
            ax.plot(t_axis, at_a[:, jidx] * asc, label="π_target",  lw=0.8, alpha=0.7, c="C1")
            ax.plot(t_axis, bl_a[:, jidx] * asc, label="blended",   lw=1.2, c="C2")
            _draw_phase_lines(ax, label_segments=True)
            ax.set_ylabel(f"{leg}_{jt}\noffset [rad]")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{jt_fullname[jt]} — joint offset from default pose [rad]  "
                     f"(red=swap  green=ramp-start  orange=ramp-end)")
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
        _draw_phase_lines(ax, label_segments=True)
    fig.suptitle("Body state  (red=swap  green=ramp-start  orange=ramp-end)")
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
        header = (["t", "current", "target", "vx", "h", "tilt",
                   "dFL", "dFR", "dRL", "dRR",
                   "power_W", "x_w"]
                  + [f"jp{i}" for i in range(12)]
                  + [f"jv{i}" for i in range(12)]
                  + [f"ja{i}" for i in range(12)]
                  + [f"tq{i}" for i in range(12)])
        w.writerow(header)
        for k in range(args.steps):
            row = [k * control_dt, gait_hist[k][0], gait_hist[k][1],
                   vx_a[k], h_a[k], tilt_a[k],
                   *delta_a[k].tolist(), p_a[k], x_a[k],
                   *jp_a[k].tolist(), *jv_a[k].tolist(),
                   *ja_a[k].tolist(), *tq_a[k].tolist()]
            w.writerow(row)
    print(f"  CSV saved to:   {csv_path}")

env.close()
sim_app.close()
