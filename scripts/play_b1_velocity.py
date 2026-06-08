"""
Play a PPO-trained B1 velocity-tracking policy.

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition
    python scripts/play_b1_velocity.py --checkpoint logs/ppo_b1/<run>/model_final.pt
    python scripts/play_b1_velocity.py --checkpoint <path> --num_envs 4 --steps 1000
    python scripts/play_b1_velocity.py --checkpoint <path> --teleop   # keyboard control

Teleop keys (--teleop):
    W/S   — forward / backward      (vx ± 0.1 m/s)
    A/D   — strafe left / right     (vy ± 0.1 m/s)
    Q/E   — turn left / right       (ωz ± 0.1 rad/s)
    SPACE — stop (zero all)
    R     — reset to random command
    ESC   — exit
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play B1 velocity-tracking policy")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-B1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--video_length", type=int, default=None)
parser.add_argument("--teleop", action="store_true",
                    help="Override commands with keyboard input (requires pynput).")
parser.add_argument("--demo", action="store_true",
                    help="Run scripted demo sequence covering all motion capabilities.")
parser.add_argument("--follow_cam", action="store_true",
                    help="Viewport camera follows the robot (GUI mode only, not headless).")
parser.add_argument("--cam_offset", type=float, nargs=3, default=[3.0, 0.0, 0.5],
                    metavar=("X", "Y", "Z"),
                    help="Camera offset from robot base in world frame (default: 2.5 0.0 1.2).")
args = parser.parse_args()

# Compute demo total steps early so video_length captures the full sequence
if args.demo:
    _DEMO_SEQ_EARLY = [
        (150,  0.0,  0.0,  0.0),
        (200,  0.3,  0.0,  0.0),
        (200,  0.6,  0.0,  0.0),
        (200,  0.8,  0.0,  0.0),
        (250, -0.4,  0.0,  0.0),
        (250,  0.0,  0.4,  0.0),
        (250,  0.0, -0.4,  0.0),
        (200,  0.5,  0.3,  0.0),
        (200,  0.5, -0.3,  0.0),
        (250,  0.0,  0.0,  1.0),
        (250,  0.0,  0.0, -1.0),
        (150,  0.0,  0.0,  0.0),
    ]
    args.steps = sum(d for d, *_ in _DEMO_SEQ_EARLY)

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import envs.b1_velocity_env_cfg  # noqa: F401

# ---------------------------------------------------------------------------
# Teleop keyboard state (only active with --teleop)
# ---------------------------------------------------------------------------

_STEP    = 0.1   # linear velocity increment per key press
_WZ_STEP = 0.4   # wz increment — needs to clear yaw_stability dead zone (~0.3 rad/s)
_teleop_cmd = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0, 'exit': False, 'reset_heading': False}
_teleop_active = False

if args.teleop:
    try:
        from pynput import keyboard as _pynput_kb

        def _on_press(key):
            try:
                ch = key.char
                if   ch == 'w': _teleop_cmd['vx'] = min(_teleop_cmd['vx'] + _STEP,  0.8)
                elif ch == 's': _teleop_cmd['vx'] = max(_teleop_cmd['vx'] - _STEP, -0.5)
                elif ch == 'a': _teleop_cmd['vy'] = min(_teleop_cmd['vy'] + _STEP,  0.5)
                elif ch == 'd': _teleop_cmd['vy'] = max(_teleop_cmd['vy'] - _STEP, -0.5)
                elif ch == 'q': _teleop_cmd['wz'] = min(_teleop_cmd['wz'] + _WZ_STEP,  1.0)
                elif ch == 'e': _teleop_cmd['wz'] = max(_teleop_cmd['wz'] - _WZ_STEP, -1.0)
                elif ch == 'r': _teleop_cmd.update({'vx': 0.4, 'vy': 0.0, 'wz': 0.0})
            except AttributeError:
                if key == _pynput_kb.Key.space:
                    _teleop_cmd.update({'vx': 0.0, 'vy': 0.0, 'wz': 0.0,
                                        'reset_heading': True})
                elif key == _pynput_kb.Key.esc:
                    _teleop_cmd['exit'] = True

        _listener = _pynput_kb.Listener(on_press=_on_press)
        _listener.start()
        _teleop_active = True
        print("\n  [teleop] keyboard active — W/S:fwd  A/D:strafe  Q/E:turn  SPACE:stop  ESC:quit\n")
    except ImportError:
        print("\n  [teleop] pynput not installed. Run: pip install pynput\n")
        args.teleop = False
from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

agent_cfg = B1FlatPPORunnerCfg()
env_cfg = parse_env_cfg(args.task, device=agent_cfg.device, num_envs=args.num_envs)
# Extend episode length to prevent mid-demo resets during scripted playback
if args.demo:
    env_cfg.episode_length_s = 200.0  # 200s >> longest demo sequence
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

if args.video:
    video_dir = Path(args.video)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_length = args.video_length if args.video_length is not None else args.steps
    print(f"  Video length  : {video_length} steps ({video_length * 0.02:.1f}s at 50fps)")
    _record_env = gym.wrappers.RecordVideo(
        env, video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=video_length, disable_logger=True, name_prefix="play",
    )
    _record_env.frames_per_sec = 50  # directly override — metadata pickup is unreliable
    print(f"  RecordVideo fps: {_record_env.frames_per_sec}")
    env = _record_env

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=agent_cfg.device)

print(f"\n  Task          : {args.task}")
print(f"  Checkpoint    : {args.checkpoint}")
print(f"  Envs          : {env.num_envs}")
print(f"  Steps         : {args.steps} ({args.steps * 0.02:.1f}s)\n")

# ---------------------------------------------------------------------------
# Sensor / joint index setup
# ---------------------------------------------------------------------------

robot = env.unwrapped.scene["robot"]
contact_sensor = env.unwrapped.scene["contact_forces"]

_all_body_names = robot.body_names
foot_body_ids = [_all_body_names.index(n) for n in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]

_joint_names = robot.joint_names
hip_ids   = [_joint_names.index(n) for n in ["FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint"]]
thigh_ids = [_joint_names.index(n) for n in ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]]
calf_ids  = [_joint_names.index(n) for n in ["FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint"]]

foot_ids_sensor, foot_names_sensor = contact_sensor.find_bodies(".*_foot$")
desired = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
perm = [foot_names_sensor.index(n) for n in desired if n in foot_names_sensor]
foot_ids_sensor = [foot_ids_sensor[i] for i in perm]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

obs, _ = env.get_observations()
total_reward = torch.zeros(env.num_envs, device=env.device)

# Follow-cam smoothed state (exponential moving average)
_cam_smooth_pos = None
_cam_smooth_yaw = None
_CAM_ALPHA = 0.92  # higher = smoother/slower tracking, filters out body bounce

# Demo scripted sequence — (duration_steps, vx, vy, wz, label)
# 50 Hz → 100 steps = 2s, 200 steps = 4s
_DEMO_SEQ = [
    (150,  0.0,  0.0,  0.0, "stand"),
    (200,  0.3,  0.0,  0.0, "walk fwd slow"),
    (200,  0.6,  0.0,  0.0, "walk fwd med"),
    (200,  0.8,  0.0,  0.0, "walk fwd fast"),
    (250, -0.4,  0.0,  0.0, "walk bwd"),
    (250,  0.0,  0.4,  0.0, "strafe left"),
    (250,  0.0, -0.4,  0.0, "strafe right"),
    (200,  0.5,  0.3,  0.0, "diagonal fwd-left"),
    (200,  0.5, -0.3,  0.0, "diagonal fwd-right"),
    (250,  0.0,  0.0,  1.0, "turn left"),
    (250,  0.0,  0.0, -1.0, "turn right"),
    (150,  0.0,  0.0,  0.0, "stand"),
] if args.demo else []

_demo_step    = 0   # global step counter within current segment
_demo_idx     = 0   # current segment index
_demo_cmd     = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0}   # target command
_demo_current = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0}   # actual injected command (ramped)
_DEMO_RAMP    = 0.016  # max change per step (0.016 m/s/step @ 50Hz = 0.8 m/s²)

if args.demo:
    total_demo_steps = sum(s for s, *_ in _DEMO_SEQ)
    args.steps = total_demo_steps
    print(f"\n  [demo] Scripted sequence: {len(_DEMO_SEQ)} segments, {total_demo_steps} steps "
          f"({total_demo_steps * 0.02:.1f}s)\n")

# per-step buffers (env 0)
vx_hist, vy_hist, h_hist, tilt_hist = [], [], [], []
cmd_vx_hist, cmd_vy_hist, cmd_wz_hist = [], [], []
contact_history, foot_z_history = [], []
hip_q_hist, thigh_q_hist, calf_q_hist = [], [], []
air_time_hist = []    # (steps, 4) — current air time per foot
action_hist = []      # (steps, 12) — raw actions env 0
reward_terms = {}     # term_name → list of per-step values
reset_count = 0

print(f"  {'step':>5} | {'vx':>6} {'vy':>6} | {'cvx':>6} {'cvy':>6} {'cwz':>5} | "
      f"{'wz_act':>9} {'yaw':>9} | {'h':>5} | {'gait':>4} | {'R_tot':>8}")
print("  " + "-" * 100)

_HEAD_K = 0.5  # heading_control_stiffness from parent env config (rel_heading_envs=1.0)

# Set heading_target to robot's spawn heading so heading controller has a sane reference.
# For wz=0 commands, heading control will naturally provide small corrective cwz
# (as it did during training). For wz≠0, _inject_cmd advances heading_target.
_init_cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
_init_cmd_term.heading_target[:] = robot.data.heading_w.clone()

_record_start = time.perf_counter()
for step in range(args.steps):
    def _inject_cmd(vx, vy, wz):
        cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1] = vy
        # Always write slot 2 directly so raw-wz policies see the correct command.
        # heading-cmd policies: env.step() overwrites this with K*heading_error anyway.
        # raw-wz policies: env.step() does NOT overwrite slot 2, so this is the only
        # way to inject wz. Also overrides periodic command resampling in both modes.
        cmd_term.vel_command_b[:, 2] = wz
        # heading_target injection for heading-cmd policies (ignored in raw-wz mode):
        current_heading = robot.data.heading_w  # [num_envs], world-frame yaw
        if abs(wz) > 1e-4:
            cmd_term.heading_target[:] = current_heading + wz / _HEAD_K
        else:
            cmd_term.heading_target[:] = current_heading

    # --- Demo: advance scripted sequence ---
    if args.demo and _demo_idx < len(_DEMO_SEQ):
        dur, vx, vy, wz, label = _DEMO_SEQ[_demo_idx]
        _demo_cmd['vx'], _demo_cmd['vy'], _demo_cmd['wz'] = vx, vy, wz
        _demo_step += 1
        if _demo_step >= dur:
            _demo_idx += 1
            _demo_step  = 0
            if _demo_idx < len(_DEMO_SEQ):
                print(f"  [demo] → {_DEMO_SEQ[_demo_idx][4]}")
            else:
                print("  [demo] → done")

    # --- Teleop: read keyboard state ---
    if _teleop_active and _teleop_cmd['exit']:
        print("\n  [teleop] ESC pressed — stopping.")
        break
    if _teleop_active and _teleop_cmd['reset_heading']:
        # SPACE was pressed — snap heading_target to current heading so robot stops turning
        _rt_term = env.unwrapped.command_manager._terms["base_velocity"]
        _rt_term.heading_target[:] = robot.data.heading_w.clone()
        _teleop_cmd['reset_heading'] = False

    # Inject before step (policy sees this command in obs)
    if args.demo:
        # Ramp current command toward target to avoid step changes
        for k in ('vx', 'vy', 'wz'):
            diff = _demo_cmd[k] - _demo_current[k]
            ramp = min(abs(diff), _DEMO_RAMP) * (1 if diff >= 0 else -1)
            _demo_current[k] += ramp
        _inject_cmd(_demo_current['vx'], _demo_current['vy'], _demo_current['wz'])
    elif _teleop_active:
        _inject_cmd(_teleop_cmd['vx'], _teleop_cmd['vy'], _teleop_cmd['wz'])

    _t0 = time.perf_counter()
    with torch.no_grad():
        actions = policy(obs)
    obs, reward, dones, extras = env.step(actions)
    total_reward += reward
    reset_count += int(dones.sum().item())
    # Rate-limit to real-time when recording video so playback matches sim view
    if args.video:
        _remaining = 0.02 - (time.perf_counter() - _t0)
        if _remaining > 0:
            time.sleep(_remaining)

    # Re-inject after step to override any command resampling triggered by env reset
    if args.demo:
        _inject_cmd(_demo_current['vx'], _demo_current['vy'], _demo_current['wz'])
    elif _teleop_active:
        _inject_cmd(_teleop_cmd['vx'], _teleop_cmd['vy'], _teleop_cmd['wz'])

    d = robot.data

    # Follow camera: smooth 3rd-person, yaw-tracking, no body-bounce
    if args.follow_cam:
        pos_w  = d.root_pos_w[0].cpu().numpy()
        quat_w = d.root_quat_w[0].cpu().numpy()  # [x, y, z, w]

        # Extract robot yaw
        x, y, z, w = quat_w
        raw_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Initialise smoothed state on first step
        if _cam_smooth_pos is None:
            _cam_smooth_pos = pos_w.copy()
            _cam_smooth_yaw = raw_yaw

        # Exponential moving average — filters gait oscillation
        _cam_smooth_pos = _CAM_ALPHA * _cam_smooth_pos + (1 - _CAM_ALPHA) * pos_w
        # Wrap-safe yaw smoothing
        dyaw = (raw_yaw - _cam_smooth_yaw + np.pi) % (2 * np.pi) - np.pi
        _cam_smooth_yaw = _cam_smooth_yaw + (1 - _CAM_ALPHA) * dyaw

        cos_y, sin_y = np.cos(_cam_smooth_yaw), np.sin(_cam_smooth_yaw)
        ox, oy, oz = -args.cam_offset[0], args.cam_offset[1], args.cam_offset[2]
        cam_offset_world = np.array([cos_y * ox - sin_y * oy,
                                     sin_y * ox + cos_y * oy,
                                     oz])

        # Camera at smoothed position + offset; look at robot face (ahead + chest height)
        eye    = _cam_smooth_pos + cam_offset_world
        target = _cam_smooth_pos + np.array([cos_y * 0.5, sin_y * 0.5, 0.1])
        env.unwrapped.sim.set_camera_view(eye=tuple(eye.tolist()), target=tuple(target.tolist()))

    vx = d.root_lin_vel_b[0, 0].item()
    vy = d.root_lin_vel_b[0, 1].item()
    vz = d.root_lin_vel_b[0, 2].item()
    h  = d.root_pos_w[0, 2].item()
    tilt = torch.sum(torch.square(d.projected_gravity_b[0, :2])).item()

    # Actual yaw rate and heading
    # Isaac Lab quaternion convention: [w, x, y, z]
    wz_act = d.root_ang_vel_b[0, 2].item()
    quat_w = d.root_quat_w[0].cpu().numpy()
    _w, _x, _y, _z = quat_w
    yaw_deg = np.degrees(np.arctan2(2.0*(_w*_z + _x*_y), 1.0 - 2.0*(_y*_y + _z*_z)))

    # Commanded velocity (env 0)
    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0].cpu()
    cvx, cvy, cwz = cmd[0].item(), cmd[1].item(), cmd[2].item()
    err_v = ((vx - cvx)**2 + (vy - cvy)**2) ** 0.5

    # Contact state
    contact_time = contact_sensor.data.current_contact_time[0, foot_ids_sensor]
    in_contact = (contact_time > 0.0).cpu().numpy()

    # Current air time per foot
    air_time = contact_sensor.data.current_air_time[0, foot_ids_sensor].cpu().numpy()

    # Foot world z
    foot_z = d.body_pos_w[0, foot_body_ids, 2].cpu().numpy()

    # Joint positions
    jp = d.joint_pos[0].cpu().numpy()

    # Actions (env 0, first 12 dims)
    act = actions[0, :12].cpu().numpy()

    # Individual reward terms from extras["log"]
    if "Episode_Reward" in str(extras.get("log", {})):
        for k, v in extras["log"].items():
            if "Episode_Reward" in k:
                term = k.replace("Episode_Reward/", "")
                reward_terms.setdefault(term, []).append(float(v) if hasattr(v, "__float__") else v)

    vx_hist.append(vx);   vy_hist.append(vy);   h_hist.append(h);   tilt_hist.append(tilt)
    cmd_vx_hist.append(cvx); cmd_vy_hist.append(cvy); cmd_wz_hist.append(cwz)
    contact_history.append(in_contact)
    foot_z_history.append(foot_z)
    air_time_hist.append(air_time)
    hip_q_hist.append(jp[hip_ids])
    thigh_q_hist.append(jp[thigh_ids])
    calf_q_hist.append(jp[calf_ids])
    action_hist.append(act)

    if (step + 1) % 50 == 0:
        gait_str = "".join("█" if c else "·" for c in in_contact)
        print(f"  {step+1:5d} | {vx:+6.3f} {vy:+6.3f} | {cvx:+6.3f} {cvy:+6.3f} {cwz:+5.2f} | "
              f"wz_act:{wz_act:+5.2f} yaw:{yaw_deg:+7.1f}° | "
              f"{h:5.3f} | {gait_str:>4} | {total_reward.mean().item():8.2f}")

# ---------------------------------------------------------------------------
# Video speed correction (sim often runs slower than 50 Hz real-time with rendering)
# ---------------------------------------------------------------------------

if args.video:
    import glob, subprocess
    _elapsed = time.perf_counter() - _record_start
    _actual_fps = args.steps / _elapsed
    _slowdown = 50.0 / _actual_fps  # factor to stretch playback time
    print(f"\n  Recording: {_elapsed:.1f}s real, {args.steps * 0.02:.1f}s sim  "
          f"(sim ran at {_actual_fps:.1f} Hz real-time)")
    if abs(_slowdown - 1.0) > 0.05:
        _vids = sorted(glob.glob(str(Path(args.video) / "play-*.mp4")))
        if _vids:
            _src = _vids[-1]
            _dst = _src.replace(".mp4", "_realtime.mp4")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", _src,
                 "-vf", f"setpts={_slowdown:.6f}*PTS",
                 "-r", f"{_actual_fps:.3f}", _dst],
                capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  Speed-corrected → {_dst}  (factor {_slowdown:.2f}x)")
            else:
                print(f"  ffmpeg failed — raw video at {args.video}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

sep = "=" * 65
print(f"\n{sep}")
print(f"  PLAYBACK SUMMARY — B1 velocity tracking")
print(sep)

vx_a  = np.array(vx_hist)
h_a   = np.array(h_hist)
cvx_a = np.array(cmd_vx_hist)
cvy_a = np.array(cmd_vy_hist)

print(f"  Steps     : {args.steps}   Resets: {reset_count}")
print(f"  Reward    : {total_reward.mean().item():.2f}")

print(f"\n  Velocity tracking (env 0):")
print(f"    vx : mean={vx_a.mean():+.3f}  std={vx_a.std():.3f}  "
      f"min={vx_a.min():+.3f}  max={vx_a.max():+.3f}")
print(f"    cmd_vx: mean={cvx_a.mean():+.3f}  cmd_vy: mean={cvy_a.mean():+.3f}")
err_v_a = np.sqrt((vx_a - cvx_a)**2 + (np.array(vy_hist) - cvy_a)**2)
print(f"    tracking err (|v - cmd|): mean={err_v_a.mean():.3f}  max={err_v_a.max():.3f}")

print(f"\n  Height: mean={h_a.mean():.3f}  std={h_a.std():.3f}  target=0.50")
print(f"  Tilt  : mean={np.array(tilt_hist).mean():.4f}  max={np.array(tilt_hist).max():.4f}")

# Actions
if action_hist:
    act_a = np.array(action_hist)  # (steps, 12)
    print(f"\n  Actions (env 0, all joints):")
    print(f"    mean |action|: {np.abs(act_a).mean():.4f}")
    print(f"    max  |action|: {np.abs(act_a).max():.4f}")
    print(f"    std  action  : {act_a.std():.4f}")

# Gait analysis
if contact_history:
    contact_arr = np.array(contact_history)  # (steps, 4)
    duty = contact_arr.mean(axis=0) * 100

    print(f"\n  Duty factor (% on ground):")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        print(f"    {lab}: {duty[i]:5.1f}%")

    print(f"\n  Footfall pattern (first 5 stance starts, s):")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        starts = np.where(np.diff(contact_arr[:, i].astype(int)) == 1)[0] + 1
        starts_s = starts * 0.02
        if len(starts_s) > 1:
            cycle = np.diff(starts_s).mean()
            freq = 1.0 / cycle if cycle > 0 else 0
            print(f"    {lab}: " + ", ".join(f"{t:.2f}" for t in starts_s[:5])
                  + f"  (cycle={cycle:.2f}s  {freq:.1f}Hz)")
        else:
            print(f"    {lab}: " + ", ".join(f"{t:.2f}" for t in starts_s[:5]))

    # Air time per foot
    if air_time_hist:
        air_arr = np.array(air_time_hist)  # (steps, 4)
        print(f"\n  Air time per foot (s) — mean of non-zero swing samples:")
        for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
            swing = air_arr[~contact_arr[:, i].astype(bool), i]
            if len(swing) > 0:
                print(f"    {lab}: mean={swing.mean():.3f}s  max={swing.max():.3f}s  "
                      f"(threshold=0.20s, target>0.20s)")

    # Foot height
    if foot_z_history:
        z_arr = np.array(foot_z_history)
        print(f"\n  Foot apex during swing (cm above floor):")
        for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
            swing_mask = ~contact_arr[:, i].astype(bool)
            if contact_arr[:, i].any() and swing_mask.any():
                floor_z = z_arr[contact_arr[:, i].astype(bool), i].mean()
                apex = (z_arr[swing_mask, i].max() - floor_z) * 100
                mean_lift = (z_arr[swing_mask, i].mean() - floor_z) * 100
                print(f"    {lab}: apex={apex:5.1f}cm  mean_lift={mean_lift:4.1f}cm")

# Joint ranges
if hip_q_hist:
    hip_q = np.array(hip_q_hist)
    thigh_q = np.array(thigh_q_hist)
    calf_q = np.array(calf_q_hist)
    print(f"\n  Joint range peak-to-peak (deg):")
    print(f"    {'leg':<5} {'hip':>7} {'thigh':>7} {'calf':>7}")
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        h_ = np.degrees(hip_q[:, i].max()   - hip_q[:, i].min())
        t_ = np.degrees(thigh_q[:, i].max() - thigh_q[:, i].min())
        c_ = np.degrees(calf_q[:, i].max()  - calf_q[:, i].min())
        print(f"    {lab:<5} {h_:7.2f} {t_:7.2f} {c_:7.2f}")

    # Default pose deviation
    default_jp = robot.data.default_joint_pos[0].cpu().numpy()
    last_jp = np.concatenate([hip_q_hist[-1], thigh_q_hist[-1], calf_q_hist[-1]])
    print(f"\n  Default joint pos vs last step (deg) — [FL FR RL RR] per type:")
    print(f"    hip   default: " + "  ".join(f"{np.degrees(default_jp[j]):+6.2f}" for j in hip_ids))
    print(f"    hip   actual : " + "  ".join(f"{np.degrees(hip_q_hist[-1][i]):+6.2f}" for i in range(4)))
    print(f"    thigh default: " + "  ".join(f"{np.degrees(default_jp[j]):+6.2f}" for j in thigh_ids))
    print(f"    thigh actual : " + "  ".join(f"{np.degrees(thigh_q_hist[-1][i]):+6.2f}" for i in range(4)))

# Gait diagram PNG
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))

    # Gait diagram
    ax = axes[0]
    for i, lab in enumerate(["FL", "FR", "RL", "RR"]):
        stance = contact_arr[:, i]
        t = np.arange(len(stance)) * 0.02
        ax.fill_between(t, i + 0.1, i + 0.9, where=stance, color="C0", alpha=0.8, step="post")
    ax.set_yticks([0.5, 1.5, 2.5, 3.5]); ax.set_yticklabels(["FL", "FR", "RL", "RR"])
    ax.set_ylabel("Foot"); ax.set_title(f"Gait — {Path(args.checkpoint).name}")
    ax.invert_yaxis(); ax.set_xlim(0, args.steps * 0.02)

    # Velocity tracking
    ax = axes[1]
    t = np.arange(args.steps) * 0.02
    ax.plot(t, vx_hist, label="vx actual", color="C0")
    ax.plot(t, cmd_vx_hist, label="cmd_vx", color="C0", ls="--", alpha=0.5)
    ax.plot(t, vy_hist, label="vy actual", color="C1")
    ax.plot(t, cmd_vy_hist, label="cmd_vy", color="C1", ls="--", alpha=0.5)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("vel (m/s)"); ax.legend(fontsize=7, ncol=4); ax.set_xlim(0, args.steps * 0.02)

    # Height
    ax = axes[2]
    ax.plot(t, h_hist, color="C2")
    ax.axhline(0.50, color="r", ls="--", lw=0.8, label="target 0.50m")
    ax.set_ylabel("height (m)"); ax.set_xlabel("time (s)")
    ax.legend(fontsize=7); ax.set_xlim(0, args.steps * 0.02)

    out_png = "logs/gait_diagram_ppo_b1.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"\n  Saved → {out_png}")
except ImportError:
    pass

env.close()
sim_app.close()
