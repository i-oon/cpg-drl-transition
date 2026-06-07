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
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import envs.b1_velocity_env_cfg  # noqa: F401

# ---------------------------------------------------------------------------
# Teleop keyboard state (only active with --teleop)
# ---------------------------------------------------------------------------

_STEP = 0.1   # velocity increment per key press
_teleop_cmd = {'vx': 0.0, 'vy': 0.0, 'wz': 0.0, 'exit': False}
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
                elif ch == 'q': _teleop_cmd['wz'] = min(_teleop_cmd['wz'] + _STEP,  1.0)
                elif ch == 'e': _teleop_cmd['wz'] = max(_teleop_cmd['wz'] - _STEP, -1.0)
                elif ch == 'r': _teleop_cmd.update({'vx': 0.4, 'vy': 0.0, 'wz': 0.0})
            except AttributeError:
                if key == _pynput_kb.Key.space:
                    _teleop_cmd.update({'vx': 0.0, 'vy': 0.0, 'wz': 0.0})
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
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

if args.video:
    video_dir = Path(args.video)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_length = args.video_length if args.video_length is not None else args.steps
    env = gym.wrappers.RecordVideo(
        env, video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=video_length, disable_logger=True, name_prefix="play",
    )

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
      f"{'err_v':>5} | {'h':>5} {'tilt':>6} | {'gait':>4} | {'R_tot':>8}")
print("  " + "-" * 88)

for step in range(args.steps):
    # --- Teleop: inject keyboard command into command manager ---
    if _teleop_active:
        if _teleop_cmd['exit']:
            print("\n  [teleop] ESC pressed — stopping.")
            break
        cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
        cmd_override = torch.tensor(
            [[_teleop_cmd['vx'], _teleop_cmd['vy'], _teleop_cmd['wz']]],
            device=env.device,
        ).repeat(env.num_envs, 1)
        cmd_term.vel_command_b[:] = cmd_override

    with torch.no_grad():
        actions = policy(obs)
    obs, reward, dones, extras = env.step(actions)
    total_reward += reward
    reset_count += int(dones.sum().item())

    d = robot.data
    vx = d.root_lin_vel_b[0, 0].item()
    vy = d.root_lin_vel_b[0, 1].item()
    vz = d.root_lin_vel_b[0, 2].item()
    h  = d.root_pos_w[0, 2].item()
    tilt = torch.sum(torch.square(d.projected_gravity_b[0, :2])).item()

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
              f"{err_v:5.3f} | {h:5.3f} {tilt:6.4f} | {gait_str:>4} | "
              f"{total_reward.mean().item():8.2f}")

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
