"""
Structured evaluation for DEPLOY_LOG.md.

Runs each command condition (C1–C13) for N parallel episodes, collects all
metrics defined in DEPLOY_LOG.md, and prints filled-in result blocks ready
to paste into the log.

Usage:
    conda activate env_isaaclab
    cd ~/cpg-drl-transition

    # Full S1 evaluation (20 episodes × 1000 steps per condition):
    python scripts/eval_b1_velocity.py \\
        --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \\
        --stage S1

    # Quick smoke-test (5 episodes, 200 steps, 3 conditions):
    python scripts/eval_b1_velocity.py \\
        --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \\
        --stage S1 --num_episodes 5 --steps 200 --conditions C1 C3 C10 C11

    # Append result block directly to DEPLOY_LOG.md:
    python scripts/eval_b1_velocity.py \\
        --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \\
        --stage S1 --append

Columns in the per-condition table printed during the run:
    cond | fall% | e_vx  e_vy  e_wz | yaw_drift | h_mean±std | duty FL FR RL RR
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing — must happen before AppLauncher
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Structured B1 policy evaluation for DEPLOY_LOG.md")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task",        type=str, default="Isaac-Velocity-Flat-Unitree-B1-RawWz-v0")
parser.add_argument("--checkpoint",  type=str, required=True)
parser.add_argument("--stage",       type=str, default="S1",
                    help="Stage label written into the output block (S1–S6).")
parser.add_argument("--simulator",   type=str, default="Isaac Sim 4.5")
parser.add_argument("--actuator",    type=str, default="Ideal PD (kp=300, kd=5)")
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Number of parallel episodes per condition.")
parser.add_argument("--steps",       type=int, default=1000,
                    help="Steps per episode (1000 = 20 s at 50 Hz).")
parser.add_argument("--conditions",  nargs="+", default=None,
                    metavar="ID",
                    help="Run a subset, e.g. --conditions C1 C3 C10 C11. Default: all C1–C13.")
parser.add_argument("--tau_sat",     type=float, default=100.0,
                    help="Torque saturation threshold in N·m (default 100, B1 peak ~140).")
parser.add_argument("--append",      action="store_true",
                    help="Append the result block to DEPLOY_LOG.md after printing.")
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---------------------------------------------------------------------------
# Late imports (after AppLauncher)
# ---------------------------------------------------------------------------
import math
import datetime
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

import isaaclab_tasks  # noqa: F401
import envs.b1_velocity_env_cfg  # noqa: F401

from envs.b1_velocity_ppo_cfg import B1FlatPPORunnerCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# ---------------------------------------------------------------------------
# Command conditions  (id, label, vx, vy, wz)
# vx/vy/wz = None → C13: use env's built-in command sampling (random omni)
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    ("C1",  "stand still",       0.0,  0.0,  0.0),
    ("C2",  "walk fwd slow",     0.3,  0.0,  0.0),
    ("C3",  "walk fwd med",      0.6,  0.0,  0.0),
    ("C4",  "walk fwd fast",     0.8,  0.0,  0.0),
    ("C5",  "walk bwd",         -0.4,  0.0,  0.0),
    ("C6",  "strafe left",       0.0,  0.4,  0.0),
    ("C7",  "strafe right",      0.0, -0.4,  0.0),
    ("C8",  "diag fwd-left",     0.5,  0.3,  0.0),
    ("C9",  "diag fwd-right",    0.5, -0.3,  0.0),
    ("C10", "turn left",         0.0,  0.0,  1.0),
    ("C11", "turn right",        0.0,  0.0, -1.0),
    ("C12", "fwd + turn left",   0.4,  0.0,  0.5),
    ("C13", "random omni",       None, None, None),
]

_HEAD_K = 0.5  # heading_control_stiffness matching training env

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_cmd(cmd_term, robot, vx, vy, wz):
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.vel_command_b[:, 2] = wz
    current_heading = robot.data.heading_w
    if abs(wz) > 1e-4:
        cmd_term.heading_target[:] = current_heading + wz / _HEAD_K
    else:
        cmd_term.heading_target[:] = current_heading


def _quat_to_rpy(quat_wxyz: torch.Tensor):
    """[N,4] [w,x,y,z] → roll, pitch in radians (tensors of shape [N])."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    roll  = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _nan(v):
    return math.isnan(v) if isinstance(v, float) else False


def _fmt(v, dec=3, unit=""):
    return "  N/A  " if _nan(v) else f"{v:.{dec}f}{unit}"


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def run_condition(cond_id, label, vx_cmd, vy_cmd, wz_cmd,
                  steps, env, robot, contact_sensor,
                  foot_body_ids, foot_ids_sensor,
                  policy, cmd_term, tau_sat_thresh, joint_lim):
    """
    Run one command condition.  Returns a metrics dict.

    joint_lim: numpy [12, 2] array of [lo, hi] in radians, or None to skip
    jlim_viol computation.
    """
    N = env.num_envs

    obs, _ = env.reset()
    cmd_term.heading_target[:] = robot.data.heading_w.clone()

    fall_mask = torch.zeros(N, dtype=torch.bool, device=env.device)

    # Pre-allocate per-step arrays — shape (steps, N, ...) stored as float32
    vx_buf  = np.empty((steps, N), np.float32)
    vy_buf  = np.empty((steps, N), np.float32)
    wz_buf  = np.empty((steps, N), np.float32)
    cvx_buf = np.empty((steps, N), np.float32)
    cvy_buf = np.empty((steps, N), np.float32)
    cwz_buf = np.empty((steps, N), np.float32)
    h_buf   = np.empty((steps, N), np.float32)
    roll_buf  = np.empty((steps, N), np.float32)
    pitch_buf = np.empty((steps, N), np.float32)
    cont_buf  = np.zeros((steps, N, 4), bool)
    fvel_buf  = np.empty((steps, N, 4), np.float32)
    air_buf   = np.empty((steps, N, 4), np.float32)
    act_buf   = np.empty((steps, N, 12), np.float32)
    jvel_buf  = np.empty((steps, N, 12), np.float32)
    jpos_buf  = np.empty((steps, N, 12), np.float32)
    torq_buf  = np.empty((steps, N, 12), np.float32)

    for t in range(steps):
        if vx_cmd is not None:
            _inject_cmd(cmd_term, robot, vx_cmd, vy_cmd, wz_cmd)

        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        fall_mask |= dones.bool()

        if vx_cmd is not None:
            _inject_cmd(cmd_term, robot, vx_cmd, vy_cmd, wz_cmd)

        d = robot.data
        roll, pitch = _quat_to_rpy(d.root_quat_w)
        cmd = env.unwrapped.command_manager.get_command("base_velocity")  # [N, 3]
        contact_time = contact_sensor.data.current_contact_time[:, foot_ids_sensor]
        in_contact = contact_time > 0.0                                   # [N, 4]
        air_time   = contact_sensor.data.current_air_time[:, foot_ids_sensor]  # [N, 4]
        foot_vel   = d.body_lin_vel_w[:, foot_body_ids, :2].norm(dim=-1)  # [N, 4]

        vx_buf[t]  = d.root_lin_vel_b[:, 0].cpu().numpy()
        vy_buf[t]  = d.root_lin_vel_b[:, 1].cpu().numpy()
        wz_buf[t]  = d.root_ang_vel_b[:, 2].cpu().numpy()
        cvx_buf[t] = cmd[:, 0].cpu().numpy()
        cvy_buf[t] = cmd[:, 1].cpu().numpy()
        cwz_buf[t] = cmd[:, 2].cpu().numpy()
        h_buf[t]   = d.root_pos_w[:, 2].cpu().numpy()
        roll_buf[t]  = roll.cpu().numpy()
        pitch_buf[t] = pitch.cpu().numpy()
        cont_buf[t]  = in_contact.cpu().numpy()
        fvel_buf[t]  = foot_vel.cpu().numpy()
        air_buf[t]   = air_time.cpu().numpy()
        act_buf[t]   = actions[:, :12].cpu().numpy()
        jvel_buf[t]  = d.joint_vel.cpu().numpy()
        jpos_buf[t]  = d.joint_pos.cpu().numpy()
        try:
            torq_buf[t] = d.applied_torque.cpu().numpy()
        except AttributeError:
            torq_buf[t] = 0.0

    # ---- compute metrics ----
    fell_np = fall_mask.cpu().numpy()         # [N] bool
    ok      = ~fell_np                        # alive envs
    sel     = ok if ok.any() else np.ones(N, bool)  # fallback: all if all fell

    fall_rate  = float(fell_np.mean() * 100.0)
    completion = float((1.0 - fell_np.mean()) * 100.0)

    def rmse_sel(a, b, mask):
        return float(np.sqrt(((a[:, mask] - b[:, mask]) ** 2).mean()))

    e_vx = rmse_sel(vx_buf, cvx_buf, sel)
    e_vy = rmse_sel(vy_buf, cvy_buf, sel)
    e_wz = rmse_sel(wz_buf, cwz_buf, sel)

    # yaw_drift only meaningful for fixed wz=0 conditions (not C13 random)
    is_wz_zero = (wz_cmd is not None) and (abs(wz_cmd) < 1e-4)
    yaw_drift = float(np.abs(wz_buf[:, sel]).mean() * (180.0 / math.pi)) if is_wz_zero else float('nan')

    h_mean = float(h_buf[:, sel].mean())
    h_std  = float(h_buf[:, sel].std())
    roll_rms  = float(np.sqrt((roll_buf[:, sel]  ** 2).mean()))
    pitch_rms = float(np.sqrt((pitch_buf[:, sel] ** 2).mean()))

    duty = cont_buf[:, sel, :].mean(axis=(0, 1)) * 100.0   # [4]

    cont_sel = cont_buf[:, sel, :]
    fvel_sel = fvel_buf[:, sel, :]
    slip_vals = fvel_sel[cont_sel]
    foot_slip = float(slip_vals.mean()) if slip_vals.size > 0 else 0.0

    air_sel  = air_buf[:, sel, :]
    swing_vals = air_sel[(~cont_sel) & (air_sel > 1e-4)]
    air_mean = float(swing_vals.mean()) if swing_vals.size > 0 else 0.0

    act_sel = act_buf[:, sel, :]
    act_diff = np.diff(act_sel, axis=0)       # (steps-1, ok, 12)
    act_rate = float(np.sqrt(((act_diff / 0.02) ** 2).mean()))

    jv = jvel_buf[:, sel, :]
    qd_rms = float(np.sqrt((jv ** 2).mean()))

    tau = torq_buf[:, sel, :]
    tau_rms = float(np.sqrt((tau ** 2).mean()))
    tau_sat = float((np.abs(tau) > tau_sat_thresh).any(axis=-1).mean() * 100.0)

    jp = jpos_buf[:, sel, :]
    if joint_lim is not None:
        lo, hi = joint_lim[:, 0], joint_lim[:, 1]
        viol = ((jp < lo) | (jp > hi)).any(axis=-1)
        jlim_viol = float(viol.mean() * 100.0)
    else:
        jlim_viol = float('nan')

    return dict(
        cond_id=cond_id, label=label,
        vx_cmd=vx_cmd, vy_cmd=vy_cmd, wz_cmd=wz_cmd,
        fall_rate=fall_rate, completion=completion,
        e_vx=e_vx, e_vy=e_vy, e_wz=e_wz, yaw_drift=yaw_drift,
        h_mean=h_mean, h_std=h_std,
        roll_rms=roll_rms, pitch_rms=pitch_rms,
        duty=duty, foot_slip=foot_slip, air_mean=air_mean,
        act_rate=act_rate, qd_rms=qd_rms,
        tau_rms=tau_rms, tau_sat=tau_sat, jlim_viol=jlim_viol,
    )


# ---------------------------------------------------------------------------
# Aggregate across conditions
# ---------------------------------------------------------------------------

def aggregate(results):
    """Compute stage-level summary metrics from per-condition results."""
    def mean_of(key, cids=None):
        rs = [r for r in results if (cids is None or r["cond_id"] in cids)]
        vals = [r[key] for r in rs if not _nan(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    agg = {k: mean_of(k) for k in [
        "fall_rate", "completion", "e_vx", "e_vy", "e_wz",
        "h_mean", "h_std", "roll_rms", "pitch_rms",
        "act_rate", "qd_rms", "tau_rms", "tau_sat", "jlim_viol",
        "foot_slip", "air_mean",
    ]}
    # Tracking metrics for specific condition sets (matching DEPLOY_LOG thresholds)
    agg["e_vx_C3C4"]   = mean_of("e_vx",  ("C3", "C4"))
    agg["e_vy_C6C7"]   = mean_of("e_vy",  ("C6", "C7"))
    agg["e_wz_C10C11"] = mean_of("e_wz",  ("C10", "C11"))
    agg["yaw_drift"]   = mean_of("yaw_drift")   # already NaN for wz≠0 conditions

    duties = [r["duty"] for r in results if r["duty"] is not None]
    agg["duty"] = np.mean(duties, axis=0).tolist() if duties else None

    return agg


# ---------------------------------------------------------------------------
# Verdict helper
# ---------------------------------------------------------------------------

_PASS_THRESHOLDS = {
    "fall_rate":   ("<",  5.0),
    "completion":  (">", 90.0),
    "e_vx_C3C4":  ("<",  0.20),
    "e_vy_C6C7":  ("<",  0.25),
    "e_wz_C10C11":("<",  0.30),
    "yaw_drift":   ("<",  5.0),
    "h_mean_lo":   (">",  0.49),
    "h_mean_hi":   ("<",  0.57),
    "roll_rms":    ("<",  0.05),
    "pitch_rms":   ("<",  0.05),
    "tau_sat":     ("<", 20.0),
    "jlim_viol":   ("<",  1.0),
}

_FAIL_THRESHOLDS = {
    "fall_rate":   (">=", 10.0),
    "completion":  ("<",  80.0),
    "e_vx_C3C4":  (">=",  0.35),
    "e_vy_C6C7":  (">=",  0.40),
    "e_wz_C10C11":(">=",  0.50),
    "yaw_drift":   (">=", 10.0),
    "tau_sat":     (">=", 40.0),
    "jlim_viol":   (">=",  5.0),
}


def _verdict(agg):
    agg_check = dict(agg)
    agg_check["h_mean_lo"] = agg.get("h_mean", float('nan'))
    agg_check["h_mean_hi"] = agg.get("h_mean", float('nan'))

    for key, (op, thr) in _FAIL_THRESHOLDS.items():
        v = agg_check.get(key, float('nan'))
        if not _nan(v):
            if op == ">=" and v >= thr: return "FAIL"
            if op == "<"  and v <  thr: return "FAIL"

    warnings = []
    for key, (op, thr) in _PASS_THRESHOLDS.items():
        v = agg_check.get(key, float('nan'))
        if not _nan(v):
            ok = (v < thr) if op == "<" else (v > thr)
            if not ok:
                warnings.append(key)

    return "PASS" if not warnings else f"WARNING ({', '.join(warnings)})"


# ---------------------------------------------------------------------------
# DEPLOY_LOG block formatter
# ---------------------------------------------------------------------------

def format_deploy_block(stage, simulator, actuator, checkpoint,
                        cond_ids_run, agg, ref_agg=None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    w = 80

    def gap_str(v, ref_v):
        if ref_agg is None or ref_v is None or _nan(v) or _nan(ref_v) or ref_v == 0:
            return ""
        g = abs(v - ref_v)
        gp = g / abs(ref_v) * 100.0
        return f"  ({gp:4.0f}% vs S1)"

    lines = []
    sep = "=" * w
    lines.append(sep)
    lines.append(f"Stage: {stage}   Simulator: {simulator}")
    lines.append(f"Actuator: {actuator}")
    lines.append(f"Policy: {checkpoint}")
    lines.append(f"Conditions run: {' '.join(cond_ids_run)}  /  R1 (nominal)")
    lines.append(f"Date: {now}")
    lines.append(sep)
    lines.append("")

    # Stability overview
    lines.append(f"  Fall rate          : {_fmt(agg['fall_rate'], 1, '%'):>8}   (pass < 5%)")
    lines.append(f"  Episode completion : {_fmt(agg['completion'], 1, '%'):>8}   (pass > 90%)")
    lines.append("")

    # Tracking
    lines.append("  Tracking RMSE:")
    rv_vx  = ref_agg["e_vx_C3C4"]   if ref_agg else None
    rv_vy  = ref_agg["e_vy_C6C7"]   if ref_agg else None
    rv_wz  = ref_agg["e_wz_C10C11"] if ref_agg else None
    rv_yd  = ref_agg["yaw_drift"]    if ref_agg else None
    lines.append(f"    e_vx  (C3/C4)    : {_fmt(agg['e_vx_C3C4'],   3, ' m/s'):>12}{gap_str(agg['e_vx_C3C4'],   rv_vx)}")
    lines.append(f"    e_vy  (C6/C7)    : {_fmt(agg['e_vy_C6C7'],   3, ' m/s'):>12}{gap_str(agg['e_vy_C6C7'],   rv_vy)}")
    lines.append(f"    e_wz  (C10/C11)  : {_fmt(agg['e_wz_C10C11'], 3, ' rad/s'):>12}{gap_str(agg['e_wz_C10C11'], rv_wz)}")
    lines.append(f"    yaw_drift        : {_fmt(agg['yaw_drift'],    2, ' deg/s'):>12}{gap_str(agg['yaw_drift'],   rv_yd)}")
    lines.append("")

    # Stability
    lines.append("  Stability:")
    lines.append(f"    h_mean +/- h_std : {_fmt(agg['h_mean'], 3)} +/- {_fmt(agg['h_std'], 3)} m")
    lines.append(f"    roll_rms         : {_fmt(agg['roll_rms'],  4, ' rad'):>12}")
    lines.append(f"    pitch_rms        : {_fmt(agg['pitch_rms'], 4, ' rad'):>12}")
    lines.append("")

    # Gait
    lines.append("  Gait:")
    if agg.get("duty"):
        d = agg["duty"]
        lines.append(f"    duty FL/FR/RL/RR : {d[0]:.1f}%  {d[1]:.1f}%  {d[2]:.1f}%  {d[3]:.1f}%")
    else:
        lines.append(f"    duty FL/FR/RL/RR : N/A")
    lines.append(f"    foot_slip        : {_fmt(agg['foot_slip'], 4, ' m/s'):>12}")
    lines.append(f"    air_time_mean    : {_fmt(agg['air_mean'],  3, ' s'):>12}")
    lines.append("")

    # Actuator
    lines.append("  Actuator:")
    lines.append(f"    act_rate_rms     : {_fmt(agg['act_rate'], 2, ' rad/s'):>12}")
    lines.append(f"    qd_rms           : {_fmt(agg['qd_rms'],   2, ' rad/s'):>12}")
    lines.append(f"    tau_rms          : {_fmt(agg['tau_rms'],  2, ' N·m'):>12}")
    lines.append(f"    tau_sat          : {_fmt(agg['tau_sat'],  1, '%'):>12}")
    lines.append(f"    jlim_viol        : {_fmt(agg['jlim_viol'],1, '%'):>12}")
    lines.append("")

    # Gap vs S1 (only shown when ref_agg provided)
    if ref_agg:
        lines.append("  Gap vs S1 (key metrics):")
        def gline(label, v, rv, unit=""):
            if _nan(v) or rv is None or _nan(rv):
                return f"    {label:<18}: N/A"
            g   = abs(v - rv)
            gp  = g / abs(rv) * 100.0 if rv != 0 else float('nan')
            gps = f"{gp:.0f}%" if not _nan(gp) else "N/A"
            return f"    {label:<18}: {g:.4f}{unit}  ({gps})"
        lines.append(gline("fall_rate gap",  agg["fall_rate"],    ref_agg["fall_rate"],    " pp"))
        lines.append(gline("e_vx gap",       agg["e_vx_C3C4"],    ref_agg["e_vx_C3C4"],   " m/s"))
        lines.append(gline("e_vy gap",       agg["e_vy_C6C7"],    ref_agg["e_vy_C6C7"],   " m/s"))
        lines.append(gline("e_wz gap",       agg["e_wz_C10C11"],  ref_agg["e_wz_C10C11"], " rad/s"))
        lines.append(gline("yaw_drift gap",  agg["yaw_drift"],    ref_agg["yaw_drift"],    " deg/s"))
        lines.append(gline("h_mean gap",     agg["h_mean"],       ref_agg["h_mean"],       " m"))
        lines.append("")

    verdict = _verdict(agg)
    lines.append(f"  Verdict: [ {verdict} ]")
    lines.append("  Notes:")
    lines.append("    -")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Filter conditions if --conditions was specified
    if args.conditions:
        requested = set(args.conditions)
        conditions = [c for c in ALL_CONDITIONS if c[0] in requested]
        if not conditions:
            print(f"[eval] No matching conditions for {args.conditions}")
            sys.exit(1)
    else:
        conditions = ALL_CONDITIONS

    print(f"\n  Stage       : {args.stage}")
    print(f"  Simulator   : {args.simulator}")
    print(f"  Actuator    : {args.actuator}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Task        : {args.task}")
    print(f"  Episodes    : {args.num_episodes}  ×  {args.steps} steps "
          f"({args.steps * 0.02:.1f}s each)")
    print(f"  Conditions  : {' '.join(c[0] for c in conditions)}")
    print(f"  τ_sat thr   : {args.tau_sat} N·m\n")

    # Build env with long episode length so only falls cause resets
    agent_cfg = B1FlatPPORunnerCfg()
    env_cfg = parse_env_cfg(
        args.task, device=agent_cfg.device, num_envs=args.num_episodes
    )
    env_cfg.episode_length_s = max(args.steps * 0.02 + 10.0, 60.0)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Scene references
    robot          = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    cmd_term       = env.unwrapped.command_manager._terms["base_velocity"]

    all_body_names   = robot.body_names
    foot_body_ids    = [all_body_names.index(n) for n in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]

    _sids, _snames = contact_sensor.find_bodies(".*_foot$")
    desired_order  = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    perm           = [_snames.index(n) for n in desired_order if n in _snames]
    foot_ids_sensor = [_sids[i] for i in perm]

    # Joint position limits: try soft limits from robot data, fall back to None
    joint_lim = None
    try:
        lim_t = robot.data.soft_joint_pos_limits[0].cpu().numpy()  # [12, 2]
        if lim_t.shape == (12, 2):
            joint_lim = lim_t
    except (AttributeError, IndexError):
        pass

    # ---------------------------------------------------------------------------
    # Per-condition progress table header
    # ---------------------------------------------------------------------------
    hdr = (f"  {'cond':>4} | {'label':<18} | "
           f"{'fall%':>5} | {'e_vx':>6} {'e_vy':>6} {'e_wz':>6} | "
           f"{'yaw°/s':>7} | {'h_mn':>5}±{'h_sd':>5} | "
           f"duty FL  FR  RL  RR")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_results = []

    for cond_id, label, vx_cmd, vy_cmd, wz_cmd in conditions:
        r = run_condition(
            cond_id, label, vx_cmd, vy_cmd, wz_cmd,
            args.steps, env, robot, contact_sensor,
            foot_body_ids, foot_ids_sensor,
            policy, cmd_term,
            args.tau_sat, joint_lim,
        )
        all_results.append(r)

        d = r["duty"]
        yd = r["yaw_drift"]
        yd_str = f"{yd:7.2f}" if not _nan(yd) else "    N/A"
        print(
            f"  {cond_id:>4} | {label:<18} | "
            f"{r['fall_rate']:5.1f} | "
            f"{r['e_vx']:6.3f} {r['e_vy']:6.3f} {r['e_wz']:6.3f} | "
            f"{yd_str} | "
            f"{r['h_mean']:5.3f}+/-{r['h_std']:5.3f} | "
            f"{d[0]:5.1f} {d[1]:5.1f} {d[2]:5.1f} {d[3]:5.1f}"
        )

    # ---------------------------------------------------------------------------
    # Aggregate + format DEPLOY_LOG block  (must happen before closing sim)
    # ---------------------------------------------------------------------------
    agg = aggregate(all_results)
    cond_ids_run = [r["cond_id"] for r in all_results]

    block = format_deploy_block(
        stage=args.stage,
        simulator=args.simulator,
        actuator=args.actuator,
        checkpoint=args.checkpoint,
        cond_ids_run=cond_ids_run,
        agg=agg,
    )

    print(f"\n\n{'=' * 80}")
    print("  DEPLOY_LOG.md RESULT BLOCK")
    print('=' * 80)
    print(block)

    if args.append:
        deploy_log = Path(__file__).parent.parent / "DEPLOY_LOG.md"
        with open(deploy_log, "a", encoding="utf-8") as f:
            f.write(f"\n\n<!-- appended by eval_b1_velocity.py -->\n")
            f.write("```\n")
            f.write(block)
            f.write("\n```\n")
        print(f"\n  Appended to {deploy_log}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
