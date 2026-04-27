"""
Custom MDP reward terms for B1 velocity-tracking that aren't in stock
Isaac Lab. Targeted at fixing degenerate gaits where the policy converges
to using fewer than four legs.

Functions follow the Isaac Lab MDP signature:
    f(env, ...other params...) -> torch.Tensor of shape (num_envs,)
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def excessive_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float = 1.0,
) -> torch.Tensor:
    """Penalize feet that have been in the air for longer than `max_air_time`.

    Targets the "permanently raised leg" pathology: a 2-leg trot where one or
    two feet are held off the ground for the entire episode. The stock
    `feet_air_time` rewards lifting feet, but caps reward at the threshold —
    it doesn't penalize feet that lift and never come down.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(air_time - max_air_time, min=0.0)
    return torch.sum(excess, dim=1)


def excessive_contact_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float = 0.5,
) -> torch.Tensor:
    """Penalize feet that have been planted for longer than `max_contact_time`.

    Symmetric counterpart to `excessive_air_time`: punishes the "permanently
    planted leg" pathology where a single foot bears the body's weight while
    others swing or are dragged. With weight -1.0 and max_contact_time=0.5,
    a foot planted continuously for 14 s (e.g. RR at 70% duty over 20 s)
    contributes ~13 / step penalty — incompatible with the velocity reward.

    Together, excessive_air_time and excessive_contact_time bracket every
    foot to swing-stance cycles no longer than (max_air + max_contact).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(contact_time - max_contact_time, min=0.0)
    return torch.sum(excess, dim=1)


def must_move_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.05,
    actual_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize standing still when commanded to move.

    Returns +1 per env when |cmd_xy| > cmd_threshold but |actual_xy|
    < actual_threshold (commanded to move but not moving). Closes the
    "stand and flick a leg" walk-reward exploit by making standing
    catastrophic when the command is non-zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = asset.data.root_lin_vel_b[:, :2]
    cmd_mag = cmd.norm(dim=1)
    actual_mag = actual.norm(dim=1)
    return ((cmd_mag > cmd_threshold) & (actual_mag < actual_threshold)).float()


def true_walk_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward 'lateral-sequence walk' coordination: exactly 3 feet in stance.

    Walk's defining signature: at any moment exactly 3 feet are planted and
    1 foot is swinging (75% duty per leg). This is mutually exclusive with
    trot (2 feet planted, 2 in air on diagonals) and bound (2 feet planted,
    2 in air on fore-aft pairs).

    Returns +1 when exactly 3 feet are in contact, 0 otherwise. Combined
    with gait_phase_match_reward using walk's phase schedule [0, 25, 12, 37]
    at period 50, this produces a true 1-3-2-4 lateral sequence walk.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    num_in_contact = in_contact.sum(dim=1)
    return (num_in_contact == 3).float()


def true_pace_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward pace coordination AND penalize trot/bound coordinations.

    Pace requires FL+RL synced (left lateral pair) AND FR+RR synced
    (right lateral pair).
    Trot has FL+RR / FR+RL diagonal sync.
    Bound has FL+FR / RL+RR fore-aft sync.

    Score = (pace pair sync count) − (trot pair sync count)
    Pure pace : +2 (lateral pairs synced, diagonal pairs split)
    Pure trot : −2 (diagonals synced, laterals split)
    Pure bound: ~0 (fore-aft synced — laterals and diagonals both mostly split)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    left_sync = (in_contact[:, 0] == in_contact[:, 2]).float()   # FL+RL
    right_sync = (in_contact[:, 1] == in_contact[:, 3]).float()  # FR+RR
    diag_a_sync = (in_contact[:, 0] == in_contact[:, 3]).float() # FL+RR
    diag_b_sync = (in_contact[:, 1] == in_contact[:, 2]).float() # FR+RL
    return (left_sync + right_sync) - (diag_a_sync + diag_b_sync)


def true_bound_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward bound coordination AND penalize trot coordination.

    Bound requires FL+FR synced AND RL+RR synced (fore-aft pairs).
    Trot has FL+RR synced AND FR+RL synced (diagonal pairs).

    Score = (bound pair sync count) - (trot pair sync count), in [-2, +2].
    Pure bound  : +2 (bound pairs synced, trot pairs split)
    Pure trot   : -2 (trot pairs synced, bound pairs split)
    All planted / all airborne : 0 (all four match each other equally)
    Mixed       : varies

    Combined with gait_phase_match_reward which targets the bound schedule,
    this term explicitly punishes the trot local optimum that PPO kept
    finding when only phase_match was active.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    front_sync = (in_contact[:, 0] == in_contact[:, 1]).float()  # FL+FR
    rear_sync = (in_contact[:, 2] == in_contact[:, 3]).float()   # RL+RR
    diag_a_sync = (in_contact[:, 0] == in_contact[:, 3]).float() # FL+RR
    diag_b_sync = (in_contact[:, 1] == in_contact[:, 2]).float() # FR+RL
    return (front_sync + rear_sync) - (diag_a_sync + diag_b_sync)


def joint_lr_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize left/right joint motion asymmetry within front and rear pairs.

    Sums |velocity magnitude difference| between (FL, FR) and (RL, RR).
    For each leg, "velocity magnitude" is the sum of (hip² + thigh² + calf²)
    joint velocities — a scalar that captures total motion intensity.

    Targets the per-leg specialization PPO converges to on B1 (e.g., FL
    hip rotating 4° while FR hip rotates 9°). Bilateral symmetry should
    hold for any gait pattern (trot, bound, pace, walk) since B1's mass
    distribution is L/R symmetric. Front/rear differences are NOT
    penalized — those are physically required by B1's 0.8/1.0 thigh
    default asymmetry.

    Stateless (uses current joint velocities). Variance gradient pushes
    PPO toward equal-effort L/R legs. Steer config should override
    weight to 0 since asymmetric L/R is the whole point of turning.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel
    names = asset.joint_names

    def leg_vel_sq(leg: str) -> torch.Tensor:
        idx = [names.index(f"{leg}_{j}_joint") for j in ["hip", "thigh", "calf"]]
        idx_t = torch.tensor(idx, device=env.device, dtype=torch.long)
        return (qd[:, idx_t] ** 2).sum(dim=1)

    fl_sq = leg_vel_sq("FL")
    fr_sq = leg_vel_sq("FR")
    rl_sq = leg_vel_sq("RL")
    rr_sq = leg_vel_sq("RR")

    return (fl_sq - fr_sq).abs() + (rl_sq - rr_sq).abs()


def gait_phase_match_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    leg_phase_offsets: list[int],
    stance_fraction: float = 0.5,
    period_steps: int = 30,
) -> torch.Tensor:
    """Reward foot contacts that agree with a target periodic gait schedule.

    Uses env.episode_length_buf as a per-env phase clock. For each leg with
    its assigned phase offset:
        target_in_stance(t) := ((t + offset) mod period) < stance_threshold
    Reward is the count of legs whose actual contact state matches target.

    This forces specific gait COORDINATION PATTERNS, not just instantaneous
    constraints. The policy cannot game the reward by locking-pair-airborne
    or 3-leg-flicking because those states violate the time-evolving target.

    Phase offset conventions for common gaits (period=30 → 1.5 Hz):
      Trot  [0, 15, 15,  0]  diagonal pairs (FL+RR sync, FR+RL sync)
      Bound [0,  0, 15, 15]  fronts together, rears together, antiphase
      Pace  [0, 15,  0, 15]  left pair sync, right pair sync, antiphase
      Walk  [0, 25, 12, 37]  lateral sequence 1-3-2-4 (period=50, stance 0.75)

    Args:
        sensor_cfg: contact sensor with body_names selecting feet (FL,FR,RL,RR).
        leg_phase_offsets: 4 integer phase offsets (in env steps).
        stance_fraction: target duty factor (fraction of cycle in stance).
        period_steps: full cycle length in env steps.

    Returns:
        (num_envs,) tensor — max value = 4.0 (all legs match target).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    step = env.episode_length_buf  # (E,) long tensor

    leg_offsets = torch.tensor(
        leg_phase_offsets, dtype=torch.long, device=env.device,
    )  # (4,)

    phase = (step.unsqueeze(1) + leg_offsets.unsqueeze(0)) % period_steps  # (E, 4)
    stance_threshold = int(stance_fraction * period_steps)
    target_in_stance = phase < stance_threshold  # (E, 4) bool

    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    actual_in_stance = contact_time > 0.0  # (E, 4) bool

    return (target_in_stance == actual_in_stance).float().sum(dim=1)  # (E,)


def bound_coordination_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward TRUE bound pattern: ALL of one pair down, ALL of the other up.

    A real bound alternates between:
      (A) both fronts planted AND both rears airborne
      (B) both fronts airborne AND both rears planted

    Earlier XOR-only version fired whenever ONE pair was both-planted and
    the OTHER pair was "not both planted" — but "not both planted" includes
    {one up, one down} and {both up}. Policy exploited (one up, one down):
    keep RR permanently planted + briefly lift RL → rear "not both planted"
    → reward. Three legs permanently planted, one foot flicks.

    Strict version requires both members of each pair to AGREE in state
    AND the two pairs to be in OPPOSITE states. Closes every loophole:
      - All 4 down: front_stance=T, rear_stance=T, rear_swing=F → 0
      - All 4 up: front_swing=T, rear_swing=T, but only one half of OR
        passes → also penalized by excessive_air_time.
      - 3 planted + 1 flick: pair containing the flicker fails its AND → 0
      - Diagonal pair (trot): no pair AND succeeds → 0
      - True bound: exactly one half of the OR fires → +1 reward

    Returns ∈ {0, 1}. Body order expected: [FL, FR, RL, RR].
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    not_in_contact = ~in_contact
    front_stance = in_contact[:, 0] & in_contact[:, 1]
    front_swing = not_in_contact[:, 0] & not_in_contact[:, 1]
    rear_stance = in_contact[:, 2] & in_contact[:, 3]
    rear_swing = not_in_contact[:, 2] & not_in_contact[:, 3]
    pattern_a = front_stance & rear_swing       # fronts down, rears up
    pattern_b = front_swing & rear_stance       # fronts up, rears down
    return (pattern_a | pattern_b).float()


def pace_coordination_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward TRUE pace pattern: ALL of one lateral pair down, OTHER all up.

    Symmetric to bound but with left/right pairs instead of front/rear:
      (A) FL+RL planted AND FR+RR airborne
      (B) FL+RL airborne AND FR+RR planted

    Closes the same "3 planted + 1 flick" loophole that affected the XOR
    version. Returns ∈ {0, 1}. Body order expected: [FL, FR, RL, RR].
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    not_in_contact = ~in_contact
    left_stance = in_contact[:, 0] & in_contact[:, 2]
    left_swing = not_in_contact[:, 0] & not_in_contact[:, 2]
    right_stance = in_contact[:, 1] & in_contact[:, 3]
    right_swing = not_in_contact[:, 1] & not_in_contact[:, 3]
    pattern_a = left_stance & right_swing
    pattern_b = left_swing & right_stance
    return (pattern_a | pattern_b).float()


def duty_factor_target_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target: float = 0.5,
) -> torch.Tensor:
    """Penalize per-foot duty factor deviation from `target`.

    Per-foot duty estimate is computed from the most recently completed
    phases:  duty ≈ last_contact_time / (last_contact_time + last_air_time).

    Closes the "lock one pair planted forever, flick the other pair" exploit
    found in bound/pace runs. With true-pace alternation each foot has
    duty ≈ 0.5 and the penalty is zero. With one pair locked at 95% duty
    and the opposite pair at 5%, each foot contributes ~0.16 → total
    ~0.65 / step, big enough to push the policy out of the locked basin.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_contact = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    duty = last_contact / (last_contact + last_air + 1e-6)
    return torch.sum((duty - target) ** 2, dim=1)


def air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize variance of `last_air_time` across feet.

    Targets the "one leg cycling at half rate" pathology: 3 legs trot at
    ~0.4 s swing while the 4th leg swings ~0.75 s, exploiting load
    asymmetry. The other timing penalties (excessive_air_time, etc.) bound
    each foot independently — none of them notice that legs are cycling
    at *different* rates.

    `last_air_time` is each foot's most-recently-completed air-phase
    duration. A symmetric trot has variance ≈ 0; a 3+1 asymmetric gait
    where one leg's swing is 2× the others gives variance ~0.025 s².
    With weight -5.0 that's ~0.13/step penalty when asymmetric, ~0/step
    when symmetric — exactly the gradient needed to pull FL into line
    without hurting the working three legs.

    Note: signal updates only at landings, so it's sparse over time but
    accumulates strongly across episodes.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return last_air_time.var(dim=1)


def short_swing_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_swing_time: float = 0.3,
) -> torch.Tensor:
    """Penalize touchdowns that happen after very short air phases.

    Targets the "rapid tap-tap-tap" pathology where one foot bounces on the
    ground at high frequency (~5 Hz) to provide support reaction force,
    instead of swinging properly. Each tap individually is too brief to trip
    excessive_contact_time, but the AGGREGATE pattern is non-walking.

    Fires only at the moment of first contact. If the prior air phase
    (last_air_time) was shorter than min_swing_time, contributes
    (min_swing_time - last_air_time) per offending foot. With weight -2.0
    and min_swing_time=0.3, a foot tapping every 0.2 s (last_air_time≈0.05 s)
    pays ~0.5 per landing → ~2.5 per second per offending foot.

    Real walking has swing phases of 0.3-0.5 s, so this term is silent on
    healthy gaits and only bites on pneumatic-hopper-style exploits.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    short_air = torch.clamp(min_swing_time - last_air_time, min=0.0)
    return (first_contact.float() * short_air).sum(dim=1)
