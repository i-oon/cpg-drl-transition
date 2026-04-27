# CLAUDE.md - Project Context for AI Assistant

**Project:** CPG-DRL Transition Control  
**Full Title:** Transition-Aware CPG-RBF Locomotion Control with DRL-Based Continuous Gait Switching  
**Student:** Disthorn Suttawet (66340500019)  
**Course:** FRA 503 Deep Reinforcement Learning  
**Timeline:** Weeks 10-15 (6 weeks)

---

## Project Overview

This project implements a two-phase system for smooth gait transitions in a quadruped robot:

- **Phase 1:** Train three base gaits (walk, trot, steer) using a CPG-RBF neural controller optimized with PI^BB.
- **Phase 2:** Train a DRL policy (PPO) that learns per-RBF-neuron blending coefficients for smooth transitions between gaits.

**Novel Contribution:** Per-leg, phase-aware transition learning via residual 
corrections. Instead of uniform blending (single α for all legs), we learn 
leg-specific corrections (Δα_FL, Δα_FR, Δα_RL, Δα_RR) on top of a baseline ramp. 
The corrections adapt to:
1. B1's morphological asymmetry (front/rear kinematic differences)
2. CPG phase state (stance vs swing → different blending aggression)
3. Robot stability (torque feedback → slow down if unstable)

This enables smooth task transitions (forward ↔ turn ↔ fast) while preserving 
the modularity of independently-trained base behaviors.

**Reference:** Thor et al. [CPG-RBFN framework](https://github.com/MathiasThor/CPG-RBFN-framework)

---

## Development Environment

### Hardware
- **GPU:** NVIDIA RTX 4070 Ti SUPER (16.7GB VRAM)
- **CPU:** Intel i5-14400F (10 cores, 16 threads)
- **RAM:** 32GB
- **OS:** Ubuntu (with conda)

### Software Stack
- **Python:** 3.10.19
- **PyTorch:** 2.5.1+cu121
- **Isaac Lab:** 0.36.3 (installed at `~/IsaacLab/`)
- **Isaac Sim:** 4.5.0.0
- **RSL-RL:** 2.2.4 (use this for PPO, NOT Stable-Baselines3)
- **Conda env:** `env_isaaclab` (always activate this)

### Robot Platform
- **Model:** Unitree B1 quadruped (12 DOF: 4 legs × 3 joints)
- **Mass:** ~50 kg
- **Asset:** Custom USD from URDF, registered as `UNITREE_B1_CFG` in `isaaclab_assets.robots.unitree`
- **USD location:** `~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd`
- **Simulation:** Flat terrain only

---

## B1 Robot Specifics

### Joint Axis Convention (CRITICAL)
```
hip_joint   → ABDUCTION (side-to-side lateral splay)
              Evidence: default FL=+0.1, FR=-0.1 (left/right mirror)
              CPG W column 0 — keep small (~±2°)

thigh_joint → FLEXION (forward-backward swing) ← PRIMARY WALKING JOINT
              Evidence: default all positive (+0.8 front, +1.0 rear)
              CPG W column 1 — dominant for locomotion (~±12-15°)

calf_joint  → KNEE BEND
              Evidence: default all -1.5 rad (-86°)
              CPG W column 2 — foot clearance during swing (~±9-12°)
```

### Default Joint Positions (asymmetric front/rear)
```
Front thighs: +0.8 rad (+45.8°)     ← less forward lean
Rear thighs:  +1.0 rad (+57.3°)     ← more forward lean
All calves:   -1.5 rad (-85.9°)
Front hips:   +0.1 / -0.1 rad       ← slight outward splay
Rear hips:    +0.1 / -0.1 rad
```
The front/rear thigh asymmetry (0.8 vs 1.0) causes different leg behavior when the same CPG offset is applied — rear legs tend to stay planted longer.

### Actuator Configuration
```python
# In ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
DCMotorCfg(
    effort_limit=280.0,       # B1 motor spec (NOT A1's 23.7!)
    saturation_effort=280.0,
    velocity_limit=21.0,
    stiffness=200.0,          # Scaled for ~50 kg (cf. H1=200 at ~47 kg)
    damping=5.0,
)
```
**WARNING:** Original B1 config was copied from A1 (12 kg) with stiffness=25, effort=23.7. At those values, joints CANNOT track position targets — the robot's mass overpowers the PD controller.

### Contact Sensor
- Foot contact: pattern `.*_foot$` (NOT `.*_calf$` which matches lower leg bone)
- The B1 USD has both `FL_calf` and `FL_foot` links; contact detection must be on the actual foot pads

---

## Project Structure

```
cpg-drl-transition/
├── envs/
│   ├── __init__.py
│   └── unitree_b1_env.py      # Main DirectRLEnv with batched CPG
├── networks/
│   ├── __init__.py
│   ├── cpg_rbf.py             # CPG-RBF network (single-env, pure PyTorch)
│   └── transition_policy.py   # PPO policy for Phase 2 (TBD)
├── algorithms/
│   ├── __init__.py
│   ├── pibb_trainer.py        # PI^BB optimizer (softmax-weighted, per-joint noise)
│   └── ppo_trainer.py         # PPO trainer for Phase 2 (TBD)
├── configs/
│   ├── phase1_walk.yaml
│   ├── phase1_trot.yaml
│   ├── phase1_bound.yaml
│   └── phase2_transition.yaml # (TBD)
├── weights/                    # Trained W matrices (.npy)
├── logs/phase1/               # TensorBoard logs (timestamped per run)
├── scripts/
│   ├── train_phase1_walk.py
│   ├── train_phase1_trot.py
│   ├── train_phase1_bound.py
│   ├── train_ppo_walk.py      # PPO-on-W detour (experiment #8, did NOT work — kept for reference)
│   ├── play_gait.py           # Playback with gait diagram + reward breakdown
│   ├── play_ppo.py            # PPO-trained policy playback (paired with train_ppo_walk.py)
│   ├── diagnose_cpg.py        # Debug: joint targets, limits, W breakdown
│   ├── visualize_cpg.py       # CPG phase offset visualization
│   └── test_env.py            # Live 7-check environment test
├── tests/
│   ├── test_cpg_rbf.py        # 21 tests
│   └── test_environment.py    # 23 tests (mocks Isaac Lab)
├── CLAUDE.md
├── README.md
└── pytest.ini
```

---

## Technical Specifications

### CPG-RBF Architecture

#### SO(2) Oscillator — Pre-computed (LocoNets approach)

Instead of stepping the oscillator each control tick, integrate it ONCE to
build a lookup table, then at runtime the phase is just an integer index.

```python
# 1. Integrate one full period at env startup:
o(t+1) = tanh(α · R(Δφ) · o(t))
R(Δφ) = [[cos(Δφ), sin(Δφ)], [-sin(Δφ), cos(Δφ)]]
α = 1.01,  Δφ = 2π · f · dt          # f = 1.0 Hz, dt = 0.02 s
# Starting from (-0.197, 0), trace until y wraps back → ~51 steps / period.

# 2. At runtime:
#   self._phase_idx : int in [0, period)   shared across all envs
#   self._phase_idx = (self._phase_idx + 1) % self._period
```

No trig, no exp in the hot loop. Phase is INTEGER, never float.

#### RBF Layer — KENNE Lookup Table

```python
# H = 20, σ² = 0.04
# Centers are sampled FROM THE TRAJECTORY (NOT on the unit circle):
ci    = np.linspace(1, period, H + 1, dtype=int)[:-1]   # 20 time indices
c_i   = (x_traj[ci], y_traj[ci])                        # at radius ~0.197

# Pre-compute (period+1, H) table of RBF activations:
KENNE[t, i] = exp( -‖(x_traj[t], y_traj[t]) − c_i‖² / σ² )

# At runtime — one index lookup per leg:
rbf_leg = KENNE[(phase_idx + leg_step_offset[leg]) % period]   # (H,)
```

`leg_step_offset` converts the YAML `phase_offsets` (radians) into integer
phase-step shifts once at startup. For walk at period=51: `[0, 26, 13, 38]`.

#### Current Encoding: Shared Indirect (20×3)

```python
W ∈ ℝ^(20, 3)    # shared across all 4 legs — 60 params total
                 # columns: (hip, thigh, calf)

# Per-leg joint targets: each leg uses the SAME W on its phase-shifted RBF
for leg k in {FL, FR, RL, RR}:
    raw[k]     = rbf_leg[k] @ W                 # (3,)
joint_offset   = tanh(raw.flatten())            # (12,) bounded [-1, 1] rad
joint_target   = default_joint_pos + joint_offset
```

Why shared W: CLAUDE.md experiment #1 is the only encoding that produced
stable trot (diagonal sync requires identical W). Shared W is a *feature*
(forces coordination), not a limitation. Front/rear duty asymmetry remains
(~25-30 % gap), caused by B1's asymmetric thigh defaults (0.8 rad front /
1.0 rad rear). Acceptable for Phase 2.

#### W Initialization: Cosine Walking Prior (Thor-style)

Random `uniform(-0.1, 0.1)` init produced W norm ~0.8 after 2000 PIBB iters
— PIBB barely moved from init because every exploratory step risks a fall
(negative reward) before any walking emerges. Seed with a walking prior so
iteration 1 already produces a (rough) gait:

```yaml
pibb:
  init_mode: cosine        # or "random"
```

```python
# In pibb_trainer.py: algorithms/pibb_trainer.py
angles     = 2π · arange(H) / H
W[:, 0]    = 0                           # hip: PIBB discovers
W[:, 1]    = 0.05 · cos(angles)          # thigh: forward swing
W[:, 2]    = 0.04 · sin(angles)          # calf: LAGS thigh by π/2
# Calf extended (foot planted) during stance, flexed (foot lifted) during swing.
# Do NOT use cos(angles + π/2) — that's −sin(angles), i.e. inverted cycle:
# foot tries to lift during mid-stance and plant during mid-swing → net-zero
# stride, robot steps in place with vx ≈ 0. (Observed experimentally.)
```

**Amplitude tuning trap:** RBF activations on the trajectory overlap ~4
neighbours → raw ≈ 4 × W. To hit the CLAUDE.md target of ±12° thigh / ±9°
calf after tanh, use W amplitudes 0.05 / 0.04, NOT 0.20 / 0.15. Verify
with a quick numpy sim of `tanh(KENNE @ W)` before launching training.

#### Phase 2 Blending (unchanged for indirect W)

```python
# Three frozen W matrices from Phase 1 — all shape (20, 3):
W_forward, W_fast, W_turn ∈ ℝ^(20, 3)

# Blend at OUTPUT level (post-tanh) for every leg:
out_cur[k] = tanh(rbf_leg[k] @ W_current)    # (3,) per leg
out_tgt[k] = tanh(rbf_leg[k] @ W_target)     # (3,) per leg
blended[k] = (1 − α_k) · out_cur[k] + α_k · out_tgt[k]
```

tanh is nonlinear, so blending W first and tanh after gives a different
(and worse) result. Always tanh, then blend.

---

### PI^BB Optimization

```python
# Thor et al. cost-weighted averaging:
s_i = exp(h · (R_i - R_min) / (R_max - R_min + 1e-8))
p_i = s_i / Σ s_i
W  += Σ_i (p_i · ε_i)
σ  *= decay

# Per-joint noise scaling (prevents hip explosion):
noise[:, :, 0] *= 0.1    # hip: 10× less perturbation
noise[:, :, 1] *= 1.0    # thigh: full exploration
noise[:, :, 2] *= 0.8    # calf: moderate
```

Current hyperparameters:
```yaml
temperature: 10.0
σ_init: 0.02-0.045
σ_decay: 0.995-0.998
init_var_boost: 2.0
num_envs: 256-512
episode_length: 200-500
max_iterations: 1000-2000
CPG frequency: 1.0 Hz
```

---

### Reward Function (7 terms, hybrid LocoNets + B1 stability)

Based on LocoNets' simple reward structure, adapted for B1's mass:

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| `forward_vel` | raw `vx` (linear) | 2.0 | Forward motion (dominant) |
| `flat_orient` | `‖gravity_xy‖²` | 2.0 | Body tilt (continuous, not threshold) |
| `height_error` | `\|h − 0.42\|` | 8.0 | Height maintenance (absolute) |
| `lin_vel_z` | `vz²` | 2.0 | Vertical bouncing |
| `yaw_rate` | `ω_yaw²` | 0.1 | Heading drift |
| `action_rate` | `‖target_t − target_{t-1}‖²` | 0.001 | Smoothness |
| `energy` | `Σ joint_vel²` | 1e-7 | Efficiency (scaled for B1 mass) |

**Why not LocoNets' 6-term reward directly?**
LocoNets uses threshold-based penalties (heading > 0.45? → penalty). This works for their
75mm gecko robot that can't fall over. B1 (50 kg) needs continuous orientation and height
penalties. The reward structure is LocoNets-inspired but adapted.

**Why no gait_match?**
With direct encoding (no phase offsets in CPG), gait_match rewards stepping-in-place without
forward motion. The robot exploits it. Task-based gaits don't need footfall enforcement.

**Per-task reward variations:**
- **Forward**: standard reward above
- **Fast**: higher w1 (3.0) to incentivize speed
- **Turn**: replace yaw penalty with yaw REWARD (encourage rotation)

---

### Phase 2: Residual Transition Learning

Phase 2 uses **residual learning** instead of full PPO. A baseline function handles
the bulk of the transition, and a small MLP learns per-leg corrections.

#### Baseline Function (phase-based linear ramp)
```python
cycles_elapsed = (φ - φ_trigger) / (2π)
α_baseline = min(1.0, cycles_elapsed / 3.0)    # 3 CPG cycles (~3s at 1 Hz)
# Same α for all legs — simple linear ramp from 0 to 1
```
This works even without the MLP (Δα=0 → pure linear transition).

#### Residual MLP
```python
# Input: extended state (44D)
state = [
    joint_pos_rel(12), joint_vel(12),           # proprioception
    body_orientation(4),                         # quaternion
    sin(φ), cos(φ),                              # CPG phase
    g_current_onehot(3), g_target_onehot(3),     # gait IDs
    α_baseline(4),                               # Same baseline for all legs (scalar, repeated 1→4 for network input)
    Δα_prev(4),                                  # previous correction
]

# MLP: [44 → 128 → 128 → 4], ReLU hidden, Tanh output
raw = MLP(state)                                  # (4,) unbounded
Δα = 0.2 * tanh(raw)                             # per-leg correction ∈ [-0.2, +0.2]
```

#### Per-Leg Output Blending
```python
# Final alpha per leg
α_final_L = clip(α_baseline + Δα_L, 0, 1)

# IMPORTANT: blend at OUTPUT level (after tanh), not W level.
# tanh is nonlinear: tanh(blend(W1,W2)) ≠ blend(tanh(W1), tanh(W2))
output_current = tanh(RBF @ W_current)   # (12,) bounded [-1, 1]
output_target  = tanh(RBF @ W_target)    # (12,) bounded [-1, 1]

# Per-leg blending on the 12D output vector:
blended[0:3]  = (1 - α_FL) · output_current[0:3]  + α_FL · output_target[0:3]   # FL
blended[3:6]  = (1 - α_FR) · output_current[3:6]  + α_FR · output_target[3:6]   # FR
blended[6:9]  = (1 - α_RL) · output_current[6:9]  + α_RL · output_target[6:9]   # RL
blended[9:12] = (1 - α_RR) · output_current[9:12] + α_RR · output_target[9:12]  # RR

# Joint command:
joint_target = default_joint_pos + blended    # still bounded since blend of [-1,1] ∈ [-1,1]
```

**Why per-leg (4D) instead of per-RBF (20D):**
- Per-RBF: controls WHEN in the cycle to blend (temporal). Same for all legs.
- Per-leg: controls WHICH legs transition first (spatial). Matches B1's asymmetry.
- B1's front/rear kinematic asymmetry means legs genuinely need different transition rates.
- 4D is much easier for a small MLP to learn than 20D.

#### Training (gradient-based, not RL)
```python
# Loss function
L = λ_torque · ‖τ - τ_baseline‖²       # minimize torque spikes
  + λ_track  · ‖θ_cmd - θ_actual‖²      # joint tracking
  + λ_orient · Var(roll, pitch)          # stability
  + λ_fall   · (1 - I(h > 0.2))         # don't fall
  + λ_smooth · ‖Δα_t - Δα_{t-1}‖²      # smooth corrections

# Optimizer: Adam, lr=3e-4
# Train through differentiable simulation (Isaac Lab)
# ~1-2 days training (faster than PPO's 5-7 days)
```

#### Phase 1 Gait Storage
Three frozen W matrices from Phase 1 (task-based, **indirect** encoding):
- **Walk/Forward**: W_walk ∈ ℝ^(20×3) — moderate speed locomotion
- **Trot/Fast**:    W_trot ∈ ℝ^(20×3) — high speed locomotion
- **Steer/Turn**:   W_steer ∈ ℝ^(20×3) — forward + yaw rotation

Each W is a single (20×3) matrix, 60 params, shared across all 4 legs (per-leg
timing comes from the KENNE phase offsets). Gait difference comes from both W
AND the `phase_offsets` in the YAML — walk uses lateral-sequence, trot uses
diagonal antiphase, steer inherits walk's offsets.

During Phase 2, BOTH W_current and W_target are evaluated through the CPG
pipeline independently, then per-leg blending happens at the output level
(see code snippet above).

---

## Encoding Experiments Summary

Ten configurations tested for Phase 1. Full results in README.md.

| #  | Encoding / Init                                 | Params | Walk result                              | Trot works? | Key finding |
|----|-------------------------------------------------|--------|------------------------------------------|-------------|-------------|
| 1  | Indirect + phase offsets (step-by-step osc.)    | 60     | 0.17 m/s ✓                               | **0.21 m/s ✓** | Only trot that works — shared W required for diagonal sync |
| 2  | Semi-indirect (no noise)                        | 120    | Unstable                                 | No          | Hip explosion |
| 3  | Semi-indirect (+ per-joint noise)               | 120    | 0.23 m/s ✓                               | No          | Walk improved, trot sync lost |
| 4  | Per-gait pairing                                | 120    | 0.16 m/s                                 | Weak        | Diagonal pair ≠ front/rear pair |
| 5  | Per-leg + unified 0.9/0.9 defaults              | 240    | Symmetric but slow                       | No (inverted) | Wrong defaults shift problem |
| 6  | Per-leg + original 0.8/1.0 defaults             | 240    | 0.16 m/s                                 | No (L/R asym) | Shared W constraint needed |
| 7  | Direct (LocoNets-style, no phase offsets)       | 240    | 0.5-0.8 m/s, lunges + falls              | Not tested  | PIBB can't discover timing from scratch |
| 8  | PPO + direct W head (RSL-RL)                    | 240 W  | Bang-bang W → motor saturation, flips    | —           | Unbounded policy output + 50 Hz control = instability; clamping + rate penalty didn't fix |
| 9  | Direct + LocoNets pre-compute, shared phase     | 240    | 0.28 m/s mean, FL/RR near-static         | —           | Per-leg phase never applied (`_leg_step_offsets` unused bug) |
| 10 | Direct + LocoNets pre-compute, per-leg phase    | 240    | 0.04 m/s, FL planted 84%, face-trip      | —           | Per-leg phase works but 240 W params still too large to search |
| 11 | **Shared indirect + LocoNets KENNE + random init** | **60** | ~0.10 m/s, FL stalls after disturbance   | —           | W barely moved from init (norm 0.83 vs 0.45 at init) — PIBB stuck in starting basin |
| 12 | **Shared indirect + LocoNets KENNE + cosine init** *(current)* | **60** | *pending first run*                      | —           | Thor-style prior seeds W at a walking basin so PIBB only refines |

**Conclusion:** More parameters consistently breaks trot and (on B1) makes
walk harder for PIBB to discover. Shared indirect is the only encoding with
historical success on all three gaits, but needs a *walking-basin W_init*
to converge in finite iterations — cold random init leaves PIBB frozen.

**PPO on W directly (experiment #8) does not work**: unbounded policy output
at 50 Hz produces bang-bang W values, motors slam to limits, robot flips.
Clamping ±2.0 + `w_action_rate` + energy penalty at `1e-4` all tried — same
failure mode. Classical residual/high-level PPO (action = joint offsets with
CPG as a prior feed-forward) was NOT tested and may still be viable for
Phase 2 if residual-baseline approach fails; keep this in mind.

---

## Lessons Learned / Pitfalls

### Actuator Strength
B1 needs stiffness=200, effort=280 (not A1's 25/23.7). At A1 values, joints don't move at all.

### Joint Axis Convention
Hip is ABDUCTION, not forward swing. Thigh drives walking.

### W_init Phase
Must use `cos(angle)` for thigh, NOT `sin(angle)`. sin peaks at mid-stance → pushes body backward.

### CPG Frequency
0.3 Hz → 1.67s swing per leg (slow motion). 1.0 Hz → 0.5s swing (natural cadence).

### Action Scaling / tanh
Either use action_scale=0.25 (indirect) or tanh output (direct). Without bounding, W explosion.

### Offset from Default
`default_joint_pos + offset`. W=0 = standing pose, not collapsed joints.

### Foot Contact Detection
Use `.*_foot$` not `.*_calf$`. Calf is the lower leg bone, not the foot pad.

### Height Penalty
Use absolute error, not squared. Squared is negligible for small deviations.

### Velocity Reward Design
- Linear vx → robot runs as fast as possible (lunging)
- Gaussian exp(-4×(vx-target)²) → robot becomes unstable with direct encoding
- Clamped min(vx, target) → no penalty for going fast, doesn't slow down
- V-shaped vx - 2×max(0, vx-target) → penalizes overshoot but still not enough
- For indirect encoding, Gaussian at exp(-4×...) worked adequately

### gait_match Exploit
With direct encoding, gait_match rewards stepping-in-place without forward motion.
The robot scores +0.5/step from phase timing while vx ≈ 0. Must remove gait_match
when using direct encoding.

### LocoNets Reward Scaling
LocoNets uses torque penalty 2.5e-5 for a gecko robot (stiffness=1.0). B1 with
stiffness=200 produces 10,000× more torque → penalty must be 1e-7 not 2.5e-5.

### Semi-Indirect Breaks Trot
Front/rear pairing gives FL and RR different W → diagonal pair sync lost.
Only indirect (shared W) preserves trot's diagonal coordination.

### LocoNets KENNE: Centers On Trajectory, NOT Unit Circle
`RBF_i = exp(-‖s − c_i‖²/σ²)`. With α=1.01 + tanh, the limit cycle sits at
radius ~0.197, NOT 1.0. If you place centers on the unit circle
(`c_i = (cos(2πi/H), sin(2πi/H))`), every trajectory point is ~0.8 away
from the nearest center → `exp(-0.64/0.04) ≈ 10⁻⁷`. RBF activations are
effectively zero and W can't drive joints at all. Centers MUST be sampled
from the actual trajectory: `c_i = (x_traj[ci], y_traj[ci])`.

### Integer Phase Offsets, Not Runtime Radians
Converting YAML `phase_offsets` (radians) into runtime rotation matrices
accumulates floating-point drift and costs trig per step. Convert once
at startup into integer step offsets (`round(offset / (2π) * period)`)
and index into the KENNE table. Walk at period=51: `[0, 26, 13, 38]`.

### `_leg_step_offsets` Must Actually Be Applied
Computing leg offsets at startup is useless if `_step_cpg_batch` calls
`KENNE[self._phase_idx]` once and broadcasts to all joints — every leg
ends up on the same phase. Index KENNE per leg:
`KENNE[(phase_idx + leg_step_offsets[k]) % period]`. Experiment #9
failed for exactly this reason — the fix is one line.

### PIBB Needs A Good W_init on Heavy Robots
On B1 (50 kg), cold `uniform(-0.1, 0.1)` init leaves PIBB stuck near
origin for thousands of iterations. Every exploratory step has a fair
chance of tripping the robot, the orientation penalty dominates
rewards, and PI^BB's softmax update moves W almost nowhere. Seed W
with a Thor-style cosine prior so iteration 1 already walks (poorly)
and PIBB only has to refine — see "W Initialization" above.

### Cosine Prior Amplitude vs RBF Overlap
A naïve amplitude of W=0.20 on thigh produces ±40° motion, not ±12°,
because multiple RBFs are active simultaneously on the KENNE trajectory
(raw ≈ 4× W). Always verify amplitudes with a numpy simulation
(`tanh(KENNE @ W)`) before launching training. Current tuned values:
`W[:,1] = 0.05·cos(θ)`, `W[:,2] = 0.04·sin(θ)` → ±12° thigh, ±10° calf.

### Cosine Prior Phase: Calf LAGS Thigh
The walking cycle wants foot planted during stance and lifted during
swing. With `thigh = cos(θ)`, thigh is max-forward at θ=0 (touchdown)
and max-back at θ=π (lift-off). Calf (knee-flex) must be near zero at
touchdown/lift-off, maximally EXTENDED at mid-stance (θ=π/2), and
maximally FLEXED at mid-swing (θ=3π/2). That's `calf = sin(θ)` —
calf LAGS thigh by π/2. Using `cos(θ + π/2) = −sin(θ)` INVERTS the
cycle: foot lifts during mid-stance, pushes down during mid-swing,
giving net-zero stride (robot rocks in place at vx≈0 with 75% duty).
PIBB can't fix an inverted phase — it amplifies W without flipping
signs. Observed in a training run; fix was a one-character edit.

### PPO on W Directly Doesn't Work (Experiment #8)
Tried PPO where the policy outputs 240 W values. At 50 Hz control and
unbounded policy output, W oscillates wildly → tanh saturates → motors
snap limit-to-limit → robot flips in ~4 steps. Tried W clamping ±2.0,
action-rate penalty, energy penalty up to 1e-4 — all failed the same
way. Policies also found degenerate exploits (e.g. hold both left legs
in the air, spin using right-side pair). Don't put PPO in the W loop;
if PPO is needed for Phase 1 in the future, target joint offsets with
a CPG feed-forward bias — NOT W directly.

### Per-Joint Noise Scaling
Hip gets 10× less PIBB noise (scale=0.1) than thigh (1.0) and calf (0.8).
Prevents hip explosion in any encoding with >60 params.

### Logging
Isaac Lab reconfigures Python root logger. Use `print()` for training progress.
TensorBoard logs go to timestamped subdirectories for run comparison.

---

## Quick Reference Commands

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

python -m pytest tests/ -v                          # unit tests
python scripts/test_env.py --headless                # live env test
python scripts/train_phase1_walk.py --headless       # train forward task
python scripts/train_phase1_trot.py --headless       # train fast task
python scripts/train_phase1_bound.py --headless      # train turn task
python scripts/play_gait.py --gait walk              # play with gait diagram + reward breakdown
python scripts/diagnose_cpg.py --headless            # debug W/joints
python scripts/visualize_cpg.py                      # CPG phase plots
tensorboard --logdir logs/phase1                     # training curves
```

---

## Implementation Progress

### Week 10: Setup ✅
- [x] Isaac Lab + Isaac Sim verified
- [x] CPG-RBF network — 21/21 tests
- [x] Environment wrapper — 44/44 tests (23 env + 21 CPG)
- [x] Live environment test — 7/7 checks
- [x] CPG phase visualization
- [x] YAML configs for all 3 gaits

### Week 11: Phase 1 Training (IN PROGRESS)
- [x] PI^BB optimizer (Thor et al. softmax, per-joint noise)
- [x] B1 actuator fix (stiffness 25→200)
- [x] Reward iteration (11-term → 6-term → 7-term hybrid)
- [x] Encoding experiments: indirect → semi-indirect → per-leg → direct
- [x] LocoNets analysis: tanh output, direct encoding, simpler reward
- [x] PPO-on-W detour (experiment #8) — failed, kept as `scripts/train_ppo_walk.py` reference
- [x] LocoNets pre-computed CPG (KENNE lookup, integer phase, centers on trajectory)
- [x] Per-leg integer phase offsets wired into `_step_cpg_batch`
- [x] Revert to shared indirect (20×3) encoding — 60 params
- [x] Cosine walking prior for W_init (configurable via `pibb.init_mode`)
- [x] Tests updated for new shapes (44/44 passing)
- [ ] Train forward/fast/turn task-based gaits (cosine prior run pending)
- [ ] Gait quality validation via play + duty factors

### Week 12-15: Remaining
- [ ] Baseline validation (4 transition tests)
- [ ] Phase 2 residual transition learning (per-leg Δα MLP)
- [ ] Analysis and documentation

---

**Last Updated:** 2026-04-24
**Status:** Week 11 — Training task-based gaits with **shared indirect (20×3) encoding + LocoNets pre-computed KENNE + Thor cosine W_init prior**. About to launch first cosine-prior training run on walk; if successful, replicate for trot/steer. Phase 2 stays as residual learning with (20×3) W matrices.
