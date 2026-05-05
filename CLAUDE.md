# CLAUDE.md — Project Context for AI Assistant

**Project:** CPG-DRL Transition Control
**Full title:** Transition-Aware CPG-RBF Locomotion Control with DRL-Based Continuous Gait Switching
**Student:** Disthorn Suttawet (66340500019)
**Course:** FRA 503 Deep Reinforcement Learning
**Timeline:** Weeks 10–15 (6 weeks)
**Status (2026-04-26):** Phase 1 complete — 4 PPO velocity-tracking policies trained with distinct leg-pair coordination patterns. Phase 2 scaffolding pending.

---

## Current Project State (READ FIRST)

### Phase 1 — DONE

Four working PPO policies, saved at `logs/phase1_final/`:

| Gait | Coordination | Source run |
|---|---|---|
| **trot.pt** | Diagonal pair sync (FL+RR / FR+RL) | `logs/ppo_b1/trot_v2/` |
| **bound.pt** | Fore-aft pair sync (FL+FR / RL+RR) | `logs/ppo_b1/bound_v4/` |
| **pace.pt** | Lateral pair sync (FL+RL / FR+RR) | `logs/ppo_b1/pace_v2/` |
| **steer.pt** | Asymmetric turning trot | `logs/ppo_b1/steer_v2/` |

These are PPO-trained **velocity-tracking policies** (NOT CPG-RBF). They output 12-D joint position offsets.

### Phase 2 — NEXT

Scaffold a per-leg residual MLP that blends between any two of the four base policies. See "Phase 2 Design" section below.

### CPG-RBF approach — ABANDONED (kept as legacy)

The original Phase 1 design used CPG-RBF + PI^BB. After 12 encoding experiments and ~3 weeks of iteration, this approach was abandoned in favor of pure PPO. The legacy code (`envs/unitree_b1_env.py`, `algorithms/pibb_trainer.py`, etc.) is kept intact for reference but is not the path forward. Phase 2's research contribution is unaffected — see "The CPG-RBF Story" below.

---

## Development Environment

### Hardware
- **GPU:** NVIDIA RTX 4070 Ti SUPER (16.7 GB VRAM)
- **CPU:** Intel i5-14400F (10 cores, 16 threads)
- **RAM:** 32 GB
- **OS:** Ubuntu

### Software stack
- **Python:** 3.10.19
- **PyTorch:** 2.5.1+cu121
- **Isaac Lab:** 0.36.3 (`~/IsaacLab/`)
- **Isaac Sim:** 4.5.0.0
- **isaaclab-tasks:** 0.10.27
- **isaaclab-rl:** 0.1.3
- **rsl-rl-lib:** 2.2.4 (note: stock train.py wants 2.3.0 but 2.2.4 has compatible API)
- **Conda env:** `env_isaaclab` (always activate)

### Robot platform
- **Model:** Unitree B1 quadruped (12 DOF: 4 legs × 3 joints)
- **Mass:** ~50 kg
- **Asset:** `UNITREE_B1_CFG` from `isaaclab_assets.robots.unitree`
- **USD:** `~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd`
- **Trunk body name:** **`trunk`** (NOT `base` like Go2 — important for body_names parameters!)
- **Foot link names:** `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot` (use `.*_foot$` regex, not `.*_calf`)
- **Terrain:** Flat plane only

---

## B1-Specific Configuration

### Joint axis convention (CRITICAL)

```
hip_joint   → ABDUCTION (lateral splay)
              Default: FL=+0.1, FR=-0.1 (mirror left/right)
              Role: minor lateral balance, kept small in policies

thigh_joint → FLEXION (forward-backward swing) ← PRIMARY WALKING JOINT
              Default: F[L,R]=+0.8, R[L,R]=+1.0 (front/rear asymmetric!)
              Role: dominant for forward locomotion

calf_joint  → KNEE BEND
              Default: -1.5 rad (-86°)
              Role: foot clearance during swing
```

The +0.2 rad asymmetry between front-thighs (0.8) and rear-thighs (1.0) is the source of several emergent policy quirks (rear-heavy duty in bound, front-leap pattern in initial bound attempts). This is morphological, not fixable in software.

### Actuator config (PROJECT-LOCAL deep-copy of UNITREE_B1_CFG)

The shared `UNITREE_B1_CFG` from isaaclab_assets uses original specs (stiffness=200, effort=280). Our project does a **`copy.deepcopy`** at module level in `envs/b1_velocity_env_cfg.py` and overrides:

```python
UNITREE_B1_CFG.actuators["base_legs"].stiffness = 400.0     # was 200 — 200 sags 9 cm under body weight
UNITREE_B1_CFG.actuators["base_legs"].damping   = 10.0       # was 5 — proportional ratio
UNITREE_B1_CFG.init_state.pos[2] = 0.50                      # was 0.42 — feet were 7.7 cm under ground at default joints
```

**CRITICAL: never mutate the shared isaaclab_assets cfg directly** — it would silently break the legacy CPG-RBF env. Deep-copy first, then mutate.

### Body name override

Stock `LocomotionVelocityRoughEnvCfg` uses `body_names="base"` (Go2's renamed trunk). B1's USD keeps the URDF original `trunk`. **Every** `body_names="base"` inherited from the parent class needs override:

```python
self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
```

If you ever see "ValueError: Cfg requires 1 body but 0 matched" — this is the cause.

### Domain randomization scaled for B1

| Param | Stock (Go2) | B1 override | Why |
|---|---|---|---|
| `add_base_mass` distribution | (−5, 5) kg | **(−10, 10) kg** | Proportional to 50 kg mass |
| `reset_base.velocity_range` (z, roll, pitch) | ±0.5 | **0** | 50 kg body at 0.5 m/s z-vel face-plants in ~100 ms |
| `reset_robot_joints.position_range` | (0.5, 1.5) | **(1.0, 1.0)** | Spawn at default; ±50% randomization triggers immediate falls |
| `base_contact.threshold` | 1 N | **50 N** | Settling produces 20-40 N transients on 50 kg body |
| `env_spacing` | 2.5 m | **3.5 m** | B1 footprint is ~1.7× Go2's |
| `gpu_max_rigid_patch_count` | 10·2¹⁵ | **20·2¹⁵** | Larger bodies → more contact patches |

---

## Phase 1 PPO Architecture

### High-level

- **Manager-based RL** via `isaaclab_tasks.manager_based.locomotion.velocity.LocomotionVelocityRoughEnvCfg` (parent)
- **Algorithm:** PPO via RSL-RL `OnPolicyRunner`
- **Action space:** 12-D joint position offsets, scaled 0.25, added to default joint pose
- **Observation space:** ~48-D (base velocities, projected gravity, velocity command, joint pos/vel rel, last action)
- **Control rate:** 50 Hz (sim dt 0.005 s × decimation 4)
- **Episode length:** 20 s (1000 control steps)

### PPO hyperparameters (`envs/b1_velocity_ppo_cfg.py`)

```python
B1FlatPPORunnerCfg:
    num_steps_per_env = 24
    max_iterations = 3000           # default; per-gait overrides via --max_iterations
    save_interval = 50
    policy:
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        init_noise_std = 1.0
        activation = "elu"
    algorithm:
        clip_param = 0.2
        entropy_coef = 0.005        # was 0.01; reduced after smoke showed bang-bang noise risk
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
```

These match unitree_rl_lab's `BasePPORunnerCfg` for Go2 with `entropy_coef` halved as a precaution against B1's heavier dynamics.

### Reward stack — base config (`B1FlatEnvCfg`)

Inherited from stock `LocomotionVelocityRoughEnvCfg` plus:

| Term | Weight | Purpose |
|---|---:|---|
| `track_lin_vel_xy_exp` (stock) | +1.5 (overrides 1.0) | Forward velocity tracking |
| `track_ang_vel_z_exp` (stock) | +0.75 (overrides 0.5) | Yaw tracking |
| `feet_air_time` (stock) | +0.25 | Encourage swings ≥ threshold |
| `flat_orientation_l2` (stock) | −2.5 | Body upright |
| `lin_vel_z_l2` (stock) | −2.0 | Vertical motion |
| `ang_vel_xy_l2` (stock) | −0.05 | Roll/pitch rate |
| `dof_acc_l2` (stock) | −1.25e-7 (halved from -2.5e-7) | Joint acc smoothness |
| `dof_torques_l2` (stock) | −1e-6 (was -2e-4) | **Critical** — Go2's value would saturate B1 |
| `action_rate_l2` (stock) | −0.01 (back from -0.025) | Action smoothness |
| `base_height_l2` | **−50** | Body at 0.42 m |
| `hip_deviation_l1` | **−0.2** | Hip near default (penalty on `*_hip_joint` only) |
| `feet_slide` (stock) | **−0.25** | No dragging planted feet |
| `excessive_air_time` | **−1.0** (max_air=0.5 s) | No leg held airborne forever |
| `excessive_contact_time` | **−1.0** (max_contact=0.5 s) | No leg planted forever |
| `short_swing_penalty` | **−2.0** (min_swing=0.3 s) | No tap-tap-tap exploit |
| `air_time_variance_penalty` | **−1.0** | Force all 4 legs to cycle at similar rates |
| `joint_lr_symmetry_penalty` | **−0.02** | L/R bilateral symmetry |
| `undesired_contacts` | (removed) | Stock pattern was Anymal-style `.*THIGH` |

Custom terms in [envs/b1_velocity_mdp.py](envs/b1_velocity_mdp.py).

### Per-gait overrides

Each gait config overrides the base with gait-specific knobs. Below: full state per gait.

#### `B1FlatTrotEnvCfg` (current trot baseline)
- Command range: `lin_vel_x=(0.4, 0.7)`, `ang_vel_z=(-0.2, 0.2)`
- Inherits all base rewards
- Result: clean diagonal trot, duty 30/65/35/65, body 0.43, vx 0.48 m/s

#### `B1FlatBoundEnvCfg` (after 4 reward iterations)
- Command range: `lin_vel_x=(0.3, 0.7)`
- **Phase-match:** offsets `[0, 0, 15, 15]`, period 30, stance 0.5, weight 2.0
- **`true_bound_reward` (custom):** anti-trot, pro-bound — score `(front_pair_sync + rear_pair_sync) − (FL+RR sync + FR+RL sync)`. Pure trot pays −2/step, pure bound +2/step. Weight 3.0.
- **Stability relaxed** (bound has natural fore-aft pitch):
  - `lin_vel_z_l2` weight: −2.0 → **−0.5**
  - `ang_vel_xy_l2`: −0.05 → **−0.02**
  - `flat_orientation_l2`: −2.5 → **−0.5**
- **`base_height_l2` tightened:** −50 → **−150** (was used at -50 originally; squat 6.5 cm at v3, fixed at v4)
- **`duty_factor_target_penalty` (custom):** target=0.5, weight −1.0 (closes lock-pair exploit)
- **`bound_coordination_reward`:** weight 0 (legacy, superseded by phase-match + true_bound)

#### `B1FlatPaceEnvCfg` (one-shot success thanks to bound recipe)
- Command range: `lin_vel_x=(0.3, 0.6)`
- **Phase-match:** offsets `[0, 15, 0, 15]`, period 30, stance 0.5, weight 2.0
- **`true_pace_reward` (custom):** anti-trot, pro-pace — same XOR structure as true_bound but for lateral pairs. Weight 3.0.
- **Stability relaxed** (pace has body roll):
  - `lin_vel_z_l2`: −1.5 (back from −0.5 after foot apex was 27 cm; pace_v2 baseline used −0.5)
  - `ang_vel_xy_l2`: −0.02
  - `flat_orientation_l2`: −0.5
- **`base_height_l2`:** −150 (prevent squat)
- **`joint_lr_symmetry_penalty` DISABLED:** weight = 0
  - **CRITICAL:** pace has inherent instantaneous L/R asymmetry (left side legs swing while right side stance). The bilateral L/R penalty would constantly fight pace coordination. This was the missing ingredient that made pace work first try with the bound recipe.
- **`duty_factor_target_penalty`:** target=0.5, weight −3.0
- **`pace_coordination_reward`:** weight 0 (legacy)

#### `B1FlatSteerEnvCfg`
- Command range: `lin_vel_x=(0.1, 0.4)`, `ang_vel_z=(0.4, 1.0)`
- Inherits trot rewards
- **`joint_lr_symmetry_penalty` DISABLED:** weight = 0 (asymmetric L/R is required for turning)
- `air_time_variance_penalty.weight = -0.3` (relaxed)

#### `B1FlatWalkEnvCfg` (NOT in final set — failed multiple iterations)
- Command range: `lin_vel_x=(0.1, 0.25)`
- **Phase-match:** offsets `[0, 25, 12, 37]`, period 50, stance 0.75, weight 2.0
- **`true_walk_reward` (custom):** +1 when exactly 3 feet in stance
- **`must_move_penalty` (custom):** −3 when commanded but not moving
- After 6 versions, walk converged either to slow trot, rapid mincing, FR-pivot 3-leg-walk, or low squat. Decision was made to ship without walk and use bound/pace as the alternative coordinations.

### Custom MDP functions (`envs/b1_velocity_mdp.py`)

10 custom reward functions developed across the engineering iterations:

```python
joint_lr_symmetry_penalty(env, asset_cfg)
    → bilateral L/R motion-magnitude difference within front and rear pairs

excessive_air_time(env, sensor_cfg, max_air_time)
    → penalty for foot airborne longer than max_air_time

excessive_contact_time(env, sensor_cfg, max_contact_time)
    → penalty for foot planted longer than max_contact_time

short_swing_penalty(env, sensor_cfg, min_swing_time)
    → penalty at touchdown if last_air_time < min_swing_time

air_time_variance_penalty(env, sensor_cfg)
    → variance of last_air_time across 4 feet

duty_factor_target_penalty(env, sensor_cfg, target)
    → per-foot (last_contact / (last_contact + last_air) − target)²

bound_coordination_reward(env, sensor_cfg)         # legacy, superseded
pace_coordination_reward(env, sensor_cfg)          # legacy, superseded

true_bound_reward(env, sensor_cfg)
    → (FL+FR sync + RL+RR sync) − (FL+RR sync + FR+RL sync) ∈ [−2, +2]

true_pace_reward(env, sensor_cfg)
    → (FL+RL sync + FR+RR sync) − (FL+RR sync + FR+RL sync) ∈ [−2, +2]

true_walk_reward(env, sensor_cfg)
    → +1 when exactly 3 feet in contact, 0 else

gait_phase_match_reward(env, sensor_cfg, leg_phase_offsets, stance_fraction, period_steps)
    → uses env.episode_length_buf as phase clock; +1 per leg whose contact state
      matches target schedule; max +4/step

must_move_penalty(env, command_name, asset_cfg, cmd_threshold, actual_threshold)
    → +1 when |cmd_xy| > 0.05 but |actual_xy| < 0.05
```

---

## The CPG-RBF Story (legacy, why it didn't work)

The original Phase 1 design followed Thor et al. 2021's CPG-RBF + PI^BB framework. After 12 encoding experiments across ~3 weeks (Week 10 + first half of Week 11), the approach was abandoned. The lessons are still relevant for understanding Phase 2's design.

### The architecture

```
SO(2) oscillator: o(t+1) = tanh(α · R(Δφ) · o(t)),  α=1.01, Δφ=2π·f·dt
                  → integrate one period at startup → KENNE lookup table (period+1, 20)
                  → no trig in hot loop, phase is integer

RBF layer:        20 Gaussian neurons with σ²=0.04
                  centers c_i = (x_traj[ci_i], y_traj[ci_i])  ← from trajectory, NOT unit circle
                  KENNE[t, i] = exp(−‖(x_traj[t], y_traj[t]) − c_i‖² / σ²)

Encoding:         shared indirect (20 × 3): same W for all 4 legs
                  per-leg timing comes from integer phase offsets in KENNE indexing

Output:           tanh(rbf_leg @ W) → joint offsets bounded [−1, 1] rad

Optimizer:        PI^BB (softmax-weighted black-box gradient):
                  s_i = exp(h · (R_i − R_min) / (R_max − R_min))
                  p_i = s_i / Σ s_i
                  W += Σ_i p_i · ε_i  (weighted perturbation)
                  σ *= decay
```

### 12 encoding experiments

| # | Encoding | Result |
|---|---|---|
| 1 | Indirect (20×3) + step-by-step oscillator | Walk 0.17 m/s ✓, Trot 0.21 m/s ✓ — **best CPG result ever** |
| 2 | Semi-indirect (no noise) | Hip joint explodes to 54° (isotropic noise grows hip in the easy direction) |
| 3 | Semi-indirect + per-joint noise | Walk 0.23 m/s, but trot diagonal sync broken |
| 4 | Per-gait pairing (front-pair / rear-pair) | Diagonal pair ≠ front/rear pair → trot fails |
| 5 | Per-leg + unified 0.9/0.9 thigh defaults | Symmetric but inverted (rear at 0.9 too retracted) |
| 6 | Per-leg + original 0.8/1.0 defaults | L/R asymmetric, no symmetry constraint |
| 7 | Direct (LocoNets-style, 20×12) + PIBB | 0.5–0.8 m/s lunging + falls (240 params too many) |
| 8 | PPO outputting W directly | Bang-bang motor saturation, robot flips in 4 steps |
| 9 | Direct + LocoNets KENNE pre-compute, shared phase | FL/RR near-static (per-leg phase bug — `_leg_step_offsets` unused) |
| 10 | Direct + per-leg phase | FL planted 84%, face-trip falls |
| 11 | Shared indirect + KENNE + random init | W barely moved from init; PIBB stuck |
| 12 | Shared indirect + cosine W_init prior | Best CPG: 0.088 m/s, asymmetric, falls under disturbance |

### The deal-breakers

1. **B1 is too heavy for PIBB.** At 50 kg, every exploratory step risks a fall (large negative reward), and PI^BB's softmax update barely moves W. Cold init policies stay near origin for thousands of iterations.
2. **Direct encoding makes trot impossible.** Only shared-W indirect encoding produces stable diagonal coordination. But it's still vulnerable to morphological asymmetry exploits.
3. **Phase coupling is hard to discover from scratch.** Even with cosine walking prior (Thor's approach) and properly-tuned RBF activations, the policy converged to "lunge-fall-recover" rather than stable cyclic gait.
4. **PPO + W-as-action is catastrophic.** Encoding Experiment #8: at 50 Hz, unbounded policy output puts wild values into W → tanh saturates → motors snap to limits → robot flips. Clamping ±2.0 + action_rate penalty + energy penalty up to 1e-4 all tried — same failure mode every time.

### Critical CPG-RBF gotchas (preserved for reference)

- **RBF centers on trajectory, NOT unit circle.** Tanh oscillator's limit cycle is at radius ~0.197, not 1.0. Centers on unit circle → exp(−0.64/0.04) ≈ 10⁻⁷ activations → W can't drive joints.
- **Integer phase offsets, not runtime radians.** Convert YAML offsets once at startup: `int(round((offset / (2π)) · period))`.
- **`_leg_step_offsets` MUST be applied per leg.** Indexing `KENNE[phase_idx]` once and broadcasting to all joints breaks per-leg phase. Fix: `KENNE[(phase_idx + leg_offsets[k]) % period]`.
- **Cosine W_init phase: calf LAGS thigh by π/2.** `thigh = cos(θ)`, `calf = sin(θ)`. Using `cos(θ + π/2) = −sin(θ)` inverts the cycle → foot lifts during stance, giving net-zero stride.
- **Cosine W amplitude must account for RBF overlap.** Naive 0.20 produces ±40° (not ±12°) because multiple RBFs are active simultaneously on the trajectory. Tuned values: `W[:,1] = 0.05·cos(θ)`, `W[:,2] = 0.04·sin(θ)`.

### Why we pivoted to PPO velocity tracking

By Week 11 it was clear that perfecting CPG-RBF gait quality on B1 would consume the entire timeline. Phase 2's research contribution is per-leg residual transition learning — Phase 1 just needs working base policies. Whether they come from CPG-RBF or PPO is irrelevant for Phase 2's claim.

The pivot moved Phase 1 from "research-grade CPG-RBF tuning" to "engineering-grade PPO velocity tracking" — well-trodden territory (legged_gym, Isaac Lab stock tasks). Phase 2 stays exactly as designed, just blends between PPO actions instead of CPG W matrices.

---

## Phase 1 PPO Engineering Lessons

These are the failure modes encountered during PPO training, and the fixes that worked. Useful when adding new gaits or revising existing ones.

### Failure 1: Velocity tracking std too loose
- Default `track_lin_vel_xy_exp` uses `std=sqrt(0.25)=0.5` → at vx=0 with command=0.18, reward = exp(-0.123) = 0.88
- Policy learns to stand still and collect 88% of max velocity reward
- Fix: tighten std to 0.25 (sqrt(0.0625)) for slow-velocity gaits; bump weight 1.0 → 1.5 or 3.0

### Failure 2: Crawling exploit (body sags 0.18 m)
- No height penalty in stock cfg — Go2 doesn't need one (small robot)
- B1 with no height target sags to half default height while still tracking velocity
- Fix: `base_height_l2(target_height=0.42)` weight −50 (or −150 for bound, −200 for walk attempts)

### Failure 3: 2-leg trot pathology
- Without `excessive_*_time` constraints, policy converges to "diagonal pair planted, other diagonal permanently airborne"
- Fix: `excessive_air_time(max=0.5)` + `excessive_contact_time(max=0.5)` together force every foot into ≤1 s cycles

### Failure 4: Tap-tap-tap exploit
- Excessive_*_time uses cumulative `current_air_time` / `current_contact_time` which reset on transitions
- Policy taps a planted foot at ~5 Hz to reset the timer continuously
- Fix: `short_swing_penalty(min_swing_time=0.3)` — at each touchdown, penalize if last_air_time < 0.3 s

### Failure 5: 3+1 asymmetric trot
- All per-foot bounds OK individually, but FL cycles at half rate of FR/RL/RR
- Fix: `air_time_variance_penalty` — penalize variance of `last_air_time` across the 4 feet

### Failure 6: Bilateral L/R asymmetry
- After fixing variance, FL hip rotates 4° while FR hip rotates 9° — same per-leg "speed" but different per-leg roles
- Fix: `joint_lr_symmetry_penalty` — penalize `|FL_vel² − FR_vel²| + |RL_vel² − RR_vel²|`
- **NOTE:** disable for steer (asymmetric turning) and pace (lateral pair sync makes instantaneous L/R asymmetry inherent)

### Failure 7: Coordination-reward exploits
- Defining "bound" as "front pair == rear pair NOT" (XOR) → policy levitates (False==False=True)
- Defining as `(FL∧FR) ⊕ (RL∧RR)` (strict pair AND, then XOR) → policy locks one pair planted, flicks one leg of other pair
- **The truly bulletproof anti-trot bound reward:** `(FL∧FR) + (RL∧RR) − (FL∧RR) − (FR∧RL)` ∈ [−2, +2]. Penalizes trot diagonal sync, rewards bound pair sync, neutral on other states.
- Same pattern applied to pace via `true_pace_reward`.

### Failure 8: Stability terms fight body pitch/roll
- Bound has natural fore-aft pitch (body bobs during leap)
- Pace has natural side-to-side roll (body sways)
- Stock `flat_orientation_l2 = -2.5`, `lin_vel_z_l2 = -2.0`, `ang_vel_xy_l2 = -0.05` actively penalize these motions → policy compensates by squatting low to minimize body motion
- Fix: relax all three for bound and pace (typically −0.5 / −0.5 / −0.02). Tighten `base_height_l2` correspondingly to prevent the squat trade-off.

### Failure 9: Walking failed despite the recipe
- Walk's defining feature: 1-3-2-4 lateral sequence with 75% duty per leg
- `true_walk_reward = +1 if 3 feet in stance else 0` — fires for "3-stance flick" patterns too (stand still + flick one leg)
- `must_move_penalty` added — fires +1 when commanded > 0.05 m/s but actual < 0.05 m/s
- Even with these, policy converged to:
  - 6 Hz mincing (each leg cycles every 0.16 s — 4× target frequency)
  - 3-leg walk (FR planted at 98% as a pivot)
  - Low-squat 1 Hz uniform-duty walk (0.376 m height vs 0.42 target)
- After 6 versions, walk was dropped. Bound + pace + steer cover the coordination diversity needs.

### Trot is PPO's natural attractor on quadrupeds
- All "non-trot" gaits (bound, pace, walk) require explicit anti-trot rewards because PPO finds trot first.
- Bound succeeded after 4 iterations of reward tightening.
- Pace succeeded **first try** by applying the bound recipe wholesale (bound's trot-detection generalizes to lateral-pair-detection trivially).
- Walk never succeeded — likely a B1 morphology + reward landscape limitation, not a fixable bug.

---

## Phase 2 Design (NEXT)

### Architecture

```
                    ┌─────────────────────────────┐
                    │  Per-leg Residual MLP       │
                    │  [obs(48) → 128 → 128 → 4]  │
                    │  outputs Δα ∈ [-0.2, +0.2]  │
                    └─────────┬───────────────────┘
                              │ (Δα_FL, Δα_FR, Δα_RL, Δα_RR)
                              ▼
   π_current ─────┐    ┌──────────────────────┐
   π_target  ─────┼───▶│ Per-leg blending     │──▶ joint_targets → B1
   α_baseline ────┘    │ α_k = α_base + Δα_k  │
   (linear ramp 3 s)   └──────────────────────┘
```

### Action blending (post-policy, at output level)

```python
# For each control step:
output_current = π_current(obs)          # frozen base policy A
output_target  = π_target(obs)           # frozen base policy B
α_baseline = min(1.0, cycles_elapsed / 3.0)  # 3 CPG-equivalent cycles
for leg_k in {FL, FR, RL, RR}:
    α_k = clip(α_baseline + Δα_k, 0, 1)
    blended[3k:3k+3] = (1 − α_k) · output_current[3k:3k+3]
                     +      α_k  · output_target[3k:3k+3]
joint_target = default_joint_pos + 0.25 · blended
```

### Why per-leg matters for this gait set

Trot, bound, and pace have **fundamentally different leg-pair sync structures**:
- Trot: FL+RR diagonal, FR+RL diagonal
- Bound: FL+FR fore-aft, RL+RR fore-aft
- Pace: FL+RL lateral, FR+RR lateral

During trot→bound transition: FL was synced with RR (its diagonal partner). It must now sync with FR (its front-pair partner). RR's sync partner must change from FL to RL. **A scalar α can interpolate joint positions but cannot dynamically swap which legs sync with which** — that requires per-leg α values that can be temporarily asymmetric during the transition.

This is the architectural argument for Phase 2's residual MLP.

### Implementation plan (4 files, ~600 lines)

| File | Role |
|---|---|
| `envs/b1_phase2_env_cfg.py` | DirectRLEnv config — defines obs (48-D), action (4-D Δα), reward terms |
| `envs/b1_phase2_env.py` | DirectRLEnv class — loads 4 frozen policies, does per-leg blending |
| `envs/b1_phase2_ppo_cfg.py` | RSL-RL PPO config for residual MLP |
| `scripts/train_b1_phase2.py` | Training launcher |
| `scripts/play_b1_phase2.py` | Playback with manual gait switching |

### Open implementation questions (need user decision)

1. **Base policy loading:** standalone `.pt` (option 1, slower) or `torch.jit.ScriptModule` (option 2, faster). Recommend option 1 for first iteration.
2. **Transition scope:** train all 12 directed pairs from day 1, or trot↔bound only first as a smoke test?
3. **Reward design:** velocity tracking + stability + smoothness, or include a "transition cleanness" term that rewards minimal Δα magnitude during steady-state and allows large Δα only during transitions?

---

## Quick Reference Commands

```bash
# Setup
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Pre-flight (always before launching)
nvidia-smi && pgrep -f "python.*play\|python.*train\|isaac\|kit" | xargs -r kill -9

# Tests
python -m pytest tests/ -q                                 # 44/44

# Phase 1 — train
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Trot-v0 \
    --max_iterations 1500 --run_name trot_v2

python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Bound-v0 \
    --max_iterations 4000 --run_name bound_v4

python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Pace-v0 \
    --max_iterations 4000 --run_name pace_v2

python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Steer-v0 \
    --max_iterations 1500 --run_name steer_v2

# Phase 1 — play (each writes to logs/gait_diagram_ppo_b1.png)
python scripts/play_b1_velocity.py \
    --task Isaac-Velocity-Flat-Unitree-B1-{Trot,Bound,Pace,Steer}-Play-v0 \
    --checkpoint logs/phase1_final/{trot,bound,pace,steer}.pt \
    --num_envs 16 --steps 1000

# Tensorboard
tensorboard --logdir logs/ppo_b1
```

---

## Project Structure

```
cpg-drl-transition/
├── envs/
│   ├── unitree_b1_env.py           # CPG-RBF DirectRLEnv (legacy)
│   ├── b1_velocity_env_cfg.py      # PPO velocity-tracking env configs (CURRENT)
│   ├── b1_velocity_ppo_cfg.py      # RSL-RL PPO hyperparameters
│   ├── b1_velocity_mdp.py          # 10 custom reward functions
│   ├── b1_phase2_env_cfg.py        # (TODO) Phase 2 env config
│   ├── b1_phase2_env.py            # (TODO) Phase 2 env class
│   └── b1_phase2_ppo_cfg.py        # (TODO) Phase 2 PPO config
├── networks/
│   └── cpg_rbf.py                  # CPG-RBF network (legacy)
├── algorithms/
│   └── pibb_trainer.py             # PI^BB optimizer (legacy)
├── scripts/
│   ├── train_b1_velocity.py        # Phase 1 PPO train (CURRENT)
│   ├── play_b1_velocity.py         # Phase 1 PPO play with diagnostics (CURRENT)
│   ├── train_phase1_{walk,trot,bound}.py    # PIBB CPG training (legacy)
│   ├── play_gait.py                # CPG-RBF playback (legacy)
│   ├── visualize_cpg.py            # CPG phase visualization (legacy)
│   ├── train_b1_phase2.py          # (TODO)
│   └── play_b1_phase2.py           # (TODO)
├── configs/
│   └── phase1_{walk,trot,bound}.yaml    # CPG-RBF YAML configs (legacy)
├── logs/
│   ├── ppo_b1/<run_name>/          # PPO training output (per-run dirs)
│   ├── phase1_final/{trot,bound,pace,steer}.pt   # Final 4 policies
│   └── gait_{walk,trot,bound,pace,steer}.png   # Saved gait diagrams
├── weights/                        # CPG-RBF W matrices (.npy, legacy)
├── tests/                          # 44 unit tests
├── README.md                       # User-facing documentation
├── CLAUDE.md                       # This file
└── pytest.ini
```

---

## Critical Pitfalls / Gotchas

1. **Trunk body name is `trunk`, not `base`.** Override every `body_names="base"` inherited from stock cfgs.
2. **Spawn z must be 0.50, not 0.42.** Default joint angles put feet 7.7 cm below ground; raise spawn or termination triggers.
3. **`base_contact.threshold` must be ≥ 50 N.** Settling produces 20-40 N transients on a 50 kg body.
4. **`dof_torques_l2` weight must be ~−1e-6 for B1**, not Go2's −2e-4. B1's effort 280 vs Go2's ~23 means torque-squared is ~150× larger.
5. **Never mutate shared `UNITREE_B1_CFG` directly.** Always `copy.deepcopy` first; the legacy CPG-RBF env imports the same shared cfg.
6. **Per-leg foot detection in playback:** use `current_contact_time > 0`, not single-frame net forces. Single-frame snapshots miss brief 1-2 step contacts.
7. **`joint_lr_symmetry_penalty` MUST be disabled for pace and steer.** Pace has inherent instantaneous L/R asymmetry; steer requires it.
8. **For coordination rewards: use anti-other-coordination, not just pro-target.** Pure trot needs to PAY a penalty to incentivize bound/pace, not just receive less reward. `true_bound_reward` and `true_pace_reward` use signed `(target_pair_sync − trot_pair_sync)`.
9. **Foot apex during swing** is determined by calf flexion, NOT body height. Bumping `lin_vel_z_l2` doesn't reduce foot lift; it makes the body squat instead.
10. **Phase-match reward target schedule** uses `env.episode_length_buf` as the phase clock — this means all envs in a parallel batch share the same target phase. That's intentional (ensures consistent target across envs) but means the reward is partial-match for any policy not aligned to that specific phase. Trot policy at bound target scores ~50% match (2/4 legs).

---

## Last Updated

**Date:** 2026-04-26
**Phase 1 status:** ✅ Complete — 4 PPO policies (trot, bound, pace, steer) trained and validated
**Phase 2 status:** Architecture decided, scaffolding pending user confirmation on:
  (a) base-policy loading approach (standalone .pt vs torch.jit)
  (b) transition scope (trot↔bound first vs all 12 pairs)
  (c) reward design (velocity tracking + stability + transition smoothness)
