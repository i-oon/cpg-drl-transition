# B1 Omnidirectional Velocity Tracking — PPO

**Robot:** Unitree B1 (12 DOF, ~63 kg)
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0
**Goal:** Single omnidirectional velocity-tracking policy controlled via joystick (vx, vy, ωz), deployable to real hardware.

---

## Quick Start

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Train
python scripts/train_b1_velocity.py --headless --num_envs 4096

# Play
python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/<run>/model_final.pt --num_envs 4
```

---

## Environment

**File:** `envs/b1_velocity_env_cfg.py`

Inherits `LocomotionVelocityRoughEnvCfg` (Isaac Lab). Flat terrain only.
Reward stack is based on the Go2 known-good baseline — only changes from Go2 are listed below.

### B1-Specific Fixes
| Issue | Fix |
|---|---|
| Go2 uses `base` body name, B1 uses `trunk` | Override all `body_names="base"` → `"trunk"` |
| Feet spawn 7.7 cm below ground at default angles | Spawn height +8 cm |
| PhysX settling contact transient triggers termination | Base contact threshold 1 N → 50 N |
| Go2 mass randomization ±5 kg is ±33% | Scale to ±10 kg for B1's ~50 kg body |
| Stock reset velocity ±0.5 m/s face-plants 50 kg body | Zero initial velocity |
| Stock joint reset ±50% causes spawn falls | ±5% of default pose |

### PD Gains
```python
stiffness = 300.0  # N·m/rad — hardware-validated standing reference
damping   = 5.0    # N·m·s/rad — matches real B1 standing ref (kd=10 was overdamped)
```
At max offset (0.25 rad): τ = 300 × 0.25 = 75 N·m — just under the 80 N·m hardware torque cap.
Uniform across all 12 joints (single `base_legs` group). Per-joint-type split deferred.

### Default Joint Positions
```
hip  :  FL/RL = +0.03 rad,  FR/RR = -0.03 rad   (hardware home pose)
thigh:  FL/FR = 1.08 rad,   RL/RR = 0.90 rad    (rear reduced from 1.08)
calf :  all   = -1.94 rad                         (hardware home pose)
```
Rear thigh reduced from hardware home (1.08 rad) because the rear hip sits 0.69 m behind
the front hip — at equal positive (backward) thigh angles the rear foot extends far further
behind the body than the front foot, producing a visually extended/backward rear leg posture.
0.90 rad is a first-try reduction; front stays at 1.08 rad (hardware home, looks correct).

### Command Ranges (Omnidirectional)
```python
lin_vel_x = (-0.5, 0.8)
lin_vel_y = (-0.5, 0.5)   # full lateral coverage
ang_vel_z = (-1.0, 1.0)   # real turning capability
rel_standing_envs = 0.05  # 5% standing for deceleration / yaw-in-place
```

### Reward Stack (current)

**From Go2 baseline (unchanged weights):**
| Term | Weight | Notes |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.5 | |
| `track_ang_vel_z_exp` | +0.75 | |
| `flat_orientation_l2` | −2.5 | |
| `feet_slide` | −0.1 | Body names changed to `.*_foot$` for B1 |
| `lin_vel_z_l2` | −2.0 | inherited |
| `ang_vel_xy_l2` | −0.05 | inherited |
| `dof_pos_limits` | −10.0 | inherited |

**From Go2 baseline (weight adjusted for B1 hardware):**
| Term | Go2 weight | B1 weight | Reason |
|---|---|---|---|
| `feet_air_time` | +0.1, thresh=0.5s | +0.1, thresh=**0.1s** | 0.3s unachievable for heavy legs; lowered to turn reward positive |
| `dof_acc_l2` | −2.5e-7 | **−1.25e-7** | Halved — heavy B1 legs need swing freedom |
| `dof_torques_l2` | −2e-4 | **−1.0e-6** | ~200× lighter — B1 motors 12× stronger than Go2 |
| `action_rate_l2` | −0.1 | **−0.01** | Go2 value crushes B1 learning (dominated reward) |

**B1-specific additions (not in Go2):**
| Term | Weight | Why added | When added |
|---|---|---|---|
| `base_height_l2` | −50.0, target=**0.53 m** | Matches sim natural default height (URDF trunk origin at default joints). Real mid-trunk in home pose is 0.55–0.58 m (URDF origin sits below visual center). Target raised 0.42→0.46→0.50→0.53 m as stance improved; 0.53 m eliminates forced crouching that overloaded rear calves | v4 |
| `excessive_air_time` | −1.0, max=**0.5s** | Catches permanently-airborne leg. 0.15s tried in v9 — conflicted with `feet_air_time` 0.1s threshold, policy escaped via rapid taps (<0.1s). Reverted to 0.5s; use `duty_factor_target` instead for duty bounding. | v6 |
| `duty_factor_target` | −2.0, target=**0.5** | Directly penalises per-leg duty deviation from 50%. At 25% duty: (0.25−0.5)²×4 feet=0.25/step ≈ 250/episode. No threshold conflict with `feet_air_time`. | v10 |
| `excessive_contact_time` | −1.0, max=0.5s | Catches permanently-planted leg exploit (observed: single leg at duty >80%) | v6 |
| `hip_deviation_l1` | −0.2, joints=`.*_hip` | Hip does lateral splay only — observed ±8-11° swing vs hardware default ±1.72°. Anchors hip near zero without conflicting with height target (calf adjusts for height). | v6 |
| `rear_thigh_deviation_l1` | −0.15, joints=`R._thigh` | Rear thighs drifting toward horizontal (observed: calf P2P 37-41° vs front 20-24° — policy satisfying height via calf bend instead of thigh). Geometrically safe: calf still free to achieve height target. | v7 |

**Removed from Go2:**
| Term | Reason |
|---|---|
| `undesired_contacts` | References `Head_*` links that B1 does not have |
| `joint_position_penalty` (full 12-joint) | Conflicts with `base_height_l2` — default pose gives 0.53 m but height target ≠ 0.53 m. The full penalty is geometrically incompatible. Per-joint-type deviation terms (`hip_deviation_l1`, `rear_thigh_deviation_l1`) are safe because they anchor joints that don't bear the height DoF (calf adjusts freely). |
| `joint_lr_symmetry_penalty` | Fires on every trot step (FL/FR always in opposite phases during trot) |

### Domain Randomization
| Randomization | Range | Mode | Covers |
|---|---|---|---|
| Trunk mass | ±10 kg | reset | Payload, sensor mounting |
| External force/torque | stock | interval | Disturbance |
| Push robot | stock | interval | Sudden shoves |
| **Actuator gains** | ±25% kp and kd | reset | Sim-to-real gain mismatch |
| Joint init pose | ±5% of default | reset | Starting configuration variety |

*Note: joint friction DR disabled — shape bug in this Isaac Lab version (`randomize_joint_parameters` with `slice(None)` produces `[N,1,12]` instead of `[N,12]`).*

---

## Training History

### What was tried and why

| Run | Key config | Outcome | Why changed |
|---|---|---|---|
| v1 | Go2 base, no extra terms | **Spider-man** — squat to 0.30 m, 3 legs mostly airborne (FL=0%, FR=0%, RL=0.6%) | Missing `joint_position_penalty` from Go2 config |
| v2 | + `joint_position_penalty` (−0.7) | **Tap-tapping** — height fixed to 0.53 m (too tall), all 4 legs cycling but 5 Hz micro-steps, foot apex 1.9–3.0 cm | Default URDF pose = 0.53 m, `joint_pos` pulls up; `feet_air_time` threshold 0.3 s unachievable |
| v3 | + `base_height_l2` (−50, target 0.42) + `joint_lr_symmetry` (−0.05) + `excessive_air_time` (0.5s) | **Conflict storm** — `joint_pos` vs `base_height_l2` fighting (−282 vs −202), reward total 1323 | `joint_pos` pulls to 0.53 m, height penalty pulls to 0.42 m — irreconcilable |
| v4 | Remove `joint_pos`, keep height + symmetry (−0.01) + air time | **3-leg exploit (RL)** — height solved (0.426 m ✓), but RL permanently airborne (duty 20%, mean air 1.9 s) | `joint_lr_symmetry` at −0.01 too weak; RL=20% vs RR=53% |
| v5 | + `excessive_contact_time` (0.5s) + symmetry → −0.05 + `feet_air_time` threshold 0.1s | **Symmetry penalty fights trot** — `joint_lr_symmetry` −255/episode (fires on every trot step since FL/FR always in opposite phases), body rocking | `joint_lr_symmetry` penalises instantaneous L/R differences which are always large in trot |
| v6 | Remove `joint_lr_symmetry`, keep `excessive_air_time` + `excessive_contact_time` + `hip_deviation_l1` (−0.2) | **4-leg trot** — all legs cycling (FL 60.6%, FR 61.0%, RL 48.3%, RR 45.1%), vel tracking err_v mean 0.031, height 0.509 m ✓. Rear thigh drooping (calf P2P 37-41° vs front 20-24°) | Add `rear_thigh_deviation_l1` to anchor rear thighs |
| v7 | Height target 0.50→**0.53 m** (sim natural default) | **Rear calf worse** (40→47° P2P); rear duty dropped 45→30%. Fine-tuning preserved the calf-dominant pattern; height increase required more calf travel at 0.53m. | Height alone doesn't fix rear leg geometry — deep local optimum requires retrain |
| v8 | Rear thigh default 1.08→**0.90 rad** + retrain from scratch | Rear thigh angle fixed (52° vs 62°), thigh P2P improved (RL 32°, RR 24° vs 13-18°). But rear duty regressed to 25-27% — `excessive_air_time` 0.5s never fires at natural gait swing ~0.18s | Tighten `excessive_air_time` 0.5→0.15s |
| v9 | `excessive_air_time` max 0.5s→**0.15s** | Tap-tap shuffling gait: air time mean 0.031–0.053s, duty 57–81%. Policy escaped by tapping < 0.15s — avoids penalty but barely lifts feet. Confirmed conflict with `feet_air_time` 0.1s threshold. | Revert to 0.5s; add `duty_factor_target_penalty` instead |
| **v10 (current)** | Revert `excessive_air_time` to 0.5s + add `duty_factor_target` (−2.0, target=0.5) | *Training planned* | `duty_factor_target` directly penalises per-leg duty deviation from 50% with no threshold conflict. |

### Key lessons

- **`joint_position_penalty` from Go2 is incompatible with a height target** — Go2's default pose = standing height, B1's default pose = 0.53 m. Once you add `base_height_l2`, remove the full 12-joint `joint_pos`. Per-joint-type anchors on joints that don't govern height (hip, thigh) are safe because calf can still adjust freely.
- **`feet_air_time` threshold must be achievable** — at threshold=0.3 s and natural gait frequency of 5 Hz (0.08 s air phases), every touchdown is penalised. Policy learns to never lift feet. Threshold 0.1 s is closer to actual air time.
- **`joint_lr_symmetry_penalty` fights trot** — trot inherently has FL/FR in opposite phases. Instantaneous L/R velocity differences are always large. Use `excessive_air_time` + `excessive_contact_time` instead.
- **3-leg exploits need dual bounding** — `excessive_air_time(0.5s)` catches permanently-airborne legs; `excessive_contact_time(0.5s)` catches permanently-planted legs. Both needed.

---

## Hardware Reference (Real B1)

From `references/b1_interface/`:

### Control Law
```
τ = kp × (q_des − q) − kd × q̇ + tau_feedforward
τ = clamp(τ, −80, +80)  N·m
```
Computed in **software at 50 Hz**, sent as pure torque (`Kp=0, Kd=0` in motor cmd). Not onboard PD — this is the main sim-to-real gap.

### Real B1 PD Gain Reference
| Mode | Hip kp | Hip kd | Thigh kp | Thigh kd | Knee kp | Knee kd |
|---|---|---|---|---|---|---|
| Standing (hardware-validated) | 300 | 5 | 200 | 5 | 300 | 5 |
| Dynamic locomotion (CPG-RBF) | 110 | 1.5 | 1150 | 1.5 | 125 | 0.7 |
| SDK default | 400 | 20 | 400 | 20 | 400 | 20 |

Dynamic gains not usable with PPO directly — thigh kp=1150 × action_scale=0.25 = 287 N·m >> 80 N·m cap.

### Joint Order (Unitree SDK)
```
[RF, LF, RH, LH] × [hip, thigh, calf]
= [FR, FL, RR, RL] × [hip, thigh, knee]
```
Left legs (LF, LH) flip hip direction sign. Offset and direction conversion handled by `myb1.cpp::control2robotspace()`. **Must verify mapping against Isaac Lab order before deployment.**

### Safety Limits
- Joint speed cutoff: 15.7 rad/s
- Temperature cutoff: 80°C
- Thigh safety window: [−1.2, +0.8] rad (tighter than URDF limits)
- Soft-start ramp: 10 s from zero to full gains on enable

### Control Frequency
50 Hz — matches Isaac Lab training frequency.

---

## Deployment

The deployment node lives in `Sim2Real-B1/b1_deployment/`. It reads `/B1/*` topics,
assembles the policy observation, runs inference at 50 Hz, and publishes joint targets.

### BEFORE FIRST DEPLOYMENT — one-time code fix needed

`/B1/imu_ang_vel` (gyroscope) is not yet published by `b1_interface`. This is required
for the `base_ang_vel` policy observation. Add it to `b1_ros2_interface.cpp` before any
hardware policy run:

```cpp
// In constructor — add publisher:
imu_gyro_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/B1/imu_ang_vel", 1);

// In publishRobotState() — add after body_rpy block:
auto gyro_msg = geometry_msgs::msg::Vector3();
gyro_msg.x = b1_->state.imu.gyroscope[0];
gyro_msg.y = b1_->state.imu.gyroscope[1];
gyro_msg.z = b1_->state.imu.gyroscope[2];
imu_gyro_pub_->publish(gyro_msg);

// In header (b1_ros2_interface.hpp) — add publisher declaration:
rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr imu_gyro_pub_;
```

Then rebuild: `cd ~/Sim2Real-B1/b1_ws && colcon build && source install/setup.bash`

Note: actuator net training data does NOT need this — angular velocity is body-level and
irrelevant to joint motor dynamics. Only needed at policy deployment time.

---

### Step 0 — Pre-flight checklist
- [ ] `/B1/imu_ang_vel` added to `b1_interface` and rebuilt (see above)
- [ ] Robot in **basic (low-level) mode**: remote `L2+B` → `L1+L2+START`
- [ ] PC wired to robot dock ethernet, static IP `192.168.123.162/24`
- [ ] Confirm reachability: `ping 192.168.123.10` (low-level board)
- [ ] `b1_interface` running in `mode:=real` (connects to `.10:8007`)
- [ ] Gains ramped: the deployment node must soft-start over **10 s** from zero to full kp/kd

---

### Step 1 — Joint order remap

The `/B1` API and Isaac Lab use **different joint orders**.

```
/B1 API order :  FR  FL  RR  RL   × [hip, thigh, calf]   (indices 0–11)
Isaac Lab order: FL  FR  RL  RR   × [hip, thigh, calf]   (indices 0–11)

Remap (API index → Isaac Lab index):
  FR hip/thigh/calf  (0,1,2)   → IL indices (3,4,5)
  FL hip/thigh/calf  (3,4,5)   → IL indices (0,1,2)
  RR hip/thigh/calf  (6,7,8)   → IL indices (9,10,11)
  RL hip/thigh/calf  (9,10,11) → IL indices (6,7,8)

Python one-liner (API → IL):
  IL_FROM_API = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
  obs_joint_pos = api_joint_pos[IL_FROM_API]

  IL_TO_API = [3,4,5, 0,1,2, 9,10,11, 6,7,8]   # same permutation (self-inverse)
  api_target = il_action[IL_TO_API]
```

---

### Step 2 — Coordinate space conversion

The `/B1` API uses **control space**. The policy uses **URDF/joint space**.

```
URDF = direction × control + offset

  Joint   | direction (left) | direction (right) | offset
  --------|-----------------|-------------------|--------
  hip     |      +1         |       -1          | +0.0    (±0.03 rad at home)
  thigh   |      +1         |       +1          | +0.85
  calf    |      +1         |       +1          | -1.56

Convert /B1/joint_position (control) → policy obs (URDF):
  hip_urdf   =  ±1 × hip_ctrl   + 0.0
  thigh_urdf =  +1 × thigh_ctrl + 0.85
  calf_urdf  =  +1 × calf_ctrl  - 1.56

Convert policy action (URDF) → /B1/joint_target (control):
  hip_ctrl   = (hip_urdf   - 0.0 ) / (±1)
  thigh_ctrl = (thigh_urdf - 0.85) / 1
  calf_ctrl  = (calf_urdf  + 1.56) / 1

Home pose check (verify before moving):
  control space : [0.17, 0.23, -0.38] per leg
  URDF space    : hip ±0.03 rad, front thigh 1.08 rad, rear thigh 0.90 rad, calf -1.94 rad

NOTE: rear thigh default is 0.90 rad (URDF) = 0.05 control — NOT 0.23 like front.
```

---

### Step 3 — Observation assembly (50 Hz)

Build the 1×N obs vector in this order (matches Isaac Lab training):

| # | Obs term | Source | Conversion needed |
|---|---|---|---|
| 1 | `base_lin_vel` (3) | Kinematic estimator (see Step 4) | body frame |
| 2 | `base_ang_vel` (3) | `/B1/imu_ang_vel` gyro topic — **must add to b1_interface** (SDK has `state.imu.gyroscope[3]`, not yet published) | body frame, rad/s |
| 3 | `projected_gravity` (3) | `/B1/body_rpy` → rotation matrix | see Step 5 |
| 4 | `velocity_commands` (3) | Joystick (vx, vy, ωz) | m/s, rad/s |
| 5 | `joint_pos` (12) | `/B1/joint_position` → URDF, minus default | Step 2 + subtract default |
| 6 | `joint_vel` (12) | `/B1/joint_velocity` | already rad/s |
| 7 | `actions` (12) | Previous policy output | stored in node |
| 8 | `foot_contact` (4) | `/B1/foot_contact` | **Raw force int → threshold to binary**: `(force > 20).astype(float)`, order FL FR RL RR |

**Total obs dim = 3+3+3+3+12+12+12+4 = 52**

**Why 50 Hz?** `b1_ros2_interface.cpp` publishes all `/B1/*` topics on a 20 ms wall timer
(50 Hz). UDP send/recv run at 1 kHz internally but state is only pushed to ROS at 50 Hz.
The policy was also trained at 50 Hz (sim `dt = 0.02 s`). The deployment node timer must
match: `control_period = 1.0 / 50`. Running faster than 50 Hz gains nothing because
observations only update at 50 Hz; running slower introduces phase lag.

---

### Step 4 — Kinematic velocity estimator

Estimates body linear velocity from foot Jacobians when feet are in contact.
No extra hardware needed — uses `/B1/joint_position`, `/B1/joint_velocity`, `/B1/foot_contact`, `/B1/body_rpy`.

```
For each foot i in contact:
    v_foot_world ≈ 0   (foot not slipping)
    v_body_world = −R_body × J_i(q) × q̇_i

v_body_estimate = mean over all feet in contact
v_body_body_frame = R_body^T × v_body_estimate

Fallback: if no feet in contact, hold last estimate (or integrate IMU accel short-term).
```

`R_body` = rotation matrix from `/B1/body_rpy` (roll r, pitch p, yaw y):
```python
R = Rz(y) @ Ry(p) @ Rx(r)   # extrinsic ZYX = intrinsic XYZ
```

`J_i(q)` = foot Jacobian for leg i. Compute analytically from B1 URDF link lengths
(thigh_length ≈ 0.35 m, calf_length ≈ 0.35 m) or use Pinocchio.

---

### Step 5 — Projected gravity from body_rpy

```python
# /B1/body_rpy gives (roll, pitch, yaw) in rad
R = rotation_matrix_from_rpy(roll, pitch, yaw)  # 3×3
gravity_world = [0, 0, -1]                       # unit gravity in world frame
projected_gravity = R.T @ gravity_world           # body-frame gravity vector
```

This is the 3-vector the policy was trained on.

---

### Step 6 — Safety

```
Before publishing any joint target:
  - Clamp each action to ±0.25 rad (action_scale) around the default joint pos
  - Torque cap: τ = kp × (q_des − q) − kd × q̇,  clamp to ±80 N·m
  - Joint speed: reject target if |q̇| > 15.7 rad/s on any joint
  - Thigh safety window: [−1.2, +0.8] rad control space, [−0.35, +1.65] URDF space
  - Emergency stop: any base contact > 50 N → send home pose, stop publishing

Soft-start (first 10 s after enable):
  scale = t / 10.0  (0 → 1)
  target = home_pose + scale × (policy_output − home_pose)
```

---

## File Structure

```
envs/
  b1_velocity_env_cfg.py   — environment + reward config
  b1_velocity_mdp.py       — custom reward functions (exploit fixes)
  b1_velocity_ppo_cfg.py   — PPO hyperparameters (RSL-RL)

scripts/
  train_b1_velocity.py     — main training entry point
  play_b1_velocity.py      — playback + gait diagnostics + PNG
  test_env.py              — environment sanity check

references/
  b1_interface/            — real B1 hardware SDK + ROS2 interface
  go2_velocity_env_cfg.py  — Go2 reference (known-good baseline)
  *.pdf                    — related papers

logs/
  ppo_b1/                  — training runs (timestamped)
  phase1_final/            — legacy Phase 1 checkpoints
```

---

## Known Issues / Next Steps

- [ ] **v7 training pending** — verify `rear_thigh_deviation_l1` fixes rear thigh drooping without introducing height-target conflict
- [ ] Test all command directions in playback (forward, backward, lateral, diagonal, turn)
- [ ] Verify policy survives gain DR range (kp ±25%) in playback
- [ ] Per-joint-type actuator split (hip/thigh/knee separate kp/kd) for better locomotion quality
- [ ] Terrain curriculum for real-world robustness
- [ ] Deployment bridge (after policy is ready)
