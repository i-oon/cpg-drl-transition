# B1 Omnidirectional Velocity Tracking — PPO

**Robot:** Unitree B1 (12 DOF, ~63 kg)
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0
**Goal:** Single omnidirectional velocity-tracking policy controlled via joystick (vx, vy, ωz), deployable to real hardware.

---

## Quick Start

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Train (heading-cmd variant — default)
python scripts/train_b1_velocity.py --headless --num_envs 4096

# Train (raw-wz variant)
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-RawWz-v0

# Resume from checkpoint
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --resume logs/ppo_b1/<run>/model_final.pt

# Play
python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/<run>/model_final.pt \
    --task <task-id> --teleop --follow_cam --num_envs 1

# Structured evaluation (all 13 command conditions, 20 episodes each → DEPLOY_LOG.md)
python scripts/eval_b1_velocity.py \
    --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \
    --stage S1 --headless --append
```

---

## Two Policy Variants

Two gym environments are registered, producing policies with different wz command interfaces:

| | `heading-cmd` | `raw-wz` |
|---|---|---|
| **Gym task** | `Isaac-Velocity-Flat-Unitree-B1-v0` | `Isaac-Velocity-Flat-Unitree-B1-RawWz-v0` |
| **Play task** | `Isaac-Velocity-Flat-Unitree-B1-Play-v0` | `Isaac-Velocity-Flat-Unitree-B1-RawWz-Play-v0` |
| **wz training** | `heading_command=True` — policy sees `K × heading_error` | `heading_command=False` — policy sees raw wz directly |
| **Inject wz in sim** | Set `heading_target` every step | Set `vel_command_b[:,2]` directly |
| **Inject wz on hardware** | `heading_target = current_yaw + wz / K` each tick | `vel_command_b[:,2] = wz` directly |
| **wz dead zone** | ~0.3 rad/s (from `yaw_stability` penalty) | ~0.3 rad/s (same penalty applies) |
| **Deploy complexity** | Higher — heading_target math required | Lower — direct injection |

The `play_b1_velocity.py` script handles both automatically — it sets `vel_command_b[:,2]` AND `heading_target` every step, so the correct one takes effect depending on which mode the policy was trained in.

---

## Current Checkpoints

| Checkpoint | Variant | Iters | Notes |
|---|---|---|---|
| `logs/ppo_b1/yaw_stable_BEST/model_final.pt` | heading-cmd | ~12000 | No falls. All 6 DOF. Heading-stable during strafe. |
| `logs/ppo_b1/raw_wz_FINAL/model_final.pt` | raw-wz | ~10000 | No falls. All 6 DOF. Simpler deployment. **Preferred for hardware.** |
| `logs/ppo_b1/turn_base/model_final.pt` | heading-cmd | 7000 | Pre-yaw-stability. Good turning. Useful as fine-tune base. |

---

## Environment

**File:** `envs/b1_velocity_env_cfg.py`

Inherits `LocomotionVelocityRoughEnvCfg` (Isaac Lab). Flat terrain only.
Reward stack is based on the Go2 known-good baseline — only B1-specific changes are listed below.

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
behind the body than the front foot. 0.90 rad is a modest reduction that visually corrects
the rear leg geometry without deviating from hardware home pose significantly.

### Command Ranges (Omnidirectional)
```python
lin_vel_x = (-0.5, 0.8)
lin_vel_y = (-0.5, 0.5)   # full lateral coverage
ang_vel_z = (-1.0, 1.0)   # real turning capability
rel_standing_envs = 0.05  # 5% standing for deceleration / yaw-in-place
```

### Reward Stack

**From Go2 baseline (unchanged weights):**
| Term | Weight | Notes |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.5 | |
| `track_ang_vel_z_exp` | **+1.5** | Raised from +0.75. At +0.75 turning was unprofitable against `duty_factor_target` at −2.0. At +1.5 it matches linear tracking and drives symmetric improvement. |
| `flat_orientation_l2` | −2.5 | |
| `feet_slide` | −0.1 | Body names changed to `.*_foot$` for B1 |
| `lin_vel_z_l2` | −2.0 | inherited |
| `ang_vel_xy_l2` | −0.05 | inherited |
| `dof_pos_limits` | −10.0 | inherited |

**From Go2 baseline (weight adjusted for B1 hardware):**
| Term | Go2 weight | B1 weight | Reason |
|---|---|---|---|
| `feet_air_time` | +0.1, thresh=0.5s | +0.1, thresh=**0.1s** | 0.5s unachievable for heavy legs; lowered to make reward positive |
| `dof_acc_l2` | −2.5e-7 | **−1.25e-7** | Halved — heavy B1 legs need swing freedom |
| `dof_torques_l2` | −2e-4 | **−1.0e-6** | ~200× lighter — B1 motors 12× stronger than Go2 |
| `action_rate_l2` | −0.1 | **−0.01** | Go2 value crushes B1 learning (dominated reward) |

**B1-specific additions (not in Go2):**
| Term | Weight | Why added |
|---|---|---|
| `base_height_l2` | −50.0, target=**0.53 m** | Policy crouched below sim natural height (0.53 m) to minimize other penalties. Explicitly anchors trunk height. Target raised iteratively 0.42→0.53 m as stance improved. |
| `excessive_air_time` | −1.0, max=**0.5s** | B1 found a stable exploit: balance on 3 legs and hold the 4th up indefinitely — still earns velocity reward. Penalizes any foot airborne beyond 0.5s. |
| `excessive_contact_time` | −1.0, max=**0.5s** | Symmetric exploit: drag one leg as a passive post while 3 others do all work. Penalizes any foot planted beyond 0.5s. Together with `excessive_air_time`, forces all four legs to cycle. |
| `duty_factor_target` | **−1.0**, target=**0.5** | Tight `excessive_air_time` threshold (0.15s) created a conflict with `feet_air_time` 0.1s threshold — policy escaped by tap-tapping just below 0.1s. This directly penalizes per-leg duty deviation from 50% using completed cycle times, no threshold to exploit. Weight −2.0→−1.0: at −2.0 turning was blocked (asymmetric gaits needed for turning were penalized too heavily). |
| `hip_deviation_l1` | −0.2, joints=`.*_hip` | B1 hips do lateral splay only; thigh/calf handle propulsion. Policy swung hips ±8–11° vs hardware default ±1.72°. Anchors hips near zero without conflicting with height target. |
| `rear_thigh_deviation_l1` | −0.15, joints=`R._thigh` | Policy satisfied height target by bending rear calves excessively rather than using the thigh (rear calf P2P 37–41° vs front 20–24°). Anchors rear thighs so calves can stride normally. |
| `yaw_stability` | **−1.0**, sigma²=**0.04** | During strafe the policy rotated its body to face the velocity vector instead of pure-sidelstepping. Penalizes yaw rate when wz command is near zero: `wz² × exp(−cwz²/0.04)`. Gate fades to 2% at \|cwz\|=0.4 — shuts off during real turn commands. **Creates ~0.3 rad/s wz dead zone** (intentional — prevents drift). First attempt (weight=−2.0, sigma²=0.09) caused falls; sigma too wide fought turns at cwz=0.3. |

**Removed from Go2:**
| Term | Reason |
|---|---|
| `undesired_contacts` | References `Head_*` links that B1 does not have |
| `joint_position_penalty` (full 12-joint) | Conflicts with `base_height_l2` — irreconcilable geometry. Per-joint-type anchors (`hip_deviation_l1`, `rear_thigh_deviation_l1`) are safe because they target joints that don't govern height. |
| `joint_lr_symmetry_penalty` | Fires on every trot step — FL/FR are always in opposite phases during trot |

### Domain Randomization
| Randomization | Range | Mode | Covers |
|---|---|---|---|
| Trunk mass | ±10 kg | reset | Payload, sensor mounting |
| External force/torque | stock | interval | Disturbance |
| Push robot | stock | interval | Sudden shoves |
| **Actuator gains** | ±25% kp and kd | reset | Sim-to-real gain mismatch |
| Joint init pose | ±5% of default | reset | Starting configuration variety |
| **Foot friction** | static 0.5–1.25, dynamic 0.4–1.0 | reset | Floor surface variability (tile → rubber mat) |
| **Joint friction** | +0.0–0.05 Nm·s/rad per joint | reset | Motor-to-motor friction variation |

*Note: joint friction DR uses three separate event terms (hip / thigh / calf) to work around a shape bug in this Isaac Lab version where `randomize_joint_parameters` with `slice(None)` produces `[N,1,12]` instead of `[N,12]`.*

---

## Training History

| Run | Key config | Outcome | Why changed |
|---|---|---|---|
| Go2 base | Go2 base, no extra terms | **Spider-man** — squat to 0.30 m, 3 legs mostly airborne | Missing `joint_position_penalty` from Go2 config |
| + joint_pos | + `joint_position_penalty` (−0.7) | **Tap-tapping** — 5 Hz micro-steps, foot apex 1.9–3.0 cm | `feet_air_time` threshold 0.3s unachievable |
| + height | + `base_height_l2` + `joint_lr_symmetry` + `excessive_air_time` | **Conflict storm** — `joint_pos` vs `base_height_l2` fighting | Irreconcilable geometry targets |
| remove joint_pos | Remove `joint_pos`, keep height + symmetry + air time | **3-leg exploit** — RL permanently airborne (duty 20%) | `joint_lr_symmetry` too weak |
| + contact_time | + `excessive_contact_time` + symmetry −0.05 + air time 0.1s | **Symmetry penalty fights trot** — −255/episode, body rocking | L/R differences always large in trot |
| remove symmetry | Remove `joint_lr_symmetry`, add `hip_deviation_l1` | **4-leg trot** ✓ — vel tracking 0.031 err, height 0.509 m. Rear thigh drooping. | Add `rear_thigh_deviation_l1` |
| height 0.53 m | Height target 0.50→0.53 m | **Rear calf worse** (40→47° P2P), rear duty dropped 45→30% | Deep local optimum — requires retrain |
| rear thigh 0.90 | Rear thigh default 1.08→0.90 rad + retrain | Rear thigh fixed. But rear duty 25–27% — `excessive_air_time` 0.5s never fires | Tighten to 0.15s |
| air_time 0.15s | `excessive_air_time` 0.5→0.15s | **Tap-tap shuffling** — policy escaped by tapping <0.1s (conflicts with `feet_air_time`) | Add `duty_factor_target` instead |
| + duty_factor | Revert air_time to 0.5s + `duty_factor_target` (−2.0) | Good gait. **Turning broken** — sidesteps instead of rotates. ang tracking 959 vs linear 1457 | Raise ang weight, lower duty weight |
| **turn_base** | `track_ang_vel_z_exp` +1.5, `duty_factor_target` −1.0. 7000 iters. | **Turning fixed** ✓ — all 6 DOF working. Heading drifts ±90° during strafe. | Add `yaw_stability_penalty` |
| yaw_stable FAILED | `yaw_stability` (−2.0, sigma²=0.09) from turn_base. ang weight →1.0. | **Falls** — base_contact 0.042. sigma too wide, fought turns at cwz=0.3. | Tighten sigma, restore ang weight |
| **yaw_stable_BEST** | `yaw_stability` (−1.0, sigma²=0.04), ang weight 1.5. Fine-tuned turn_base iter 7000→11997. | **No falls** ✓. Turning maintained. Strafe drift reduced. | Checkpoint: `yaw_stable_BEST` |
| **raw_wz_v1** | `heading_command=False` + foot/joint friction DR. 5000 iters from scratch. | **No falls** ✓. Linear 1450, angular 1448 — matches yaw_stable_BEST. Simpler deployment. | Continue training |
| **raw_wz_FINAL** | Continued from raw_wz_v1, iter 5000→9998. | **No falls** ✓. Same tracking quality. **Preferred for hardware deployment.** | Checkpoint: `raw_wz_FINAL` |

### Key Lessons

- **`joint_position_penalty` from Go2 is incompatible with a height target** — Go2 default pose = standing height, B1 default pose = 0.53 m. Once you add `base_height_l2`, remove the full 12-joint penalty. Per-joint-type anchors on joints that don't govern height are safe.
- **`feet_air_time` threshold must be achievable** — at 0.3s and natural gait ~0.08s air phases, every touchdown is penalized. Policy learns to never lift feet. 0.1s threshold works.
- **`joint_lr_symmetry_penalty` fights trot** — trot inherently has FL/FR in opposite phases. Use `excessive_air_time` + `excessive_contact_time` instead.
- **3-leg exploits need dual bounding** — `excessive_air_time` catches permanently-airborne legs; `excessive_contact_time` catches permanently-planted legs. Both needed.
- **Turning requires reward balance** — `track_ang_vel_z_exp` must match `track_lin_vel_xy_exp`. At 0.75 vs 1.5, turning is unprofitable when `duty_factor_target` penalizes asymmetric gaits.
- **`yaw_stability` sigma must be tight** — sigma²=0.09 still gates at 37% at cwz=0.3, fighting turns. sigma²=0.04 fades to 2% at cwz=0.4 — only active when the robot genuinely shouldn't rotate.
- **`yaw_stability` creates a wz dead zone** — the gate teaches the policy to suppress rotation for small cwz. Dead zone ≈ 0.3 rad/s. Apply dead zone + rescale in joystick mapping at deployment.
- **Raw-wz trains just as well as heading-cmd** — with the same reward stack, raw-wz reached identical tracking metrics in fewer iters and is significantly easier to deploy.

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
Left legs (LF, LH) flip hip direction sign. **Must verify mapping against Isaac Lab order before deployment.**

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

### Recommended policy for hardware

Use **`raw_wz_FINAL`** — identical tracking performance to `yaw_stable_BEST`, simpler wz injection (no heading_target math):

```
logs/ppo_b1/raw_wz_FINAL/model_final.pt
task: Isaac-Velocity-Flat-Unitree-B1-RawWz-v0
```

---

### BEFORE FIRST DEPLOYMENT — one-time code fix needed

`/B1/imu_ang_vel` (gyroscope) is not yet published by `b1_interface`. Required for `base_ang_vel` policy observation. Add to `b1_ros2_interface.cpp`:

```cpp
// In constructor:
imu_gyro_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/B1/imu_ang_vel", 1);

// In publishRobotState() after body_rpy block:
auto gyro_msg = geometry_msgs::msg::Vector3();
gyro_msg.x = b1_->state.imu.gyroscope[0];
gyro_msg.y = b1_->state.imu.gyroscope[1];
gyro_msg.z = b1_->state.imu.gyroscope[2];
imu_gyro_pub_->publish(gyro_msg);

// In header (b1_ros2_interface.hpp):
rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr imu_gyro_pub_;
```

Then rebuild: `cd ~/Sim2Real-B1/b1_ws && colcon build && source install/setup.bash`

---

### wz command injection

The two policy variants require different wz injection. **Always set `vel_command_b[:,2]` every control step** — the command manager periodically resamples commands (~every 10 s), which would override a one-time injection.

**raw-wz policy (recommended):**
```python
def inject_cmd(cmd_term, vx, vy, wz):
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    cmd_term.vel_command_b[:, 2] = wz  # direct — env.step() does NOT overwrite this
```

**heading-cmd policy:**
```python
_HEAD_K = 0.5  # heading_control_stiffness from parent config

def inject_cmd(cmd_term, robot, vx, vy, wz):
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = vy
    # env.step() overwrites vel_command_b[:,2] with K*heading_error — inject via heading_target
    current_heading = robot.data.heading_w
    if abs(wz) > 1e-4:
        cmd_term.heading_target[:] = current_heading + wz / _HEAD_K
    else:
        # Reset every step — prevents heading error from accumulating and firing
        # correction bursts at segment transitions (robot turns when wz=0)
        cmd_term.heading_target[:] = current_heading
```

**Symptoms of wrong injection (heading-cmd policy with direct vel_command_b[:,2]):**
- Robot ignores turn commands
- Robot turns unexpectedly at segment transitions
- Turning behavior appears random and unrelated to commands

### wz dead zone (both policies)

`yaw_stability_penalty` (sigma²=0.04) creates an effective dead zone below ~0.3 rad/s.
Apply dead zone + rescale in joystick mapping:

```python
def map_wz(joystick_val, max_wz=1.0, dead=0.3):
    if abs(joystick_val) < dead:
        return 0.0
    sign = 1.0 if joystick_val > 0 else -1.0
    return sign * (abs(joystick_val) - dead) / (1.0 - dead) * max_wz
```

---

### Step 0 — Pre-flight checklist
- [ ] `/B1/imu_ang_vel` added to `b1_interface` and rebuilt (see above)
- [ ] Robot in **basic (low-level) mode**: remote `L2+B` → `L1+L2+START`
- [ ] PC wired to robot dock ethernet, static IP `192.168.123.162/24`
- [ ] Confirm reachability: `ping 192.168.123.10` (low-level board)
- [ ] `b1_interface` running in `mode:=real` (connects to `.10:8007`)
- [ ] Gains ramped: deployment node must soft-start over **10 s** from zero to full kp/kd

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
  hip     |      +1         |       -1          | +0.0
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

| # | Obs term | Source | Conversion needed |
|---|---|---|---|
| 1 | `base_lin_vel` (3) | Kinematic estimator (see Step 4) | body frame |
| 2 | `base_ang_vel` (3) | `/B1/imu_ang_vel` gyro — **must add to b1_interface** | body frame, rad/s |
| 3 | `projected_gravity` (3) | `/B1/body_rpy` → rotation matrix | see Step 5 |
| 4 | `velocity_commands` (3) | Joystick (vx, vy, ωz) | m/s, rad/s |
| 5 | `joint_pos` (12) | `/B1/joint_position` → URDF, minus default | Step 2 + subtract default |
| 6 | `joint_vel` (12) | `/B1/joint_velocity` | already rad/s |
| 7 | `actions` (12) | Previous policy output | stored in node |
| 8 | `foot_contact` (4) | `/B1/foot_contact` | threshold to binary: `(force > 20).float()`, order FL FR RL RR |

**Total obs dim = 3+3+3+3+12+12+12+4 = 52**

Policy runs at 50 Hz (sim dt = 0.02 s). `/B1/*` topics publish at 50 Hz. Running faster gains nothing; running slower introduces phase lag.

---

### Step 4 — Kinematic velocity estimator

```
For each foot i in contact:
    v_foot_world ≈ 0   (foot not slipping)
    v_body_world = −R_body × J_i(q) × q̇_i

v_body_estimate = mean over all feet in contact
v_body_body_frame = R_body^T × v_body_estimate

Fallback: if no feet in contact, hold last estimate (or integrate IMU accel short-term).
```

`R_body` from `/B1/body_rpy` (roll r, pitch p, yaw y): `R = Rz(y) @ Ry(p) @ Rx(r)`

`J_i(q)` = foot Jacobian. Compute analytically (thigh_length ≈ 0.35 m, calf_length ≈ 0.35 m) or use Pinocchio.

---

### Step 5 — Projected gravity from body_rpy

```python
R = rotation_matrix_from_rpy(roll, pitch, yaw)  # 3×3
gravity_world = [0, 0, -1]
projected_gravity = R.T @ gravity_world           # body-frame gravity vector
```

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

## Sim-to-Real Evaluation

Deployment readiness is tracked in **`DEPLOY_LOG.md`** through 6 stages:

| Stage | Environment | Status |
|---|---|---|
| S1 | Isaac Sim 4.5 + Ideal PD | ✅ Complete — PASS |
| S2 | Isaac Sim 5.0 + Ideal PD | Pending |
| S3 | MuJoCo + Ideal PD | Pending |
| S4 | MuJoCo + Actuator net (sim) | Pending |
| S5 | MuJoCo + Actuator net (real data) | Pending |
| S6 | Real B1 | Pending |

**S1 reference numbers** (20 episodes × 1000 steps × 13 conditions):

| Metric | Value |
|---|---|
| Fall rate | 0.0% |
| e_vx (C3/C4) | 0.152 m/s |
| e_vy (C6/C7) | 0.121 m/s |
| e_wz (C10/C11) | 0.209 rad/s |
| Yaw drift (wz=0) | 3.01°/s |
| Height | 0.535 ± 0.006 m |
| Duty FL/FR/RL/RR | 57.6 / 58.1 / 49.9 / 51.3% |
| tau_sat (>100 N·m) | 13.7% |

Run evaluation for any stage:
```bash
python scripts/eval_b1_velocity.py \
    --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \
    --stage S1 --headless --append
```

---

## File Structure

```
envs/
  b1_velocity_env_cfg.py   — env + reward config (both heading-cmd and raw-wz variants)
  b1_velocity_mdp.py       — custom reward functions
  b1_velocity_ppo_cfg.py   — PPO hyperparameters (RSL-RL)

scripts/
  train_b1_velocity.py     — training entry point (--task selects variant)
  play_b1_velocity.py      — teleop + demo + video recording (works with both variants)
  eval_b1_velocity.py      — structured evaluation for DEPLOY_LOG.md (all 13 conditions)
  test_env.py              — environment sanity check

references/
  b1_interface/            — real B1 hardware SDK + ROS2 interface
  go2_velocity_env_cfg.py  — Go2 reference (known-good baseline)

logs/
  ppo_b1/
    yaw_stable_BEST/       — heading-cmd policy, ~12k iters
    raw_wz_FINAL/          — raw-wz policy, ~10k iters (preferred for hardware)
    raw_wz_v1/             — raw-wz policy, 5k iters (intermediate)
    turn_base/             — heading-cmd pre-yaw-stability (fine-tune base)

DEPLOY_LOG.md              — sim-to-real evaluation log (S1–S6 results + pass/fail)
```

---

## Play Script

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# --- raw-wz policy (recommended) ---

# Teleop
python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \
    --task Isaac-Velocity-Flat-Unitree-B1-RawWz-Play-v0 \
    --teleop --follow_cam --num_envs 1

# Demo
python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \
    --task Isaac-Velocity-Flat-Unitree-B1-RawWz-Play-v0 \
    --demo --follow_cam --num_envs 1

# Demo + record video
python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/raw_wz_FINAL/model_final.pt \
    --task Isaac-Velocity-Flat-Unitree-B1-RawWz-Play-v0 \
    --demo --follow_cam --num_envs 1 --video logs/videos/

# --- heading-cmd policy ---

python scripts/play_b1_velocity.py \
    --checkpoint logs/ppo_b1/yaw_stable_BEST/model_final.pt \
    --task Isaac-Velocity-Flat-Unitree-B1-Play-v0 \
    --teleop --follow_cam --num_envs 1
```

> **Always pass `--task` to match the policy variant.** Using the wrong task causes incorrect wz behavior — the policy receives the wrong command type.

### Teleop keys
| Key | Action |
|---|---|
| W / S | forward / backward (±0.1 m/s per press, max ±0.8/0.5) |
| A / D | strafe left / right (±0.1 m/s per press, max ±0.5) |
| Q / E | turn left / right (**±0.4 rad/s per press** — large step to clear ~0.3 rad/s dead zone) |
| SPACE | stop + reset heading target (clears residual cwz from prior turns) |
| ESC | quit |

### Scripted demo sequence (`--demo`)

| Steps | vx | vy | wz | Segment |
|---|---|---|---|---|
| 150 | 0.0 | 0.0 | 0.0 | stand |
| 200 | 0.3 | 0.0 | 0.0 | walk fwd slow |
| 200 | 0.6 | 0.0 | 0.0 | walk fwd med |
| 200 | 0.8 | 0.0 | 0.0 | walk fwd fast |
| 250 | −0.4 | 0.0 | 0.0 | walk backward |
| 250 | 0.0 | +0.4 | 0.0 | strafe left |
| 250 | 0.0 | −0.4 | 0.0 | strafe right |
| 200 | 0.5 | +0.3 | 0.0 | diagonal fwd-left |
| 200 | 0.5 | −0.3 | 0.0 | diagonal fwd-right |
| 250 | 0.0 | 0.0 | +1.0 | turn left |
| 250 | 0.0 | 0.0 | −1.0 | turn right |
| 150 | 0.0 | 0.0 | 0.0 | stand |

Each step = 0.02 s sim time (50 Hz). Total: 2650 steps = 53 s.

### Follow camera
`--follow_cam` tracks the robot in 3rd-person front view (yaw-only EMA smoothing).
`--cam_offset X Y Z` (default: `3.0 0.0 0.5` — 3.0 m behind, 0.5 m up). Look-at target z=0.1 m keeps feet visible.

### Video recording
`--video <dir>` saves a video. Isaac Sim with rendering runs slower than 50 Hz real-time, making raw recordings appear sped up. The script corrects this automatically:
```
actual_fps = total_steps / elapsed_wall_time
ffmpeg -vf "setpts=(50/actual_fps)*PTS" -r actual_fps raw.mp4 → raw_realtime.mp4
```
Corrected file is `*_realtime.mp4`. Skipped if sim ran within 5% of 50 Hz.

### Isaac Lab quaternion convention
Isaac Lab uses **[w, x, y, z]** (scalar-first), NOT [x, y, z, w].
```python
quat_w = robot.data.root_quat_w[0].cpu().numpy()
_w, _x, _y, _z = quat_w
yaw = np.arctan2(2.0*(_w*_z + _x*_y), 1.0 - 2.0*(_y*_y + _z*_z))
```

---

## Known Issues / Next Steps

**Policy status:**
- [x] All 6 DOF commands working (vx, vy, wz — forward, backward, strafe, diagonal, turn)
- [x] Heading stable during strafe (`yaw_stability_penalty`)
- [x] No falls across all conditions (S1 fall rate 0%)
- [x] Two policy variants trained and tested — `raw_wz_FINAL` preferred for deployment

**Tracking limitations (from S1 eval):**
- [ ] **Strafe e_vy = 0.12 m/s** at vy=0.4 cmd — policy undershoots lateral velocity, capability limit not a bug
- [ ] **Slow forward (vx=0.3) e_vx = 0.07 m/s** — weak exp reward incentive at low speed, robot often appears to stand
- [ ] **Yaw drift 3.0°/s at wz=0** — exceeds <2°/s source target. Root cause: `yaw_stability` dead zone. Acceptable for hardware but monitor in S2–S6.
- [ ] **C10/C11 turn asymmetry** — e_wz varies between left/right turns across runs (stochastic, no fixed seed)

**Deployment pipeline:**
- [ ] **S2: Run `eval_b1_velocity.py --stage S2`** in Isaac Sim 5.0 before hardware (API compat check)
- [ ] **S3–S5: MuJoCo bridge + actuator net** — see `DEPLOY_LOG.md`
- [ ] Joystick wz dead zone mapping not yet implemented in deployment node — see wz dead zone section above
- [ ] `/B1/imu_ang_vel` not yet published by `b1_interface` — one-time fix required (see Deployment section)
- [ ] Verify joint order remap + coordinate conversion on real robot before moving
- [ ] Deploy to real hardware — `Sim2Real-B1/b1_deployment/` — only after S1–S5 pass in `DEPLOY_LOG.md`

**Future improvements:**
- [ ] Per-joint-type actuator gains (hip/thigh/knee separate kp/kd)
- [ ] Terrain curriculum for real-world robustness
