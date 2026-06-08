# Sim-to-Sim Policy Transfer for Learned Legged Locomotion
**Isaac Gym → MuJoCo | Command Switching / Transient Response Study**

---

Sim-to-real transfer for legged locomotion policies faces significant challenges due to differences in simulation environments and real-world hardware, including discrepancies in actuator dynamics, contact modeling, and real-time constraints.

Sim-to-sim transfer serves as a crucial intermediate step, enabling us to evaluate a learned policy's ability to adapt to different simulation environments before real-world deployment. This helps identify issues such as mismatches in actuator stiffness, friction models, and solver behavior, which could lead to instability during transient behaviors like command switching.

---

This repository accompanies **Exam 2** of the Sim2Real Internship Candidate Exam from VISTEC. The objective is to analyze **sim-to-sim policy transfer mismatch** for learned legged locomotion policies under **contact-rich dynamics**, with focus on **transient responses induced by command switching**. A locomotion policy is trained in **Isaac Gym (Sim A)** and transferred **without retraining** to **MuJoCo (Sim B)**.

Note : [Experimental Report]((Exam2)_Sim-to-Sim_Report_Disthorn_Suttawet.pdf)

<p align="center">
    <img width=45% src="sources/play_isaacgym_1.gif">
    <img width=41% src="sources/deploy_mujoco_1.gif">
    </br> Played policy on Isaac Gym then deployed to MuJoCo
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings Summary](#key-findings-summary)
- [Research Questions](#research-questions)ฟ
- [Robot Platform](#robot-platform)
- [Simulators and Configuration](#simulators-and-configuration)
  - [Sim-to-Sim Deployment Checklist](#sim-to-sim-deployment-checklist)
  - [Interface Alignment and Parity Verification](#interface-alignment-and-parity-verification)
- [Observation Space](#observation-space-48-dimensions)
- [Command Switching Scenarios](#command-switching-scenarios)
- [Evaluation Metrics](#evaluation-metrics)
- [Transient Response Analysis](#transient-response-analysis)
  - [S1 Stop Metrics](#s1-stop-transient-metrics-vx-06--00)
  - [S2 Turn Metrics](#s2-turn-transient-metrics-wz-00--10)
  - [S3 Lateral Metrics](#s3-lateral-transient-metrics-vy-03---03)
  - [Transient Summary](#transient-response-summary)
- [Experimental Results](#experimental-results)
  - [Stage 0: Sanity Checks](#stage-0-baseline-parity-sanity-checks)
  - [Stage 1: Baseline Performance](#stage-1-baseline-performance-steady-state--transient)
  - [Stage 1.5: Parameter Ablation](#stage-15-parameter-ablation-one-factor-at-a-time)
  - [Stage 2: Foot Friction Sweep](#stage-2-foot-friction-sweep)
  - [Stage 3: Observation Delay](#stage-3-observation-delay)
  - [Stage 3.5: Motor Command Delay](#stage-35-motor-command-delay)
- [Summary of Mismatch Sources](#summary-of-mismatch-sources)
- [Conclusions](#conclusions)
- [Bonus: Mismatch Reduction](#bonus-mismatch-reduction-via-learned-actuator-models)
  - [Approach 1: ActuatorNet](#approach-1-actuatornet-direct-torque-prediction)
  - [Approach 2: Residual Learning](#approach-2-residual-learning-pd--learned-correction)
  - [Approach 3: Domain Randomization](#approach-3-domain-randomization)
  - [Final Comparison](#final-comparison-all-approaches)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Key Findings Summary

| Finding | Impact | Section |
|---------|--------|---------|
| **9.4% vx tracking gap** in steady-state | Baseline mismatch exists | [Stage 1](#stage-1-baseline-performance-steady-state--transient) |
| **20ms delay causes FALL** during turns | Critical for sim-to-real | [Stage 3](#stage-3-observation-delay) |
| **Kp=30 in MuJoCo ≈ Kp=20 in Isaac Gym** | 50% stiffness difference | [Stage 1.5](#stage-15-parameter-ablation-one-factor-at-a-time) |
| **Foot friction is bottleneck**, not floor | μ_foot=0.8 reduces pitch 88% | [Stage 2](#stage-2-foot-friction-sweep) |
| **Opposite yaw conventions** between sims | Sign difference in wz | [Stage 1](#stage-1-key-observation-divergent-yaw-behavior) |
| **DR-trained settling 58% faster** than PD | 160ms vs 380ms in S1 Stop | [Transient Analysis](#s1-stop-transient-metrics-vx-06--00) |
| **Residual 84% faster vx rise** in S2 Turn | 300ms vs 1880ms | [Transient Analysis](#s2-turn-transient-metrics-wz-00--10) |
| **Residual 97% faster vx rise** in S3 Lateral | 40ms vs 1280ms | [Transient Analysis](#s3-lateral-transient-metrics-vy-03---03) |
| **Data diversity > R² score** | ActuatorNet V2 (94.55%) beats V1 (99.21%) | [Bonus](#actuatornet-v2-policy-driven-excitation-data) |
| **Hwangbo excitation > policy-driven** | ActuatorNet V3 fixes V2's S2 instability | [Bonus](#actuatornet-v3-hwangbo-style-excitation-fixed-s2-instability) |
| **ActuatorNet V3 reduces S2 pitch 84%** | 17.8° → 2.8° | [Bonus](#actuatornet-v3-hwangbo-style-excitation-fixed-s2-instability) |
| **DR-trained best stability** | Consistently lowest torque & excursions | [Bonus](#approach-3-domain-randomization) |
| **DR-trained reduces S2 roll 42%** | 6.0° → 3.5° | [Bonus](#approach-3-domain-randomization) |
| **DR-trained reduces S3 roll 44%** | 4.1° → 2.3° | [Bonus](#approach-3-domain-randomization) |

---

## Overview

This repository accompanies **Exam 2** of the Sim2Real Internship Candidate Exam from VISTEC.

The objective is to analyze **sim-to-sim policy transfer mismatch** for learned legged locomotion policies under **contact-rich dynamics**, with focus on **transient responses induced by command switching**.

A locomotion policy is trained in **Isaac Gym (Sim A)** and transferred **without retraining** to **MuJoCo (Sim B)**.

<p align="center">
    <img width=45% src="sources/play_isaaclab.gif">
    <img width=45% src="sources/play_isaacgym.gif">
    </br> Similarity in IsaacLab and Isaac Gym
</p>

While we have verified that Isaac Lab can produce deployable policies to MuJoCo with equivalent performance of Isaac Gym, we selected **Isaac Gym as our primary framework** due to its simpler sim-to-sim pipeline requiring no joint order remapping and its wide adoption in legged locomotion research.

---

## Research Questions

1. **Why do locomotion policies that exhibit similar steady-state performance diverge significantly during transient command switches?**

2. **Which sim-to-sim mismatches (actuator stiffness, solver dynamics, contact models) are amplified during high-acceleration maneuvers?**

3. **Can transient response metrics serve as a more sensitive indicator of sim-to-real transferability than steady-state tracking error?**

---

## Robot Platform

<p align="center">
    <img width=60% src="sources/Robots_Unitree_Go2.png">
    </br> The Unitree Go2 is a quadrupedal mobile robot designed for research, field operations, education, and industrial tasks.
</p>

- **Source:** [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
- **Robot:** Unitree Go2 Quadruped
- **DOF:** 12 (3 joints × 4 legs)
- **Joint Order Isaac Gym:** FL, FR, RL, RR (hip, thigh, calf per leg)
- **Joint Order MuJoCo qpos:** FL, FR, RL, RR (same as Isaac Gym)
- **Actuator Order MuJoCo ctrl:** FR, FL, RR, RL (requires remapping)

The Go2 uses **quasi-direct drive (QDD) actuators** with low gear ratio (~6-10:1), placing its dynamics complexity between platforms studied in prior sim-to-real work:

| Paper | Robot | Actuator Type | Modeling Approach |
|-------|-------|--------------|-------------------|
| Tan et al. (2018) | Minitaur | Direct-drive (no gears) | Analytical model (T = Kt·I) |
| Hwangbo et al. (2019) | ANYmal | Series-Elastic (SEA) | Learned ActuatorNet |
| Peng et al. (2018) | Laikago | Proprioceptive / QDD | Domain randomization |
| Kumar et al. (2021) | Unitree A1 | QDD | RMA adaptation |

The Go2 sits in the **middle ground**: simple enough that basic PD control works for steady-state, but with sufficient non-linearities (gear friction, torque saturation) that become problematic during transient maneuvers. This makes it an ideal testbed for comparing all three paradigms — analytical models (Tan), learned actuator models (Hwangbo), and adaptive policies (Kumar).

---

## Simulators and Configuration

### Sim A: Isaac Gym

| Parameter | Value |
|-----------|-------|
| Physics Engine | PhysX |
| sim.dt | 0.005s (200 Hz) |
| decimation | 4 |
| **Policy rate** | **50 Hz** (0.02s) |
| Kp (stiffness) | 20 N·m/rad |
| Kd (damping) | 0.5 N·m·s/rad |
| action_scale | 0.25 |
| Friction | 1.0 |
| Quaternion format | **(x, y, z, w)** |
| **Torque limits** | **±30 N·m** (implicit clipping) |

### Sim B: MuJoCo

| Parameter | Value |
|-----------|-------|
| Physics Engine | MuJoCo |
| timestep | 0.005s (200 Hz) |
| decimation | 4 |
| **Policy rate** | **50 Hz** (0.02s) |
| Kp (stiffness) | 20 N·m/rad |
| Kd (damping) | 0.5 N·m·s/rad |
| action_scale | 0.25 |
| Floor friction | 1.0 |
| Robot feet friction | 0.4 |
| Quaternion format | **(w, x, y, z)** |
| **Torque limits** | None (must add manually) |

---

### Sim-to-Sim Deployment Checklist

Before deploying an Isaac Gym policy to MuJoCo, the following must be verified:

| # | Step | Details |
|---|------|---------|
| 1 | **Quaternion reorder** | Isaac Gym `(x,y,z,w)` → MuJoCo `(w,x,y,z)` |
| 2 | **Actuator remapping** | MuJoCo `ctrl` uses FR, FL, RR, RL order |
| 3 | **Velocity frame** | MuJoCo world frame → body frame via `quat_rotate_inverse` |
| 4 | **Observation scales** | Match exactly: lin_vel×2.0, ang_vel×0.25, dof_vel×0.05 |
| 5 | **Torque clipping** | Add explicit `±30 N·m` clip (Isaac Gym does this implicitly) |
| 6 | **Heading command mode** | If `heading_command=True`: set `commands[:,3]` not `[:,2]` |
| 7 | **Zero-action stability** | Robot must hold stance without policy — height ~0.27m, roll/pitch < 3° |
| 8 | **Observation parity** | Gravity diff < 0.03, lin_vel diff < 0.05 vs Isaac Gym |

See [Interface Alignment and Parity Verification](#interface-alignment-and-parity-verification) for full implementation details.

---

### Interface Alignment and Parity Verification

Before measuring physics mismatch, the policy interface must be correctly implemented in MuJoCo. Three critical differences must be resolved — failure to address any of them produces incorrect observations or actions, making it impossible to isolate physics mismatch from interface bugs.

**1. Quaternion Convention:**
```python
# Isaac Gym: (x, y, z, w)
# MuJoCo:    (w, x, y, z)  ← reorder before computing gravity/velocity
qw, qx, qy, qz = d.qpos[3:7]
```

**2. Actuator Remapping:**
```python
# Isaac Gym: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
# MuJoCo:    FR(0-2), FL(3-5), RR(6-8), RL(9-11)
d.ctrl[0:3]  = tau[3:6]   # FR
d.ctrl[3:6]  = tau[0:3]   # FL
d.ctrl[6:9]  = tau[9:12]  # RR
d.ctrl[9:12] = tau[6:9]   # RL
```

**3. Velocity Frame Transformation:**
```python
# MuJoCo gives world frame → transform to body frame
base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
```

**Parity Checks (all must pass before proceeding):**

| Check | Status | Result |
|-------|--------|--------|
| Zero-Action Stability | ✓ Pass | Height stable ~0.27m, roll/pitch < 3° |
| Observation Parity | ✓ Pass | Gravity diff < 0.03, lin_vel diff < 0.05 |
| Joint Order Verification | ✓ Pass | qpos order matches, ctrl order remapped |

---

## Observation Space (48 dimensions)

```
obs[0:3]   = base_lin_vel * 2.0           # Linear velocity (body frame)
obs[3:6]   = base_ang_vel * 0.25          # Angular velocity (body frame)
obs[6:9]   = projected_gravity            # Gravity in body frame
obs[9:12]  = cmd * cmd_scale              # Command [vx, vy, wz]
obs[12:24] = (joint_pos - default) * 1.0  # Joint positions
obs[24:36] = joint_vel * 0.05             # Joint velocities
obs[36:48] = last_action                  # Previous action
```

**Critical:** Velocities must be in **body frame**, not world frame.

---

## Command Switching Scenarios

| Scenario | Before (t<3s) | After (t≥3s) | Test Focus |
|----------|---------------|--------------|------------|
| **S1 Stop** | vx=0.6, vy=0, wz=0 | vx=0, vy=0, wz=0 | Sudden stop |
| **S2 Turn** | vx=0.4, vy=0, wz=0 | vx=0.4, vy=0, wz=1.0 | Sudden turn |
| **S3 Lateral** | vx=0.3, vy=+0.3, wz=0 | vx=0.3, vy=-0.3, wz=0 | Direction flip |

### Visual Representation

**S1 Stop Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
     cmd = [0.6, 0, 0]    cmd = [0, 0, 0]
     (walk straight)       (stop instantly!)
```

**S2 Turn Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
     cmd = [0.4, 0, 0]    cmd = [0.4, 0, 1.0]
     (walk straight)       (turn instantly!)
```

**S3 Lateral Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
  cmd = [0.3, 0.3, 0]    cmd = [0.3, -0.3, 0]
   (walk diagonal)      (flip direction instantly!)
```

---

## Evaluation Metrics

### Steady-State Metrics
- Velocity tracking error (vx, vy, wz)
- RMSE

### Transient / Shock Metrics
- **Peak joint torque** — maximum |τ| during transient
- **Max roll/pitch** — body stability during command switch
- **Velocity overshoot** — how much velocity exceeds target
- **Fall detection** — roll/pitch > 1.0 rad or height < 0.15m

### Comparative Metrics
- Δ(peak torque) = MuJoCo − Isaac Gym
- Δ(max pitch) = MuJoCo − Isaac Gym

All experiments are conducted in **closed-loop without retraining**. Each episode lasts 6s and begins from a **fixed nominal stance**. Velocity commands are applied as piecewise-constant signals with switches at t_s = 3.0s. The policy executes at 50 Hz via zero-order hold over a 200 Hz physics simulation.

An episode terminates early if the base height falls below 0.15m or roll/pitch exceeds 1.0 rad. Each configuration is evaluated over **N = 10 episodes** with identical initial conditions.

> **Reproducibility note:** Episodes use fixed initial conditions (identical robot pose and zero velocity) without random seed variation across runs. This is intentional — the objective is to isolate *physics-engine mismatch* between Isaac Gym and MuJoCo, not to characterize policy variance. The N = 10 repetitions confirm deterministic consistency within each configuration rather than sampling across initial states.

---

## Transient Response Analysis

This section provides detailed transient response metrics for command switching scenarios, measuring rise time, settling time, overshoot, and peak values **between each sim-to-sim gap reduction strategy**.

<p align="center">
    <img width=80% src="plots/transient_overview_all.png">
    </br> Transient response comparison between methods across all 3 scenarios in MuJoCo
</p>

### S1 Stop: Transient Metrics (vx: 0.6 → 0.0)

<p align="center">
    <img width=80% src="plots/transient_S1_comparison.png">
    </br><img width=80% src="plots/transient_S1_metrics.png">
    </br> Transient response comparison between methods in S1 scenario in MuJoCo
</p>

| Metric | PD Only | PD + Residual | DR-trained | ActuatorNet V3 | Best |
|--------|---------|---------------|------------|----------------|------|
| **vx Rise time (10-90%)** | 300 ms | 200 ms | **140 ms** | 180 ms | DR-trained |
| **vx Settling time (5%)** | 380 ms | 240 ms | **160 ms** | 240 ms | DR-trained |
| **vx Overshoot** | 0.5% | 0.9% | **-0.4%** | -0.8% | DR-trained |
| **wz Settling time** | 160 ms | **40 ms** | 60 ms | 240 ms | Residual |
| **Peak torque** | 15.93 N·m | 16.16 N·m | **13.85 N·m** | 14.03 N·m | DR-trained |
| **Peak pitch** | 4.7° | 4.8° | 5.2° | **1.5°** | ActuatorNet V3 |
| **Peak roll** | 3.1° | **2.4°** | 2.8° | 2.5° | Residual |

**Key Observations:**
- DR-trained achieves **fastest settling** (160ms vs 380ms for PD) — **58% improvement**
- DR-trained has **lowest peak torque** (13.85 N·m) — **13% reduction** from PD
- Residual has **fastest wz settling** (40ms) and **best roll stability** (2.4°)
- ActuatorNet V3 has **exceptional pitch performance** (1.5°) in S1

---

### S2 Turn: Transient Metrics (wz: 0.0 → 1.0)

<p align="center">
    <img width=80% src="plots/transient_S2_comparison.png">
    </br><img width=80% src="plots/transient_S2_metrics.png">
    </br> Transient response comparison between methods in S2 scenario in MuJoCo
</p>

| Metric | PD Only | PD + Residual | DR-trained | ActuatorNet V3 | Best |
|--------|---------|---------------|------------|----------------|------|
| **vx Rise time** | 1880 ms | 300 ms | 880 ms | **20 ms** | ActuatorNet V3 |
| **wz Rise time** | N/A | N/A | N/A | 80 ms | ActuatorNet V3 |
| **wz Settling time** | N/A | N/A | N/A | N/A | — |
| **wz Overshoot** | -17.1% | **-10.1%** | -23.3% | 26.3% | Residual |
| **Peak torque** | 15.22 N·m | 14.59 N·m | **13.23 N·m** | 13.34 N·m | DR-trained |
| **Peak pitch** | 4.7° | 4.6° | 3.9° | **2.8°** | ActuatorNet V3 |
| **Peak roll** | 6.0° | 5.4° | **3.5°** | 5.1° | DR-trained |

**Key Observations:**
- Residual has **fastest vx rise time** (300ms vs 1880ms) — **84% faster**
- Residual has **smallest wz overshoot** (-10.1% vs -17% to -23%)
- DR-trained has **best stability** (pitch -17%, roll -42% vs PD)
- ActuatorNet V3 has **best pitch** (2.8°) but struggles with overshoot
- Turn scenario never fully settles for PD-based methods

---

### S3 Lateral: Transient Metrics (vy: +0.3 → -0.3)

<p align="center">
    <img width=80% src="plots/transient_S3_comparison.png">
    </br><img width=80% src="plots/transient_S3_metrics.png">
    </br> Transient response comparison between methods in S3 scenario in MuJoCo
</p>

| Metric | PD Only | PD + Residual | DR-trained | ActuatorNet V3 | Best |
|--------|---------|---------------|------------|----------------|------|
| **vx Rise time** | 1280 ms | **40 ms** | 1120 ms | 520 ms | Residual |
| **vy Rise time** | 620 ms | **420 ms** | 620 ms | — | Residual |
| **wz Settling time** | N/A | N/A | 2800 ms | — | DR-trained |
| **Peak torque** | 14.02 N·m | 13.95 N·m | **12.18 N·m** | 13.88 N·m | DR-trained |
| **Peak pitch** | 5.3° | 5.2° | 4.8° | **4.5°** | ActuatorNet V3 |
| **Peak roll** | 4.1° | 5.5° | **2.3°** | 6.3° | DR-trained |

**Key Observations:**
- Residual has **extremely fast vx rise** (40ms) — **97% faster** than PD
- DR-trained consistently has **lowest peak torque** (12.18 N·m) — **13% reduction**
- DR-trained has **best roll stability** (2.3° vs 4.1°) — **44% reduction**
- ActuatorNet V3 performs well overall but shows high roll (6.3°)

---

### Transient Response Summary

| Controller | S1 Stop | S2 Turn | S3 Lateral | Overall |
|------------|---------|---------|------------|---------|
| **PD Only** | Baseline | High overshoot | Baseline | Baseline |
| **PD + Residual** | Fast wz settling | **Best wz overshoot** | **Fastest vx rise** | Best speed |
| **DR-trained** | **Best overall** | **Best stability** | **Best stability** | **Best robustness** |
| **ActuatorNet V3** | Best pitch | Good pitch | Good pitch | Specialized |

**Recommendations:**
- **For best stability/robustness:** DR-trained Policy
- **For best tracking speed:** PD + Residual Learning
- **For specialized pitch control:** ActuatorNet V3 (when systematic data available)
- **For baseline comparison:** PD Only

**Overall Findings:**

1. **DR-trained excels at stability metrics** — consistently lowest torque (12–14 N·m), pitch (3.9–5.2°), and roll (2.3–3.5°)
2. **Residual excels at response speed** — fastest rise times in S2 (300ms vx) and S3 (40ms vx)
3. **ActuatorNet V3 provides specialized benefits** — exceptional pitch control when systematic excitation data is available
4. **Trade-off exists:** DR-trained offers best stability at higher training complexity; Residual offers faster response at moderate complexity; ActuatorNet V3 requires systematic data collection; PD Only is simple but has highest peak values
5. **S2 Turn is most challenging** — no method achieves full settling
6. **Transient metrics reveal critical differences** not visible in steady-state tracking

---

## Experimental Results

### Stage 0: Baseline Parity (Sanity Checks)

**Goal:** Ensure policy interface is correctly implemented before measuring physics mismatch.

| Check | Status | Result |
|-------|--------|--------|
| Zero-Action Stability | ✓ Pass | Height stable ~0.27m, roll/pitch < 3° |
| Observation Parity | ✓ Pass | Gravity diff < 0.03, lin_vel diff < 0.05 |
| Joint Order Verification | ✓ Pass | qpos order matches, ctrl order remapped |

**Key implementation fixes:** see [Interface Alignment and Parity Verification](#interface-alignment-and-parity-verification).

---

### Stage 1: Baseline Performance (Steady-State + Transient)

> Before attributing sim-to-sim mismatch to any specific source, we first establish a baseline comparison under nominal and well-controlled conditions. This step answers a fundamental question: *does a gap exist at all when both simulators are configured as similarly as possible?*

**Goal:** Measure sim-to-sim gap under nominal conditions (flat ground, default friction, no noise/delay) using PD Controllers in both simulators.

#### Steady-State Baseline (cmd: vx=0.5 m/s, 10 seconds)

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|-----|
| vx mean | 0.498 m/s | 0.451 m/s | **-0.047 (9.4%)** |
| vx RMSE | 0.079 | 0.066 | -0.013 |
| vy error | 0.011 | 0.010 | -0.001 |
| wz error | 0.216 | 0.007 | -0.209 |
| Torque mean | 2.56 N·m | 2.97 N·m | +0.41 |
| Torque max | 17.29 N·m | 15.71 N·m | -1.58 |

#### S1: Stop Shock (vx: 0.6 → 0.0)

<p align="center">
    <img width=45% src="sources/isaacgym_s1.gif">
    <img width=45% src="sources/mujoco_s1.gif">
    </br> S1 scenario: straight walk then stop
</p>

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|-----|
| Steady-State vx | 0.610 m/s | 0.566 m/s | -0.044 |
| Transient vx mean | 0.065 m/s | 0.040 m/s | -0.025 |
| Peak torque | 15.26 N·m | 16.29 N·m | +1.03 |
| Max pitch | 3.8° | 4.8° | +1.0° |
| Fallen | No | No | — |

#### S2: Turn Shock (wz: 0.0 → 1.0)

<p align="center">
    <img width=45% src="sources/isaacgym_s2.gif">
    <img width=45% src="sources/mujoco_s2.gif">
    </br> S2 scenario: straight walk then turn
</p>

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|-----|
| Steady-State vx | 0.407 m/s | 0.353 m/s | -0.054 |
| Transient wz mean | -0.178 rad/s* | 0.682 rad/s | — |
| Peak torque | 11.37 N·m | 15.32 N·m | **+3.95** |
| Max pitch | 3.0° | 5.0° | **+2.0°** |
| Fallen | No | No | — |

*Note: Isaac Gym wz has opposite sign convention — see below.

#### S3: Lateral Flip (vy: +0.3 → -0.3)

<p align="center">
    <img width=45% src="sources/isaacgym_s3.gif">
    <img width=45% src="sources/mujoco_s3.gif">
    </br> S3 scenario: lateral direction flip
</p>

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|-----|
| Steady-State vx | 0.327 m/s | 0.263 m/s | -0.064 |
| Transient vx mean | 0.298 m/s | 0.263 m/s | -0.035 |
| Peak torque | 12.75 N·m | 13.77 N·m | +1.02 |
| Max pitch | 3.8° | 5.2° | +1.4° |
| Fallen | No | No | — |

<p align="center">
   <img src="plots/all_scenarios_overview.png" width="400">
   </br> S1, S2, and S3 scenario comparison between Isaac Gym and MuJoCo
</p>

#### Stage 1 Key Observation: Divergent Yaw Behavior

A significant behavioral difference was observed in S2 Turn:

| Period | Isaac Gym wz | MuJoCo wz | Note |
|--------|-------------|-----------|------|
| t=0-1s (init) | **-0.807 rad/s** | +0.018 rad/s | Isaac rotates without command! |
| t=1-3s (pre-switch) | -0.422 rad/s | -0.004 rad/s | Isaac still rotating |
| t=3-4s (post-switch) | -0.196 rad/s | **+0.710 rad/s** | Opposite directions |
| Total yaw change | **-117.3°** (CCW) | **+118.0°** (CW) | Same magnitude, opposite sign |

**Raw Data Ranges:**
- Isaac Gym wz: [-1.098, -0.032] rad/s — **always negative**
- MuJoCo wz: [-0.112, +0.839] rad/s — **mostly positive after switch**

> ⚠️ **Note on Plots:** In comparison plots, Isaac Gym wz sign is flipped (`-wz`) to visually align the turning behavior. The raw data shows opposite signs.

<p align="center">
   <img src="plots/S2_turn_comparison.png" width="400">
   </br> S2 scenario comparison between Isaac Gym and MuJoCo
</p>

*S2 scenario is primarily used in subsequent experiments due to its clear transient behavior.*

---

#### Root Cause Analysis: Heading Command Mode

We investigated why Isaac Gym shows rotation when command wz=0:

**Initial Observation:**

| Test | Command wz | Actual wz | Result |
|------|-----------|-----------|--------|
| Zero action (no policy) | 0 | **0.000** | ✓ Environment OK |
| Policy + set wz=0 directly | 0 | **-0.554** | ✗ Unexpected rotation |

**Root Cause Discovery:**

The config uses `heading_command = True`:
```python
# In legged_robot.py post_physics_step:
if self.cfg.commands.heading_command:
    self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
```

This means `commands[:, 2]` (wz) is **NOT a direct command** — it is **computed from heading error**. The policy was trained to track **heading**, not angular velocity. Setting `commands[:, 2] = 0` directly gets overwritten by the heading controller.

**Verification:**

| Test | Method | wz (cmd=0) | Result |
|------|--------|------------|--------|
| ✗ Wrong | Set `commands[:, 2] = 0` | -0.55 rad/s | wz overwritten |
| ✓ Correct | Set `commands[:, 3] = current_heading` | **-0.009 rad/s** | No rotation |

**Fixed Results:**

| Scenario | wz Before Switch | wz After Switch | Status |
|----------|------------------|-----------------|--------|
| S1 Stop | -0.003 rad/s | -0.036 rad/s | ✓ |
| S2 Turn | -0.009 rad/s | -0.778 rad/s | ✓ |
| S3 Lateral | -0.008 rad/s | +0.005 rad/s | ✓ |

**Key Lesson:** Always check `env_cfg.commands.heading_command` before setting commands. If `heading_command = True`, set `commands[:, 3]` (target heading); if `False`, set `commands[:, 2]` (angular velocity). This is **not a bug** — it is a different control mode that requires a different command interface.

**Interpretation:** Both robots rotate approximately the same magnitude (~118°) but in **opposite directions**, indicating a sign convention difference between PhysX and MuJoCo for yaw rate.

> At this stage, it remains unclear which physical factors are responsible for the observed transient divergence.

---

### Stage 1.5: Parameter Ablation (One-Factor-at-a-Time)

> To identify the source of the transient mismatch observed in Stage 1, we vary individual simulator parameters in MuJoCo while holding all others fixed.

**Goal:** Identify causal sources of mismatch by varying parameters individually in MuJoCo.

#### 1. Kp (Stiffness) Sweep

| Kp | vx mean | vx error | Notes |
|----|---------|----------|-------|
| 10 | 0.101 m/s | 0.399 | Nearly crawling |
| **20 (baseline)** | **0.451 m/s** | **0.049** | Default |
| **30** | **0.466 m/s** | **0.034** | **Best match to Isaac Gym** |
| 40 | 0.407 m/s | 0.093 | Worse (overshoot) |

**Finding:** Kp=30 in MuJoCo best matches Isaac Gym (Kp=20) performance. MuJoCo requires ~50% higher stiffness to achieve equivalent actuator response.

<p align="center">
   <img src="plots/kp_ablation.png" width="400">
   </br> Kp=30 in MuJoCo best matches Isaac Gym
</p>

#### 2. Kd (Damping) Sweep

| Kd | vx mean | wz overshoot | Max pitch | Max roll | Peak torque |
|----|---------|-------------|-----------|----------|-------------|
| 0.3 | 0.348 m/s | -0.167 | 4.5° | 6.3° | 14.82 N·m |
| **0.5 (baseline)** | **0.353 m/s** | **-0.161** | **5.0°** | **5.1°** | **15.32 N·m** |
| 0.8 | 0.349 m/s | -0.079 | 5.0° | 6.9° | 15.25 N·m |
| 1.0 | 0.344 m/s | -0.090 | 4.7° | 5.1° | 15.22 N·m |

<p align="center">
   <img src="plots/kd_ablation.png" width="600">
   </br> Kd sweep — minimal effect compared to Kp
</p>

**Finding:** Kd has minimal effect on both tracking and transient stability. Varying Kd by 3× (0.3→1.0) produces less than 0.1 rad/s difference in wz overshoot with no fall in any condition. This confirms that **actuator stiffness (Kp), not damping (Kd), is the dominant PD gain mismatch** in this sim-to-sim transfer.

#### 3. Mass (Base Link) Perturbation

| Base mass | Δmass | vx (steady) | wz overshoot | Max pitch | Peak torque |
|-----------|-------|------------|-------------|-----------|-------------|
| 5.921 kg | -1 kg | 0.363 m/s | -0.220 | 3.8° | 13.63 N·m |
| **6.921 kg (baseline)** | **0** | **0.353 m/s** | **-0.161** | **5.0°** | **15.32 N·m** |
| 7.921 kg | +1 kg | 0.358 m/s | +0.374 | **15.5°** | **23.11 N·m** |
| 8.921 kg | +2 kg | 0.237 m/s | -0.049 | 11.2° | 21.56 N·m |

<p align="center">
   <img src="plots/mass_ablation.png" width="600">
   </br> Adding +1 kg nearly triples peak pitch
</p>

**Finding:** Mass perturbation has a significant and **asymmetric** effect on transient stability. Adding just +1 kg (14% increase) nearly triples peak pitch (15.5° vs 5.0°) and increases peak torque by 51%, while removing -1 kg has minimal impact. This asymmetry validates the Domain Randomization approach, which randomizes mass by [-1 kg, +2 kg] during training to explicitly handle such variation.

#### 4. Joint Friction (Viscous + Coulomb) Sweep

MuJoCo's go2.xml includes joint-level friction parameters (`damping` for viscous friction, `frictionloss` for Coulomb friction) that have no direct equivalent in Isaac Gym's PhysX engine.

| Condition | damping | frictionloss | vx (steady) | wz overshoot | Max roll | Max pitch | Peak torque |
|-----------|---------|-------------|------------|-------------|----------|-----------|-------------|
| No friction | 0.0 | 0.0 | 0.361 m/s | -0.147 | **9.3°** | 5.0° | 14.79 N·m |
| **Baseline** | **0.1** | **0.2** | **0.353 m/s** | **-0.161** | **5.1°** | **5.0°** | **15.32 N·m** |
| High friction | 0.3 | 0.5 | 0.326 m/s | -0.054 | **3.4°** | 4.7° | 15.02 N·m |

<p align="center">
   <img src="plots/joint_friction_ablation.png" width="600">
   </br> Joint friction acts as a passive damper
</p>

**Finding:** Joint friction has a moderate effect, primarily on **roll stability**. Removing all joint friction increases peak roll by 82% (9.3° vs 5.1°). This represents a **structural mismatch** between simulators — MuJoCo models joint friction explicitly, while Isaac Gym's PhysX does not expose equivalent parameters.

#### 5. dt (Timestep) Sweep

| dt | decimation | Policy rate | vx mean | vx error |
|----|------------|-------------|---------|----------|
| 0.002 | 10 | 50 Hz | 0.410 | 0.090 |
| 0.005 | 4 | 50 Hz | 0.451 | 0.049 |
| 0.01 | 2 | 50 Hz | 0.408 | 0.092 |

**Finding:** Timestep has minimal effect when policy rate is kept constant at 50 Hz.

#### 6. Floor Friction Sweep

| Floor friction | vx mean | vx error |
|----------------|---------|----------|
| 0.5 | 0.451 | 0.049 |
| 1.0 | 0.451 | 0.049 |
| 1.5 | 0.451 | 0.049 |

**Finding:** Floor friction has no measurable effect across the entire tested range.

> This surprising result raised a critical follow-up question: Is friction truly irrelevant, or are we varying the wrong parameter?

---

### Stage 2: Foot Friction Sweep

The floor friction result motivated testing **foot friction** directly. MuJoCo uses the geometric mean of contacting surfaces:

$$\mu_{effective} \approx \sqrt{\mu_{floor} \times \mu_{foot}}$$

Because the default foot friction is low (μ_foot=0.4), the effective friction remains low (≈0.63) even when floor friction is increased significantly. Increasing floor friction beyond real-world ranges (0.5–1.25 per Tan et al., 2018) would produce a physically unrealistic environment, defeating the purpose of sim-to-sim validation.

**Method:** Vary foot friction μ_foot ∈ {0.2, 0.4, 0.8} while holding floor friction constant at 1.0.

#### S2: Turn Shock (wz: 0.0 → 1.0)

| Metric | μ_foot=0.2 | μ_foot=0.4 (baseline) | μ_foot=0.8 |
|--------|------------|----------------------|------------|
| Steady-State vx | 0.294 m/s | 0.353 m/s | 0.353 m/s |
| wz overshoot | **+1.868 rad/s** | -0.161 rad/s | -0.254 rad/s |
| Max roll | **17.8°** | 5.1° | 5.3° |
| Max pitch | **25.1°** | 5.0° | **3.0°** |
| Peak torque | **27.57 N·m** | 15.32 N·m | **13.65 N·m** |

<p align="center">
   <img src="plots/foot_friction_ablation.png" width="600">
   <br> Foot friction sweep: 0.2, 0.4, 0.8 with floor friction fixed at 1.0
</p>

#### Stage 2 Findings

1. **Foot friction is the bottleneck**, not floor friction.
2. **High friction (μ=0.8) improves stability:** max roll 70% lower, max pitch 88% lower, peak torque 50% lower vs μ=0.2.
3. **Low friction (μ=0.2) causes severe instability:** wz overshoot nearly 2× target, peak torque nearly doubles, robot approaches fall conditions.

> **Interpretation:** When foot friction is low, the system enters a **slip-prone regime** where lateral forces cannot be fully transmitted to the ground. This confirms that **friction mismatch between Isaac Gym (μ≈1.0) and MuJoCo (μ_foot=0.4) contributes significantly to the observed sim-to-sim gap**.

---

### Stage 3: Observation Delay

> All previous experiments assume perfect and instantaneous state feedback. Real robots operate under sensing latency — we now test whether transient sensitivity is further amplified by observation delays.

**Goal:** Evaluate sensitivity to sensing latency during transient command switches.

**Method:** Introduce observation delay of 0, 1, 2 policy steps (0, 20, 40 ms) in MuJoCo.

**Scenario:** S2 Turn (most challenging based on Stage 1 results)

#### Results

<p align="center">
   <img width=45% src="sources/mujoco_s2.gif">
   <img width=45% src="sources/delay_20ms.gif">
   <br> No latency vs. 20ms latency
</p>

| Metric | 0 ms | 20 ms | 40 ms |
|--------|------|-------|-------|
| Steady-State vx | 0.353 m/s | 0.330 m/s | 0.322 m/s |
| wz overshoot | -0.161 rad/s | **+1.348 rad/s** | +0.974 rad/s |
| Max roll | 5.1° | **167.8°** | 18.8° |
| Max pitch | 5.0° | **35.1°** | 28.4° |
| Peak torque | 15.32 N·m | **27.57 N·m (+80%)** | 26.21 N·m (+71%) |
| **Fallen** | No | **YES** | No |

<p align="center">
   <img src="plots/delay_ablation.png" width="500">
   <br> Observation delay: 0, 1, 2 policy steps
</p>

#### Stage 3 Key Finding: 20ms Delay Causes Fall

**Just 20ms of observation delay (1 policy step) causes the robot to fall.**

This is a critical finding for sim-to-real transfer: real robots typically have 10–50ms sensing latency, transient maneuvers are highly sensitive to delay, and steady-state metrics do not predict this failure mode.

> **Interpretation:** Observation delay causes the policy to act on outdated state information. During rapid transient maneuvers, even small delays lead to over-correction, oscillation, and eventual fall. Interestingly, 40ms delay does not cause a fall — suggesting **chaotic sensitivity** where specific delay values interact with gait timing in unpredictable ways.

---

### Stage 3.5: Motor Command Delay

> Stage 3 demonstrated that observation delay is critically destabilizing. Real robots also experience actuator command delay. We now test this output-side counterpart to complete the latency analysis.

**Goal:** Evaluate sensitivity to motor command latency and compare with observation delay from Stage 3.

**Method:** Buffer target joint positions by 0, 1, 2 policy steps (0, 20, 40 ms) before applying PD control. The policy observes correct state but actuators execute outdated commands.

**Scenario:** S2 Turn

#### Results

<p align="center">
   <img width=45% src="sources/mujoco_s2.gif">
   <img width=45% src="sources/motor_delay_40s.gif">
   <br> No motor latency vs. 40ms motor latency
</p>

| Motor delay | wz overshoot | Max pitch | Max roll | Peak torque | Fallen |
|-------------|-------------|-----------|----------|-------------|--------|
| 0 ms | -0.161 | 5.0° | 5.1° | 15.32 N·m | No |
| 20 ms (1 step) | +0.241 | 6.1° | 5.3° | 19.44 N·m | No |
| 40 ms (2 steps) | +0.246 | 18.9° | 73.5° | exploded | **YES** |

#### Stage 3.5 Key Finding: Motor Delay is Less Critical than Observation Delay

| Delay | Observation (Stage 3) | Motor command (Stage 3.5) |
|-------|----------------------|--------------------------|
| 0 ms | Stable | Stable |
| 20 ms | **FELL** | Stable |
| 40 ms | Stable* | **FELL** |

*40ms observation delay shows chaotic sensitivity — see Stage 3.

20ms motor delay is survivable while 20ms observation delay caused immediate fall. This asymmetry makes physical sense: observation delay causes compounding errors (policy thinks robot hasn't turned yet, keeps commanding turn), while motor delay still allows the policy to observe correct state and make appropriate decisions — only execution is late.

<p align="center">
   <img src="plots/motor_delay_ablation.png" width="600">
   <br> Observation delay vs. motor delay comparison
</p>

> **Implication for sim-to-real:** Prioritize **low-latency state estimation** (IMU, joint encoders) over actuator response time. For the Go2's QDD actuators with typical command latency of ~5–10ms, motor delay alone is unlikely to cause instability. This aligns with Tan et al.'s findings on the Minitaur, where modeling latency (~15–19ms) was crucial for preventing oscillations during sim-to-real transfer.

---

## Summary of Mismatch Sources

### Physics Mismatch (Ablated)

| Source | Effect on Tracking | Effect on Stability | Recommendation |
|--------|-------------------|---------------------|----------------|
| **Kp (stiffness)** | High | Medium | Use Kp=30 in MuJoCo to match Isaac Gym Kp=20 |
| **Kd (damping)** | Low | Low | Not a significant factor (Kp dominates) |
| **Mass perturbation** | Medium | **High** | +1kg nearly triples pitch; asymmetric response |
| **Joint friction** | Low | Medium | Affects roll stability; structural mismatch between sims |
| **Foot friction** | Medium | **High** | Match μ_foot to real robot (~0.6–0.8 for rubber) |
| **Floor friction** | None | None | Not a significant factor (bottleneck is foot) |
| **Timestep** | Low | Low | Keep policy rate constant |
| **Observation delay** | Medium | **Critical** | 20ms causes fall in S2 Turn |
| **Motor command delay** | Low | **High** | Less critical than obs delay; 20ms survivable, 40ms causes fall |
| **Torque clipping** | Medium | High | Isaac Gym clips at ±30 N·m (implicit) |
| **Velocity-dependent dynamics** | Medium | High | Isaac Gym has implicit vel-dependent compensation; residual learning captures this |

### Implementation Pitfalls

| Source | Description | Fix |
|--------|-------------|-----|
| **Quaternion convention** | Isaac Gym (x,y,z,w) vs MuJoCo (w,x,y,z) | Reorder before computing gravity/velocity |
| **Yaw sign convention** | Opposite rotation direction between sims | Document and flip sign in comparison plots |
| **Actuator ordering** | MuJoCo ctrl uses FR,FL,RR,RL | Remap: `ctrl[0:3]=tau[3:6]`, etc. |
| **Heading command mode** | `heading_command=True` overwrites `commands[:,2]` | Set `commands[:,3]` (target heading), not `[:,2]` (wz) |

### Known but Not Ablated

| Source | Referenced In | Description | Expected Impact |
|--------|-------------|-------------|-----------------|
| **Contact solver differences** | Implicit in all papers | PhysX uses iterative Gauss-Seidel; MuJoCo uses Newton-based convex optimization | High — likely root cause of Kp stiffness gap, but cannot be isolated without modifying solver internals |
| **Ground compliance** | Peng et al. (2020) | Rigid vs. soft contact modeling differences | Low — both simulators use rigid flat ground in this study |

> **Note:** The contact solver difference is likely the most impactful untested source, as it affects how forces propagate through the kinematic chain during contact. The observed Kp mismatch (Stage 1.5) and foot friction sensitivity (Stage 2) are both downstream effects of different solver behaviors.

---

## Conclusions

### Key Findings

1. **Steady-state metrics mask critical instability exposed by transient maneuvers.**
   While the baseline velocity tracking error differed by only ~9% during steady walking, the S2 Turn scenario revealed significant divergences in stability. Sim-to-sim validation must prioritize transient response analysis over steady-state averages.

2. **Simulators enforce conflicting coordinate conventions.**
   Isaac Gym and MuJoCo exhibit opposite yaw rotation directions for the same positive $w_z$ command. The magnitude of rotation (~118°) was consistent, indicating a coordinate system difference rather than a policy failure.

3. **Solver stiffness discrepancy necessitates gain tuning ($K_p$).**
   Matching actuator performance required increasing $K_p$ in MuJoCo (30 vs. 20 in Isaac Gym), suggesting Isaac Gym's PhysX engine behaves "softer" than MuJoCo's Newton-based solver.

4. **Contact stability is bottlenecked by foot friction, not floor friction.**
   Varying floor friction had no effect due to MuJoCo's geometric mean calculation. Increasing $\mu_{foot}$ to 0.8 resolved the bottleneck, reducing peak torque by 50% and pitch oscillation by 88%.

5. **Observation latency is the critical failure mode.**
   A mere 20ms observation delay caused immediate falls during turning maneuvers, whereas motor command delays of the same magnitude were survivable.

### Answers to Research Questions

**Q1: Why do policies diverge during transients?**
> Transient states expose implicit simulator behaviors dormant during steady-state. Differences in contact solver stiffness and observation timing amplify errors when the policy commands rapid corrections. Isaac Gym's implicit torque clipping (±30 N·m) acts as a "silent safety net" absent in MuJoCo, causing destabilizing torque spikes in the target environment.

**Q2: Which mismatches are amplified during high-acceleration maneuvers?**
> Observation delay and foot friction are the most amplified mismatches. A 20ms delay, negligible during walking, becomes catastrophic during rapid turns due to phase lag in the feedback loop. Low foot friction creates a "slip-prone regime" only when lateral forces peak during transient maneuvers.

**Q3: Can transient metrics serve as better transferability indicators?**
> **Yes.** Steady-state metrics proved deceptive, showing only a ~9% gap even when the policy was critically unstable. Transient metrics (overshoot, rise time, peak torque) successfully revealed the system's fragility to latency and friction. "Survival during Transients" is a far superior predictor of sim-to-real robustness than "Average Tracking Error."

### Recommendations for Sim-to-Real Transfer

1. **Tune Kp/Kd gains** — MuJoCo requires ~50% higher Kp than Isaac Gym
2. **Match foot friction** to real robot contact properties (rubber ≈ 0.6–0.8)
3. **Add observation delay** (20–40ms) during training/validation
4. **Test turning maneuvers** — they expose the largest gaps
5. **Verify sign conventions** for angular velocities and quaternions
6. **Use transient metrics** (peak torque, max pitch) in addition to tracking error
7. **Add torque clipping** in MuJoCo to match Isaac Gym behavior (±30 N·m)

---

## Bonus: Mismatch Reduction via Learned Actuator Models

This section explores using neural network-based actuator models to reduce sim-to-sim mismatch.

### Approach 1: ActuatorNet (Direct Torque Prediction)

**Concept:** Replace PD control entirely with a learned model that predicts torque from Isaac Gym data.

**Reference:** [actuator_net](https://github.com/sunzhon/actuator_net) — originally designed for sim-to-real transfer.

---

#### ActuatorNet V1: Normal Walking Data (Failed)

**Method:**
1. Collected 30,000 timesteps × 12 motors = 360,000 samples from **normal walking only**
2. Input: `[pos_error, pos_error_t-1, pos_error_t-2, vel, vel_t-1, vel_t-2]` → output: `torque`
3. Trained MLP with R² = 99.21%
4. Deployed in MuJoCo replacing PD control

<p align="center">
<img src="plots/actuator_net.png" width="400">
<br> actuator_net UI and usage example
</p>

| Metric | ActuatorNet V1 | PD Control |
|--------|----------------|------------|
| vx error | 0.277 m/s | **0.049 m/s** |
| Torque max | 26.2 N·m | **15.7 N·m** |

**Problem:** High R² (99.21%) but poor deployment — overfit to normal walking dynamics, data didn't cover extreme cases.

---

#### ActuatorNet V2: Policy-Driven Excitation Data

**Excitation Data Collection:**
```
Phase 1: Normal walking (varied commands)     - 300,000 samples
Phase 2: Random joint perturbations           - 150,000 samples
Phase 3: High-speed commands                  - 150,000 samples
Phase 4: Sudden command switches              - 150,000 samples
─────────────────────────────────────────────────────────────
Total: 750,000 samples (25x more than V1)
```

| Metric | V1 (Normal) | V2 (Excitation) | Improvement |
|--------|-------------|-----------------|-------------|
| Samples | 30,000 | **750,000** | **25x** |
| Velocity range | ±20 rad/s | **±37 rad/s** | 1.85x |
| Torque range | ±26 N·m | **±35.5 N·m** | Reaches limits |

- Architecture: MLP [100 units, 4 layers] with softsign activation
- R² = 94.55% (lower than V1 but on diverse data = better generalization)

<p align="center">
    <img width=50% src="sources/actuator_netv2_s2.gif">
    <br> ActuatorNet V2 with S2 command switch
</p>

| Metric | PD Control | ActuatorNet V2 | Change |
|--------|------------|----------------|--------|
| Status | Stable | **Unstable** | — |
| Torque max | **15.32 N·m** | 28.49 N·m | +86% ✗ |
| Max pitch | **5.0°** | 17.8° | +256% ✗ |
| Max roll | **5.1°** | 15.3° | +200% ✗ |

**V2 Limitation:** Still struggles with transient turn commands (S2) — policy-driven data does not cover the full actuator dynamics space.

---

#### ActuatorNet V3: Hwangbo-Style Excitation (Fixed S2 Instability)

**Reference:** Hwangbo et al., "[Learning agile and dynamic motor skills for legged robots](https://www.science.org/doi/10.1126/scirobotics.aau5872)" (Science Robotics 2019)

The key insight: **actuator data should be collected independently from the locomotion policy**, using systematic excitation signals that cover the full operating range.

| Aspect | V2 (Policy-Driven) | V3 (Hwangbo-Style) |
|--------|-------------------|-------------------|
| Data source | Policy locomotion | **Systematic excitation** |
| Sinusoidal sweep | ✗ None | ✓ 0.5–10 Hz |
| Chirp (freq sweep) | ✗ None | ✓ 0.5→10 Hz |
| Torque saturation probing | Partial | ✓ Deliberate |
| Decoupled from task | ✗ No | ✓ Yes |
| Identifiability | Low | **High** |

**Hwangbo Excitation Data Collection:**
```
Phase 1: Low-frequency sinusoids (0.5-2 Hz)   -  60,000 samples
Phase 2: Mid-frequency sinusoids (2-5 Hz)      -  60,000 samples
Phase 3: High-frequency sinusoids (5-10 Hz)    -  60,000 samples
Phase 4: Chirp sweep (0.5→10 Hz)               - 120,000 samples
Phase 5: Torque saturation probing             -  60,000 samples
Phase 6: Multi-sine (mixed frequencies)        -  60,000 samples
─────────────────────────────────────────────────────────────
Total: 420,000 samples with full frequency coverage
```

**Simulated Actuator Dynamics** (to make learning non-trivial):
```python
class ActuatorDynamics:
    motor_time_constant = 0.02s    # First-order lag
    viscous_friction    = 0.1      # N·m/(rad/s)
    coulomb_friction    = 0.5      # N·m
    noise_std           = 0.1      # N·m
```

- Features: `[pos_error, pos_error_t-1, pos_error_t-2, vel, vel_t-1, vel_t-2]`
- R² = 99.80%, training stopped early at epoch 44

**Results V3 - S1 Stop:**

| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Rise time | 200 ms | **180 ms** | -10% ✓ |
| Settling time | 260 ms | **240 ms** | -8% ✓ |
| Peak torque | **12.15 N·m** | 14.03 N·m | +15% |
| Peak pitch | 4.6° | **1.5°** | **-67%** ✓ |
| Peak roll | 2.2° | 2.5° | +14% |

**Results V3 - S2 Turn:**

<p align="center">
   <img width=45% src="sources/actuator_netv2_s2.gif">
   <img width=45% src="sources/actuator_netv3_s2.gif">
   <br> ActuatorNet V2 vs. V3
</p>

| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Status | **Unstable** | ✓ **Stable** | **Fixed!** |
| Peak torque | 28.49 N·m ✗ | **13.34 N·m** | **-53%** ✓ |
| Peak pitch | 17.8° ✗ | **2.8°** | **-84%** ✓ |
| Peak roll | 15.3° ✗ | **5.1°** | **-67%** ✓ |
| wz Rise time | 320 ms | **80 ms** | -75% ✓ |

**Results V3 - S3 Lateral:**

| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Peak torque | **13.56 N·m** | 13.88 N·m | +2% |
| Peak pitch | 5.1° | **4.5°** | -12% |
| Peak roll | **2.9°** | 6.3° | +117% |

---

#### ActuatorNet Conclusions

<p align="center">
    <img width=80% src="plots/actuatornet_s2_v2_v3_fix.png">
    <br> ActuatorNet V2 and V3 compared to PD Controller
</p>

| Version | Data Type | Normal Walking | Constant Turn | Command Switch (S2) |
|---------|-----------|----------------|---------------|---------------------|
| **V1** | Normal walking | ✗ Worse | — | — |
| **V2** | Policy excitation | ✓ Comparable | ✓ Stable | ✗ **Unstable** |
| **V3** | Hwangbo excitation | ✓ Comparable | ✓ Stable | ✓ **Stable** |

**Key Insights:**
1. **Data diversity matters more than R²** — V1 had 99.21% R² but failed; V2 had 94.55% but works better
2. **Systematic excitation > policy-driven** — V3's Hwangbo-style collection fixed V2's S2 instability
3. **Frequency coverage is critical** — sinusoidal sweeps and chirp signals ensure full actuator dynamics coverage
4. **History features help** — V3 uses `[pos_error_t-2:t, vel_t-2:t]` for temporal modeling

**Recommendation:** Use Hwangbo-style systematic excitation with sinusoidal sweeps (0.5–10 Hz), chirp signals, and torque saturation probing. For best overall robustness, prefer DR-trained over ActuatorNet.

---

### Approach 2: Residual Learning (PD + Learned Correction)

**Concept:** Instead of replacing PD control, learn a residual correction term:

```
τ_mujoco = τ_pd + Δτ_learned
```

**Method:**
1. Collected residual data: `Δτ = τ_isaac - τ_pd` from Isaac Gym
2. Residual statistics: mean=0.011 N·m, std=0.810 N·m, range=[-19.7, +18.0] N·m
3. Trained ResidualNet: input `[pos_error, velocity]` → output `Δτ`, test RMSE: 0.577 N·m

**Results - Steady-State (vx=0.5 m/s):**

| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| vx error | 0.049 m/s | 0.043 m/s | **-12%** ✓ |
| Torque max | 15.71 N·m | 15.42 N·m | -2% |

**Results - Continuous Turn Command (vx=0.4, wz=1.0):**

<p align="center">
    <img width=45% src="sources/pd_continuous_turn.gif">
    <img width=45% src="sources/pd_with_residual_continuous_turn.gif">
    <br> PD Controller vs. PD + Residual Learning
</p>

| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| vx error | 0.093 m/s | 0.052 m/s | **-44%** ✓ |
| wz error | 0.395 rad/s | 0.338 rad/s | **-14%** ✓ |
| Status | **FALLEN** | Stable | **Fixed!** ✓ |

**Results - S2 Command Switch:**

| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| Torque max | 15.32 N·m | 14.64 N·m | **-4.4%** ✓ |
| Max pitch | 5.0° | 4.4° | **-12%** ✓ |

---

#### Analysis: What Does Residual Learning Actually Learn?

**Torque Analysis:**

| Metric | τ_pd (computed) | τ_isaac (actual) |
|--------|-----------------|------------------|
| Range | -46.6 to +29.1 N·m | **-28.6 to +30.0 N·m** |

Isaac Gym applies **torque clipping** at ~30 N·m limits. But clipping alone is NOT sufficient:

| Approach | Turn Command Status |
|----------|---------------------|
| PD Only | **FALLEN** |
| PD + Clipping (±30) | **FALLEN** |
| PD + Residual | **Stable** ✓ |

**Velocity-Dependent Compensation:**

| Velocity Range | Mean Residual | Interpretation |
|----------------|---------------|----------------|
| \|vel\| < 2 rad/s | -0.061 N·m | Negligible |
| \|vel\| 2–5 rad/s | **+0.345 N·m** | Positive compensation |
| \|vel\| 5–10 rad/s | **+1.462 N·m** | Large positive compensation |

Residual learning captures **both** torque clipping AND velocity-dependent compensation. During turns, joint velocities increase significantly, and Isaac Gym has implicit velocity-dependent dynamics that simple clipping cannot replicate.

---

### Approach 3: Domain Randomization

**Concept:** Train policy with domain randomization (DR) and privileged critic to learn robust behaviors that generalize across simulator differences.

**Reference:** Kumar et al., "[RMA: Rapid Motor Adaptation for Legged Robots](https://ashish-kmr.github.io/rma-legged-robots/)" (CoRL 2021) — this draws from Phase 1 (Base Policy Training), constructing robust policies through training in highly diverse environments.

**Method:**
1. Created `go2_rma` environment with domain randomization: friction [0.3, 1.5], mass [-1kg, +2kg], Kp/Kd [0.8×, 1.2×]
2. Asymmetric actor-critic: actor uses 48-dim obs, critic uses 58-dim privileged obs
3. 5000 iterations, final reward: 33.77, tracking accuracy: 94–95%
4. Deployment: actor policy only in MuJoCo

**Results - Baseline (vx=0.5 m/s):**

| Metric | Original Policy | DR-trained Policy | Change |
|--------|-----------------|------------|--------|
| vx error | **0.049 m/s** | 0.072 m/s | +47% ✗ |
| Torque max | 15.71 N·m | **13.01 N·m** | **-17%** ✓ |

**Results - Continuous Turn Command (vx=0.4, wz=1.0):**

<p align="center">
    <img width=45% src="sources/pd_continuous_turn.gif">
    <img width=45% src="sources/randomization_policy.gif">
    </br> Original Policy vs. DR-trained Policy
</p>

| Metric | Original Policy | DR Policy | Change |
|--------|-----------------|------------|--------|
| Status | **FALLEN** | **Stable** | **Fixed!** ✓ |
| vx error | N/A | 0.033 m/s | — |
| Torque max | exploded | 14.36 N·m | — |

**Results - S2 Command Switch:**

| Metric | PD Only | DR Policy | Change |
|--------|---------|-----------|--------|
| Torque max | 15.22 N·m | **13.23 N·m** | **-13%** ✓ |
| Max pitch | 4.7° | **3.9°** | **-17%** ✓ |
| Max roll | 6.0° | **3.5°** | **-42%** ✓ |
| vx error (after) | 0.040 m/s | **0.029 m/s** | **-28%** ✓ |

**Key Insight:** Domain randomization makes the policy robust to simulator differences by learning to handle variations in friction, mass, and gains — which implicitly covers MuJoCo's different dynamics.

---

### Final Comparison: All Approaches

**Transient stability across all scenarios:**

| Approach | S1 Pitch | S1 Roll | S1 Torque | S2 Pitch | S2 Roll | S2 Torque | S3 Pitch | S3 Roll | S3 Torque | Complexity |
|----------|----------|---------|-----------|----------|---------|-----------|----------|---------|-----------|------------|
| **PD Only** | 4.7° | 3.1° | 15.93 N·m | 4.7° | 6.0° | 15.22 N·m | 5.3° | 4.1° | 14.02 N·m | Low |
| **PD + Residual** | 4.8° | **2.4°** | 16.16 N·m | 4.6° | 5.4° | 14.59 N·m | 5.2° | 5.5° | 13.95 N·m | Medium |
| **DR Policy** | 5.2° | 2.8° | **13.85 N·m** | **3.9°** | **3.5°** | **13.23 N·m** | **4.8°** | **2.3°** | **12.18 N·m** | High |
| **ActuatorNet V3** | **1.5°** | 2.5° | 14.03 N·m | **2.8°** | 5.1° | 13.34 N·m | 4.5° | 6.3° | 13.88 N·m | Medium |

**Stability across all deployment scenarios:**

| Approach | Baseline vx error | Constant Turn | S2 Switch | S2 Pitch | S2 Roll | S2 Torque | Complexity |
|----------|-------------------|---------------|-----------|----------|---------|-----------|------------|
| **PD Only** | 0.049 m/s | ✗ **FELL** | ✓ Stable | 4.7° | 6.0° | 15.22 N·m | Low |
| **PD + Clipping** | — | ✗ **FELL** | — | — | — | — | Low |
| **ActuatorNet V1** | 0.277 m/s | — | — | — | — | — | Medium |
| **ActuatorNet V2** | 0.062 m/s | ✓ Stable | ✗ **Unstable** | 17.8° ✗ | 15.3° ✗ | 28.49 N·m ✗ | Medium |
| **ActuatorNet V3** | ~0.06 m/s | ✓ Stable | ✓ **Stable** | **2.8°** ✓ | 5.1° | 13.34 N·m | Medium |
| **PD + Residual** | 0.043 m/s | ✓ Stable | ✓ Stable | 4.6° | 5.4° | 14.59 N·m | Medium |
| **DR Policy** | 0.072 m/s | ✓ Stable | ✓ Stable | **3.9°** | **3.5°** | **13.23 N·m** | High |

**Key Improvements vs PD Only:**

| Metric | Best Method | Improvement |
|--------|-------------|-------------|
| S1 Pitch | ActuatorNet V3 | 4.7° → 1.5° (**68%**) |
| S1 Torque | DR-trained | 15.93 → 13.85 N·m (**13%**) |
| S2 Pitch | ActuatorNet V3 | 4.7° → 2.8° (**40%**) |
| S2 Roll | DR-trained | 6.0° → 3.5° (**42%**) |
| S2 Torque | DR-trained | 15.22 → 13.23 N·m (**13%**) |
| S3 Roll | DR-trained | 4.1° → 2.3° (**44%**) |
| S3 Torque | DR-trained | 14.02 → 12.18 N·m (**13%**) |

**Recommendations:**
- **For best overall robustness:** DR-trained Policy
- **For fastest tracking response:** PD + Residual Learning
- **For specialized pitch control:** ActuatorNet V3 (when systematic data available)
- **Avoid:** ActuatorNet V1/V2 without proper excitation data
- **Avoid:** Simple torque clipping (does not prevent falls)

---

## Usage

### Prerequisites

**1. Clone repository:**
```bash
git clone https://github.com/i-oon/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion.git
cd Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion
```

**2. Download Isaac Gym:**

**IMPORTANT**: Isaac Gym is **NOT included** in this repository due to NVIDIA's license. You must download it separately.

**Step 2.1: Download from NVIDIA**
1. Go to: https://developer.nvidia.com/isaac-gym
2. Sign up / Log in with NVIDIA Developer account (free)
3. Download **Isaac Gym Preview 4** (file: IsaacGym_Preview_4_Package.tar.gz)

**Step 2.2: Extract and place in project**
```bash
# Go to your Downloads folder (adjust path if needed)
cd ~/Downloads

# Extract the archive (This creates a folder called 'isaacgym')
tar -xf IsaacGym_Preview_4_Package.tar.gz

# Move it into the unitree_rl_gym directory
mv isaacgym ~/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/

# Verify it's in the right place
ls ~/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/isaacgym/
# Expected output: docs  python  LICENSE  README.md  ...
```

**Alternative: If you already have Isaac Gym installed elsewhere**
```bash
# Create a symbolic link instead of copying
cd ~/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/unitree_rl_gym/
ln -s /path/to/your/existing/isaacgym isaacgym
```


**3. Install dependencies:**
```bash
cd ~/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion
# Make the script executable
chmod +x setup.sh
# Run the setup (This will create a 'unitree_rl' conda env)
./setup.sh
```
Regarding Dependency Conflicts:
During installation, you may encounter ERROR: pip's dependency resolver... regarding numpy version conflicts with gymnasium or pandas. **This is expected and safe to ignore.** 
The project specifically pins certain library versions to maintain compatibility with the Isaac Gym binaries (Python 3.8).
As long as the Sanity Check at the end of the script displays the following, the environment is fully functional:
- ✓ PyTorch 2.4.1+cu121 (CUDA: True)
- ✓ Isaac Gym OK
- ✓ MuJoCo 3.2.3 OK


**4. Verification:**
```bash
conda activate unitree_rl
# Check if all engines are ready
python -c "import torch; print(f'PyTorch OK (CUDA: {torch.cuda.is_available()})')"
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"
python -c "import mujoco; print('MuJoCo OK')"
```


**5. actuator_net installation:**
```bash
# Navigate to the project root
cd Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion
# Install ActuatorNet requirements
pip install -r actuator_net/requirements.txt
```


**If CUDA errors occur:**
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with matching CUDA version
# See: https://pytorch.org/get-started/locally/
```

### Training

```bash
# Standard policy
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=5000

# DR-trained policy (with domain randomization)
python legged_gym/scripts/train.py --task=go2_rma --headless --max_iterations=5000
```

### Isaac Gym Evaluation

```bash
python unitree_rl_gym/legged_gym/scripts/play.py --task=go2 --num_envs=1
python unitree_rl_gym/legged_gym/scripts/play_logging.py --task=go2
python unitree_rl_gym/legged_gym/scripts/play_cmd_switch.py --task=go2 --scenario S2_turn
```

### MuJoCo Deployment

```bash
# Basic deployment
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2.py go2.yaml

# With metric logging
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_logging.py go2.yaml --duration 10

# Command switching scenarios
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py go2.yaml --scenario S2_turn

# Observation delay test (Stage 3)
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_delay.py go2.yaml --scenario S2_turn --delay 1

# Motor command delay test (Stage 3.5)
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_motor_delay.py go2.yaml --scenario S2_turn --motor_delay 1
```

### Parameter Ablation (Stage 1.5)

```bash
# Kp sweep
for kp in go2_kp_low go2 go2_kp_high go2_kp_40; do
    python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py ${kp}.yaml --scenario S2_turn --no_viewer
done

# Kd sweep
for kd in go2_kd_03 go2 go2_kd_08 go2_kd_10; do
    python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py ${kd}.yaml --scenario S2_turn --no_viewer
done

# Mass perturbation
for mass in go2_mass_m1 go2 go2_mass_p1 go2_mass_p2; do
    python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py ${mass}.yaml --scenario S2_turn --no_viewer
done

# Joint friction sweep
for jf in go2_jfric_none go2 go2_jfric_high; do
    python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py ${jf}.yaml --scenario S2_turn --no_viewer
done

# Motor command delay sweep (Stage 3.5)
for delay in 0 1 2; do
    python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_motor_delay.py go2.yaml --scenario S2_turn --motor_delay ${delay} --no_viewer
done
```

### Mismatch Reduction Controllers

```bash
# PD + Residual Learning
python unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco_go2_residual.py go2.yaml --duration 10 --cmd 0.5 0.0 0.0
python unitree_rl_gym/deploy/deploy_mujoco/deploy_residual_cmd_switch.py --scenario S2_turn

# DR-trained Policy
python unitree_rl_gym/deploy/deploy_mujoco/deploy_rma_cmd_switch.py --scenario S2_turn

# ActuatorNet V3 transient analysis
python unitree_rl_gym/deploy/deploy_mujoco/deploy_transient_actuator_net_v3.py S2 --headless
```

### Data Collection and Training (Bonus)

```bash
# Hwangbo-style excitation data
python unitree_rl_gym/legged_gym/scripts/collect_hwangbo_excitation.py --task=go2

# Residual data
python unitree_rl_gym/legged_gym/scripts/collect_residual_data.py --task=go2

# Train residual network
python unitree_rl_gym/legged_gym/scripts/train_residual_net.py
```

### Generate Plots

```bash
python unitree_rl_gym/legged_gym/scripts/plot_results.py
python unitree_rl_gym/legged_gym/scripts/plot_transient_analysis.py
```

---

## Project Structure

```
unitree_rl_gym/
│
├── legged_gym/
│   ├── envs/go2/
│   │   ├── go2_config.py                              # Base Go2 configuration
│   │   ├── go2_rma_config.py                          # DR-trained config (domain randomization)
│   │   └── go2_rma_env.py                             # DR-trained env (privileged observations)
│   │
│   └── scripts/
│       ├── train.py                                   # Policy training
│       ├── play.py / play_logging.py                  # Playback & logging
│       ├── play_cmd_switch.py                         # Command switching in Isaac Gym
│       ├── collect_actuator_data_for_actuator_net.py  # ActuatorNet V1 data
│       ├── collect_excitation_data.py                 # Policy-driven excitation (V2)
│       ├── collect_hwangbo_excitation.py              # Hwangbo-style excitation (V3)
│       ├── collect_residual_data.py                   # Residual Δτ data
│       ├── train_actuator_net_v2.py                   # Train ActuatorNet V2
│       └── train_residual_net.py                      # Train residual network
│
├── deploy/deploy_mujoco/
│   ├── configs/
│   │   ├── go2.yaml                                   # Base MuJoCo config (Kp=20, Kd=0.5)
│   │   ├── go2_rma.yaml                               # DR-trained policy config
│   │   ├── go2_kp_*.yaml                              # Kp ablation configs (10, 30, 40)
│   │   ├── go2_kd_*.yaml                              # Kd ablation configs (0.3, 0.8, 1.0)
│   │   ├── go2_mass_*.yaml                            # Mass ablation configs (-1kg, +1kg, +2kg)
│   │   ├── go2_jfric_*.yaml                           # Joint friction ablation configs
│   │   └── go2_foot_*.yaml                            # Foot friction ablation configs
│   │
│   ├── deploy_mujoco_go2.py                           # Basic deployment
│   ├── deploy_mujoco_go2_logging.py                   # With metric logging
│   ├── deploy_mujoco_go2_cmd_switch.py                # Command switching
│   ├── deploy_mujoco_go2_delay.py                     # Observation delay test (Stage 3)
│   ├── deploy_mujoco_go2_motor_delay.py               # Motor command delay test (Stage 3.5)
│   ├── deploy_mujoco_go2_residual.py                  # PD + Residual Learning
│   ├── deploy_mujoco_go2_clipping.py                  # Explicit torque clipping
│   ├── deploy_mujoco_go2_actuator_net.py              # ActuatorNet V1
│   ├── deploy_mujoco_go2_actuator_net_v2.py           # ActuatorNet V2
│   ├── deploy_transient_actuator_net_v2.py            # ActuatorNet V2 transient
│   ├── deploy_transient_actuator_net_v3.py            # ActuatorNet V3 transient
│   ├── deploy_transient_analysis.py                   # General transient metrics
│   ├── deploy_rma_cmd_switch.py                       # DR-trained command switching
│   └── sanity_check_*.py                              # Validation scripts
│
├── unitree_mujoco/unitree_robots/go2/
│   ├── go2.xml                                        # Base Go2 robot model
│   ├── scene_flat.xml                                 # Flat terrain (baseline)
│   ├── scene_foot_02.xml                              # μ_foot = 0.2
│   ├── scene_foot_08.xml                              # μ_foot = 0.8
│   ├── scene_mass_m1.xml                              # Base mass -1 kg
│   ├── scene_mass_p1.xml                              # Base mass +1 kg
│   ├── scene_mass_p2.xml                              # Base mass +2 kg
│   ├── scene_jfric_none.xml                           # No joint friction
│   ├── scene_jfric_high.xml                           # High joint friction
│   └── variants/
│       ├── go2_mass_m1.xml                            # Go2 base mass 5.921 kg (-1 kg)
│       ├── go2_mass_p1.xml                            # Go2 base mass 7.921 kg (+1 kg)
│       ├── go2_mass_p2.xml                            # Go2 base mass 8.921 kg (+2 kg)
│       ├── go2_jfric_none.xml                         # damping=0, frictionloss=0
│       └── go2_jfric_high.xml                         # damping=0.3, frictionloss=0.5
│
├── scripts/
│   └── plot_results.py                                # Generate comparison plots
│
├── logs/
│   ├── rough_go2/Jan30_23-48-03_/model_5000.pt       # Trained standard policy
│   ├── go2_rma/Jan31_20-18-52_/model_5000.pt         # Trained DR policy
│   ├── sim2sim/                                       # Baseline & ablation logs
│   │   ├── cmd_switch/                                # Command switching logs
│   │   ├── delay/                                     # Observation delay logs
│   │   └── motor_delay/                               # Motor command delay logs
│   ├── transient_analysis/                            # Transient response logs
│   ├── residual_net.pt                                # Trained residual model
│   ├── residual_scaler.pkl                            # Residual feature scaler
│   ├── actuator_net_v2.pt                             # Trained ActuatorNet V2
│   ├── actuator_net_v2_scaler_X.pkl                   # V2 input scaler
│   └── actuator_net_v2_scaler_y.pkl                   # V2 output scaler
│
├── plots/                                             # Generated figures
└── README.md
```

**External dependency:**
```
~/actuator_net/app/resources/
├── hwangbo_excitation_data.csv   # Hwangbo excitation data (V3)
├── actuator.pth                  # Trained ActuatorNet V3 model
├── scaler.pkl                    # V3 input scaler
└── motor_data.pkl                # Processed training data
```

---

## Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Isaac Gym)                         │
│  train.py → model_5000.pt → export → policy_1.pt               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              SIMULATORS & CONFIGURATION                         │
│  Interface alignment · Parity checks · Deployment checklist     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 1: BASELINE COMPARISON                    │
│  Isaac Gym ←→ MuJoCo | Scenarios: S1 Stop, S2 Turn, S3 Lateral │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 1.5: PARAMETER ABLATION (One Factor at a Time)     │
│  Kp · Kd · Mass · Joint friction · dt · Floor friction          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: FOOT FRICTION SWEEP                       │
│  μ_foot: 0.2, 0.4, 0.8 | Finding: μ_foot=0.8 reduces pitch 88%│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: OBSERVATION DELAY                         │
│  0 / 20 / 40 ms | Finding: 20ms causes FALL in S2 Turn         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 3.5: MOTOR COMMAND DELAY                       │
│  0 / 20 / 40 ms | Finding: less critical than obs delay        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              BONUS: MISMATCH REDUCTION                          │
│  ActuatorNet V1→V2→V3 · Residual Learning · DR-trained Policy  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Author


Disthorn Suttawet | FIBO, KMUTT | January 2026
