# B1 Policy Deployment Evaluation Log

**Policy:** `logs/ppo_b1/raw_wz_FINAL/model_final.pt`
**Trained in:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0
**Variant:** raw-wz (`heading_command=False`)
**Policy is fixed.** All stages use the same weights. Only the simulator and actuator model change.

---

## Purpose

This document evaluates how the same fixed policy behaves as the simulation environment and
actuator model change. The goal is not to compare policies — it is to measure the physics gap
between simulators and between simulation and real hardware, to determine whether the policy is
ready for real B1 deployment.

**Core question:** How much does performance degrade as we move away from the training environment?

---

## Evaluation Stages

| Stage | Simulator | Actuator | Purpose |
|---|---|---|---|
| **S1** | Isaac Sim 4.5 | Ideal PD (kp=300, kd=5) | **Source baseline** |
| **S2** | Isaac Sim 5.0 | Ideal PD (kp=300, kd=5) | Prove Isaac 4.5 → 5.0 gap is acceptable |
| **S3** | MuJoCo | Ideal PD (kp=300, kd=5) | Measure physics/contact sim-to-sim gap |
| **S4** | MuJoCo | Actuator net (sim-trained) | Does actuator modeling close the MuJoCo gap? |
| **S5** | MuJoCo | Actuator net (real B1 data) | Does real actuator net match hardware behavior? |
| **S6** | **Real B1** | Real hardware | Ground truth — hardware deployment |

**Decision gate:** S6 is only attempted after S1–S5 show acceptable gaps.

---

## Experiment Matrix

For each stage, run all command test conditions and all robustness conditions.
Every cell = 20 episodes × 1000 steps (20 s sim time) unless noted.

### Command Test Matrix

| ID | vx (m/s) | vy (m/s) | wz (rad/s) | Description |
|---|---|---|---|---|
| C1 | 0.0 | 0.0 | 0.0 | Stand still |
| C2 | 0.3 | 0.0 | 0.0 | Walk forward slow |
| C3 | 0.6 | 0.0 | 0.0 | Walk forward medium |
| C4 | 0.8 | 0.0 | 0.0 | Walk forward fast |
| C5 | −0.4 | 0.0 | 0.0 | Walk backward |
| C6 | 0.0 | +0.4 | 0.0 | Strafe left |
| C7 | 0.0 | −0.4 | 0.0 | Strafe right |
| C8 | 0.5 | +0.3 | 0.0 | Diagonal fwd-left |
| C9 | 0.5 | −0.3 | 0.0 | Diagonal fwd-right |
| C10 | 0.0 | 0.0 | +1.0 | Turn left |
| C11 | 0.0 | 0.0 | −1.0 | Turn right |
| C12 | 0.4 | 0.0 | +0.5 | Forward + turn left |
| C13 | random | random | random | Random omnidirectional (full range) |

### Robustness Test Conditions

| ID | Perturbation | Value | Purpose |
|---|---|---|---|
| R1 | None | — | Nominal |
| R2 | Payload | +10 kg trunk mass | Sensor/equipment mounting |
| R3 | Payload | +20 kg trunk mass | Heavy payload limit |
| R4 | Floor friction | μ_static = 0.4 | Tile / smooth floor |
| R5 | Floor friction | μ_static = 0.8 | Nominal concrete |
| R6 | Floor friction | μ_static = 1.2 | Rubber mat |
| R7 | External push | 80 N lateral, 0.1 s | Disturbance recovery |
| R8 | Actuator gain noise | kp/kd ±25% | Real hardware gain uncertainty |
| R9 | Observation noise | ±0.2 m/s lin_vel, ±0.05 rad/s ang_vel | Sensor noise |

---

## Metrics

All metrics computed per episode, then averaged across 20 episodes per condition.

### Tracking

| Metric | Symbol | Unit | How computed |
|---|---|---|---|
| Forward velocity RMSE | `e_vx` | m/s | `sqrt(mean((vx_actual − vx_cmd)²))` |
| Lateral velocity RMSE | `e_vy` | m/s | `sqrt(mean((vy_actual − vy_cmd)²))` |
| Yaw rate RMSE | `e_wz` | rad/s | `sqrt(mean((wz_actual − wz_cmd)²))` |
| Yaw drift at wz=0 | `yaw_drift` | °/s | mean \|d(yaw)/dt\| when wz_cmd=0 |

### Stability

| Metric | Symbol | Unit | How computed |
|---|---|---|---|
| Fall rate | `fall_rate` | % | episodes with base_contact > 50 N / total |
| Episode completion | `completion` | % | episodes reaching 1000 steps without fall |
| Base height mean | `h_mean` | m | mean trunk height |
| Base height std | `h_std` | m | std trunk height |
| Roll RMS | `roll_rms` | rad | `sqrt(mean(roll²))` |
| Pitch RMS | `pitch_rms` | rad | `sqrt(mean(pitch²))` |

### Gait

| Metric | Symbol | Unit | How computed |
|---|---|---|---|
| Duty factor per leg | `duty_FL/FR/RL/RR` | % | contact time / episode time per leg |
| Foot slip rate | `slip` | m/s | mean foot velocity magnitude while in contact |
| Air time mean | `air_mean` | s | mean swing phase duration per foot |

### Actuator / Joint

| Metric | Symbol | Unit | How computed |
|---|---|---|---|
| Action rate RMS | `act_rate` | rad/s | `sqrt(mean((a_t − a_{t-1})² / dt²))` |
| Joint velocity RMS | `qd_rms` | rad/s | RMS of all 12 joint velocities |
| Joint jerk RMS | `qdd_rms` | rad/s² | RMS of joint acceleration estimates |
| Torque RMS | `tau_rms` | N·m | RMS of all 12 joint torques |
| Torque saturation | `tau_sat` | % | timesteps with \|τ\| > 75 N·m / total |
| Joint limit violations | `jlim_viol` | % | timesteps with any joint outside URDF limits |

### Sim-to-Sim Gap

For each metric `m`, the gap relative to the S1 source baseline:

```
gap(m) = |metric_SX(m) − metric_S1(m)|
gap_pct(m) = gap(m) / metric_S1(m) × 100%
```

Report `gap_pct` for every metric at every stage.

---

## Pass / Fail Criteria

### Per-Stage Thresholds

| Metric | Source (S1) target | Pass threshold | Fail threshold |
|---|---|---|---|
| Fall rate | < 2% | < 5% | ≥ 10% |
| Episode completion | > 98% | > 90% | < 80% |
| `e_vx` (C3, C4) | < 0.10 m/s | < 0.20 m/s | ≥ 0.35 m/s |
| `e_vy` (C6, C7) | < 0.15 m/s | < 0.25 m/s | ≥ 0.40 m/s |
| `e_wz` (C10, C11) | < 0.15 rad/s | < 0.30 rad/s | ≥ 0.50 rad/s |
| `yaw_drift` (C1–C9) | < 2 °/s | < 5 °/s | ≥ 10 °/s |
| `h_mean` | 0.53 ± 0.01 m | 0.53 ± 0.04 m | outside 0.45–0.60 m |
| `roll_rms` | < 0.02 rad | < 0.05 rad | ≥ 0.10 rad |
| `pitch_rms` | < 0.02 rad | < 0.05 rad | ≥ 0.10 rad |
| `duty_FL/FR/RL/RR` | 45–60% | 35–70% | any leg < 20% or > 85% |
| `tau_sat` (>100 N·m) | < 5% | < 20% | ≥ 40% |
| `jlim_viol` | 0% | < 1% | ≥ 5% |

### Gap Thresholds (relative to S1)

| Gap | Verdict |
|---|---|
| < 10% on all key metrics | ✅ **Pass** — simulator is equivalent |
| 10–25% on any key metric | ⚠️ **Warning** — investigate before proceeding |
| > 25% on any key metric | ❌ **Fail** — gap too large, do not proceed to next stage |

Key metrics for gap decision: `fall_rate`, `e_vx`, `e_vy`, `e_wz`, `yaw_drift`, `h_mean`.

---

## What Each Comparison Proves

### S2 − S1: Isaac Sim 4.5 vs 5.0 (same policy, same actuator)
**Proves:** The PhysX contact solver update between Isaac Sim versions does not meaningfully
change how the policy behaves. If gap is small, we can use Isaac Sim 5.0 for future development
without retraining. If gap is large, the policy must be fine-tuned in 5.0 before hardware.

### S3 − S1: MuJoCo vs Isaac Sim 4.5 (ideal PD in both)
**Proves:** How different the two simulators' physics engines are under identical ideal control.
MuJoCo and PhysX have fundamentally different contact and solver implementations. This gap is
the **baseline physics gap** — the lower bound on sim-to-real error that cannot be closed by
actuator modeling alone.

### S4 − S3: MuJoCo + actuator net vs MuJoCo ideal PD
**Proves:** Whether adding a sim-trained actuator model (to capture PD lag, bandwidth limits,
and torque saturation) reduces the observed MuJoCo gap. If S4 is not better than S3, the
actuator net provides no benefit and may even destabilize the policy.

### S5 − S4: Real B1 actuator net vs sim-trained actuator net
**Proves:** How much of the sim-to-real actuator gap is captured by real hardware data. A
smaller S5−S4 gap means the real B1 actuator behavior is close to what the sim-trained net
models — the remaining gap is geometry and contact, not actuation. A large S5−S4 gap means
real motor dynamics differ significantly from sim and must be addressed in training.

### S6 − S5: Real B1 vs best simulation estimate
**Proves:** The irreducible sim-to-real gap after all modeling improvements. This is the final
check. If S6 metrics are within pass thresholds, the policy is ready to deploy.

---

## Decision Gate Summary

```
S1 complete (source baseline established)
    ↓
S2 gap < threshold?  YES → Isaac 5.0 safe for future work
                     NO  → Retrain or fine-tune in Isaac 5.0 before hardware
    ↓
S3 gap < threshold?  YES → Physics gap is acceptable
                     NO  → Identify which metrics fail; consider MuJoCo DR in training
    ↓
S4 gap < S3 gap?     YES → Actuator net helps; use S4 for future sim evals
                     NO  → Actuator net hurts or is neutral; keep ideal PD for now
    ↓
S5 gap < threshold?  YES → Real actuator net is accurate; proceed to hardware
                     NO  → Collect more real data; retrain actuator net
    ↓
S6 pass?             YES → Policy ready for operation
                     NO  → Identify failure mode; retrain with identified gap as DR
```

---

## Results Table Template

Copy one block per stage. Fill in after running each evaluation.

```
================================================================================
Stage: S__   Simulator: ___________   Actuator: ___________   Date: __________
================================================================================

Condition: C__ / R__

  Fall rate          : _____%   (pass < 5%)
  Episode completion : _____%   (pass > 90%)

  Tracking RMSE:
    e_vx  (C3/C4)   : _____ m/s
    e_vy  (C6/C7)   : _____ m/s
    e_wz  (C10/C11) : _____ rad/s
    yaw_drift       : _____ °/s

  Stability:
    h_mean ± h_std  : _____ ± _____ m
    roll_rms        : _____ rad
    pitch_rms       : _____ rad

  Gait:
    duty FL/FR/RL/RR: __%  __%  __%  __%
    foot_slip       : _____ m/s
    air_time_mean   : _____ s

  Actuator:
    act_rate_rms    : _____ rad/s
    qd_rms          : _____ rad/s
    tau_rms         : _____ N·m
    tau_sat         : _____%
    jlim_viol       : _____%

  Gap vs S1 (key metrics):
    fall_rate gap   : _____ pp
    e_vx gap        : _____ m/s  (____%)
    e_vy gap        : _____ m/s  (____%)
    e_wz gap        : _____ rad/s (____%)
    yaw_drift gap   : _____ °/s  (____%)
    h_mean gap      : _____ m    (____%)

  Verdict: [ PASS / WARNING / FAIL ]
  Notes:
    -
================================================================================
```

---

## Stage S1 — Source Baseline (Isaac Sim 4.5, Ideal PD)

```
================================================================================
Stage: S1   Simulator: Isaac Sim 4.5   Actuator: Ideal PD (kp=300, kd=5)
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
Conditions run: C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13  /  R1 (nominal)
Episodes: 20 per condition x 1000 steps (20 s)   Date: 2026-06-09 02:03
================================================================================

  Fall rate          :      0.0%   (pass < 5%)
  Episode completion :    100.0%   (pass > 90%)

  Tracking RMSE:
    e_vx  (C3/C4)    :    0.152 m/s
    e_vy  (C6/C7)    :    0.121 m/s
    e_wz  (C10/C11)  :    0.209 rad/s
    yaw_drift        :     3.01 deg/s

  Stability:
    h_mean +/- h_std :  0.535 +/- 0.006 m
    roll_rms         :    0.0196 rad
    pitch_rms        :    0.0278 rad

  Gait:
    duty FL/FR/RL/RR :  57.6%  58.1%  49.9%  51.3%
    foot_slip        :    0.056 m/s
    air_time_mean    :    0.056 s

  Actuator:
    act_rate_rms     :   17.91 rad/s
    qd_rms           :    0.99 rad/s
    tau_rms          :   33.22 N·m
    tau_sat (>100Nm) :    13.7%
    jlim_viol        :     0.0%

  Per-condition detail:
    cond | e_vx  e_vy  e_wz | yaw°/s | h_mean | duty FL   FR   RL   RR
    C1   | 0.046 0.048 0.058 |   2.62 | 0.535  | 58.0  58.9  49.8  51.5
    C2   | 0.072 0.048 0.062 |   2.77 | 0.536  | 57.8  58.6  50.3  51.5
    C3   | 0.111 0.044 0.060 |   2.66 | 0.536  | 57.5  58.3  50.0  51.8
    C4   | 0.192 0.050 0.063 |   2.81 | 0.537  | 57.3  57.2  49.6  52.1
    C5   | 0.086 0.052 0.064 |   2.84 | 0.534  | 57.3  58.8  49.8  52.0
    C6   | 0.050 0.127 0.074 |   3.40 | 0.536  | 58.0  57.0  50.8  51.9
    C7   | 0.047 0.115 0.074 |   3.34 | 0.534  | 57.4  58.1  49.2  52.0
    C8   | 0.143 0.100 0.071 |   3.20 | 0.535  | 58.7  57.5  50.1  50.3
    C9   | 0.119 0.090 0.076 |   3.44 | 0.535  | 56.4  58.6  49.7  51.4
    C10  | 0.049 0.060 0.181 |    N/A | 0.535  | 58.6  57.7  49.2  49.8
    C11  | 0.046 0.053 0.238 |    N/A | 0.535  | 57.2  57.7  49.2  51.2
    C12  | 0.055 0.053 0.079 |    N/A | 0.535  | 57.7  58.1  51.0  50.9
    C13  | 0.068 0.067 0.081 |    N/A | 0.536  | 57.2  58.0  49.6  50.6

  Verdict: [ PASS ]
  Notes:
    - yaw_drift 3.0°/s at wz=0 passes <5°/s threshold but is above the <2°/s
      source target. Expected: yaw_stability_penalty dead zone (~0.2 rad/s)
      means the policy is never penalized for this level of drift.
      Use 3.01°/s as the S1 reference for S2-S6 gap comparisons.
    - Front/rear duty asymmetry (~8%): FL/FR ~58%, RL/RR ~50%.
      Consistent across all conditions — B1 mass distribution effect.
    - e_wz C10 vs C11: 0.181 vs 0.238 rad/s — turn-right harder to track
      than turn-left in this run. Variation is stochastic (no fixed seed).
    - tau_sat 13.7% at 100 N·m threshold — passes <20%.
      B1 joint effort limits ~140 N·m peak.
================================================================================
```

---

## Stage S2 — Isaac Sim 5.0, Ideal PD

*Fill in after running S2 evaluation.*

```
================================================================================
Stage: S2   Simulator: Isaac Sim 5.0   Actuator: Ideal PD (kp=300, kd=5)
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
================================================================================

[ Results pending ]

  Decision: Isaac 5.0 safe for future development?  [ YES / NO ]

================================================================================
```

---

## Stage S3 — MuJoCo, Ideal PD

*Fill in after running S3 evaluation.*

```
================================================================================
Stage: S3   Simulator: MuJoCo   Actuator: Ideal PD (kp=300, kd=5)
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
================================================================================

[ Results pending ]

  Decision: Physics gap acceptable?  [ YES / NO ]

================================================================================
```

---

## Stage S4 — MuJoCo, Sim-Trained Actuator Net

*Fill in after running S4 evaluation.*

```
================================================================================
Stage: S4   Simulator: MuJoCo   Actuator: Actuator net (sim-trained)
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
================================================================================

[ Results pending ]

  Decision: Actuator net reduces gap vs S3?  [ YES / NO ]

================================================================================
```

---

## Stage S5 — MuJoCo, Real B1 Actuator Net

*Fill in after collecting real B1 data and training actuator net.*

```
================================================================================
Stage: S5   Simulator: MuJoCo   Actuator: Actuator net (real B1 data)
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
================================================================================

[ Results pending ]

  Decision: Real actuator net gap acceptable? Proceed to hardware?  [ YES / NO ]

================================================================================
```

---

## Stage S6 — Real B1 Hardware

*Only fill in after S1–S5 pass.*

```
================================================================================
Stage: S6   Platform: Real Unitree B1   Date: __________
Policy: logs/ppo_b1/raw_wz_FINAL/model_final.pt
Pre-flight checklist: [ ] imu_ang_vel published  [ ] low-level mode  [ ] gains ramped
================================================================================

[ Results pending ]

  Decision: Policy ready for operation?  [ YES / NO ]
  Failure modes observed:
    -

================================================================================
```

---

## Notes and Observations

*Record unexpected findings, hardware issues, or config changes during evaluation.*

| Date | Stage | Note |
|---|---|---|
| | | |
