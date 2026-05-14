# Transition-Aware Quadruped Locomotion: A Study of Residual Correction Spaces

**Course:** FRA 503 — Deep Reinforcement Learning
**Student:** Disthorn Suttawet (66340500019)
**Robot:** Unitree B1 quadruped (12 DOF, ~50 kg)
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0
**Deadline:** 20 May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background and Design Motivation](#2-background-and-design-motivation)
3. [Phase 1 — Base Gait Policy Generation](#3-phase-1--base-gait-policy-generation)
4. [Phase 2 — Residual Transition Learning](#4-phase-2--residual-transition-learning)
5. [Development History — From Prototype to Design-Space Study](#5-development-history--from-prototype-to-design-space-study)
6. [Systematic Design-Space Study (2×2 Ablation)](#6-systematic-design-space-study-22-ablation)
7. [Experiments and Metrics](#7-experiments-and-metrics)
8. [Results](#8-results)
9. [Discussion](#9-discussion)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Reproducibility](#11-reproducibility)
12. [B1 Robot Configuration](#12-b1-robot-configuration)

---

## 1. Project Overview

### Motivation

Quadruped robots operating across terrains and tasks require different gait patterns — trots for cruising, bounds for speed, paces for lateral coordination. Switching between these gaits without losing balance or wasting energy is a fundamental deployment challenge. Gaits with different leg-pair coordination structures (diagonal vs fore-aft vs lateral) cannot be linearly interpolated at the joint level — naive switching creates kinematic shocks that can stagger or destabilize a 50 kg robot.

### Problem Statement

A hand-designed **smoothstep** blending schedule already outperforms discrete switching on transition continuity. But it still produces velocity reversal on hard gait-pair transitions. The question is whether a small learned residual correction can close this gap without requiring a full learned policy.

### Final Research Question

> **How do residual correction space (α-space vs q-space) and action dimension (4D per-leg vs 12D per-joint) affect jerk reduction, velocity reversal, energy cost, and robustness during quadruped gait transition?**

This question is investigated through a systematic **2×2 ablation**: output space (α vs q) × action dimension (4D vs 12D). The study reveals trade-offs rather than a single winner: all residual variants reduce transition-window jerk, but velocity safety rankings are gait-phase-dependent.

### Contributions

- Formulated gait transition as a **residual correction problem** on top of frozen policy blending, comparing q-space (joint-position corrections Δq) and α-space (blending-schedule corrections Δα) as correction targets.
- Developed a stable **residual-learning recipe** (time-gating, asymmetric sigmoid clamp, jerk penalty, sparsity) through a v1–v10 prototype sequence using the simpler 4D α-space model before expanding to the full design space.
- Conducted a **2×2 design-space study** (α vs q) × (4D vs 12D) with all other factors held fixed, revealing that both output space and action dimension affect jerk, reversal, and robustness in ways that interact with gait-phase diversity.
- Demonstrated that all four residual variants reduce transition-window jerk vs Smoothstep (−6% to −16% canonical; −6% to −20% at N=60). Velocity safety is gait-phase-dependent: Residual-α 12D achieves zero reversal at the canonical fixed gait phase; Residual-q 4D achieves zero reversal across 60 diverse gait phases.

### Key Results

**Canonical evaluation** (seed=42, 2500 steps, 6 directed gait-pair transitions):

| Method | vx_mean | vx_min | CoT | **jerk_TRANS** |
|---|---:|---:|---:|---:|
| Discrete Switch | +0.435 | −0.195 | 2.793 | 11361 |
| Linear Ramp | +0.390 | −0.206 | 1.955 | 7441 |
| Smoothstep Ramp | +0.415 | −0.096 | 2.090 | 8508 |
| Residual-α 4D | +0.430 | −0.086 | 2.171 | 7617 |
| Residual-q 4D | +0.416 | −0.024 | 2.158 | 7320 |
| Residual-q 12D | +0.408 | −0.122 | 2.064 | 7719 |
| Residual-α 12D | +0.427 | +0.004 | 2.105 | 7951 |

**Multi-seed evaluation** (N=60: 10 seeds × 6 gait pairs, `--randomize_start`):

| Method | jerk_TRANS mean | vs Smoothstep | reversal rate | worst vx_min |
|---|---:|---:|---:|---:|
| Discrete Switch | 10166 | +11.7% | 18% | −0.527 |
| Smoothstep Ramp | 9102 | — | **55%** | −0.236 |
| Residual-α 12D | 8570 | −5.8% | 30% | −0.217 |
| Residual-α 4D | 8185 | −10.1% | 7% | −0.167 |
| Residual-q 4D | **7619** | **−16.3%** | **0%** | **+0.072** |
| Residual-q 12D | 7305 | −19.7% | 38% | −0.429 |

**The robust finding** (holds at both N=6 and N=60): all four residual variants beat Smoothstep on transition-window jerk. **The gait-phase-dependent finding**: velocity reversal safety rankings differ between evaluations — Res-α 12D has zero reversal at the fixed canonical gait phase but 30% reversal across diverse phases; Res-q 4D has the reverse pattern (mild reversal at canonical, zero reversal at N=60). **Base-swap validation**: the learned residual is schedule-calibrated — the MLP reduces Smoothstep jerk by −6.5% but raises Linear Ramp jerk by +12.2%, confirming it is a genuine contextual correction rather than a generic transition policy.

---

## 2. Background and Design Motivation

### Quadruped Gait Transition Problem

Trot, bound, and pace have fundamentally different leg-pair synchrony structures:
- **Trot** — diagonal pairs in phase (FL+RR, FR+RL)
- **Bound** — fore-aft pairs in phase (FL+FR, RL+RR)
- **Pace** — lateral pairs in phase (FL+RL, FR+RR)

The midpoint between "FL+RR planted" (trot) and "FL+FR planted" (bound) is **not a valid quadruped configuration**. Naive blending of joint targets produces incoherent commands that destabilize the body.

### Why Naive Discrete Switching Fails

The simplest strategy is to set α = 1 immediately at the switch command. The result is a kinematic shock — joint targets jump discontinuously:

![Discrete switch spike](logs/phase2/discrete_spike.png)

At switch time (t = 0): max joint velocity spikes to **16.4 rad/s** (6.2× steady-state), jerk peaks at **19 189 rad/s³**, and forward velocity drops sharply. Generated by `scripts/plot_discrete_spike.py`.

### Why Passive Blending Helps but Is Limited

A smoothstep schedule already eliminates endpoint kinematic kicks through zero-derivative endpoints (`dα/dt = 0` at t=0 and t=T). In the canonical seed=42 run, Smoothstep reduces jerk_TRANS from 11361 to 8508 and reduces the worst vx_min from −0.195 to −0.096. However it still produces **velocity reversal** on hard gait-pair transitions — the robot momentarily moves backward during a trot↔bound or pace↔bound switch.

### Why Residual Learning Is Appropriate

A residual MLP trained on top of Smoothstep only needs to learn the **missing correction** — not the full transition from scratch. This is a smaller, safer, and more interpretable problem:
- **Smaller search space**: the MLP corrects a reasonable schedule, not an arbitrary function.
- **Built-in safety fallback**: Δα = 0 exactly recovers Smoothstep.
- **Time-gated**: the residual is forced to zero outside the transition window, so source and target gaits run untouched during steady-state holds.
- **Interpretable**: the correction Δα directly shows where and when the MLP disagrees with the baseline schedule.

### Why Smoothstep Is the Residual Baseline

Smoothstep is chosen as the baseline for four reasons:

1. **Deterministic and fair.** Smoothstep contains no learned parameters. Any gain over Smoothstep is entirely attributable to the residual MLP.
2. **Zero-derivative endpoints.** `α = x²(3−2x)` has `dα/dt = 0` at both x=0 and x=1, removing kinematic kicks at ramp-start and ramp-end. Linear ramp lacks this property.
3. **Clean counterfactual.** Setting `Δα = 0` exactly recovers Smoothstep. This makes the contribution of the learned residual directly measurable.
4. **Strong enough to be meaningful.** Smoothstep is a competitive passive baseline — it outperforms discrete switching on jerk and reversal. Improving on it indicates a genuine gain from the learned correction.

*"We choose Smoothstep as the residual baseline because it is deterministic, interpretable, and already removes endpoint discontinuities through zero endpoint slope. Setting Δα = 0 exactly recovers Smoothstep, so the effect of the learned residual can be measured directly."*

### Why Residual Learning Instead of a Fully Learned Blending Policy

A fully learned blending policy — one that outputs the entire α schedule without a structured prior — would face a harder problem:
- It must learn when to start, how fast to ramp, how to keep α monotonic, and how to avoid unsafe midpoints, all simultaneously.
- It has no built-in safety fallback: if training fails or generalizes poorly, there is no graceful degradation.
- It can corrupt steady-state gaits unless explicitly prevented.

The residual formulation constrains the policy to learn only the correction on top of a working baseline:

*"We do not ask the policy to invent the full blending equation because this makes the transition problem unnecessarily unconstrained. A fully learned scheduler could collapse to fast switching, delay the transition, or corrupt steady-state gaits. Instead, Smoothstep provides a stable prior and the residual learns only the missing correction. This makes the problem smaller, safer, and easier to interpret."*

---

## 3. Phase 1 — Base Gait Policy Generation

### Initial Direction — CPG-RBF + PIBB

The original Phase 1 design used a **CPG-RBF (Central Pattern Generator + Radial Basis Function)** controller optimized with **PI^BB (Thor et al. 2021)**. After ~3 weeks of iteration and 12 documented encoding experiments, this approach was abandoned.

The structural failure was a **representation mismatch**: the indirect RBF encoding uses one weight matrix W (20×3, 60 params) shared across all 4 legs — per-leg differences come only from integer phase offsets. B1 has a 0.2 rad asymmetry between front thighs (0.8 rad) and rear thighs (1.0 rad). No single W can simultaneously produce the correct swing arc for both front and rear leg pairs.

**Experiment A** (old weights, corrected env): vx = 0.000 m/s — robot stands still.
**Experiment B** (retrain in corrected env): vx = +0.091 m/s — oscillatory lunge. Compare: PPO trot achieves +0.434 m/s.

| Metric | A: old weights | B: retrained | PPO trot |
|---|---:|---:|---:|
| mean vx | 0.000 m/s | +0.091 m/s | 0.434 m/s |
| FL/FR/RL/RR duty | 28/100/100/100% | 85/77/47/50% | ~40/33/39/65% |
| locomotion | stands still | oscillatory lunge | stable trot |

### Pivot to PPO Base Gait Policies

Four PPO velocity-tracking policies trained on flat terrain. Stored at `logs/phase1_final/`.

| Gait | Coordination | Duty FL/FR/RL/RR | Body height | Cycle | Speed |
|---|---|---:|---:|---:|---:|
| **trot_v2** | Diagonal (FL+RR / FR+RL) | 40/33/39/65% | 0.43 m | 1.6 Hz | 0.5 m/s |
| **bound_v4** | Fore-aft (FL+FR / RL+RR) | 65/65/33/34% | 0.39 m | 2.5 Hz | 0.5 m/s |
| **pace_v2** | Lateral (FL+RL / FR+RR) | 30/69/30/69% | 0.40 m | 2.5 Hz | 0.45 m/s |
| **steer_v2** | Asymmetric trot for turning | 39/16/27/35% | 0.42 m | 1.7 Hz | 0.25 m/s |

**Gait quality caveat.** These are velocity-tracking policies, not biologically faithful gaits. Duty cycles deviate from natural locomotion (e.g., trot FR at 33% stance vs biological ~50%). Phase 2's transition-smoothness results hold on top of these policies, but the base policies themselves should not be presented as accurate gait models.

Phase 2 uses only trot, bound, and pace. Steer is excluded because its training range (`yaw ∈ (0.4, 1.0)`) is incompatible with Phase 2's fixed `yaw=0` command.

### Phase 1 Gait Diagrams

Foot contact bars (blue = stance, white = swing):

**Trot** — diagonal pairs: FR+RL co-swing while FL+RR co-swing.

![Trot gait diagram](logs/gait_trot_v2.png)

**Bound** — fore-aft pairs: FL+FR swing together, RL+RR swing together.

![Bound gait diagram](logs/gait_bound_v4.png)

**Pace** — lateral pairs: FL+RL swing together, FR+RR swing together.

![Pace gait diagram](logs/gait_pace_v2.png)

---

## 4. Phase 2 — Residual Transition Learning

### Architecture Overview

**Two-phase design:**

```
                    ┌──────────────────────────────┐
                    │  Per-joint Residual MLP      │
                    │  [obs(45) → 128 → 128 → 12]  │
                    │  outputs Δα ∈ [0, +0.3]      │
                    │  ELU activation              │
                    └─────────┬────────────────────┘
                              │ (Δα_0, Δα_1, …, Δα_11) per joint
                              ▼
   π_current ─────┐    ┌──────────────────────┐
   π_target  ─────┼───▶│ Per-joint blending   │──▶ joint_targets → B1
   α_baseline ────┘    │ α_j = α_base + Δα_j  │
   (3 s smoothstep)    │ × time-gating mask   │
                       └──────────────────────┘
```

### Per-Joint Blending Math

```python
# 1. MLP forward pass — per-joint residual
delta_alpha_raw = MLP(obs)                                   # (12,)
delta_alpha     = sigmoid(delta_alpha_raw) × delta_alpha_max  # ∈ [0, +0.3]

# 2. Time-gating: residual is zero outside transition window
in_window       = (transition_start − pad) ≤ t ≤ (transition_end + pad)
delta_alpha     = delta_alpha if in_window else 0

# 3. Baseline schedule (smoothstep — Hermite 3x²−2x³)
x              = clamp((t − transition_start_s) / transition_duration_s, 0, 1)
alpha_baseline = x*x*(3 − 2*x)     # dα/dt = 0 at endpoints → no kinematic kick

# 4. Per-joint α and blending
for joint_j in {0 … 11}:
    α_j = clamp(alpha_baseline + delta_alpha[j], 0, 1)
    blended[j] = (1 − α_j) · π_current(obs)[j]
               +      α_j · π_target(obs)[j]

# 5. Joint commands
joint_target = default_joint_pos + 0.25 × blended
```

### Why Per-Joint, Not Per-Leg Scalar

Trot, bound, and pace have different leg-pair sync structures. During trot→bound, FL must decouple from its diagonal partner (RR) and recouple with its fore-pair partner (FR). Per-leg 4D α allows independent leg timing. Per-joint 12D further allows each hip, thigh, and knee joint to transition at its own rate. The 2×2 ablation (Section 6) shows that within α-space, 12D achieves zero velocity reversal at the canonical fixed gait phase (N=6) — a property that the 4D variant cannot provide at that phase.

### Observation Space (45-D)

```
base_lin_vel       (3)   robot's linear velocity in body frame
base_ang_vel       (3)   robot's angular velocity in body frame
projected_gravity  (3)   gravity direction in body frame
joint_pos_rel      (12)  joint angles relative to default pose
joint_vel          (12)  joint velocities
last_residual      (4)   per-leg mean of 12D output (obs stays 45-D)
gait_current_oh    (3)   one-hot encoding of current source gait
gait_target_oh     (3)   one-hot encoding of target gait
alpha_baseline     (1)   current α from baseline schedule
cycles_elapsed     (1)   time elapsed in episode (1 Hz CPG-equivalent)
```

### Reward Function (Training)

| Term | Weight | Description |
|---|---:|---|
| Velocity tracking | +1.5 | `exp(−‖cmd_xy − vel_xy‖² / 0.25)` |
| Yaw tracking | +0.75 | `exp(−(cmd_yaw − ωz)² / 0.25)` |
| Body orientation | −2.0 | `‖projected_gravity_xy‖²` — steady-state upright |
| Orientation (in-window) | −8.0 | Same term ×4 during transition window only |
| Body height | −50.0 | `(h − 0.42)²` |
| Δα smoothness | −0.15 | `‖Δα_t − Δα_{t−1}‖²` — step-to-step action rate |
| Joint acceleration | −2.5×10⁻⁷ | `‖q̈‖²` — defensive PD-overshoot penalty |
| Joint jerk | −1×10⁻¹⁰ | `‖(q̈_t − q̈_{t−1})/dt‖²` — primary smoothness signal |
| Δα sparsity | −3.0 | `‖Δα‖²` — pushes residual toward zero unless earning reward |
| Alive bonus | +0.5 | Per-step survival bonus |

*Note: the sp05_jw2 retrained variants used jerk weight −2×10⁻¹⁰ and sparsity −0.5.*

### Action Space and Network Architecture

| Variant | Action dim | Output | Clamp |
|---|---:|---|---|
| Residual-α 4D | 4 | Per-leg Δα | `sigmoid(a) × 0.3 → [0, 0.3]` |
| Residual-q 4D | 4 | Per-leg Δq (shared across hip/thigh/calf) | `tanh(a) × 0.25` |
| Residual-α 12D | 12 | Per-joint Δα | `sigmoid(a) × 0.3 → [0, 0.3]` |
| Residual-q 12D | 12 | Per-joint Δq | `tanh(a) × 0.25` |

**Network (all variants):**

```
Input (45-D) → Linear(45, 128) → ELU → Linear(128, 128) → ELU → Linear(128, action_dim)
                           Actor and Critic share this depth (separate heads)
```

### Training Setup

**Episode configuration:**

| Parameter | Value |
|---|---|
| Episode length | 10 s (500 control steps at 50 Hz) |
| Parallel environments | 1024 |
| Control dt | 0.02 s (physics dt = 0.005 s, render interval = 4) |
| Velocity command | vx = 0.4, vy = 0, yaw = 0 m/s |
| Transition start hold | Uniform(1.5, 3.5) s — random gait-phase at switch |
| Transition duration | Fixed 3.0 s |
| Termination | Base contact > 50 N OR `‖gravity_xy‖² > 1.0` |

**PPO hyperparameters:**

| Parameter | Value |
|---|---|
| Algorithm | PPO (rsl_rl) |
| Steps per env per update | 24 |
| Mini-batches | 4 |
| Learning epochs per update | 5 |
| Learning rate | 5×10⁻⁴ (adaptive schedule) |
| Clip parameter ε | 0.2 |
| Entropy coefficient | 0.005 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Desired KL | 0.01 |
| Max grad norm | 1.0 |
| Init noise std | 0.5 |
| Training iterations | 3000 (sp05_jw2 variants) |

### Explainability Properties

1. **Counterfactual is free.** Setting `Δα = 0` reduces to pure Smoothstep. Differences between `Δα = 0` and `Δα = MLP(obs)` directly measure the learned contribution.
2. **Per-joint attribution.** Each `Δα_j` independently shows how much joint j was advanced beyond the baseline.
3. **Bounded safety.** `Δα ∈ [0, 0.3]` — α never falls below Smoothstep, never overshoots by more than 0.3.
4. **Sparsity makes intervention visible.** `Δα ≈ 0` during steady-state holds; grows only during the active ramp window.

---

## 5. Development History — From Prototype to Design-Space Study

### Why v1–v10 Used Residual-α 4D

The v1–v10 sequence was a **framework development phase**, not the final architectural claim. The 4D per-leg model was used as a controllable prototype because it is simpler, faster to debug, and easier to interpret than 12D. The goal was to make residual transition learning work at all — the question of the best architecture came later.

### Problems Solved in v1–v10

| Version | Problem | Fix |
|---|---|---|
| v1–v2 | Standstill exploit (alive bonus dominates) | Velocity tracking weight ×3 |
| v2–v3 | No time-gating — MLP corrupts source gait | Hard zero outside window |
| v3–v4 | Steer policy out-of-distribution | Drop steer from Phase 2 |
| v4–v5 | Base policies queried with wrong `last_action` | Per-policy `_base_last_actions` buffer |
| v5–v7 | Wrong smoothness metric (`jacc_RMS` not jerk) | Replace with `(q̈_t − q̈_{t-1})/dt` |
| v7 | Symmetric `tanh × 0.8` clamp → delay-rush exploit | Asymmetric `sigmoid × 0.3` in v10 |
| v8–v9 | Jerk weight too high → compressed-jump strategy | Sweep to `−1e-10` |
| v10 | No sparsity → MLP saturates Δα throughout window | Add `−3.0 · ‖Δα‖²` |

**v7 was the previous intermediate headline** (smoothstep α + per-leg Δα ∈ [−0.8, +0.8]). Re-evaluation under the correct jerk metric revealed v7 was using a "delay-rush" exploit: outputting Δα < 0 in early ramp (delaying α below smoothstep) and Δα > 0 in late ramp. This compressed mid-α blending time but produced velocity dips (vx_min = −0.045 m/s). The symmetric clamp hid this behind the jacc_RMS metric.

**v10 closed the failure mode** with the asymmetric sigmoid clamp: `Δα ∈ [0, 0.3]`. The MLP can only advance α above smoothstep, never delay it. v10 (Residual-α 4D prototype) achieves +0.433 m/s, vx_min ≈ 0, jerk below Smoothstep. This stable recipe was then used for the 2×2 design-space study.

### From Residual Prototype to Design-Space Study

Once v10 established a stable residual-learning recipe, the project expanded the research question:

> *"If residual learning works on top of Smoothstep, what output space and action dimension gives the best result?"*

Silver et al. (2018) defined the residual as a **joint-position correction** Δq. This project asks whether correcting **blending weights α** is preferable — keeping the output within the valid interpolation between two frozen gaits by construction — and whether per-joint (12D) corrections outperform per-leg (4D) corrections.

*"The v1–v10 sequence should be read as the development of a stable residual-learning recipe, not as the final architectural claim. We used the 4D α model as a controllable prototype to debug the residual framework. Once the recipe was stable, we evaluated the broader design space. The 2×2 ablation revealed that α-space × 12D performs best at the canonical evaluation — but the full picture requires both evaluation levels."*

### Key Architecture Decisions (v10 Recipe — Applied to All Variants)

| Component | Value |
|---|---|
| Δα clamp | `sigmoid(action) × 0.3` → ∈ [0, 0.3] per joint |
| Time-gating | Hard zero outside transition window |
| Smoothness reward | `−1e-10 · Σ((q̈_t − q̈_{t-1})/dt)²` (jerk, not jacc) |
| α schedule | Smoothstep `x²(3−2x)` with zero-derivative endpoints |
| Sparsity | `−3.0 · Σ Δα²` |
| Action rate | `−0.15 · Σ ‖Δα_t − Δα_{t-1}‖²` |
| Orientation boost | ×4 inside transition window |
| Policy net | 45 → 128 → 128 → **12** (or 4 for 4D variant), ELU |

### Residual Diagnostic Plots (Residual-α 12D)

Plots generated from the canonical seed=42 run (`logs/phase2/residual_alpha_12d/diag/`):

**Gait diagram** — foot contact + vx across 6 directed transitions:

![Gait diagram — Residual-α 12D](logs/phase2/residual_alpha_12d/diag/gait_diagram.png)

**Per-leg mean Δα over time** — MLP output summarized per leg (4D mean of 12D for readability):

![Delta alpha](logs/phase2/residual_alpha_12d/diag/delta_alpha.png)

The MLP is mostly silent during steady-state holds (Δα ≈ 0) and activates during the transition window. Rear legs (RL, RR) tend to show larger corrections on trot↔bound transitions.

**Body state** — vx, height, tilt:

![Body state](logs/phase2/residual_alpha_12d/diag/body_state.png)

**Blend thigh joints** — per-joint α traces across transitions:

![Blend thigh](logs/phase2/residual_alpha_12d/diag/blend_thigh.png)

---

## 6. Systematic Design-Space Study (2×2 Ablation)

### Research Question

After establishing a working residual recipe with Residual-α 4D, two design dimensions remained open:

1. **Output space**: Should the residual correct blending weights (α) or joint positions (q)?
2. **Action dimension**: Should corrections be per-leg (4D, one scalar broadcast to all joints in the leg) or per-joint (12D, independent per joint)?

### The 2×2 Design Space

|  | **4D** (per-leg scalar) | **12D** (per-joint) |
|---|---|---|
| **α-space** (blending weight) | Residual-α 4D | **Residual-α 12D** ← main α-space 12D variant |
| **q-space** (joint position) | Residual-q 4D | Residual-q 12D |

All four variants share: same smoothstep baseline, same time-gating, same sparsity weight (`−3.0`), same jerk reward (`−1e-10`), same network size (128×128), same training budget. Only output space and action dimension differ.

**Why α-space is more structured:** Residual-α keeps the command on the interpolation path between frozen policy outputs. The correction `Δα ∈ [0, 0.3]` advances the blend — the output stays within the convex combination of two frozen policies rather than adding unconstrained joint offsets. This is more interpretable and bounded, but it does not guarantee dynamic safety for every gait phase — the N=60 evaluation shows Res-α 12D still produces reversal on 30% of windows. Residual-q adds directly to joint targets with no such interpolation structure, which can deviate further from safe gait states, but conservative action magnitude (as in Res-q 4D) can compensate empirically.

**A note on Residual-q 4D:** The 4D q-space design broadcasts one scalar Δq to all three joints in a leg (hip, thigh, calf). Applying the same correction to joints with different angular ranges and mechanical roles is physically less interpretable than per-joint correction. However, this uniformity constraint appears to be implicitly conservative — the network learns small corrections that do not cause reversal across diverse gait phases, producing the best N=60 combined result (lowest jerk among zero-reversal methods). Residual-q 4D is included for 2×2 completeness; q-space results should be read together with Residual-q 12D to separate the effect of output space from action dimension.

### 2×2 Canonical Evaluation (seed=42, 2500 steps)

| Variant | vx_mean | vx_std | vx_min | tilt_max | h_mean | CoT | **jerk_TRANS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Residual-α 4D | +0.430 | 0.113 | −0.086 | 0.189 | 0.408 | 2.171 | 7617 |
| Residual-q 4D | +0.416 | 0.100 | −0.024 | 0.200 | 0.417 | 2.158 | 7320 |
| **Residual-α 12D** | **+0.427** | **0.109** | **+0.004** | 0.190 | 0.407 | **2.105** | **7951** |
| Residual-q 12D | +0.408 | 0.130 | −0.122 | 0.207 | 0.414 | 2.064 | 7719 |

**Transition-window zoom — 2×2 (trot→bound, seed=42):**

![Transition zoom — ablation](logs/phase2/transition_zoom_ablation.png)

Blue = α-space; Orange = q-space. Solid = 4D; Dashed = 12D (best). Max |joint velocity| at t=0: Res-α 4D 2.12 rad/s, Res-α 12D 2.11 rad/s, Res-q 12D 2.37 rad/s, Res-q 4D 2.15 rad/s. Generated by `scripts/plot_transition_zoom.py --mode ablation`.

### Interpretation

- **All four residual variants beat Smoothstep (8508)** on jerk_TRANS. The residual recipe works.
- **α-space × 12D (Residual-α 12D)** is the safety-preferred method **at the canonical evaluation (N=6)**: the **only variant with zero velocity reversal** at the canonical fixed gait phase (vx_min = +0.004), with competitive jerk (7951) and comparable CoT (2.105 vs 2.090). *Caveat: at N=60, Res-α 12D has 30% reversal rate; Res-q 4D achieves zero reversal.*
- **Within α-space (canonical N=6)**: 4D (7617) achieves lower jerk than 12D (7951) but at the cost of mild velocity reversal (vx_min = −0.086). 12D pays a small jerk penalty for the zero-reversal guarantee at the canonical gait phase.
- **Within q-space (canonical N=6)**: both variants achieve lower raw jerk (7320, 7719) but have velocity reversal; neither achieves zero reversal at the canonical gait phase.
- **Within q-space, 12D does not improve reversal vs 4D**: vx_min worsens from −0.024 to −0.122 at canonical N=6 (though at N=60 both improve substantially, with Res-q 4D reaching zero reversal).
- **The structural property of output space**: α-space keeps the command on the interpolation path between frozen policy outputs; q-space adds directly to joint targets with no such constraint. This structural difference provides interpretability and bounded correction, but does not fully determine empirical safety across all gait phases — see Section 9 discussion.

---

## 7. Experiments and Metrics

### Methods Compared

| Method | Description | Action |
|---|---|---|
| **(a) Discrete Switch** | α = 1 instantly at switch. No blending. | — |
| **(b) Linear Ramp** | α ramps linearly over 3 s. | — |
| **(c) Smoothstep Ramp** | α follows x²(3−2x) over 3 s. | — |
| **(d) Residual-1D** | Smoothstep + scalar Δα broadcast to all 4 legs. | 1-D tanh |
| **(e) Residual-α 4D** | Smoothstep + per-leg Δα, asymmetric clamp [0, 0.3]. | 4-D sigmoid |
| **(f) Residual-q 4D** | Smoothstep + per-leg scalar Δq broadcast to 3 joints per leg. | 4-D tanh |
| **(g) Residual-q 12D** | Smoothstep + 12-D joint correction Δq (Silver et al. 2018). | 12-D tanh |
| **(h) Residual-α 12D** | Smoothstep + per-joint Δα, asymmetric clamp [0, 0.3]. Main α-space 12D variant. | 12-D sigmoid |

### Metrics

**Primary metric — `jerk_TRANS`:** Jerk RMS (rad/s³) measured only inside the 3 s transition window. All methods run identical frozen base policies during steady-state holds — transition-window jerk is the only period where blending strategies differ. Jerk (rate of change of acceleration) is used as a proxy for abrupt joint-command changes and transition harshness.

**Why not `jerk_ALL`:** Jerk during steady-state holds is identical across methods (same frozen base policies). Aggregating over the full episode dilutes the signal.

**Secondary metrics:**
- `vx_min` — minimum forward velocity during any transition window. Negative = velocity reversal (robot momentarily moves backward).
- `vx_mean`, `vx_std` — tracking quality and consistency.
- `CoT` (Cost-of-Transport) — energy efficiency.
- `tilt_max` — maximum body tilt (orientation stability).
- Fall / episode termination count.

### Evaluation Protocol

**Canonical Evaluation (seed=42):** Fixed seed, fixed gait sequence (trot→bound→pace→trot→pace→bound), 2500 steps, 8 s per segment. All methods run under identical conditions.

**Per-Gait-Pair Analysis (N=6):** For each method, jerk_TRANS is computed separately for each of the 6 directed gait-pair transitions in the canonical episode. This gives 6 data points per method — one per gait pair — and reveals the per-pair difficulty hierarchy. It is reported alongside the mean to show spread across gait pairs.

**Multi-Seed Robustness Evaluation (N=60):** The play script supports `--randomize_start`, which samples `_transition_start_s` from the training range [1.5, 3.5] s using a seed-isolated `np.random.default_rng`. Different seeds hit the switch at different gait phases, producing genuine jerk variation. Running 10 seeds × 6 gait pairs gives N=60 transition windows per method. Two bugs were found and fixed before this experiment was valid: (1) IsaacLab resets numpy's global RNG during env init, so `np.random.uniform` returned the same value for all seeds — fixed by using an isolated Generator; (2) the discrete baseline had `_transition_start_steps` hardcoded to `int(2.0/dt)`, ignoring the sampled hold time — fixed by using `_current_hold_s`. The multi-seed results use `bash scripts/run_seed_experiment_v2.sh` followed by `analyze_seed_experiment.py --source seeds`.

---

## 8. Results

### Discrete Spike Analysis

![Discrete switch spike](logs/phase2/discrete_spike.png)

At switch time: max joint velocity 16.4 rad/s (6.2× steady-state), jerk 19 189 rad/s³, velocity dip and reversal. Generated by `scripts/plot_discrete_spike.py`.

### Single-Seed Canonical Result (seed=42)

Full table — all methods, 6-pair evaluation:

| Method | vx_mean | vx_std | vx_min | tilt_max | h_mean | CoT | **jerk_TRANS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Discrete Switch | +0.435 | 0.108 | −0.195 | 0.234 | 0.409 | 2.793 | 11361 |
| Linear Ramp | +0.390 | 0.157 | −0.206 | 0.184 | 0.404 | 1.955 | 7441 |
| Smoothstep Ramp | +0.415 | 0.129 | −0.096 | 0.187 | 0.405 | 2.090 | 8508 |
| Residual-α 4D | +0.430 | 0.113 | −0.086 | 0.189 | 0.408 | 2.171 | 7617 |
| Residual-q 4D | +0.416 | 0.100 | −0.024 | 0.200 | 0.417 | 2.158 | 7320 |
| Residual-q 12D | +0.408 | 0.130 | −0.122 | 0.207 | 0.414 | 2.064 | 7719 |
| **Residual-α 12D** | **+0.427** | **0.109** | **+0.004** | 0.190 | 0.407 | **2.105** | **7951** |

*All four residual variants beat Smoothstep on jerk_TRANS. In this canonical evaluation, Residual-α 12D is the only method with zero velocity reversal (vx_min +0.004 vs Smoothstep −0.096) — this zero-reversal property holds at the fixed canonical gait phase but does not generalise across diverse phases (N=60: 30% reversal rate). It reduces jerk by 6.5% vs Smoothstep (8508 → 7951) and 30.0% vs Discrete (11361 → 7951), with comparable CoT (2.105 vs 2.090).*

### Transition-Window Jerk Profile

![Transition jerk profile](logs/phase2/transition_jerk_profile.png)

RMS jerk in 10 equal 0.3 s bins across the 3 s window, averaged over all 6 gait-pair transitions (seed=42). Discrete Switch (red) remains elevated uniformly — the momentum discontinuity from the instant switch persists for the full 3 s recovery. Residual-α 12D (dark blue) is consistently the lowest across most bins. Generated by `scripts/plot_transition_jerk.py`.

### Baselines Comparison — Transition Zoom

![Transition zoom — baselines](logs/phase2/transition_zoom_baselines.png)

Four panels: α schedule, max |joint velocity|, joint velocity RMS, forward velocity. Aligned to each method's own transition start (t=0). Discrete spike reaches 16.4 rad/s; Residual-α 12D (blue) stays at 2.1 rad/s throughout. Generated by `scripts/plot_transition_zoom.py --mode baselines`.

### Body Acceleration Comparison

![Body acceleration](logs/phase2/compare_body_acc.png)

Body forward acceleration (dvx/dt) overlaid for Discrete Switch (red) and Residual-α 12D (blue). Purple band = transition window. Generated by `scripts/plot_body_acc_compare.py`.

### Smoothstep vs Residual-α 12D

The cleanest measure of what the MLP adds — both share the same smoothstep baseline:

| | Smoothstep | Residual-α 12D | Change |
|---|---:|---:|---:|
| vx_mean | +0.415 | **+0.427** | **+2.9%** |
| vx_std | 0.129 | **0.109** | **−15.5%** |
| vx_min | −0.096 | **+0.004** | **reversal eliminated** |
| **jerk_TRANS** | 8508 | **7951** | **−6.5%** |
| CoT | 2.090 | **2.105** | +0.7% |

The MLP adds 2.9% velocity, reduces velocity variance by 15.5%, eliminates reversal at the canonical fixed gait phase (30% reversal rate at N=60), and reduces jerk by 6.5%. CoT is comparable (within noise), with Residual-α 12D marginally higher.

<table>
<tr>
<td align="center"><b>Smoothstep Ramp</b></td>
<td align="center"><b>Residual-α 12D</b></td>
</tr>
<tr>
<td><img src="logs/phase2/baselines/smoothstep_ramp/diag/gait_diagram.png"/></td>
<td><img src="logs/phase2/residual_alpha_12d/diag/gait_diagram.png"/></td>
</tr>
</table>

### Linear vs Smoothstep (Schedule Shape Alone)

| | Linear | Smoothstep | Change |
|---|---:|---:|---:|
| vx_mean | +0.390 | **+0.415** | **+6.4%** |
| vx_std | 0.157 | **0.129** | **−17.8%** |
| vx_min | −0.206 | **−0.096** | less reversal |
| jerk_TRANS | **7441** | 8508 | linear wins on jerk |

Smoothstep's zero-derivative endpoints reduce ramp-start kinematic kicks, producing better velocity tracking and less reversal. Linear achieves lower canonical jerk (constant dα/dt avoids the smoothstep mid-ramp S-curve), but its larger velocity variance and reversal make it a weaker residual baseline. Smoothstep is chosen not because it has the lowest canonical jerk, but for endpoint continuity and velocity safety — properties that matter more when the residual learning has to correct hard transitions.

### Discrete vs Residual-α 12D

*In the canonical seed=42 run:*

| | Discrete | Residual-α 12D | Change |
|---|---:|---:|---:|
| **jerk_TRANS** | **11361** | **7951** | **−30.0%** |
| vx_min | −0.195 | **+0.004** | reversal eliminated |
| CoT | 2.793 | **2.105** | −24.7% |

*Across 6 directed gait-pair transitions (canonical seed=42):*

| | Discrete | Residual-α 12D | Change |
|---|---:|---:|---:|
| jerk_TRANS mean | 11361 | **7951** | **−30.0%** |
| Worst gait pair | 19030 | **12351** | −35.1% |

<table>
<tr>
<td align="center"><b>Discrete Switch</b></td>
<td align="center"><b>Residual-α 12D</b></td>
</tr>
<tr>
<td><img src="logs/phase2/baselines/discrete/diag/gait_diagram.png"/></td>
<td><img src="logs/phase2/residual_alpha_12d/diag/gait_diagram.png"/></td>
</tr>
</table>

### Per-Method Gait Diagrams

---

**(a) Discrete Switch** — vx dips at each switch; worst window: bound→pace, jerk_TRANS=19030.

![Discrete gait diagram](logs/phase2/baselines/discrete/diag/gait_diagram.png)

---

**(b) Linear Ramp** — jerk_TRANS 7441, but larger velocity variance and reversal; vx_min = −0.206.

![Linear ramp gait diagram](logs/phase2/baselines/linear_ramp/diag/gait_diagram.png)

---

**(c) Smoothstep Ramp** — best passive schedule; jerk_TRANS 8508, vx_min = −0.096 (reversal still occurs).

![Smoothstep gait diagram](logs/phase2/baselines/smoothstep_ramp/diag/gait_diagram.png)

---

**(e) Residual-α 4D** — per-leg Δα, asymmetric clamp. jerk_TRANS 7617 (−10.5% vs Smoothstep), vx_min −0.086 (mild reversal). MLP activates mainly during transition windows.

![Residual-α 4D gait diagram](logs/phase2/residual_alpha_4d_sp05_jw2/diag/gait_diagram.png)

---

**(h) Residual-α 12D** — per-joint Δα. jerk_TRANS 7951 (−6.5% vs Smoothstep), vx_min +0.004 (only zero-reversal method at canonical N=6; 30% reversal rate at N=60), CoT 2.105.

![Residual-α 12D gait diagram](logs/phase2/residual_alpha_12d/diag/gait_diagram.png)

---

### Per-Gait-Pair Analysis (N=6, canonical seed=42)

Each method evaluated across all 6 directed gait-pair transitions in the canonical seed=42 episode. jerk_TRANS is computed per transition, giving N=6 per method. This reveals the per-pair difficulty hierarchy.

| Method | N | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Discrete Switch | 6 | 11361 | 5426 | 4183 | 19030 |
| Smoothstep Ramp | 6 | 8508 | 2610 | 4233 | 11801 |
| Res-α 4D | 6 | 7617 | 3104 | 4607 | 13540 |
| Res-q 4D | 6 | 7320 | **1934** | **4930** | **10789** |
| **Res-α 12D** | **6** | **7951** | 3267 | 4072 | 12351 |
| Res-q 12D | 6 | 7719 | 1921 | 4490 | 10193 |

*All four residual variants beat Smoothstep (8508). Residual-α 12D has the lowest mean among zero-reversal methods (7951). q-space variants (7320, 7719) achieve lower raw jerk but at the cost of velocity reversal. Res-q 4D has the tightest per-pair spread (std=1934), but with vx_min=−0.024.*

Per-gait-pair breakdown (tro=trot, bou=bound, pac=pace):

| Method | tro→bou | bou→pac | pac→tro | tro→pac | pac→bou | bou→tro |
|---|---:|---:|---:|---:|---:|---:|
| Discrete | 14568 | 19030 | 4183 | 14937 | 10619 | 4829 |
| Smoothstep | 4233 | 9932 | 6171 | 8343 | 11801 | 10566 |
| Res-α 4D | 6797 | 9652 | 4607 | 5957 | 13540 | 5147 |
| Res-q 4D | 5933 | 10789 | 6090 | 8273 | 7905 | 4930 |
| **Res-α 12D** | **4072** | 12351 | **5523** | **8149** | 12140 | **5470** |
| Res-q 12D | 9190 | 10193 | 7965 | 8446 | 4490 | 6033 |

The hardest pairs (bou→pac, pac→bou) involve the largest coordination-structure mismatch and remain challenging for all methods.

![All-method per-gait-pair jerk boxplot](logs/phase2_seed_experiment_v2/results_all.png)

![Baselines per-gait-pair](logs/phase2_seed_experiment_v2/results_baselines.png)

![2×2 ablation per-gait-pair](logs/phase2_seed_experiment_v2/results_2x2.png)

### Base-Swap Validation (Silver et al. Style)

A key question in residual policy learning is whether the MLP has learned a genuinely residual correction — calibrated to its training base — or whether it has learned the full transition behavior and can generalize to any base.

**Experiment:** run the trained Res-α 12D checkpoint with the base schedule replaced by linear ramp at evaluation time (no retraining). The MLP still receives `alpha_base` in its observation, but now those values follow a linear rather than Smoothstep trajectory.

| | Linear ramp base | Smoothstep base |
|---|---:|---:|
| **Δα = 0** (no MLP) | 7441 | 8508 |
| **Δα = MLP output** | **8343 (+12.2%)** | **7951 (−6.5%)** |

With the **correct base (Smoothstep)**, the MLP reduces mean jerk by −6.5% (8508 → 7951). With the **wrong base (linear ramp)**, the MLP remains active (Δα_max ≈ 0.15 per joint vs. 0.30 with SS base) but its corrections are miscalibrated: it hurts on 4/6 gait pairs and raises mean jerk by +12.2% (7441 → 8343).

Per-gait-pair breakdown:

| Gait pair | LR, Δα=0 | LR + MLP | SS, Δα=0 | SS + MLP |
|---|---:|---:|---:|---:|
| trot→bound | 2540 | 5784 (+128%) | 4233 | **4072** |
| bound→pace | 8735 | 11229 (+29%) | 9932 | 12351 |
| pace→trot | 5539 | 8163 (+47%) | 6171 | **5523** |
| trot→pace | 5507 | 9770 (+77%) | 8343 | **8149** |
| pace→bound | 12805 | **8945** (−30%) | 11801 | 12140 |
| bound→trot | 9519 | **6170** (−35%) | 10566 | **5470** |
| **Mean** | **7441** | **8343** | **8508** | **7951** |

The MLP's corrections accelerate the transition in Smoothstep-specific ways (e.g., pushing α above the slow early-ramp region). When the base is already linear, those same accelerations land at structurally wrong times — producing larger jerk on most pairs.

This is a stronger result than simple shutdown: the MLP is not silenced by the wrong base; it actively misfires. The corrections are Smoothstep-calibrated and cannot generalize to a mismatched base schedule.

![Base-swap validation](logs/phase2/base_swap_validation.png)

*Panel A: effective α(t) during bound→trot — one of 2/6 pairs where LR+MLP incidentally helps. SS+MLP (blue) shows larger Δα (up to 0.30) than LR+MLP (dotted, up to ≈0.15). Panel B: per-gait-pair jerk — LR+MLP bars exceed LR+0 on 4 of 6 pairs. Highlighted pairs (pace→bound, bound→trot) are the only exceptions. Panel C: 2×2 mean summary with percentage change from no-MLP row.*

### Jerk-Weight Sweep

*(Residual-α 12D architecture, v10 recipe, canonical seed=42, 6-pair evaluation — hyperparameter selection for the final recipe.)*

To verify `rew_joint_jerk = −1e-10` is empirically optimal:

![Jerk-weight Pareto sweep](logs/phase2/sweep_jerk_pareto.png)

| Run | jerk weight | jerk_RMS | CoT | vx_mean | Behavior |
|---|---:|---:|---:|---:|---|
| `sweep_w0` | 0 | 10742 | 2.45 | +0.442 | No smoothness pressure |
| `sweep_w_low` | −2e-11 | 11084 | 2.54 | +0.440 | Penalty too small |
| **`sweep_w_med`** | **−1e-10** | **9392** | **2.32** | **+0.437** | **Sweet spot** |
| `sweep_w_hi` | −5e-10 | 11125 | 2.59 | +0.438 | Over-strong → compressed-jump |
| `sweep_w_xhi` | −1e-9 | 11430 | 2.09 | +0.405 | Collapse |

Both jerk_RMS and CoT form a U-shape with minimum at `−1e-10`.

### Duration Sweep

*(Residual-α 12D vs Smoothstep, canonical seed=42. The MLP is trained on 3 s only and evaluated at other durations without retraining.)*

![Duration sweep](logs/phase2/duration_sweep/duration_sweep.png)

| Duration | Residual-α 12D jerk_TRANS | Smoothstep jerk_TRANS | Verdict |
|---:|---:|---:|---|
| 0.5 s | 13063 | 11324 | Both fail — architectural ceiling |
| 1.0 s | ~11000 | ~11000 | Both fail |
| 2.0 s | — | — | Residual-α 12D wins |
| **3.0 s** | **7951** | **8508** | **Residual-α 12D wins (training dist)** |
| 5.0 s | ~10500 | ~10500 | Methods converge |

Three regimes: **Catastrophic (d ≤ 1 s)** — frozen-base-policy blending ceiling, both fail. **Sweet spot (d = 2–3 s)** — Residual-α 12D wins on all smoothness metrics. **Easy (d = 5 s)** — methods converge; MLP adds nothing when ramp is gentle enough.

### With vs Without Sparsity Penalty

*(Residual-α 12D architecture, canonical seed=42, 6-pair evaluation — ablating the sparsity term from the v10 recipe.)*

| | No Sparsity | With Sparsity (−3.0) |
|---|---:|---:|
| \|Δα\| mean | 0.049 | **0.004** |
| jerk_TRANS mean | 8509 | **7733** |
| jerk_TRANS std | 2672 | **1908** |
| vx_mean | 0.428 | **0.431** |

Without sparsity, the MLP saturates Δα throughout the window (11× larger output) and produces 10% higher jerk. The sparsity term is structurally necessary for "silent except when needed" behavior.

### Residual-1D vs Residual-α 12D

*(Both evaluated at canonical seed=42, 6-pair evaluation. Residual-1D broadcasts one scalar Δα to all 12 joints; Residual-α 12D uses independent per-joint corrections.)*

| | Residual-1D | Residual-α 12D | Change |
|---|---:|---:|---:|
| vx_mean | +0.411 | **+0.427** | **+3.9%** |
| vx_std | 0.134 | **0.109** | **−18.7%** |
| vx_min | −0.095 | **+0.004** | **reversal eliminated** |
| jerk_TRANS | 8606 | **7951** | **−7.6%** |

The 1D scalar cannot independently advance different legs or joints through the coordination-structure change. Per-joint 12D allows the finest-grained learned scheduling.

---

## 9. Discussion

### What Each Method Reveals

The two evaluation levels (canonical N=6, multi-seed N=60) agree on jerk but disagree on velocity safety. Each method's trade-off is described at both levels.

**Smoothstep Ramp** is the strongest passive baseline — no training required, simple to implement (CoT 2.090). At N=6 it has one of the lower jerk values (8508) among baselines and moderate velocity reversal (vx_min −0.096). At N=60 it has 55% reversal rate — the worst of any method. This reveals that Smoothstep's passive schedule produces reversal on the majority of gait-phase conditions.

**Residual-α 4D** consistently achieves ~10% jerk reduction vs Smoothstep at both N=6 (7617) and N=60 (8185). At canonical it has mild reversal (vx_min −0.086); at N=60 its reversal rate is 7% — the second lowest. It is the development prototype and shows that α-space correction with the v10 recipe reliably improves on Smoothstep. *Not the strongest method, but the most consistently well-behaved.*

**Residual-q 4D** broadcasts one scalar Δq to all three joints per leg (hip, thigh, calf) — physically less interpretable than per-joint correction. Despite this, it achieves the strongest combined result at N=60: 16% jerk reduction and **zero velocity reversal** (0/60 windows, worst vx_min=+0.072). At canonical N=6 it has mild reversal (−0.024). The uniformity constraint appears to be implicitly conservative — the network cannot produce large per-joint corrections, which limits the risk of velocity dips across diverse gait phases. *The N=60 safety result likely reflects this conservative action structure rather than a learned safety strategy.*

**Residual-q 12D** achieves the lowest mean jerk at N=60 (7305, −20%) but also the worst reversal among residual variants (38% at N=60, vx_min −0.429). Per-joint q-space corrections allow the most direct reduction in joint acceleration, but without the interpolation structure of α-space the commands can deviate significantly from the blended policy trajectory on hard transitions. *Strongest on jerk, weakest on safety — the clearest trade-off in the design space.*

**Residual-α 12D** achieves zero velocity reversal at the canonical fixed gait phase (vx_min=+0.004) and −6.5% jerk vs Smoothstep. At N=60 it shows 30% reversal rate and −5.8% jerk reduction — still beats Smoothstep, but the zero-reversal claim does not hold across diverse gait phases. Its advantage is that the α-space asymmetric clamp [0, 0.3] prevents delaying the blend below Smoothstep; the 30% reversal shows that advancing α too fast on certain phases can still cause dips. *Structurally principled, but not robustly safer than other variants.*

*"The two-level evaluation reveals that the design space trade-offs are more complex than any single evaluation can capture. The robust finding is that all residual variants beat Smoothstep on jerk. The velocity safety ranking is gait-phase-dependent. Res-q 4D performs best at N=60 on both metrics, but this may reflect implicit conservatism. Res-α 12D is structurally the most principled: any deviation from Smoothstep is bounded and interpretable. The 2×2 study shows that both output space and action dimension matter, but their effects interact with the gait-phase distribution of the evaluation."*

### Why Residual-α 12D Has the Widest Box (N=6)

The per-gait-pair boxplot (N=6 canonical) shows Residual-α 12D with the widest IQR. This reflects **gait-pair-to-gait-pair difficulty differences**, not instability.

The N=6 spread comes from the 6 gait pairs having different structural difficulty — Res-α 12D achieves large improvements on easy pairs and smaller gains on the hardest pairs.

Per-gait-pair breakdown:

| Gait pair | Res-α 12D | Res-α 4D | Winner |
|---|---:|---:|---|
| trot→bound | **4072** | 6797 | 12D −40% |
| bound→pace | 12351 | **9652** | 4D −22% |
| pace→trot | **5523** | 4607 | 4D −16% |
| trot→pace | **8149** | 5957 | 4D −27% |
| pace→bound | 12140 | **13540** | 12D +12% |
| bound→trot | **5470** | 5147 | 4D −6% |

Residual-α 12D and 4D are competitive across pairs. The wide box reflects a large gap between its best pairs (trot→bound: 4072) and hardest pairs (bound→pace: 12351). The hardest pairs involve the largest coordination-structure mismatch (fore-aft ↔ lateral) and remain challenging for all methods.

**The correct interpretation:** The wide IQR in Residual-α 12D reflects gait-pair difficulty variation — not inconsistency. Its worst-case ceiling (12351) is well below Discrete's (19030).

### α-Space vs q-Space Safety: What the Evaluations Show

Structurally, α-space keeps the command on the interpolation path between frozen policy outputs — even at the maximum Δα=0.3 the command is a convex combination of two stable policies. q-space corrections add directly to the blended joint target with no such constraint.

In practice the picture is more nuanced. At the canonical fixed gait phase (N=6), Res-α 12D achieves zero velocity reversal while Res-q 4D has mild reversal. But at N=60 across diverse gait phases, Res-q 4D achieves zero reversal (0/60 windows) while Res-α 12D shows 30% reversal. The asymmetric clamp [0, 0.3] in α-space prevents delaying below Smoothstep but does not prevent advancing the blend too fast, which can cause velocity dips on certain gait phases. The small uniform corrections learned by Res-q 4D (constrained by the per-leg design to be conservative) turn out to be implicitly safe across diverse conditions.

This finding complicates the α-vs-q narrative: structural safety bounds do not guarantee empirical safety across all gait phases, and empirical safety can emerge from design constraints that limit action magnitude.

### Why 12D Helps in α-Space

Different joints have fundamentally different roles during gait transitions. During trot→bound, the thigh joints (which drive leg swing arcs) need to transition at a different rate than the hip joints (which control lateral balance). A per-leg scalar broadcasts the same correction to hip, thigh, and calf simultaneously. Per-joint 12D allows the MLP to learn these different roles independently.

### Base-Swap Validation: the Residual Is Schedule-Calibrated

Running the trained Residual-α 12D MLP with linear ramp instead of Smoothstep at evaluation time (no retraining) confirms the learned corrections are base-specific. With the correct base (Smoothstep), the MLP reduces jerk by −6.5%. With a mismatched base (linear ramp), the MLP remains active but its corrections misfire: jerk worsens by +12.2% across 4/6 gait pairs. This is a stronger result than shutdown — the corrections are not silenced, they are miscalibrated. This rules out the possibility that the MLP has learned a generic transition policy and merely uses the base schedule as a warm-start.

---

## 10. Limitations and Future Work

### 1. Fixed Transition Duration (3 s)

Every transition uses a 3 s ramp hardcoded at training time. A multi-seed evaluation (N=60: 10 seeds × 6 gait pairs) was performed varying the gait-phase at switch time via `--randomize_start`, sampling `_transition_start_s ~ Uniform(1.5, 3.5)` s within the training distribution. The N=60 results show all residual variants beat Smoothstep on mean jerk, but reveal that velocity safety rankings are gait-phase-dependent — Res-α 12D's zero-reversal property at the canonical evaluation does not hold across diverse phases.

The transition duration itself (3 s) is fixed. A curriculum attempt (v11: duration sampled from [1.5, 5.0] s) diverged due to high return variance. Warm-starting from the fixed-duration checkpoint is the natural next step toward a policy that generalizes across timing conditions.

### 2. Uniform Smoothstep Baseline for All Gait Pairs

The smoothstep function is applied identically to all six directed transitions. Different gait pairs have fundamentally different coordination mismatches — trot→bound requires a different sync-partner swap than pace→trot — and the optimal interpolation shape likely differs per pair. The current MLP sees the gait one-hot encoding and can learn different Δα patterns per pair, but the baseline shape is global.

### 3. Base Gait Quality (Reward-Hacked Duty Cycles)

Phase 1 base policies are PPO velocity-tracking policies, not biologically faithful gaits. Duty cycles deviate significantly from natural locomotion. Adding a gait-naturalness term to Phase 1 rewards would produce better base policies and a more meaningful Phase 2 result.

### 4. Flat Terrain Only

All training and evaluation is on flat terrain. The key motivation for constrained residual blending — reducing transition jerk and limiting velocity dips — is expected to compound on uneven terrain, where base policies already face disturbances. Rough terrain training of both Phase 1 and Phase 2 is the most important generalization direction.

### 5. Simulation Only

Results are in Isaac Lab simulation. Sim-to-real transfer of the residual MLP requires: (a) base policy sim-to-real transfer, (b) verification that the residual correction remains bounded and safe on real hardware, and (c) testing that `jerk_TRANS` reduction translates to reduced mechanical wear.

---

## 11. Reproducibility

### Environment

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Kill any zombie Isaac Sim processes before launching
nvidia-smi && pgrep -f "python.*play\|python.*train\|isaac\|kit" | xargs -r kill -9
```

### Phase 1 — Train Base Policies (already done, available at `logs/phase1_final/`)

```bash
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Trot-v0 \
    --max_iterations 1500 --run_name trot_v2

python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Bound-v0 \
    --max_iterations 4000 --run_name bound_v4

python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Pace-v0 \
    --max_iterations 4000 --run_name pace_v2
```

### Phase 1 — Playback Any Base Gait

```bash
python scripts/play_b1_velocity.py \
    --task Isaac-Velocity-Flat-Unitree-B1-Trot-Play-v0 \
    --checkpoint logs/phase1_final/trot.pt \
    --num_envs 1 --num_steps 500
```

### Phase 2 — Train Residual Transition Policy

```bash
# Residual-α 12D (main α-space 12D variant)
python scripts/train_b1_phase2.py --headless --num_envs 2048 \
    --task Isaac-B1-Phase2-Alpha12D-v0 \
    --max_iterations 2000 --run_name residual_alpha_12d --seed 42

# Residual-α 4D (prototype)
python scripts/train_b1_phase2.py --headless --num_envs 2048 \
    --max_iterations 2000 --run_name phase2_v10 --seed 42
```

### Phase 2 — Canonical Playback (seed=42)

```bash
# Residual-α 12D — canonical evaluation + diagnostic plots
python scripts/play_b1_phase2.py \
    --task Isaac-B1-Phase2-Alpha12D-v0 \
    --checkpoint logs/phase2/residual_alpha_12d/model_final.pt \
    --num_envs 1 --steps 2500 --seed 42 \
    --gait_pairs trot,bound,pace,trot,pace,bound --switch_interval_s 8.0 \
    --save_csv logs/phase2/residual_alpha_12d/playback_seed42.csv \
    --save_plots logs/phase2/residual_alpha_12d/diag \
    --headless

# Residual-α 4D — canonical evaluation (sp05_jw2 retrained variant)
python scripts/play_b1_phase2.py \
    --task Isaac-B1-Phase2-Transition-v0 \
    --checkpoint logs/phase2/residual_alpha_4d_sp05_jw2/model_final.pt \
    --num_envs 1 --steps 2500 --seed 42 \
    --gait_pairs trot,bound,pace,trot,pace,bound --switch_interval_s 8.0 \
    --save_csv logs/phase2/residual_alpha_4d_sp05_jw2/playback_seed42.csv --headless
```

### Baseline Playback

```bash
python scripts/play_b1_phase2.py \
    --baseline discrete --seed 42 --steps 2500 \
    --gait_pairs trot,bound,pace,trot,pace,bound \
    --save_csv logs/phase2/baselines/discrete/playback_seed42.csv --headless

python scripts/play_b1_phase2.py \
    --baseline smoothstep_ramp --seed 42 --steps 2500 \
    --gait_pairs trot,bound,pace,trot,pace,bound \
    --save_csv logs/phase2/baselines/smoothstep_ramp/playback_seed42.csv --headless

python scripts/play_b1_phase2.py \
    --baseline linear_ramp --seed 42 --steps 2500 \
    --gait_pairs trot,bound,pace,trot,pace,bound \
    --save_csv logs/phase2/baselines/linear_ramp/playback_seed42.csv --headless
```

### 2×2 Ablation Canonical Playback

```bash
# Residual-q 4D
python scripts/play_b1_phase2.py --task Isaac-B1-Phase2-ResidualQ4D-v0 \
    --checkpoint logs/phase2/residual_q_4d/model_final.pt \
    --num_envs 1 --steps 2500 --seed 42 \
    --gait_pairs trot,bound,pace,trot,pace,bound --switch_interval_s 8.0 \
    --save_csv logs/phase2/residual_q_4d/playback_seed42.csv --headless

# Residual-q 12D
python scripts/play_b1_phase2.py --task Isaac-B1-Phase2-ActionSpace-v0 \
    --checkpoint logs/phase2/residual_q_12d/model_final.pt \
    --num_envs 1 --steps 2500 --seed 42 \
    --gait_pairs trot,bound,pace,trot,pace,bound --switch_interval_s 8.0 \
    --save_csv logs/phase2/residual_q_12d/playback_seed42.csv --headless
```

### Plot Generation

```bash
# Transition zoom — baselines and ablation
python scripts/plot_transition_zoom.py --mode baselines
python scripts/plot_transition_zoom.py --mode ablation

# Jerk profile across transition window
python scripts/plot_transition_jerk.py

# Body acceleration comparison
python scripts/plot_body_acc_compare.py

# Discrete spike figure
python scripts/plot_discrete_spike.py \
    --csv logs/phase2/baselines/discrete/playback_seed42.csv
```

### Per-Gait-Pair Analysis (N=6, canonical)

```bash
# Generate per-gait-pair boxplots from canonical seed=42 CSVs (N=6 per method)
python scripts/analyze_seed_experiment.py --mode all \
    --out logs/phase2_seed_experiment_v2/results_all.png
python scripts/analyze_seed_experiment.py --mode baselines \
    --out logs/phase2_seed_experiment_v2/results_baselines.png
python scripts/analyze_seed_experiment.py --mode ablation \
    --out logs/phase2_seed_experiment_v2/results_2x2.png
```

### Multi-Seed Robustness (N=60)

```bash
# Regenerate all seed CSVs with corrected checkpoints (requires Isaac Lab)
# Uses --randomize_start so seeds hit different gait phases at switch time
bash scripts/run_seed_experiment_v2.sh

# Analyze and plot multi-seed results (N=60 per method)
python scripts/analyze_seed_experiment.py --mode all --source seeds \
    --seed_dir logs/phase2_seed_experiment_v2 \
    --out logs/phase2_seed_experiment_v2/results_all.png
python scripts/analyze_seed_experiment.py --mode baselines --source seeds \
    --seed_dir logs/phase2_seed_experiment_v2 \
    --out logs/phase2_seed_experiment_v2/results_baselines.png
python scripts/analyze_seed_experiment.py --mode ablation --source seeds \
    --seed_dir logs/phase2_seed_experiment_v2 \
    --out logs/phase2_seed_experiment_v2/results_2x2.png
```

### Tests

```bash
python -m pytest tests/ -q    # 44/44 unit tests
```

### File Structure

```
cpg-drl-transition/
├── envs/
│   ├── b1_phase2_env_cfg.py        # Phase 2 env config (all variants)
│   └── b1_phase2_env.py            # Phase 2 env class — per-joint blending
├── scripts/
│   ├── play_b1_phase2.py           # Canonical playback + diagnostic plots
│   ├── plot_transition_zoom.py     # Baselines / ablation zoom figure
│   ├── plot_transition_jerk.py     # Jerk profile across transition window
│   ├── plot_body_acc_compare.py    # Body acceleration overlay
│   ├── plot_discrete_spike.py      # Discrete switch spike figure
│   ├── plot_base_swap.py           # Base-swap validation (Silver et al. style)
│   └── analyze_seed_experiment.py  # per-gait-pair jerk analysis (N=6 or N=60)
├── logs/
│   ├── phase1_final/               # Base policy checkpoints
│   ├── phase2/
│   │   ├── residual_alpha_12d/     # Best method checkpoint + canonical CSV + diag
│   │   ├── phase2_v10/             # Residual-α 4D (prototype)
│   │   ├── residual_q_4d/          # Residual-q 4D (ablation)
│   │   ├── residual_q_12d/         # Residual-q 12D (ablation)
│   │   └── baselines/              # Discrete / Linear / Smoothstep CSVs
│   └── phase2_seed_experiment_v2/  # multi-seed N=60 results (corrected checkpoints, --randomize_start)
└── tests/
```

### Determinism Note

All canonical `playback_seed42.csv` files were generated with:
```python
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
This guarantees identical outputs on repeated runs with the same seed.

---

## 12. B1 Robot Configuration

### Joint Axis Convention

| Joint | Axis | Default FL/FR/RL/RR | Role |
|---|---|---|---|
| `hip_joint` | Abduction (lateral splay) | +0.1 / −0.1 / +0.1 / −0.1 | Lateral balance |
| `thigh_joint` | **Flexion (fore/aft swing)** | +0.8 / +0.8 / +1.0 / +1.0 | **Primary walking driver** |
| `calf_joint` | Knee bend | −1.5 / −1.5 / −1.5 / −1.5 | Foot clearance during swing |

The +0.2 rad asymmetry between front and rear thighs directly motivates the **per-joint residual structure** — different joints need different transition rates, and a per-leg scalar cannot capture this asymmetry.

### Foot Contact Convention

`contact_forces[:, foot_ids, 2] > threshold` — vertical force on the four feet (FL, FR, RL, RR). Contact threshold: 1.0 N.

### Phase 2 Joint Order

```
j0  FL_hip    j1  FR_hip    j2  RL_hip    j3  RR_hip
j4  FL_thigh  j5  FR_thigh  j6  RL_thigh  j7  RR_thigh
j8  FL_calf   j9  FR_calf   j10 RL_calf   j11 RR_calf
```

Δα_j applies independently to each of j0–j11. The per-leg mean (`(Δα_0+Δα_4+Δα_8)/3` for FL, etc.) is stored as the 4D `last_residual` in the observation, keeping the obs space at 45-D.

---

*Generated with [Claude Code](https://claude.ai/claude-code)*
