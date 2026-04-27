# Transition-Aware Quadruped Locomotion with Per-Leg Residual Learning

**Course:** FRA 503 — Deep Reinforcement Learning
**Student:** Disthorn Suttawet (66340500019)
**Robot:** Unitree B1 quadruped (12 DOF, ~50 kg)
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0
**Deadline:** 20 May 2026

---

## Problem Statement

Quadruped robots operate across diverse terrains and tasks that demand different gait patterns — slow walks for stability, trots for cruising, bounds for speed, pace for lateral coordination. Real-world deployment requires the robot to **transition smoothly between these gaits** in response to changing commands or environmental conditions, without losing balance or wasting energy.

The transition problem is fundamentally hard because gaits with **different leg-pair coordination structures** (diagonal vs fore-aft vs lateral) cannot be linearly interpolated at the joint level — the midpoint between "FL+RR planted" (trot) and "FL+FR planted" (bound) is **not a valid quadruped configuration**. Naive blending produces incoherent joint targets that destabilize the body.

This project's central question:

> **Can a small per-leg residual network, trained on top of a hand-designed baseline transition schedule, learn to produce graceful gait transitions on a 50-kg quadruped that the baseline alone cannot achieve?**

---

## Contribution Summary

We demonstrate **per-leg residual transition learning** for a heavy quadruped (Unitree B1) operating across three fundamentally different gait coordinations:

- **Trot** — diagonal pairs in phase (FL+RR, FR+RL)
- **Bound** — fore-aft pairs in phase (FL+FR, RL+RR)
- **Pace** — lateral pairs in phase (FL+RL, FR+RR)

A residual MLP outputs a 4-D per-leg correction `Δα ∈ [-0.8, +0.8]` that is added to a hand-designed linear-ramp baseline `α_baseline ∈ [0, 1]`. The corrected α blends the outputs of two frozen base policies (one per gait) at the joint-target level. The residual is **time-gated** to be exactly zero outside the transition window — guaranteeing source and target gaits run untouched during steady-state holds — and **L2-penalized** during transitions to encourage minimal intervention.

**Key result (v4):** Across 6 directed gait transitions (trot↔bound, bound↔pace, pace↔trot), the residual policy achieves:
- Mean forward velocity tracking error of 7.5% (vx mean +0.425 m/s vs commanded +0.4 m/s)
- Zero falls across 2000 evaluation steps
- Per-leg `|Δα|max` ≤ 0.28 (well below the 0.8 budget — the MLP needs only small corrections)
- Per-leg Δα mean ≈ 0 (sparsity penalty + time-gating produces an explainable, parsimonious residual)

The architecture's explainability is direct: with `Δα = 0` (the MLP zeroed), the system reduces to pure linear-ramp blending — providing a **null-hypothesis baseline that is decomposable from the learned correction at any time step**.

---

## System Architecture

### Two-phase design

**Phase 1 — Base gait policies.**
Train four PPO velocity-tracking policies (trot, bound, pace, steer) on flat terrain. Each specializes in a distinct leg-pair coordination pattern (or directional asymmetry). Action space: 12-D joint position offsets, scaled 0.25, added to default joint pose. Trained via Isaac Lab's manager-based RL + RSL-RL `OnPolicyRunner`.

**Phase 2 — Per-leg residual transition learning.**
Freeze three of the four Phase 1 policies (trot, bound, pace) and train a per-leg residual MLP that learns to smooth transitions between any two of them. The residual MLP is the project's main research contribution.

```
                    ┌─────────────────────────────┐
                    │  Per-leg Residual MLP       │
                    │  [obs(45) → 128 → 128 → 4]  │
                    │  outputs Δα ∈ [-0.8, +0.8]  │
                    │  ELU activation             │
                    └─────────┬───────────────────┘
                              │ (Δα_FL, Δα_FR, Δα_RL, Δα_RR)
                              ▼
   π_current ─────┐    ┌──────────────────────┐
   π_target  ─────┼───▶│ Per-leg blending     │──▶ joint_targets → B1
   α_baseline ────┘    │ α_k = α_base + Δα_k  │
   (3 s linear ramp)   │ × time-gating mask   │
                       └──────────────────────┘
```

### Per-leg blending math

For each control step (50 Hz):

```python
# 1. MLP forward pass — per-leg residual
delta_alpha_raw = MLP(obs)                                   # (4,)
delta_alpha     = tanh(delta_alpha_raw) × delta_alpha_max    # ∈ [-0.8, +0.8]

# 2. Time-gating: residual is zero outside transition window
in_window       = (transition_start − pad) ≤ t ≤ (transition_end + pad)
delta_alpha     = delta_alpha if in_window else 0

# 3. Baseline schedule (linear ramp)
ramp_progress   = (t − transition_start_s) / transition_duration_s
alpha_baseline  = clamp(ramp_progress, 0, 1)

# 4. Per-leg α and blending (broadcast Δα to 3 joints per leg)
for leg_k in {FL, FR, RL, RR}:
    α_k = clamp(alpha_baseline + delta_alpha[k], 0, 1)
    blended[3k:3k+3] = (1 − α_k) · π_current(obs)[3k:3k+3]
                      +      α_k · π_target(obs)[3k:3k+3]

# 5. Joint commands
joint_target = default_joint_pos + 0.25 × blended
```

### Why per-leg, not single-α scalar

Trot, bound, and pace have **different leg-pair sync structures**. During trot→bound:
- FL was synced with RR (its diagonal partner). It must now sync with FR (its front-pair partner).
- RR was synced with FL. It must now sync with RL.

A scalar α can interpolate joint *positions* but cannot dynamically swap *which legs sync with which* — that requires per-leg α values that can be temporarily asymmetric during the transition. This is the architectural argument for the per-leg structure.

### Observation space (45-D)

```
base_lin_vel       (3)   robot's linear velocity in body frame
base_ang_vel       (3)   robot's angular velocity in body frame
projected_gravity  (3)   gravity direction in body frame (orientation cue)
joint_pos_rel      (12)  joint angles relative to default pose
joint_vel          (12)  joint velocities
last_residual      (4)   previous Δα (residual smoothness)
gait_current_oh    (3)   one-hot encoding of current source gait
gait_target_oh     (3)   one-hot encoding of target gait
alpha_baseline     (1)   current α from baseline schedule
cycles_elapsed     (1)   time elapsed in episode (1 Hz CPG-equivalent)
```

### Reward function (training)

```
+1.5  · exp(-‖cmd_xy − vel_xy‖² / 0.25)    velocity tracking
+0.75 · exp(-(cmd_yaw − ang_z)² / 0.25)     yaw tracking
-2.0  · ‖projected_gravity_xy‖²             body upright
-50.0 · (h − 0.42)²                         body height target
-0.05 · ‖Δα_t − Δα_{t−1}‖²                  Δα smoothness step-to-step
-3.0  · ‖Δα‖²                               Δα sparsity (encourages near-zero)
+0.5                                        alive bonus
```

### Per-leg residual structure → explainability

The architecture provides four explainability properties without any post-hoc analysis:

1. **Counterfactual is free.** Setting `Δα = 0` reduces the system to the pure linear-ramp baseline. We can run identical episodes with `Δα = 0` vs `Δα = MLP(obs)` and directly attribute differences to the learned correction.
2. **Per-leg attribution.** `Δα_FL = +0.18` vs `Δα_RR = −0.05` directly tells us "the MLP wanted FL to advance through the transition faster than RR" — a research-paper-figure-quality explanation.
3. **Bounded safety.** `|Δα| ≤ 0.8` means the baseline can never be overridden by more than ~80%. The system always retains a fingerprint of the hand-designed schedule.
4. **Sparsity makes the intervention visible.** With the L2 sparsity penalty, the MLP outputs `Δα ≈ 0` during steady-state holds (when α_baseline is at 0 or 1) and grows only during the active ramp window. The temporal `|Δα|(t)` plot is the research narrative figure.

---

## Phase 1 — Base Gait Policies (Done)

Four PPO velocity-tracking policies, all trained on flat terrain via `Isaac-Velocity-Flat-Unitree-B1-{Trot,Bound,Pace,Steer}-v0` tasks. Stored at `logs/phase1_final/`.

| Gait | Coordination | Duty pattern (FL/FR/RL/RR) | Body height | Foot apex | Cycle | Speed |
|---|---|---:|---:|---:|---:|---:|
| **trot_v2** | Diagonal (FL+RR / FR+RL) | 40 / 33 / 39 / 65 % | 0.43 m | 4–5 cm | 1.6 Hz | 0.5 m/s |
| **bound_v4** | Fore-aft (FL+FR / RL+RR) | 65 / 65 / 33 / 34 % | 0.39 m | 10–15 cm | 2.5 Hz | 0.5 m/s |
| **pace_v2** | Lateral (FL+RL / FR+RR) | 30 / 69 / 30 / 69 % | 0.40 m | 19–30 cm | 2.5 Hz | 0.45 m/s |
| **steer_v2** | Asymmetric trot for turning | 39 / 16 / 27 / 35 % | 0.42 m | 5–12 cm | 1.7 Hz | 0.25 m/s + 0.6 rad/s yaw |

For Phase 2 we use only the three forward gaits (trot, bound, pace). Steer is excluded because its training velocity command range (`yaw ∈ (0.4, 1.0)`) is incompatible with Phase 2's fixed `yaw = 0` command and produces out-of-distribution behavior.

### How we got here — the CPG-RBF detour

The original Phase 1 design used a **CPG-RBF (Central Pattern Generator + Radial Basis Function)** controller optimized with **PI^BB (Thor et al. 2021)**. After ~3 weeks of iteration (Week 10 + early Week 11) and 12 documented encoding experiments, this approach was abandoned in favor of pure PPO velocity tracking.

The dealbreakers:
- **B1 is too heavy for PIBB.** At 50 kg, every exploratory step risks a fall (large negative reward). PI^BB's softmax update barely moves W. Cold-init policies stay near origin for thousands of iterations.
- **Direct encoding (240 params) breaks trot.** Only shared-W indirect encoding (60 params) produces stable diagonal coordination, but it remains vulnerable to morphological asymmetry exploits.
- **PPO-on-W is catastrophic.** Putting policy output directly into a CPG W matrix at 50 Hz produces bang-bang motor saturation; the robot flips in 4 steps regardless of action clamping.

The pivot shifted Phase 1 from "research-grade CPG-RBF tuning" to "engineering-grade PPO velocity tracking" — a well-trodden path (legged_gym, Isaac Lab stock tasks). Phase 2's research contribution is *unaffected* by this change because the contribution is the **per-leg residual blending architecture**, not how the base policies are produced.

The legacy CPG-RBF code is preserved at `envs/unitree_b1_env.py` and `algorithms/pibb_trainer.py` — kept intact for reference and as a candidate "future work" direction (use CPG as a phase oracle for the residual MLP rather than as the controller).

### Phase 1 PPO engineering — failure modes encountered and fixes

Stock Isaac Lab velocity reward stack (calibrated for ~15 kg Go2) broke on B1 (~50 kg) in characteristic ways. We accumulated a stack of B1-specific reward terms and corrections through ~10 reward iterations per non-trot gait.

| B1 failure mode | Cause | Fix |
|---|---|---|
| Standstill local optimum | Track reward at vx=0 still pays 88% (std=0.5 too loose) | Tighten `track_lin_vel_xy_exp` std 0.5→0.25, bump weight 1.0→1.5 |
| Crawling exploit (body sags to 0.18 m) | No height penalty in stock | `base_height_l2(target=0.42)` weight −10 to −200 (per-gait tuning) |
| 2-leg trot pathology | No anti-pair-pathology constraint | `excessive_air_time` (max 0.5 s), `excessive_contact_time` (max 0.5 s) |
| Rapid foot-tap exploit (5 Hz on planted leg) | Cumulative time penalties don't catch frequency | `short_swing_penalty` (penalize touchdowns after <0.3 s air) |
| 3+1 asymmetric trot | Per-foot bounds OK individually | `air_time_variance_penalty` (variance of last_air_time) |
| Bilateral L/R asymmetry (FR hip 2× FL) | No L/R constraint | `joint_lr_symmetry_penalty` (\|FL_vel² − FR_vel²\| + \|RL_vel² − RR_vel²\|) |
| Lock-pair exploit (bound: rears planted forever) | Coordination reward fires when locked too | `duty_factor_target_penalty` (target 0.5 per leg) |
| Trot pretending to be bound/pace | Phase-match alone allows trot to score 50% | `true_bound_reward` / `true_pace_reward` (anti-trot, pro-target) |

All custom reward terms are in [envs/b1_velocity_mdp.py](envs/b1_velocity_mdp.py).

---

## Phase 2 — Residual Transition Learning (Main Contribution)

### Version evolution — what we tried, what failed, what worked

The Phase 2 design evolved through four major versions. Each version revealed a specific failure mode that informed the next iteration. This section documents the full evolution as a research narrative.

#### v1 — Initial design with bounded residual (`delta_alpha_max = 0.2`)

**Hypothesis:** Small per-leg corrections (≤ 0.2 magnitude) on a hand-designed linear-ramp baseline are sufficient to learn smooth transitions.

**Setup:**
- 4 base policies (trot, bound, pace, steer)
- Linear-ramp `α_baseline` over 3 s, transition starts at fixed `t=2.0 s`
- `Δα = tanh(MLP(obs)) × 0.2`
- Reward: track_lin_vel + track_ang_vel + orient + height + action_rate + alive

**Result (mean episode reward = +954, mean episode length 500/500, zero falls):**

The training metrics looked great — full-length episodes, no falls, positive total reward. But playback exposed the failure: **standstill exploit**.

| Metric | Value | Diagnosis |
|---|---:|---|
| vx mean | **+0.011 m/s** | Robot doesn't move (commanded 0.4 m/s) |
| All four `|Δα|max` | **0.197, 0.200, 0.200, 0.200** | All saturated at the bound |

**What was happening:** The linear ramp at α=0.5 produces an incoherent midpoint — averaging "trot wants FL+RR planted" with "bound wants FL+FR planted" gives joint targets that don't correspond to any valid gait. The PPO policy discovered that *standing still* during the unstable midpoint window earns the alive bonus + partial track reward, while attempting to move risks catastrophic falls. With `|Δα| ≤ 0.2`, the residual was too small to fix the underlying joint-space incoherence.

**Lesson:** Bounded residuals cannot rescue from a fundamentally incoherent baseline. The bound itself was the limiting factor.

#### v2 — Wider residual (`delta_alpha_max = 0.8`)

**Hypothesis:** Relaxing the residual bound 4× (0.2 → 0.8) gives the MLP enough magnitude to override the incoherent midpoint when needed.

**Setup:** Same as v1, but `delta_alpha_max = 0.8`.

**Result (mean reward = +1100, vx mean = +0.160 m/s, falls = 0):**

Major partial improvement. Some transition windows now show clean velocity tracking — trot↔bound segments hit vx peaks of 0.45-0.51 m/s with body height stable. But:

| Failure mode | Evidence |
|---|---|
| **Source gait isn't visible early** | During the source-only hold window (before transition starts), `|Δα|` was still 0.18-0.39 std — the MLP was applying corrections constantly, dragging target-gait influence into the source phase. The source gait never had a chance to run pure. |
| **Steer transitions completely fail** | Pace→steer segments showed vx ≈ 0 throughout. Cause: steer was trained with `yaw ∈ (0.4, 1.0)`, always with yaw — Phase 2 sends `yaw = 0`. Steer's output is out-of-distribution → garbage joint targets → blend fails. |
| **All four legs `|Δα|max` saturated at 0.56-0.80** | The MLP wants to override even harder than 0.8 allows in some moments. |

**Lesson:** Wider residual bounds enable working transitions but don't solve the architectural problem. Two distinct issues:
1. Without time-gating, the MLP intervenes constantly — even when it shouldn't.
2. Including a fourth gait with mismatched training distribution corrupts the entire system.

#### v3 — Drop steer + time-gate residual + boost sparsity

**Hypotheses:**
1. Restricting to three coordination gaits (trot, bound, pace) — all sharing similar forward-only training distributions — eliminates the OOD problem.
2. **Hard time-gating**: forcing `Δα = 0` outside the transition window guarantees source/target gaits run untouched during steady-state holds.
3. Bumping the L2 sparsity penalty (-0.5 → -3.0) discourages large `|Δα|` even within the active window.

**Setup:**
- 3 base policies (trot, bound, pace)
- Time-gating: `Δα = MLP(obs) if (t in transition_window) else 0`
- Per-env transition_start_s sampled uniformly in [1.5, 3.5] s for robustness
- Sparsity penalty `−3.0 · ‖Δα‖²`

**Result (mean reward = +3787, vx mean = +0.057 m/s, falls = 0):**

Time-gating worked: per-step printout showed `Δα = +0.000 -0.000 +0.000 +0.000` during steady-state (gating doing its job). Δα magnitudes during transitions reduced to 0.2-0.5 (from v2's 0.56-0.80).

But playback revealed a *different* failure: **steady-state stagnation**. The robot doesn't move during the source-only and target-only hold windows — even though `Δα = 0` should mean "pure source gait" or "pure target gait" output goes to the robot.

**Diagnosis (took several debug iterations):** A bug in the base-policy observation construction. Phase 1 policies were trained with `last_action` in their observation = their own previous 12-D actor output. The Phase 2 env was always feeding `last_action = zeros` to the base policies. Each policy saw "I just took a no-op action" every step → policies trained with action_rate penalty collapsed to a stationary "default pose" output during steady-state. During transitions, the body was being pushed around enough that joint velocities remained non-zero, giving the policies *something* to react to — that's why transitions worked but steady-state didn't.

**Lesson:** Frozen base policies depend on observation-format fidelity. Subtle obs-construction bugs (always-zero `last_action`) silently degrade their behavior.

#### v4 — Per-policy `last_action` history (current best)

**Fix:** Maintain a `_base_last_actions[3, num_envs, 12]` buffer in the env. Each base policy queries with its own previous output as its `last_action` observation. Buffer reset on episode reset.

**Setup:** Same as v3 + per-policy last_action buffers.

**Result (mean reward = +4887, vx mean = +0.425 m/s, falls = 0):**

This is the working policy. **Source gait runs visibly during the hold window. Target gait runs visibly after the ramp. Transitions complete without falls.**

| Metric | v1 | v2 | v3 | **v4** |
|---|---:|---:|---:|---:|
| vx mean | +0.011 | +0.160 | +0.057 | **+0.425** |
| Mean reward | +954 | +1100 | +3787 | **+4887** |
| Δα behavior | saturated at 0.2 | saturated at 0.8 | sparse but unused | **sparse and effective** |
| Source gait visible | no | corrupted | no | **yes** |
| Target gait visible | no | yes (sometimes) | no | **yes** |
| Falls | 0 | 0 | 0 | **0** |
| Per-leg \|Δα\|max | 0.197–0.200 | 0.565–0.798 | 0.183–0.530 | **0.129–0.284** |

The per-leg `|Δα|max` of 0.13-0.28 is well below the 0.8 budget — the MLP needs only small corrections, demonstrating that the residual approach is parsimonious. The mean ≈ 0 across all legs confirms the sparsity penalty + time-gating produce the expected explainability signature.

**Per-step Δα pattern across a transition (trot → bound):**

```
Step  50: trot→bound  vx=+0.343  Δα = (-0.000, -0.000, -0.000, -0.000)   ← source-only, gait visible
Step 100: trot→bound  vx=+0.407  Δα = (-0.003, -0.031, +0.015, +0.007)   ← ramp begins, residuals small
Step 150: trot→bound  vx=+0.257  Δα = (+0.013, +0.038, +0.091, +0.106)   ← mid-ramp, RL/RR engage
Step 200: trot→bound  vx=+0.329  Δα = (+0.001, +0.009, +0.094, +0.042)   ← ramp finishing
Step 300: trot→bound  vx=+0.412  Δα = (-0.000, -0.000, +0.000, -0.000)   ← target-only, gait visible
```

**Per-leg interpretation:** RL and RR show larger residual corrections than FL and FR during this transition — consistent with the morphological hypothesis. Going from trot (FL+RR diagonal) to bound (FL+FR fore-aft) requires the **rear legs** to switch their sync partner (FL → RL for rear). The MLP discovered this asymmetry and applies stronger corrections to the rear pair.

### Remaining issues for v5 (planned)

While v4 delivers the central result, transitions still have observable spikes:
- Tilt mean 0.056 (max 0.189) — body pitches during some transitions, especially trot↔bound
- Some vz spikes ±0.7 m/s during gait reorganization

These are **cosmetic, not functional** — the transitions still complete without falls and tracking is preserved. To address them in v5:
- Add joint-acceleration penalty `−0.001 · ‖q̈‖²` to discourage jerky motion
- Bump action_rate weight on Δα (-0.05 → -0.15) for smoother residual evolution
- Consider tightening `delta_alpha_max` (0.8 → 0.5) — the current `|Δα|max` of 0.28 suggests the budget is overprovisioned

---

## Planned Experiments

### Polish: graceful transition metrics (v5)

The v4 result demonstrates *successful* transitions. v5 will quantify *graceful* transitions using metrics established in legged-locomotion research:

| Metric | Formula | What it measures |
|---|---|---|
| **Cost of Transport (CoT)** | `Σ\|τ·q̇\|·dt / (m·g·distance)` | Energy efficiency — quadrupeds at trot typically have CoT ≈ 0.5-1.0; transition spikes reveal extra effort |
| **Mean forward velocity during ramp** | `mean(vx)` over [t_start, t_start + duration] | How much momentum the transition costs |
| **Joint acceleration peak** | `max(‖q̈(t)‖)` during ramp | Mechanical jerk — high values cause wear and visual ugliness |
| **Body tilt RMS during ramp** | `sqrt(mean(tilt²(t)))` | Postural stability through the transition |
| **Joint trajectory at switch boundary** | `q(t)` per joint, plotted with vertical markers at switch moments | Visual continuity check |

CoT requires `applied_torque` from `articulation.data` and integrated body distance from `root_pos_w`. Both available in Isaac Lab — no env modifications needed; metrics are computed offline from logged playback.

### Baseline experiments (4 methods, common evaluation protocol)

To establish that the per-leg residual matters, we will compare four transition-control methods on identical evaluation episodes (100 trials per directed gait pair × 6 pairs = 600 trials per method):

| Method | Description | Implementation |
|---|---|---|
| **(a) Discrete Switch** | π_target replaces π_current instantly at trigger time (no blending). | Set `Δα = 0` and `α_baseline = step(t − transition_start)`. No training needed. |
| **(b) Linear Ramp** | Pure baseline schedule, no learned residual. Single uniform α for all legs. | Set `Δα ≡ 0`. No training needed. |
| **(c) E2E PPO** | Standard PPO outputs full α (1-D scalar) without any baseline. Same backbone (128, 128). | New env config — action space 1-D, no baseline ramp, pure RL on a from-scratch transition policy. ~30 min training. |
| **(d) Residual (Ours)** | What v4 produces. Baseline + per-leg learned correction. | Already trained. |

**Comparison metrics:** survival rate (fraction completing the transition), CoT, mean forward velocity, joint acceleration peak, body tilt RMS.

**Predicted outcomes:**
- Discrete Switch likely **catastrophically fails** — instant 12-D joint target swap on a 50 kg body causes huge action jerk, predicted survival rate near 0%.
- Linear Ramp survives easy pairs (trot↔pace) but fails on the trot↔bound coordination swap (incoherent midpoint).
- E2E PPO likely matches Residual on survival but loses *explainability* — policy outputs are opaque without the baseline decomposition.
- Residual achieves comparable or better survival than E2E PPO with substantially smaller learned magnitude.

The argument for Residual is not necessarily that it scores higher than E2E PPO, but that it provides **comparable performance with full explainability** — directly supporting the contribution claim.

### Ablation studies

Repurposed for the PPO + Residual setup (the original CPG-RBF ablations partially translate):

| Ablation | What it tests | Implementation |
|---|---|---|
| **Per-leg (4-D Δα) vs single-α (1-D Δα)** | Whether the per-leg structure matters for transitioning between different leg-pair coordinations. **The central architectural claim.** | Change action dim from 4 to 1, broadcast to all legs. ~30 min retrain. |
| **Δα bound sweep (0.1, 0.2, 0.5, 0.8)** | How residual magnitude budget affects transition quality. | Already partial: v1=0.2, v2/v4=0.8. Need 0.1 and 0.5 retrains. |
| **With vs without time-gating** | Whether forcing Δα=0 outside transition window is necessary for explainability. | Toggle the gating mask in env. ~30 min retrain without gating. |
| **With vs without sparsity penalty** | Whether the sparsity term is necessary or if gating alone suffices. | Toggle sparsity weight. ~30 min retrain. |

Each ablation produces a metric row in the comparison table. Estimated total ablation time: 4 retrains × 30 min = 2 hours.

### Diagnostic plots and per-version logs

For the report, generate per-version diagnostic figures from existing v1-v4 checkpoints:

- **Figure 1: Δα(t) per leg across a transition** (4 line plots stacked) — shows the temporal pattern of per-leg corrections
- **Figure 2: Joint position(t) per joint across a transition** (12 line plots stacked, vertical markers at switch boundaries) — shows trajectory continuity
- **Figure 3: vx(t), height(t), tilt(t) across all transitions** — shows tracking quality
- **Figure 4: Method comparison bars** — survival rate, CoT, joint accel peak across (Discrete, Linear, E2E, Residual)

Implementation: extend the play script to dump per-step CSV; separate plotting script generates figures.

### Optional: CPG-RBF revisit (if time permits)

The legacy CPG-RBF infrastructure could be repurposed: not as the controller, but as a **phase oracle** for the residual MLP. The Δα MLP currently sees `α_baseline` and `cycles_elapsed` as scalars. A CPG running in parallel could provide `(sin φ, cos φ)` per leg — letting the MLP time its corrections to specific gait phases.

This is the lowest-priority item and only happens if all baselines, ablations, polish, and the report are complete with time remaining.

---

## Per-Version Log

| Version | Path | Δα bound | Time-gate | Sparsity weight | Result | Status |
|---|---|---:|:---:|---:|---|---|
| v1 | `logs/phase2/phase2_v1/` | 0.2 | no | -0.5 | Standstill exploit | Negative result |
| v2 | `logs/phase2/phase2_v2_wider/` | 0.8 | no | -0.5 | Partial transitions, source corrupted | Negative result |
| v3 | `logs/phase2/phase2_v3/` | 0.8 | yes | -3.0 | Stagnant steady-state (last_action bug) | Negative result |
| **v4** | `logs/phase2/phase2_v4/` | 0.8 | yes | -3.0 | **Clean transitions, vx mean +0.425** | **Working baseline** |
| v5 | (planned) | 0.5 | yes | -3.0 | Polish (tilt smoother, joint accel lower) | TODO |

Final v4 checkpoint copied to `logs/phase2_final/transition_policy.pt` for reproducibility.

---

## Setup & Run Commands

### Environment

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Pre-flight: kill any zombie Isaac Sim processes before launching
nvidia-smi && pgrep -f "python.*play\|python.*train\|isaac\|kit" | xargs -r kill -9
```

### Phase 1 — train the four base policies (already done, available at `logs/phase1_final/`)

```bash
# Trot (medium speed, diagonal coordination)
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Trot-v0 \
    --max_iterations 1500 --run_name trot_v2

# Bound (fore-aft pair coordination)
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Bound-v0 \
    --max_iterations 4000 --run_name bound_v4

# Pace (lateral pair coordination)
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Pace-v0 \
    --max_iterations 4000 --run_name pace_v2

# Steer (forward + turning, optional 4th gait)
python scripts/train_b1_velocity.py --headless --num_envs 4096 \
    --task Isaac-Velocity-Flat-Unitree-B1-Steer-v0 \
    --max_iterations 1500 --run_name steer_v2
```

### Phase 1 — playback any base gait

```bash
python scripts/play_b1_velocity.py \
    --task Isaac-Velocity-Flat-Unitree-B1-{Trot,Bound,Pace,Steer}-Play-v0 \
    --checkpoint logs/phase1_final/{trot,bound,pace,steer}.pt \
    --num_envs 16 --steps 1000

xdg-open logs/gait_diagram_ppo_b1.png
```

### Phase 2 — train residual transition policy

```bash
# Verify Phase 1 base policies are in place
ls logs/phase1_final/{trot,bound,pace}.pt

# Train the residual MLP — ~25-30 min on RTX 4070 Ti SUPER
python scripts/train_b1_phase2.py --headless --num_envs 2048 \
    --max_iterations 2000 --run_name phase2_v4
```

### Phase 2 — playback transition policy across cycling gaits

```bash
python scripts/play_b1_phase2.py \
    --checkpoint logs/phase2_final/transition_policy.pt \
    --num_envs 4 --steps 2000 \
    --gait_pairs trot,bound,pace \
    --switch_interval_s 8.0
```

The play printout shows per-step `(vx, vz, h, tilt, Δα_FL, Δα_FR, Δα_RL, Δα_RR)`. Watch the Δα columns transition from `+0.000` (steady-state) to small corrections (during ramp) and back to `+0.000` (post-ramp).

### Tests

```bash
python -m pytest tests/ -q                # 44/44 unit tests
```

---

## B1 Robot Configuration

### Joint axis convention

| Joint | Axis | Default FL/FR/RL/RR | Role |
|---|---|---|---|
| `hip_joint` | Abduction (lateral splay) | +0.1 / −0.1 / +0.1 / −0.1 | Lateral balance — kept small |
| `thigh_joint` | **Flexion (fore/aft swing)** | +0.8 / +0.8 / +1.0 / +1.0 | **Primary walking driver** |
| `calf_joint` | Knee bend | −1.5 / −1.5 / −1.5 / −1.5 | Foot clearance during swing |

The +0.2 rad asymmetry between front and rear thighs is responsible for several trained-policy quirks (rear-heavy duty in bound, asymmetric leg use in trot) — fundamental to B1's morphology and **directly motivates the per-leg residual structure** in Phase 2 (different legs need different transition rates).

### Foot contact convention

B1's USD uses `*_foot` link names (NOT `*_calf`). All contact-sensor patterns use `.*_foot$` (anchored).

The trunk body is named `trunk` (NOT `base` like Go2). Every `body_names="base"` inherited from Isaac Lab's stock Go2 cfg must be overridden to `"trunk"` for B1.

### Actuator overrides (project-local deep-copy)

```python
UNITREE_B1_CFG.actuators["base_legs"].stiffness = 400.0    # was 200 — 200 sags 9 cm under body weight
UNITREE_B1_CFG.actuators["base_legs"].damping   = 10.0     # proportional ratio
UNITREE_B1_CFG.init_state.pos = (0, 0, 0.50)               # was 0.42 — feet were 7.7 cm under ground at default joints
```

`base_contact.threshold = 50 N` (was 1 N) — settling produces 20-40 N transients on a 50 kg body.

---

## Project Structure

```
cpg-drl-transition/
├── envs/
│   ├── unitree_b1_env.py           # CPG-RBF DirectRLEnv (legacy)
│   ├── b1_velocity_env_cfg.py      # PPO velocity-tracking env configs (Phase 1)
│   ├── b1_velocity_ppo_cfg.py      # PPO hyperparameters (Phase 1 + Phase 2)
│   ├── b1_velocity_mdp.py          # 11 custom reward functions (Phase 1)
│   ├── b1_phase2_env_cfg.py        # Phase 2 env config (residual transition)
│   └── b1_phase2_env.py            # Phase 2 env class — base policy loading + per-leg blending
├── networks/
│   └── cpg_rbf.py                  # CPG-RBF network (legacy)
├── algorithms/
│   └── pibb_trainer.py             # PI^BB optimizer (legacy)
├── scripts/
│   ├── train_b1_velocity.py        # Phase 1 PPO training
│   ├── play_b1_velocity.py         # Phase 1 playback with diagnostics
│   ├── train_b1_phase2.py          # Phase 2 residual MLP training
│   ├── play_b1_phase2.py           # Phase 2 playback with scripted gait switches
│   ├── train_phase1_*.py           # Legacy CPG-RBF training
│   ├── play_gait.py                # Legacy CPG-RBF playback
│   └── visualize_cpg.py            # Legacy CPG phase plots
├── configs/                        # Legacy CPG-RBF YAML configs
├── logs/
│   ├── ppo_b1/<run>/               # Phase 1 PPO run dirs
│   ├── phase1_final/               # Final Phase 1 base policies (trot, bound, pace, steer)
│   ├── phase2/<run>/               # Phase 2 residual MLP run dirs (v1-v4)
│   ├── phase2_final/               # Final Phase 2 transition policy
│   └── gait_*.png                  # Saved gait diagrams (Phase 1)
├── weights/                        # Legacy CPG-RBF W matrices
├── tests/                          # 44 unit tests
├── README.md                       # This report
├── CLAUDE.md                       # AI-assistant context
└── pytest.ini
```

---

## Implementation Progress

### Week 10 — Setup ✅
- Isaac Lab + Isaac Sim + RSL-RL verified
- Custom MDP framework (44/44 tests)

### Week 11 — Phase 1 CPG-RBF attempts (abandoned)
- 12 encoding experiments documented
- PI^BB with softmax averaging, per-joint noise scaling
- LocoNets KENNE pre-compute integration
- Cosine walking prior (Thor-style)
- **Conclusion:** B1 + PIBB cannot reliably produce stable gaits in the project timeline

### Week 11 — Phase 1 PPO pivot ✅
- Manager-based RL env with B1-specific actuator config
- 11 custom reward terms for B1 mass scaling
- Trot, bound, pace, steer base policies trained and validated

### Week 12 — Phase 2 architecture ✅
- DirectRLEnv with frozen base policy loading
- Per-leg residual MLP via PPO
- v1 → v4 evolution with documented failure modes
- **v4 working baseline:** vx mean +0.425, zero falls, sparse explainable Δα

### Week 13 (current) — Phase 2 polish + experiments (in progress)
- [ ] v5 with joint-accel penalty + CoT logging
- [ ] Baseline experiments (Discrete, Linear, E2E PPO, Residual)
- [ ] Ablations (per-leg vs scalar, Δα sweep, gating, sparsity)
- [ ] Diagnostic plots and per-version comparison figures

### Week 14 — Analysis and writeup
- [ ] Compile baseline + ablation results
- [ ] Generate report figures
- [ ] Update README/CLAUDE.md with final results

### Week 15 — Final polish + (optional) CPG-RBF revisit
- [ ] Final video demo
- [ ] CPG as phase oracle for residual MLP (if time permits)

---

## Key References

- **Silver et al. 2018** — *Residual Policy Learning*. arXiv:1812.06298. The canonical "learn corrections on top of a baseline" paper. The Phase 2 architectural pattern.
- **Johannink et al. 2018** — *Residual Reinforcement Learning for Robot Control*. ICRA 2019. Same pattern, applied to manipulation.
- **Thor et al. 2021** — [CPG-RBFN framework](https://github.com/MathiasThor/CPG-RBFN-framework). SO(2) oscillator + RBF + PI^BB. Source of the original (abandoned) Phase 1 design; preserved in legacy code.
- **Siekmann et al. 2021** — *Sim-to-real bipedal locomotion with PPO*. Reference for joint-offset action design.
- **Rudin et al. 2022** — *legged_gym*. Isaac Gym quadruped training. Reward stack inspiration for B1 reward engineering.
- **Isaac Lab `Isaac-Velocity-Flat-Unitree-Go2-v0`** — Stock manager-based velocity-tracking config. Direct ancestor of `b1_velocity_env_cfg.py`.
- **unitree_rl_lab** — Unitree's manager-based RL framework (Go2/G1/H1). Reference for hyperparameters and reward weights.

---

## Acknowledgments

Project supervised in the context of FRA 503 — Deep Reinforcement Learning. The CPG-RBF detour was a productive research-process learning experience: documented failure of an interesting design provides scaffolding for the simpler PPO baseline that ultimately succeeded.
