You are helping me finalize my FRA503 project repository, code, experiment results, README, and report text.

The current final research direction is:

Transition-Aware Quadruped Locomotion: A Study of Residual Correction Spaces

The final project should NOT be framed as “we proposed per-leg residual learning from the beginning.” Instead, frame it as a systematic research study:

Before applying residual learning to quadruped gait transition, we ask:
1. Where should the residual act?
   - q-space: directly correct joint commands Δq
   - α-space: correct blending coefficient / transition timing Δα
2. What action dimension is appropriate?
   - 4D: per-leg correction
   - 12D: per-joint correction

The final result is two-level: canonical (N=6, seed=42) and multi-seed (N=60, 10 seeds × 6 pairs).
The two evaluations agree on the main finding but disagree on which method is safest.

Canonical N=6 (seed=42): Res-α 12D has zero velocity reversal; Res-q 4D has mild reversal.
Multi-seed N=60: Res-q 4D has zero reversal (0/60 windows); Res-α 12D has 30% reversal rate.

The robust primary finding (holds at both N=6 and N=60):
- ALL four residual variants beat Smoothstep on jerk_TRANS.

The gait-phase-dependent finding (N=6 and N=60 disagree):
- Velocity reversal safety depends on which gait phase is active at switch time.
- N=6 alone is insufficient to conclude about velocity safety.
- At N=60: Res-q 4D achieves zero reversal + lowest jerk among zero-reversal methods.
- The per-leg uniform correction in Res-q 4D (design flaw) acts as implicit conservatism.

Your responsibility:
You have access to the project repository. Please fix and improve the project, code, results, README, and report narrative so the whole project is internally consistent and ready for a polished academic report. You do not need to make presentation slides, but the storyline should be clear enough that slides could be made from it later.

Important final numbers to use consistently:

Single-seed canonical evaluation, seed=42, 2500 steps, 6 directed gait-pair transitions:
Baseline CSVs corrected: sigmoid bug fixed (actions=-100, Δα≈0 during holds).
Residual variants retrained with sparsity=-0.5, jerk_weight=-2e-10 where noted.

Discrete Switch:
- vx_mean = +0.435
- vx_std = 0.108
- vx_min = -0.195
- tilt_max = 0.234
- h_mean = 0.409
- CoT = 2.793
- jerk_TRANS = 11361

Linear Ramp:
- vx_mean = +0.390
- vx_std = 0.157
- vx_min = -0.206
- tilt_max = 0.184
- h_mean = 0.404
- CoT = 1.955
- jerk_TRANS = 7441

Smoothstep Ramp:
- vx_mean = +0.415
- vx_std = 0.129
- vx_min = -0.096
- tilt_max = 0.187
- h_mean = 0.405
- CoT = 2.090
- jerk_TRANS = 8508

Residual-α 4D (retrained sp05_jw2):
- vx_mean = +0.430
- vx_std = 0.113
- vx_min = -0.086
- tilt_max = 0.189
- h_mean = 0.408
- CoT = 2.171
- jerk_TRANS = 7617

Residual-q 4D (retrained sp05_jw2):
- vx_mean = +0.416
- vx_std = 0.100
- vx_min = -0.024
- tilt_max = 0.200
- h_mean = 0.417
- CoT = 2.158
- jerk_TRANS = 7320

Residual-α 12D (original checkpoint):
- vx_mean = +0.427
- vx_std = 0.109
- vx_min = +0.004
- tilt_max = 0.190
- h_mean = 0.407
- CoT = 2.105
- jerk_TRANS = 7951

Residual-q 12D (retrained sp05_jw2):
- vx_mean = +0.408
- vx_std = 0.130
- vx_min = -0.122
- tilt_max = 0.207
- h_mean = 0.414
- CoT = 2.064
- jerk_TRANS = 7719

Main single-seed claims (canonical N=6, seed=42):
- ALL four residual variants beat Smoothstep on jerk_TRANS.
- Lowest jerk: Res-q 4D (7320), but has mild velocity reversal (vx_min=-0.024).
- Res-α 12D reduces jerk_TRANS by 6.5% vs Smoothstep: 8508 → 7951.
- Res-α 12D is the only method with zero velocity reversal at canonical N=6 (vx_min=+0.004).
- Res-α 12D has comparable CoT to Smoothstep (2.105 vs 2.090, within noise).

Multi-seed claims (N=60, 10 seeds × 6 pairs, seed_experiment_v2):
- ALL four residual variants beat Smoothstep on jerk_TRANS mean.
  Smoothstep: 9102  Res-α 4D: 8185 (−10%)  Res-α 12D: 8570 (−6%)
  Res-q 4D: 7619 (−16%)  Res-q 12D: 7305 (−20%)
- Velocity reversal rates at N=60:
  Smoothstep: 55.0%  Discrete: 18.3%
  Res-α 12D: 30.0%  Res-α 4D: 7.4%
  Res-q 12D: 38.3%  Res-q 4D: 0.0%
- Res-q 4D is the ONLY method with zero velocity reversal at N=60 (worst vx_min=+0.072).
- Res-α 12D's zero-reversal property does NOT generalize: 30% reversal at N=60.
- N=6 and N=60 agree on jerk ordering but disagree on velocity safety ranking.
- The robust claim: all residual variants beat Smoothstep on jerk.
- The gait-phase-dependent claim: velocity safety depends on which gait phases are tested.

Per-gait-pair analysis results (N=6 directed transitions, canonical seed=42):

These are the 6 gait-pair jerk_TRANS values from the single deterministic seed=42 episode.
They reveal the per-pair difficulty hierarchy (which gait pairs are structurally harder).

Discrete Switch:
- N = 6
- jerk_TRANS mean = 11361
- std = 5426
- min = 4183
- max = 19030

Smoothstep Ramp:
- N = 6
- jerk_TRANS mean = 8508
- std = 2610
- min = 4233
- max = 11801

Res-α 4D:
- N = 6
- jerk_TRANS mean = 7617
- std = 3104
- min = 4607
- max = 13540

Res-q 4D:
- N = 6
- jerk_TRANS mean = 7320
- std = 1934
- min = 4930
- max = 10789

Res-α 12D:
- N = 6
- jerk_TRANS mean = 7951
- std = 3267
- min = 4072
- max = 12351

Res-q 12D:
- N = 6
- jerk_TRANS mean = 7719
- std = 1921
- min = 4490
- max = 10193

Note on seed=42 vs N=60 framing:
- seed=42 is the canonical evaluation used for all visual diagnostics (zoom plots, gait
  diagrams, delta-alpha plots). It is a legitimate deterministic baseline used for
  reproducible visual comparisons.
- The N=60 multi-seed experiment (logs/phase2_seed_experiment_v2) uses the SAME corrected
  checkpoints and sigmoid fix. It is valid and should be cited alongside N=6.
- The two evaluations agree on jerk ordering but disagree on velocity safety:
  N=6: Res-α 12D has zero reversal; N=60: Res-q 4D has zero reversal.
- This disagreement is the main finding of the multi-seed study: velocity safety is
  gait-phase-dependent and N=6 is insufficient to conclude about it.
- Do NOT claim Res-α 12D has general zero reversal. Qualify: "at the canonical seed=42
  evaluation" or "at the fixed gait phase used in the canonical run."
- The N=60 v2 data supersedes the old multi-seed data (old checkpoints, pre-sigmoid fix).

Note on Res-q 4D design flaw:
- Res-q 4D applies a single scalar Δq to all three joints in each leg (hip, thigh, calf).
- Hip, thigh, and calf operate in different angular ranges and serve different mechanical roles.
  Applying the same correction to all three is physically unreasonable.
- Res-q 4D is included for 2×2 completeness but should not be treated as a fair q-space
  ablation point. The proper q-space representative is Res-q 12D (per-joint correction).
- Res-α 4D does NOT have the same problem: a blending weight α is dimensionless and
  scale-invariant, so a per-leg α is at least internally consistent.

Important wording for claims:

CANONICAL (N=6, seed=42):
"All four residual variants beat Smoothstep on jerk_TRANS. In the canonical evaluation,
Res-α 12D is the only method with zero velocity reversal (vx_min = +0.004), achieving
−6.5% jerk reduction vs Smoothstep. However, this result reflects a single fixed gait
phase at switch time."

MULTI-SEED (N=60):
"Across 60 transition windows (10 seeds × 6 gait pairs), all residual variants beat
Smoothstep on mean jerk_TRANS (−6% to −20%). Res-q 4D achieves zero velocity reversal
across all 60 windows while reducing jerk by 16%. Res-α 12D shows 30% reversal rate at
N=60, indicating its zero-reversal property is gait-phase-specific."

HONEST COMBINED CLAIM:
"The primary finding — all residual variants beat Smoothstep on jerk — is robust across
both evaluations. The velocity safety ordering is gait-phase-dependent: the canonical
evaluation (fixed phase, N=6) favors Res-α 12D; the multi-phase evaluation (N=60) favors
Res-q 4D. The 2×2 ablation reveals unexpected complexity: the design constraint in Res-q 4D
(uniform per-leg correction) acts as implicit conservatism that benefits velocity safety
across diverse gait phases."

2×2 ablation table:

Residual-α 4D (retrained sp05_jw2):
- vx_mean +0.430
- vx_std 0.113
- vx_min -0.086
- tilt_max 0.189
- h_mean 0.408
- CoT 2.171
- jerk_TRANS 7617

Residual-q 4D (retrained sp05_jw2):
- vx_mean +0.416
- vx_std 0.100
- vx_min -0.024
- tilt_max 0.200
- h_mean 0.417
- CoT 2.158
- jerk_TRANS 7320

Residual-α 12D (original checkpoint):
- vx_mean +0.427
- vx_std 0.109
- vx_min +0.004
- tilt_max 0.190
- h_mean 0.407
- CoT 2.105
- jerk_TRANS 7951

Residual-q 12D (retrained sp05_jw2):
- vx_mean +0.408
- vx_std 0.130
- vx_min -0.122
- tilt_max 0.207
- h_mean 0.414
- CoT 2.064
- jerk_TRANS 7719

Interpretation (canonical N=6):
- ALL four residual variants beat Smoothstep on jerk_TRANS (8508). This is the main finding.
- Within α-space: 4D (7617) achieves lower jerk than 12D (7951). Both have mild reversal at
  N=6 except 12D which has zero reversal at the fixed canonical gait phase.
- Within q-space: 4D (7320) lower jerk than 12D (7719); both show reversal at N=6.
- Smoothstep has velocity reversal (vx_min=−0.096) and higher jerk than all residual variants.

Interpretation (multi-seed N=60, seed_experiment_v2):
- ALL four residual variants beat Smoothstep on mean jerk (9102).
  Res-q 4D: 7619 (−16%)  Res-q 12D: 7305 (−20%)
  Res-α 4D: 8185 (−10%)  Res-α 12D: 8570 (−6%)
- Velocity reversal rates:
  Res-q 4D: 0%  Res-α 4D: 7%  Discrete: 18%
  Res-α 12D: 30%  Res-q 12D: 38%  Smoothstep: 55%
- The safety ordering flips vs canonical: Res-q 4D has zero reversal at N=60, not Res-α 12D.
- Res-q 12D has the lowest jerk (7305) but worst reversal among residual variants (38%).
- The conservative action space of Res-q 4D (same scalar to all joints → small values)
  appears to provide implicit safety across diverse gait phases.
- Do not present other methods as simply bad. Present trade-offs at both evaluation levels.

====================================================================
REQUIRED STRUCTURE / TABLE OF CONTENTS
====================================================================

Please reorganize the README/report into a clear Table of Contents so each study can be accessed easily and in the correct order.

The Table of Contents should separate the work into logical categories:

1. Project Overview
   - Motivation
   - Problem Statement
   - Final Research Question
   - Contributions
   - Key Results

2. Background and Design Motivation
   - Quadruped gait transition problem
   - Why naive discrete switching fails
   - Why passive blending helps but is limited
   - Why residual learning is appropriate
   - Why Smoothstep is selected as the residual baseline
   - Why not train a policy to learn the whole blending equation from scratch?

3. Phase 1: Base Gait Policy Generation
   - Initial CPG-RBF + PIBB direction
   - Why CPG-RBF failed on B1
   - Pivot to PPO base gait policies
   - Trot / Bound / Pace base gait policies
   - Caveat: PPO gaits are functional but not biologically perfect

4. Phase 2: Residual Transition Learning
   - Frozen base policy blending
   - Smoothstep baseline
   - Residual correction design
   - Time-gating
   - Asymmetric clamp Δα ∈ [0, 0.3]
   - Sparsity penalty
   - Reward function
   - Final Residual-α 12D architecture

5. Development History of Residual Learning
   - v1–v10: improving the general residual model
   - Explain this as architecture/debug development, not final contribution
   - v1–v10 mainly focused on Residual-α 4D
   - v10 solved key problems: time-gating, sparsity, asymmetric clamp, jerk metric
   - After v10 stabilized the residual framework, the project expanded into systematic 2×2 ablation:
     q-space vs α-space
     4D vs 12D
   - Residual-α 12D emerged as best after the 2×2 study

6. Systematic Design-Space Study
   - Research question: Where should residual act?
   - Output space:
     q-space vs α-space
   - Action dimension:
     4D vs 12D
   - 2×2 ablation table
   - Interpretation and trade-offs

7. Experiments and Metrics
   - Experimental sequence
   - Methods compared
   - Metrics:
     jerk_TRANS
     vx_mean
     vx_std
     vx_min
     tilt_max
     CoT
     fall/termination
   - Why jerk_TRANS is the primary metric
   - Why vx_min is important for velocity reversal

8. Results
   - Discrete spike analysis
   - Single-seed canonical result
   - Smoothstep vs Residual-α 12D
   - 2×2 ablation result
   - 60-window robustness result
   - Duration sweep
   - Jerk-weight sweep
   - With/without sparsity
   - Transition zoom plots

9. Discussion
   - What Residual-α 12D improves
   - What Smoothstep still does well
   - What q-space variants reveal
   - Why 12D helps
   - Why other methods are not simply “bad”
   - Trade-offs between smoothness, energy, variance, and safety

10. Limitations and Future Work
   - Fixed 3 s duration
   - Flat terrain only
   - Simulation only
   - Base gait quality
   - Need adaptive transition timing
   - Warm-start duration curriculum
   - Rough terrain and real robot testing

11. Reproducibility
   - Setup
   - Training commands
   - Evaluation commands
   - Plot generation scripts
   - Test commands
   - File structure

Make sure the Table of Contents links to sections if Markdown is used.

====================================================================
SMOOTHSTEP BASELINE: REQUIRED EXPLANATION
====================================================================

We need a strong reason why Smoothstep is the baseline for residual learning.

Please add a dedicated section:
“Why Smoothstep is the Residual Baseline”

The explanation should include:

1. Smoothstep is a passive, deterministic, non-learned transition schedule.
   It provides a fair baseline because it contains no extra policy capacity.

2. Smoothstep has zero derivative at both endpoints:
   α = x²(3 − 2x)
   dα/dt = 0 at x = 0 and x = 1
   This reduces endpoint kinematic kicks compared with a linear ramp.

3. Smoothstep is simple and interpretable.
   If residual output Δα = 0, the system exactly becomes Smoothstep.
   Therefore, Smoothstep provides a clean counterfactual baseline:
   “What does the learned residual add?”

4. Smoothstep is strong enough to be meaningful.
   It is better than discrete switching on transition continuity, but still fails in hard cases:
   it can produce velocity reversal and higher jerk than Residual-α 12D.

5. Smoothstep gives a structured prior.
   The residual does not need to learn the entire transition from scratch.
   It only learns small corrections on top of a reasonable schedule.

Suggested wording:
“We choose Smoothstep as the residual baseline because it is deterministic, interpretable, and already removes endpoint discontinuities through zero endpoint slope. This makes it a stronger and fairer baseline than discrete switching or linear interpolation. More importantly, setting Δα = 0 exactly recovers Smoothstep, so the effect of the learned residual can be measured directly.”

====================================================================
WHY NOT LET A POLICY LEARN THE WHOLE BLENDING EQUATION?
====================================================================

Please add a section:
“Why Residual Learning Instead of Learning the Full Blending Policy?”

We need a good academic reason why we do residual learning instead of training a policy to output the entire blending equation/schedule itself.

The answer should include:

1. Full schedule learning has a larger search space.
   A policy must learn:
   - when to start transition
   - how fast to transition
   - how to keep α monotonic
   - how to avoid unsafe midpoints
   - how to preserve source and target gaits
   This is harder than learning a small bounded correction.

2. Residual learning gives a safety fallback.
   If Δα = 0, the system remains Smoothstep.
   A full learned scheduler has no such built-in fallback.

3. Residual learning preserves steady-state gait policies.
   Time-gating forces residual = 0 outside the transition window.
   This guarantees that the frozen source and target gaits are untouched during steady-state.

4. Residual learning improves interpretability.
   Δα directly shows where and when the model disagrees with the baseline.
   This is easier to analyze than an unconstrained learned blending function.

5. Residual learning reduces exploit risk.
   Earlier unconstrained/symmetric variants showed delay-rush behavior.
   The final residual uses asymmetric clamp [0, 0.3] to prevent α from falling below Smoothstep.

6. The objective of this project is not to learn any possible transition function.
   It is to study how much a small, bounded correction can improve a strong baseline.

7. Experiments and Metrics
   - Canonical Evaluation (seed=42)
   - Robustness Evaluation (10 seeds × 6 transitions = 60 windows)
   - Why seed=42 is used for visualization
   - Why multi-seed evaluation is used for robustness

8. Results
   - Canonical seed=42 result
   - 60-window robustness result

Suggested wording:
“We do not ask the policy to invent the full blending equation because this makes the transition problem unnecessarily unconstrained. A fully learned scheduler could collapse to fast switching, delay the transition, or corrupt steady-state gaits. Instead, Smoothstep provides a stable prior and the residual learns only the missing correction. This makes the problem smaller, safer, and easier to interpret.”

====================================================================
HOW TO EXPLAIN v1–v10 AND THEN 2×2 ABLATION
====================================================================

Please fix the narrative around the iteration history.

The project history should not sound like:
“We first proposed per-leg 4D, then changed our mind randomly.”

It should sound like:
“We first developed a stable residual-learning framework using the simpler 4D α-space model. During v1–v10, the goal was to solve general residual-learning problems: standstill exploit, source gait corruption, missing time-gating, wrong last-action buffer, delay-rush exploit, wrong smoothness metric, and lack of sparsity. Once the residual framework was stable, we expanded the question into a systematic design-space study over output space and action dimension.”

Please add a section:
“From Residual Prototype to Design-Space Study”

Required explanation:

1. v1–v10 should be described as the residual framework development phase.
   It mainly used Residual-α 4D because 4D is simpler, easier to debug, and easier to interpret.

2. The goal of v1–v10 was not to prove 4D was final.
   The goal was to make residual transition learning work at all.

3. v1–v10 solved general issues:
   - standstill exploit
   - residual too small
   - no time-gating
   - steer policy out-of-distribution
   - base policies queried with wrong last_action
   - symmetric clamp caused delay-rush exploit
   - jacc_RMS was the wrong smoothness metric
   - sparsity was needed for minimal intervention

4. v10 produced a stable residual framework:
   - Smoothstep baseline
   - time-gating
   - asymmetric sigmoid clamp [0, 0.3]
   - jerk penalty
   - sparsity penalty
   - no velocity reversal

5. After v10, the research question became more systematic:
   If residual learning works, what residual output space and action dimension is best?

6. This led to the 2×2 ablation:
   - Residual-q 4D
   - Residual-q 12D
   - Residual-α 4D
   - Residual-α 12D

7. The result shows:
   - α-space is safer than q-space
   - 12D α improves jerk compared with 4D α
   - Residual-α 12D is final best method



Suggested wording:
“The v1–v10 sequence should be read as the development of a stable residual-learning recipe, not as the final architectural claim. We used the 4D α model as a controllable prototype to debug the residual framework. Once the recipe was stable, we evaluated the broader design space. The 2×2 ablation then showed that the best final architecture is not the prototype, but Residual-α 12D.”

====================================================================
TRADE-OFFS: DO NOT MAKE OTHER METHODS LOOK USELESS
====================================================================

Please add a discussion that other methods are good in different ways.

Required trade-off interpretation:

Smoothstep:
- Very simple
- No training needed
- Low CoT
- Strong passive baseline
- Weakness: velocity reversal and higher jerk than Residual-α 12D

Residual-α 4D:
- Simpler than 12D
- Safe, no velocity reversal
- Useful prototype and ablation
- Weakness: higher jerk and higher CoT than 12D

Residual-q 4D:
- Lower CoT than Residual-α 4D
- Lower 60-window std than Residual-α 12D
- Weakness: velocity reversal, less safe

Residual-q 12D:
- Low std
- Tests whether direct joint correction can work
- Weakness: worst velocity reversal among residual variants and higher jerk than α-space

Residual-α 12D:
- Best main objective:
  lowest jerk_TRANS
  no reversal
  near-smoothstep CoT
- Weakness:
  not lowest variance
  more complex than 4D
  duration-specific

Suggested wording:
“The result is not that every alternative fails. Instead, each method reveals a different trade-off. Smoothstep is the strongest passive baseline and remains energy-efficient. q-space residuals can reduce some variance but are less safe because they may produce velocity reversal. α-space residuals are more structurally constrained and preserve interpolation between frozen gait policies. The 12D α variant provides the best match to our primary objective: reducing transition-window jerk while preserving forward motion.”

====================================================================
CODE / RESULT / README RESPONSIBILITIES
====================================================================

Please do the following in the repository:

1. README/report cleanup:
   - Add Table of Contents.
   - Reorder sections using the structure above.
   - Update the final title and final method to Residual-α 12D.
   - Keep Residual-α 4D as prototype/ablation, not final.
   - Keep CPG-RBF as initial failed direction, not final contribution.
   - Replace outdated results from older versions unless marked as legacy.
   - Make all claims match the latest numbers.

2. Code consistency:
   - Ensure task names, config names, and comments correctly identify final method.
   - If the final method is Alpha12D, names/comments should not imply final method is 4D.
   - Do not delete old code/checkpoints unless asked.
   - Old methods should be preserved as baselines/ablations.

3. Result consistency:
   - Ensure all tables use the latest metrics.
   - Verify script outputs match README tables.
   - If any figure/table cannot be regenerated, mark it clearly as pending/manual check.
   - Make sure figure captions identify the correct method and not old v7/v10 names.

4. Report-style writing:
   - Write in academic but clear style.
   - Avoid overclaiming.
   - Use “indicates”, “suggests”, “demonstrates in simulation”.
   - Avoid “proves”, “best in every way”, “biologically faithful”.

5. Final output:
   At the end, report:
   - files changed
   - sections reorganized
   - result tables updated
   - figures checked
   - remaining inconsistencies or missing files
   - any assumptions made


====================================================================
SEED=42 VS MULTI-SEED EVALUATION: CLARIFICATION (UPDATED)
====================================================================

Two evaluation levels are used and BOTH are valid:

1. Canonical single-seed evaluation (seed=42):
   - seed = 42, 2500 steps, 6 directed gait pairs (fixed sequence)
   - Used for all visual diagnostics: discrete spike, transition zoom, gait diagrams,
     body state plots, Δα / residual plots.
   - All methods compared under identical fixed transition sequence and initial condition.
   - Per-gait-pair jerk_TRANS (N=6) is computed from this single episode — the 6 values
     come from the 6 directed transitions in that one run.

2. Multi-seed robustness evaluation (N=60, seed_experiment_v2):
   - 10 seeds × 6 gait pairs = 60 windows per method.
   - Uses --randomize_start: _transition_start_s ~ Uniform(1.5, 3.5) s per seed.
   - Seeds hit the switch at DIFFERENT gait phases → genuine jerk and reversal variation.
   - Data in logs/phase2_seed_experiment_v2/ (corrected checkpoints + sigmoid fix).
   - This IS a valid multi-seed robustness study. Use it.
   - Note: a PREVIOUS old experiment (logs/phase2_seed_experiment/) used env_cfg.seed
     only (no --randomize_start), so all seeds produced identical results. That old
     experiment was superseded by v2. Do not cite the old experiment.

IMPORTANT: Do NOT claim Res-α 12D has general zero reversal.
At canonical N=6: Res-α 12D zero reversal (vx_min=+0.004).
At N=60: Res-α 12D 30% reversal rate; Res-q 4D ZERO reversal.
Always qualify the zero-reversal claim with the evaluation level.

Use section labels:
- “Canonical Evaluation (seed=42)” or “Canonical N=6”
- “Per-Gait-Pair Analysis (N=6)”
- “Multi-Seed Robustness Evaluation (N=60)”

Results are interpreted correctly when:
- N=6 and N=60 AGREE on jerk ordering (all residual variants beat Smoothstep).
- N=6 and N=60 DISAGREE on velocity safety ranking.
- N=6: Res-α 12D zero reversal; N=60: Res-q 4D zero reversal.
- The robust claim is jerk. The phase-dependent claim is reversal safety.
- The final claim combines both evaluations and acknowledges the disagreement.