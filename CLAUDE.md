You are helping me finalize my FRA503 project repository, code, experiment results, README, and report text.

The current final research direction is:

Transition-Aware Quadruped Locomotion with Per-Joint Residual-α Learning

The final project should NOT be framed as “we proposed per-leg residual learning from the beginning.” Instead, frame it as a systematic research study:

Before applying residual learning to quadruped gait transition, we ask:
1. Where should the residual act?
   - q-space: directly correct joint commands Δq
   - α-space: correct blending coefficient / transition timing Δα
2. What action dimension is appropriate?
   - 4D: per-leg correction
   - 12D: per-joint correction

The final result is that Residual-α 12D is the best method for the main objective:
- lowest transition-window jerk among evaluated methods
- zero velocity reversal
- only small CoT increase over smoothstep
- better suited for transition timing than direct joint-space correction

Your responsibility:
You have access to the project repository. Please fix and improve the project, code, results, README, and report narrative so the whole project is internally consistent and ready for a polished academic report. You do not need to make presentation slides, but the storyline should be clear enough that slides could be made from it later.

Important final numbers to use consistently:

Single-seed canonical evaluation, seed=42, 2500 steps, 6 directed gait-pair transitions:

Discrete Switch:
- vx_mean = +0.435
- vx_std = 0.108
- vx_min = -0.195
- tilt_max = 0.234
- h_mean = 0.409
- CoT = 2.793
- jerk_TRANS = 11361

Linear Ramp:
- vx_mean = +0.392
- vx_std = 0.155
- vx_min = -0.193
- tilt_max = 0.196
- h_mean = 0.405
- CoT = 1.962
- jerk_TRANS = 8251

Smoothstep Ramp:
- vx_mean = +0.413
- vx_std = 0.129
- vx_min = -0.078
- tilt_max = 0.189
- h_mean = 0.405
- CoT = 2.052
- jerk_TRANS = 10121

Residual-α 4D:
- vx_mean = +0.433
- vx_std = 0.104
- vx_min = +0.004
- tilt_max = 0.186
- h_mean = 0.406
- CoT = 2.436
- jerk_TRANS = 9775

Residual-α 12D:
- vx_mean = +0.427
- vx_std = 0.109
- vx_min = +0.004
- tilt_max = 0.190
- h_mean = 0.407
- CoT = 2.105
- jerk_TRANS = 7951

Main single-seed claims:
- Residual-α 12D reduces jerk_TRANS by 21.4% vs Smoothstep:
  10121 → 7951
- Residual-α 12D reduces jerk_TRANS by 30.0% vs Discrete:
  11361 → 7951
- Residual-α 12D removes velocity reversal:
  vx_min +0.004 vs Smoothstep -0.078
- Residual-α 12D costs only about 2.6% more CoT than Smoothstep:
  2.105 vs 2.052

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
- jerk_TRANS mean = 10121
- std = 2458
- min = 6136
- max = 12790

Res-q 4D:
- N = 6
- jerk_TRANS mean = 9006
- std = 2072
- min = 6445
- max = 11076

Res-q 12D:
- N = 6
- jerk_TRANS mean = 9969
- std = 1857
- min = 6672
- max = 12932

Res-α 4D:
- N = 6
- jerk_TRANS mean = 9775
- std = 2738
- min = 5489
- max = 13804

Res-α 12D:
- N = 6
- jerk_TRANS mean = 7951
- std = 3267
- min = 4072
- max = 12351

Multi-seed experiment results (N=60: 10 seeds × 6 directed transitions):

NOTE: The seed sweep was previously broken — IsaacLab internally resets numpy's global
RNG during env init, causing np.random.uniform() to return the same hold time (2.0 s)
for all seeds. Fixed in play_b1_phase2.py by using an isolated np.random.default_rng()
Generator immune to IsaacLab's internal np.random.seed() calls.
Additionally, the discrete baseline had _transition_start_steps hardcoded to int(2.0/dt),
ignoring --randomize_start. Fixed by using _current_hold_s = _sample_transition_start().
All 6 methods now produce 10/10 unique seed files.

Discrete Switch:
- N = 60
- jerk_TRANS mean = 10166
- std = 4973
- min = 3044
- max = 22861

Smoothstep Ramp:
- N = 60
- jerk_TRANS mean = 8658
- std = 3256
- min = 3325
- max = 15340

Res-α 4D:
- N = 60
- jerk_TRANS mean = 8863
- std = 2558
- min = 4698
- max = 15207

Res-q 4D:
- N = 60
- jerk_TRANS mean = 9000
- std = 2150
- min = 4436
- max = 14571

Res-α 12D:
- N = 60
- jerk_TRANS mean = 8706
- std = 2569
- min = 4269
- max = 13279

Res-q 12D:
- N = 60
- jerk_TRANS mean = 8196
- std = 2524
- min = 3106
- max = 13242

Multi-seed interpretation:
- Discrete is clearly worst: highest mean (10166) and highest max (22861).
- Res-q 12D has the lowest mean (8196) in multi-seed, but it has velocity reversal (vx_min=-0.277).
- Res-α 12D and Smoothstep are nearly tied on mean (8706 vs 8658).
- Res-α 12D has the LOWEST MAX (13279) among all methods — best worst-case ceiling.
- The canonical seed=42 advantage for Res-α 12D (7951 vs 10121) was partly a seed artifact:
  seed=42 happened to give a favorable gait phase that benefited Res-α 12D specifically.
- In multi-seed, Res-α 12D's advantage shifts from "lowest average jerk" to
  "lowest worst-case jerk + no velocity reversal."

Important wording for claims:
- CANONICAL: "Res-α 12D achieves the lowest transition-window jerk (7951) in the canonical
  seed=42 evaluation, a 21% reduction vs Smoothstep (10121)."
- MULTI-SEED: "Across 60 transition windows, Res-α 12D achieves the lowest worst-case jerk
  (max=13279), 13% below Smoothstep (max=15340), while Res-q 12D achieves the lowest mean
  (8196) but produces velocity reversal (vx_min=-0.277)."
- Do NOT claim Res-α 12D has the lowest multi-seed mean jerk. It does not.
- Correct final claim: Res-α 12D provides the best combination of low worst-case jerk,
  no velocity reversal, and near-Smoothstep CoT across all evaluated gait-phase conditions.

2×2 ablation table:

Residual-α 4D:
- vx_mean +0.433
- vx_std 0.104
- vx_min +0.004
- tilt_max 0.186
- h_mean 0.406
- CoT 2.436
- jerk_TRANS 9775

Residual-q 4D:
- vx_mean +0.410
- vx_std 0.134
- vx_min -0.149
- tilt_max 0.192
- h_mean 0.405
- CoT 2.057
- jerk_TRANS 9006

Residual-α 12D:
- vx_mean +0.427
- vx_std 0.109
- vx_min +0.004
- tilt_max 0.190
- h_mean 0.407
- CoT 2.105
- jerk_TRANS 7951

Residual-q 12D:
- vx_mean +0.410
- vx_std 0.133
- vx_min -0.277
- tilt_max 0.187
- h_mean 0.413
- CoT 2.193
- jerk_TRANS 9969

Interpretation:
- α-space is safer because it preserves interpolation between two frozen valid policies.
- q-space can be energy-efficient or lower variance in some cases, but it can produce velocity reversal.
- 12D α gives the best jerk reduction.
- 4D α is simpler and already safe, but not best on jerk.
- Smoothstep remains a strong simple baseline with low CoT, but it still has velocity reversal and higher jerk than Residual-α 12D.
- Do not present other methods as simply bad. Present trade-offs.

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
SEED=42 VS MULTI-SEED EVALUATION: REQUIRED CLARIFICATION
====================================================================

Please make the role of seed=42 and the multi-seed experiment explicit and consistent throughout the README/report.

Current evaluation uses two levels:

1. Canonical single-seed evaluation:
   - seed = 42
   - used for detailed diagnostic comparison
   - used to generate plots such as:
     - discrete spike plot
     - transition zoom
     - gait diagram
     - body state plot
     - Δα / residual plot
   - used because all methods are compared under the same fixed transition sequence and initial condition
   - useful for visual interpretation and debugging

2. Per-gait-pair analysis (N=6):
   - 6 directed gait-pair transitions from the canonical seed=42 episode
   - used for per-pair jerk_TRANS mean, std, min, max
   - this shows which gait pairs are structurally hard vs easy and is the spread reported
     alongside the canonical mean
   - A 10-seed sweep was run but found no variation: env_cfg.seed does not change
     jerk_TRANS because transition timing is pinned and domain randomization does not
     affect gait-phase at switch time. The “10-seed” files are identical copies of
     the same canonical run. Do NOT describe this as a multi-seed robustness study.

Important wording:
Do NOT present seed=42 as the only evaluation.
Frame seed=42 as the canonical diagnostic run AND the source of all per-gait-pair data.

Correct framing:
“The evaluation uses the canonical seed=42 episode. Jerk_TRANS is computed per gait
pair (N=6), revealing the per-pair difficulty hierarchy. All quantitative claims
derive from this deterministic canonical episode.”

Use section labels:
- “Canonical Evaluation (seed=42)”
- “Per-Gait-Pair Analysis (N=6)”

Do NOT use:
- “Robustness Evaluation (10 seeds × 6 transitions = 60 windows)”
- “Across 60 transition windows”
- “60-window”

If reporting spread, phrase it as:
“Across 6 directed gait-pair transitions…”

Results are interpreted correctly when:
- Residual-α 12D has the lowest mean jerk_TRANS (N=6, mean=7951).
- Residual-α 12D does NOT have the tightest per-gait-pair spread.
- q-space variants may have tighter spread but suffer velocity reversal.
- The final claim combines mean jerk, vx_min, and CoT.