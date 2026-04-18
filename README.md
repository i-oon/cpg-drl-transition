# Transition-Aware CPG-RBF Locomotion Control with Continuous Gait Switching

**Course:** FRA 503 Deep Reinforcement Learning  
**Student:** Disthorn Suttawet (66340500019)  
**Robot:** Unitree B1 quadruped (12 DOF, ~50 kg)  
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0  
**References:**
- Thor et al. [CPG-RBFN framework](https://github.com/MathiasThor/CPG-RBFN-framework) — SO(2) oscillator + RBF + PI^BB
- LocoNets (IsaacLab-LocoNets) — direct encoding + tanh clamping + simplified reward

---

## Overview

This project implements a two-phase system for smooth gait transitions in a quadruped robot:

- **Phase 1** trains three base gaits using a CPG-RBF neural controller optimized with PI^BB (black-box gradient estimation).
- **Phase 2** trains a residual MLP that learns per-leg blending corrections for smooth transitions between gaits.

**Novel contribution:** Per-leg residual corrections (Δα per leg) on top of a phase-based baseline ramp, enabling the transition to adapt to B1's morphological asymmetry (front/rear kinematic differences) and CPG phase context.

---

## Architecture

### Phase 1: CPG-RBF Base Locomotion

```
SO(2) Oscillator (α=1.01, f=1.0 Hz)
    → normalize to unit circle (limit cycle ≈ 0.196, not 1.0)
    → RBF layer (20 Gaussian neurons, σ²=0.04)
    → W matrix × tanh → 12 joint position targets
    → add to default_joint_pos
```

**Two encoding approaches tested:**

| Encoding | W shape | Phase offsets | Params | Status |
|----------|---------|---------------|--------|--------|
| **Indirect** (Thor et al.) | (20,3) shared | Per-leg rotation | 60 | **Works — stable gaits** |
| Direct (LocoNets) | (20,12) per-joint | None | 240 | Failed — robot unstable |

**Current approach:** Indirect encoding with phase offsets. The shared W forces all legs to coordinate through the same trajectory — a structural constraint that PIBB needs for the heavy B1 (50 kg). Direct encoding works for LocoNets' 75mm gecko but not for B1.

### Phase 2: Residual Transition Learning

```
Baseline:  α_base = min(1.0, cycles_elapsed / 3.0)     ← linear ramp over 3s
Residual:  Δα = 0.2 × tanh(MLP(state))                  ← per-leg correction ∈ [-0.2, +0.2]
Final:     α_L = clip(α_base + Δα_L, 0, 1)              ← per leg L

Blending (at OUTPUT level, after tanh):
  output_current = tanh(RBF @ W_current)
  output_target  = tanh(RBF @ W_target)
  blended[L] = (1 - α_L) · output_current[L] + α_L · output_target[L]
```

MLP architecture: [44D → 128 → 128 → 4D], ReLU hidden, Tanh output × 0.2.

---

## B1 Robot Configuration

### Joint Axis Convention

| Joint | Axis | Default FL/FR/RL/RR | Role |
|-------|------|---------------------|------|
| `hip_joint` | Abduction | +0.1/-0.1/+0.1/-0.1 | Lateral balance (small) |
| `thigh_joint` | **Flexion** | +0.8/+0.8/+1.0/+1.0 | **Primary walking driver** |
| `calf_joint` | Knee bend | -1.5/-1.5/-1.5/-1.5 | Foot clearance |

Front thighs (+0.8) and rear thighs (+1.0) differ by 0.2 rad — root cause of front/rear duty asymmetry with indirect encoding.

### Actuator Settings

| Parameter | B1 (correct) | A1 (wrong, was default) |
|-----------|-------------|------------------------|
| Stiffness | **200.0** N·m/rad | 25.0 |
| Damping | **5.0** N·m·s/rad | 0.5 |
| Effort limit | **280.0** N·m | 23.7 |

---

## Phase 1 Encoding Experiments

Seven encoding/configuration combinations tested:

| # | Encoding | Params | Walk | Trot | Key finding |
|---|----------|--------|------|------|-------------|
| 1 | **Indirect + phase offsets** | 60 | 0.17 m/s ✓ | **0.21 m/s ✓** | Only encoding where trot works |
| 2 | Semi-indirect (no noise) | 120 | Unstable | Broken | Hip explodes to 54° without constraints |
| 3 | Semi-indirect (+ per-joint noise) | 120 | 0.23 m/s ✓ | Broken | Front/rear pairing breaks diagonal sync |
| 4 | Per-gait pairing | 120 | 0.16 m/s | 0.15 m/s (weak) | Diagonal pair ≠ front/rear pair |
| 5 | Per-leg + unified defaults (0.9) | 240 | 0.10 m/s (symmetric!) | Broken (inverted) | Changing defaults shifts asymmetry, doesn't fix |
| 6 | Per-leg + original defaults | 240 | 0.16 m/s | Broken (L/R asym) | No symmetry constraint → PIBB finds lateral bias |
| 7 | **Direct (LocoNets)** | 240 | 0.5-0.8 m/s (unstable) | Not tested | Robot lunges/falls, no gait pattern |

**Key findings:**
1. Only indirect encoding (shared W + phase offsets) produces a working trot
2. More parameters = more freedom for PIBB to find degenerate solutions
3. B1's front/rear default asymmetry (0.8 vs 1.0 thigh) causes 31.6% duty factor gap with indirect encoding
4. Direct encoding (LocoNets style) fails because B1 is too heavy for PIBB to discover stable timing without structural constraints
5. Per-joint noise scaling (hip 10× less) is essential for any encoding >60 params

---

## Phase 1 Best Results (Indirect Encoding)

| Metric | Walk | Trot | Steer |
|--------|------|------|-------|
| Forward velocity (m/s) | 0.17 | 0.21 | 0.22 |
| Height (m) | 0.46 | 0.47 | 0.45 |
| Tilt max | 0.008 | 0.005 | 0.008 |
| FL / FR / RL / RR duty % | 45/56/80/84 | 44/44/79/74 | 50/62/76/82 |
| Front/rear gap | 31.6% | 22.5% | 23.0% |

Rear legs planted ~80% vs front ~45% due to B1's asymmetric thigh defaults. Documented as a finding, not a showstopper for Phase 2 transitions.

---

## Progress

### Week 10 — Setup & Core Components ✅

| Task | Files |
|------|-------|
| CPG-RBF network (21/21 tests) | `networks/cpg_rbf.py` |
| Environment wrapper (44/44 tests) | `envs/unitree_b1_env.py` |
| Live environment test (7/7 checks) | `scripts/test_env.py` |
| CPG phase visualization | `scripts/visualize_cpg.py` |
| YAML configs | `configs/phase1_*.yaml` |

### Week 11 — Phase 1 Training (IN PROGRESS)

| Task | Status |
|------|--------|
| PI^BB optimizer (Thor et al. softmax) | Done |
| B1 actuator fix (stiffness 25→200) | Done |
| Reward iteration (11→6→7 terms) | Done |
| 7 encoding experiments | Done — indirect wins |
| LocoNets analysis (tanh, direct, simpler reward) | Done |
| Direct encoding attempt | Done — failed for B1 |
| **Next: revert to indirect, train task-based gaits** | Pending |

### Week 12-15 — Remaining

| Task | Status |
|------|--------|
| Final Phase 1 gaits (3 tasks) | Pending |
| Baseline validation (4 transition tests) | Pending |
| Phase 2 residual MLP training | Pending |
| Analysis and documentation | Pending |

---

## Setup

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

python -m pytest tests/ -v                          # unit tests
python scripts/test_env.py --headless                # live env test
python scripts/train_phase1_walk.py --headless       # train
python scripts/play_gait.py --gait walk              # play with gait diagram
python scripts/diagnose_cpg.py --headless            # debug W/joints
tensorboard --logdir logs/phase1                     # training curves
```

---

## Bugs Found & Fixed (15)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| RBF activations = 0 | Limit cycle ≈ 0.196, centers on unit circle | Normalize before RBF |
| Joints don't move | B1 stiffness=25 (A1 values for 12 kg) | stiffness=200, effort=280 |
| Robot collapses at W=0 | Targets were absolute, not offsets | `default_pos + offset` |
| Push-up gait | Hip assumed flexion (it's abduction) | Thigh-dominant W_init |
| W_init phase wrong | sin peaks mid-stance → backward | cos for thigh |
| Foot contact on calf | `.*_calf$` = leg bone, not foot | `.*_foot$` |
| Air time target hardcoded | 0.5s negative at 1.0 Hz | Dynamic: `0.35/freq` |
| Semi-indirect hip explosion | Isotropic noise grows hip to 54° | Per-joint: hip=0.1× |
| Trot diagonal sync lost | Front/rear pairing splits FL+RR | Only indirect preserves sync |
| Unified defaults inverts problem | 0.9 makes front planted instead | Revert to 0.8/1.0 |
| Per-leg L/R asymmetry | No symmetry constraint | Shared W constrains search |
| Velocity Gaussian too sharp | exp(-16×) → robot stands still | exp(-4×) or linear |
| Height penalty squared | 10 × 0.06² = negligible | Absolute error |
| LocoNets torque scale wrong | 2.5e-5 for gecko → -141K reward on B1 | 1e-7 for B1 mass |
| gait_match exploit | Steps in place scores +0.5/step | Remove from direct encoding |
