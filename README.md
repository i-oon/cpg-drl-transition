# CPG-RBF Locomotion — Unitree B1

**Course:** FRA 503 — Deep Reinforcement Learning  
**Student:** Disthorn Suttawet (66340500019)  
**Robot:** Unitree B1 quadruped (12 DOF)  
**Simulator:** Isaac Lab 0.36.3 / Isaac Sim 4.5.0

---

## Current Status

Replicating the online-locomotion-rl (Thor et al. / Fernandez 2023) CPG-RBF + PIBB approach on B1.

| Run | Iters | Envs | R_best | Notes |
|---|---:|---:|---:|---|
| Baseline (all bugs) | 2000 | 256 | — | vx = 0.091 m/s, oscillatory lunge |
| Fixed (this repo) | 500 | 64 | 282.8 | Trot diagonals emerging, falls at ~450 steps |

Trot diagonal pattern (FL+RR vs FR+RL) confirmed in playback at 150 steps. Robot survives ~8s before losing balance. Clean-W eval still at vx ≈ 0.03–0.08 m/s — training ongoing.

---

## Architecture

### CPG-RBF Network

```
SO(2) oscillator → RBF layer (H=20) → W (20×3) → joint offsets (4 legs)
```

- **Oscillator:** SO(2) with α=1.01, φ = 2π × freq × dt rad/step
- **RBF centers:** sampled from actual oscillator trajectory at evenly-spaced timesteps (pre-computed lookup table KENNE)
- **Encoding:** indirect — shared W (20×3) for all 4 legs; per-leg timing from phase offset
- **Two-state encoding:** W evaluated at `phase_idx` (current) and `phase_idx + period//2` (half-period delayed); outputs interleaved per joint type

**Trot interleaving** (`cpg_reversed=True`):
```
FL: [now_hip, del_thigh, now_calf]   →  FL + RR in phase (diagonal pair A)
FR: [del_hip, now_thigh, del_calf]   →  FR + RL in phase (diagonal pair B)
RL: [del_hip, now_thigh, del_calf]   (reversed second pair)
RR: [now_hip, del_thigh, now_calf]
```

- `cpg_reversed=False` → pace (lateral pairs FL+RL, FR+RR)
- `cpg_reversed=True` → trot (diagonal pairs FL+RR, FR+RL)

**Action application:**
```python
joint_target = default_dof_pos + action_scale_vec * tanh(KENNE @ W)
# hip:        action_scale = 0.04
# thigh/calf: action_scale = 0.20
```

### Key bug fixes vs original implementation

| Bug | Symptom | Fix |
|---|---|---|
| 4-phase offsets → 4 independent relationships | Harder PIBB optimisation | Two-state (current + delayed) encoding |
| W_norm grew to 5+ → tanh saturation → motor_now ≈ motor_del | All legs same phase (hopping) | Hard W_norm cap at 2.0 in PIBB update |
| effort_limit=23.7 N·m vs gravity torque 19 N·m | Robot collapses (< 5 N·m dynamic budget) | effort_limit=100 N·m for simulation |
| Thigh asymmetry 0.8/1.0 front/rear | Single shared W can't satisfy both pairs | Symmetric thighs: 0.85 all four |
| Saving W from noisy rollouts, not clean eval | Training R ≫ playback R | Clean-W eval every 25 iters → `W_trot_eval_best.npy` |

---

## B1 Robot Config

```python
# DCMotorCfg — empirically validated for Isaac Lab B1 (SETUP.md §"Stock needs three overrides")
# online-locomotion-rl's Kp=1150 is software PD in FORCE mode (no effort limit) — not equivalent
stiffness   = 400.0     # N·m/rad
damping     = 10.0      # N·m·s/rad
effort_limit = 100.0    # N·m (raised from spec 23.7 to give CPG dynamic headroom)

# Default joint positions — symmetric thighs (real robot, eliminates indirect-W breakage)
FL/RL/FR/RR hip:   ±0.1 rad
all thighs:         0.85 rad
all calfs:         −1.56 rad
spawn z:            0.50 m
```

online-locomotion-rl reference values (config_b1.json, updated to match real robot):
```json
"FL/FR/RL/RR_thigh_joint": 0.85
"FL/FR/RL/RR_calf_joint": -1.56
```

---

## PIBB Config (phase1_trot.yaml)

| Parameter | Value | Note |
|---|---:|---|
| Parallel envs | 64 | 15 was too few for 60-dim W gradient |
| Episode length | 400 steps (8 s) | 3 s caused sprint-then-fall overfit |
| Max iterations | 500 | Still growing at iter 500, increase to 1000+ |
| Exploration noise σ | 0.19 | √0.036, matching online-locomotion-rl |
| Noise decay | 0.998/iter | 0.992 killed exploration by iter 200 |
| Noise boost | 1.5× (iter 1) | |
| Temperature h | 10 | |
| W_norm cap | 2.0 | Hard clip after each update — prevents tanh saturation |
| Eval interval | 25 | Clean W evaluated every 25 iters, best saved separately |
| Init mode | cosine | thigh: 0.20×cos, calf: 0.16×sin |

---

## Reward Function

```python
# Per step, accumulated by PIBB over episode
reward = (1.5  × vx                           # forward progress (signed)
        - 1.0  × |vy|                          # lateral drift
        - 1.0  × stability × 0.5 × fwd        # stability gated by forward speed
        - 1.0  × contact × 0.5 × fwd          # thigh/hip contact gated by fwd
        - 2.0  × (grav_x² + grav_y²))         # unconditional tilt — prevents drift

# stability = 1.3×|h−h_nom| + 1.3×grav_x + 1.1×grav_y + 1.3×grav_z
# h_nominal = 0.55 m
# fwd       = clamp(vx, min=0)
```

Design matches online-locomotion-rl (Rewards.py) with one addition: the unconditional tilt term (`-2.0 × tilt`) that prevents the slow lateral drift the open-loop CPG can't self-correct.

---

## Setup (new machine)

### 1. B1 USD

Follow SETUP.md to convert the B1 URDF to USD and register `UNITREE_B1_CFG` in Isaac Lab.

Quick version:
```bash
# Fix mesh paths in the project URDF
MESHDIR="$(pwd)/online-locomotion-rl/models/b1/urdf/meshes"
sed "s|meshes/|file://${MESHDIR}/|g" \
    online-locomotion-rl/models/b1/urdf/b1.urdf \
    > online-locomotion-rl/models/b1/urdf/b1_fixed.urdf

# Convert to USD
mkdir -p ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1
conda run -n env_isaaclab python ~/IsaacLab/scripts/tools/convert_urdf.py \
    online-locomotion-rl/models/b1/urdf/b1_fixed.urdf \
    ~/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/B1/b1.usd \
    --joint-stiffness 25.0 --joint-damping 0.5 --headless
```

Then add `UNITREE_B1_CFG` to `~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py` — see the block at the bottom of that file (already added on the source machine).

### 2. Environment

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition
python -m pytest tests/ -q    # should be 44/44
```

---

## Training

```bash
conda activate env_isaaclab
cd ~/cpg-drl-transition

# Kill any zombie Isaac Sim processes first
pgrep -f "python.*train\|kit" | xargs -r kill -9

python scripts/train_phase1_trot.py --headless
```

Outputs:
- `weights/W_trot.npy` — best noisy-rollout W
- `weights/W_trot_eval_best.npy` — best clean-W (use this for playback)
- `weights/W_trot_iter{N}.npy` — checkpoint every 50 iters

---

## Playback

```bash
# Use eval_best weights for honest evaluation
cp weights/W_trot_eval_best.npy weights/W_trot.npy
python scripts/play_gait.py --gait trot --num_envs 1 --steps 500 --headless
```

Output: `logs/gait_diagram_trot.png` — footfall pattern, duty factors, phase timing.

For GUI playback (Isaac Sim rendering bug on some machines — check SETUP.md):
```bash
python scripts/play_gait.py --gait trot --num_envs 1 --steps 500
```

---

## Known Issues / Next Steps

**Open issues:**
1. Robot falls at ~450 steps (8–9 s) — CPG is open-loop, can't correct lateral drift
2. Duty factors asymmetric (FL≠RR, FR≠RL) — gait not yet fully converged
3. Training R ≫ eval R gap — partly fixed by clean-W eval, but still present at low iteration counts

**Next steps:**
1. Run 1000+ iterations — sigma still alive at iter 500 (0.07), reward still climbing
2. Confirm trot diagonal pairs stabilise in gait diagram (FL+RR vs FR+RL footfall)
3. Once stable trot at ≥ 0.15 m/s: tune CPG frequency and/or action scale

**Architecture ceiling:** open-loop CPG cannot self-correct. If gait doesn't stabilise past ~10 s with more iterations, the next step is CPG + a small feedback layer (phase-modulated W based on contact state), not further PIBB tuning.

---

## File Structure

```
cpg-drl-transition/
├── envs/
│   └── unitree_b1_env.py          # B1 env + CPG-RBF (Phase 1)
├── networks/
│   └── cpg_rbf.py                 # Standalone CPGRBFNetwork (reference, not used in training)
├── algorithms/
│   └── pibb_trainer.py            # PIBB optimiser with W_norm cap + clean eval
├── scripts/
│   ├── train_phase1_trot.py       # Train trot W via PIBB
│   ├── train_phase1_{pace,bound,walk,steer}.py
│   ├── play_gait.py               # Playback + gait diagram
│   ├── visualize_cpg.py
│   └── diagnose_cpg.py
├── configs/
│   └── phase1_{trot,pace,bound,walk,walk_fixed,steer}.yaml
├── weights/
│   ├── W_trot.npy                 # best noisy-rollout W
│   └── W_trot_eval_best.npy       # best clean-W (use for playback)
├── online-locomotion-rl/          # reference implementation (Thor/Fernandez)
│   ├── configs/config_b1.json     # B1 joint defaults (updated to 0.85/−1.56)
│   └── modules/cpgrbfn2.py        # CPGRBFN class — two-state encoding source
├── tests/
│   ├── test_cpg_rbf.py
│   └── test_environment.py
├── SETUP.md                       # B1 USD conversion + Isaac Lab registration
└── README.md                      # this file
```
