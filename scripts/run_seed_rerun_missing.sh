#!/usr/bin/env bash
# Re-run only Discrete and Res-α 4D (v10) seed experiments.
#
# Root causes of stale/identical seed files (all fixed in play_b1_phase2.py):
#
#   1. Discrete: _transition_start_steps was hardcoded to int(2.0/control_dt),
#      so --randomize_start had no effect on when the α jump fired.
#      Fixed: _transition_start_steps = int(_current_hold_s / control_dt).
#
#   2. np.random global RNG: IsaacLab resets numpy's global RNG during env init,
#      overriding np.random.seed(args.seed) set at the top of the script. This
#      caused _sample_transition_start() to always return the same value (2.0).
#      Fixed: use an isolated np.random.default_rng(args.seed) Generator that is
#      immune to IsaacLab's internal np.random.seed() calls.
#
#   3. Res-α 4D (v10): run_seed_experiment.sh passed --task Isaac-B1-Phase2-v0
#      which is not registered → play script likely raised an error or used the
#      wrong env. Fixed by omitting --task (defaults to Isaac-B1-Phase2-Transition-v0).
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/run_seed_rerun_missing.sh 2>&1 | tee /tmp/rerun_missing.log

set -euo pipefail

GAIT_PAIRS="trot,bound,pace,trot,pace,bound"
STEPS=2500
SWITCH_S=8.0
ENVS=16

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PLAY="$BASE/scripts/play_b1_phase2.py"
OUT="$BASE/logs/phase2_seed_experiment"

echo ""
echo "================================================================"
echo "  Discrete Switch  (baseline — hardcoded hold fix applied)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "  → discrete  seed=$seed"
    mkdir -p "$OUT/discrete"
    PYTHONUNBUFFERED=1 python -u "$PLAY" \
        --baseline discrete \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --switch_interval_s "$SWITCH_S" \
        --gait_pairs "$GAIT_PAIRS" \
        --seed "$seed" \
        --randomize_start \
        --headless \
        --save_csv "$OUT/discrete/playback_s${seed}.csv"
done

echo ""
echo "================================================================"
echo "  Res-α 4D  (v10 — default task, no explicit --task arg)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "  → v10  seed=$seed"
    mkdir -p "$OUT/v10"
    PYTHONUNBUFFERED=1 python -u "$PLAY" \
        --checkpoint "$BASE/logs/phase2/phase2_v10/model_final.pt" \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --switch_interval_s "$SWITCH_S" \
        --gait_pairs "$GAIT_PAIRS" \
        --seed "$seed" \
        --randomize_start \
        --headless \
        --save_csv "$OUT/v10/playback_s${seed}.csv"
done

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "  Run: python scripts/analyze_seed_experiment.py --mode all --source seeds"
echo "================================================================"
