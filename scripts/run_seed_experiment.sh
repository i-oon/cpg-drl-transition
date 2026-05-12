#!/usr/bin/env bash
# Run 10-seed playback for all methods with --randomize_start.
#
# --randomize_start samples transition_start_s from the training range
# [1.5, 3.5] using the seeded numpy RNG, so different seeds produce
# different gait phases at switch time. The same seed is used for all
# methods, keeping cross-method comparisons fair.
#
# Produces:
#   logs/phase2_seed_experiment/{method}/playback_s{0..9}.csv
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/run_seed_experiment.sh 2>&1 | tee /tmp/seed_experiment.log

set -euo pipefail

GAIT_PAIRS="trot,bound,pace,trot,pace,bound"
STEPS=2500
SWITCH_S=8.0
ENVS=16

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PLAY="$BASE/scripts/play_b1_phase2.py"
OUT="$BASE/logs/phase2_seed_experiment"

run_seed() {
    local task="$1"
    local ckpt_or_baseline="$2"
    local folder="$3"
    local seed="$4"
    local outfile="$OUT/$folder/playback_s${seed}.csv"
    local extra_args="$5"

    mkdir -p "$OUT/$folder"
    echo "  → $folder  seed=$seed"
    PYTHONUNBUFFERED=1 python -u "$PLAY" \
        $extra_args \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --switch_interval_s "$SWITCH_S" \
        --gait_pairs "$GAIT_PAIRS" \
        --seed "$seed" \
        --randomize_start \
        --headless \
        --save_csv "$outfile"
}

# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Discrete Switch  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "discrete" "$seed" "--baseline discrete"
done

echo ""
echo "================================================================"
echo "  Smoothstep Ramp  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "smoothstep" "$seed" "--baseline smoothstep_ramp"
done

echo ""
echo "================================================================"
echo "  Linear Ramp  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "linear_ramp" "$seed" "--baseline linear_ramp"
done

echo ""
echo "================================================================"
echo "  Residual-α 4D  (v10)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "Isaac-B1-Phase2-v0" \
             "$BASE/logs/phase2/phase2_v10/model_final.pt" \
             "v10" "$seed" \
             "--task Isaac-B1-Phase2-v0 --checkpoint $BASE/logs/phase2/phase2_v10/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-q 4D  (Isaac-B1-Phase2-Joint4D-v0)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "residual_q_4d" "$seed" \
        "--task Isaac-B1-Phase2-Joint4D-v0 --checkpoint $BASE/logs/phase2/residual_q_4d/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-q 12D  (Isaac-B1-Phase2-ActionSpace-v0)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "action_space" "$seed" \
        "--task Isaac-B1-Phase2-ActionSpace-v0 --checkpoint $BASE/logs/phase2/residual_q_12d/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-α 12D  (Isaac-B1-Phase2-Alpha12D-v0)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "" "" "residual_alpha_12d" "$seed" \
        "--task Isaac-B1-Phase2-Alpha12D-v0 --checkpoint $BASE/logs/phase2/residual_alpha_12d/model_final.pt"
done

echo ""
echo "================================================================"
echo "  ALL DONE — CSVs written to logs/phase2_seed_experiment/"
echo "  Run: python scripts/analyze_seed_experiment.py --mode all"
echo "================================================================"
