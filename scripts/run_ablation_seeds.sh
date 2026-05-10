#!/usr/bin/env bash
# Generate 10-seed playback CSVs for the 2×2 ablation new variants.
#
# Produces:
#   logs/phase2_seed_experiment/residual_q_4d/playback_s{0..9}.csv
#   logs/phase2_seed_experiment/residual_alpha_12d/playback_s{0..9}.csv
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/run_ablation_seeds.sh 2>&1 | tee /tmp/ablation_seeds.log

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
    local ckpt="$2"
    local folder="$3"
    local seed="$4"
    local outfile="$OUT/$folder/playback_s${seed}.csv"

    mkdir -p "$OUT/$folder"
    echo "  → $folder  seed=$seed"
    PYTHONUNBUFFERED=1 python -u "$PLAY" \
        --task "$task" \
        --checkpoint "$ckpt" \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --switch_interval_s "$SWITCH_S" \
        --gait_pairs "$GAIT_PAIRS" \
        --seed "$seed" \
        --headless \
        --save_csv "$outfile"
}

echo ""
echo "================================================================"
echo "  Residual-q 4D  (Isaac-B1-Phase2-Joint4D-v0)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "Isaac-B1-Phase2-Joint4D-v0" \
             "$BASE/logs/phase2/residual_q_4d/model_final.pt" \
             "residual_q_4d" \
             "$seed"
done

echo ""
echo "================================================================"
echo "  Residual-α 12D  (Isaac-B1-Phase2-Alpha12D-v0)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "Isaac-B1-Phase2-Alpha12D-v0" \
             "$BASE/logs/phase2/residual_alpha_12d/model_final.pt" \
             "residual_alpha_12d" \
             "$seed"
done

echo ""
echo "================================================================"
echo "  ALL DONE — CSVs written to logs/phase2_seed_experiment/"
echo "================================================================"
