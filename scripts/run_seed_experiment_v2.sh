#!/usr/bin/env bash
# Run 10-seed playback for all methods with --randomize_start.
# v2: uses corrected checkpoints (sp05_jw2 retrained variants, sigmoid bug fixed).
#
# Output goes to logs/phase2_seed_experiment_v2/ to preserve old data.
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/run_seed_experiment_v2.sh 2>&1 | tee /tmp/seed_experiment_v2.log

set -euo pipefail

GAIT_PAIRS="trot,bound,pace,trot,pace,bound"
STEPS=2500
SWITCH_S=8.0
ENVS=16

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PLAY="$BASE/scripts/play_b1_phase2.py"
OUT="$BASE/logs/phase2_seed_experiment_v2"

run_seed() {
    local folder="$1"
    local seed="$2"
    local extra_args="$3"
    local outfile="$OUT/$folder/playback_s${seed}.csv"

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

echo ""
echo "================================================================"
echo "  Discrete Switch  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "discrete" "$seed" "--baseline discrete"
done

echo ""
echo "================================================================"
echo "  Linear Ramp  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "linear_ramp" "$seed" "--baseline linear_ramp"
done

echo ""
echo "================================================================"
echo "  Smoothstep Ramp  (baseline)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "smoothstep" "$seed" "--baseline smoothstep_ramp"
done

echo ""
echo "================================================================"
echo "  Smoothstep-q  (passive q-space baseline, Δq≡0)"
echo "  Identical smoothstep schedule to smoothstep_ramp but uses"
echo "  residual_mode=joint code path — confirms Smoothstep-α ≡ Smoothstep-q"
echo "  and establishes q-space starting point for Residual-q comparison."
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "smoothstep_q" "$seed" \
        "--task Isaac-B1-Phase2-Joint4D-v0 --baseline smoothstep_q"
done

echo ""
echo "================================================================"
echo "  Residual-α 4D  (sp05_jw2)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "residual_alpha_4d_sp05_jw2" "$seed" \
        "--task Isaac-B1-Phase2-Transition-v0 --checkpoint $BASE/logs/phase2/residual_alpha_4d_sp05_jw2/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-q 4D  (sp05_jw2)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "residual_q_4d_sp05_jw2" "$seed" \
        "--task Isaac-B1-Phase2-Joint4D-v0 --checkpoint $BASE/logs/phase2/residual_q_4d_sp05_jw2/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-α 12D  (original checkpoint)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "residual_alpha_12d" "$seed" \
        "--task Isaac-B1-Phase2-Alpha12D-v0 --checkpoint $BASE/logs/phase2/residual_alpha_12d/model_final.pt"
done

echo ""
echo "================================================================"
echo "  Residual-q 12D  (sp05_jw2)"
echo "================================================================"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    run_seed "residual_q_12d_sp05_jw2" "$seed" \
        "--task Isaac-B1-Phase2-ActionSpace-v0 --checkpoint $BASE/logs/phase2/residual_q_12d_sp05_jw2/model_final.pt"
done

echo ""
echo "================================================================"
echo "  ALL DONE — analyze with:"
echo "  python scripts/analyze_seed_experiment.py --source seeds \\"
echo "    --seed_dir logs/phase2_seed_experiment_v2 --mode all \\"
echo "    --out logs/phase2_seed_experiment_v2/results_all.png"
echo "================================================================"
