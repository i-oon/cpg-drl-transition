#!/usr/bin/env bash
# Run canonical seed=42 playback for all 4 V3 variants.
# Saves CSVs to logs/phase2_new_approach/<run>/playback_seed42.csv
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/playback_v3.sh 2>&1 | tee /tmp/playback_v3.log

set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PLAY="$BASE/scripts/play_b1_phase2.py"
LOGDIR="$BASE/logs/phase2_new_approach"

run_play() {
    local task="$1"
    local run_name="$2"
    local ckpt="$LOGDIR/$run_name/model_final.pt"
    local out_csv="$LOGDIR/$run_name/playback_seed42.csv"

    if [[ -f "$out_csv" && "${FORCE:-0}" != "1" ]]; then
        echo "  [skip] $run_name — playback_seed42.csv already exists"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  task     : $task"
    echo "  run_name : $run_name"
    echo "================================================================"

    python -u "$PLAY" \
        --task "$task" \
        --checkpoint "$ckpt" \
        --num_envs 1 \
        --steps 2500 \
        --seed 42 \
        --gait_pairs trot,bound,pace,trot,pace,bound \
        --switch_interval_s 8.0 \
        --transition_duration_s 3.0 \
        --save_csv "$out_csv" \
        --headless

    echo "  Saved → $out_csv"
}

run_play "Isaac-B1-Phase2-V3-Alpha4D-v0"  "schedule_residual_4d_v3"
run_play "Isaac-B1-Phase2-V3-Alpha12D-v0" "schedule_residual_12d_v3"
run_play "Isaac-B1-Phase2-V3-Joint4D-v0"  "action_residual_4d_v3"
run_play "Isaac-B1-Phase2-V3-Joint12D-v0" "action_residual_12d_v3"

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
