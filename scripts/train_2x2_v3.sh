#!/usr/bin/env bash
# Train all 4 V3 variants (V2 + foot contact 4D) sequentially.
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/train_2x2_v3.sh 2>&1 | tee /tmp/train_v3.log

set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$BASE/scripts/train_b1_phase2.py"
ITERS=3000

run_train() {
    local task="$1"
    local run_name="$2"
    local dest="$BASE/logs/phase2_new_approach/${run_name}"

    if [[ -f "$dest/model_final.pt" && "${FORCE:-0}" != "1" ]]; then
        echo "  [skip] $run_name — model_final.pt already exists"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  task     : $task"
    echo "  run_name : $run_name"
    echo "  dest     : $dest"
    echo "  iters    : $ITERS"
    echo "================================================================"

    python -u "$TRAIN" \
        --task "$task" \
        --run_name "$run_name" \
        --max_iterations "$ITERS" \
        --headless

    echo "  Saved → $dest/model_final.pt"
}

run_train "Isaac-B1-Phase2-V3-Alpha4D-v0"  "schedule_residual_4d_v3"
run_train "Isaac-B1-Phase2-V3-Alpha12D-v0" "schedule_residual_12d_v3"
run_train "Isaac-B1-Phase2-V3-Joint4D-v0"  "action_residual_4d_v3"
run_train "Isaac-B1-Phase2-V3-Joint12D-v0" "action_residual_12d_v3"

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
