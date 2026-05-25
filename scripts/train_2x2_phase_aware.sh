#!/usr/bin/env bash
# Train all 4 phase-aware 2×2 ablation variants sequentially.
# Checkpoints save directly to logs/phase2/<run_name>/ via --run_name.
# Skips any variant whose model_final.pt already exists (override with FORCE=1).
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/train_2x2_phase_aware.sh 2>&1 | tee /tmp/train_2x2_phase_aware.log

set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$BASE/scripts/train_b1_phase2.py"
ITERS=3000

run_train() {
    local task="$1"
    local run_name="$2"
    local dest="$BASE/logs/phase2/${run_name}"

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

run_train "Isaac-B1-Phase2-Alpha12D-PhaseAware-v0"    "residual_alpha_12d_phase_aware"
run_train "Isaac-B1-Phase2-Alpha4D-PhaseAware-v0"     "residual_alpha_4d_phase_aware"
run_train "Isaac-B1-Phase2-Joint4D-PhaseAware-v0"     "residual_q_4d_phase_aware"
run_train "Isaac-B1-Phase2-ActionSpace-PhaseAware-v0" "residual_q_12d_phase_aware"

echo ""
echo "================================================================"
echo "  ALL DONE"
echo "================================================================"
