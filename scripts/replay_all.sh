#!/usr/bin/env bash
# Replay all Phase 2 runs with the full 6-pair gait sequence.
#
# Produces:
#   logs/phase2/<run>/playback.csv   — per-step time-series (new alpha_base column)
#   logs/phase2/<run>/diag/          — diagnostic plots (purple transition zone)
#
# Run order: v10 → baselines → ablations → seeds
# Each run starts its own Isaac Sim instance sequentially (GPU can't run two at once).
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/replay_all.sh 2>&1 | tee /tmp/replay_all.log

set -euo pipefail

GAIT_PAIRS="trot,bound,pace,trot,pace,bound"   # all 6 directed pairs
STEPS=2500          # 2500 × 0.02 s = 50 s  →  6 segments × 8 s + 2 s padding
SWITCH_S=8.0        # seconds per segment
ENVS=16
SEED=42

BASE="$(cd "$(dirname "$0")/.." && pwd)"        # repo root
PLAY="$BASE/scripts/play_b1_phase2.py"

run() {
    local label="$1"; shift
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    PYTHONUNBUFFERED=1 python -u "$PLAY" \
        --num_envs "$ENVS" \
        --steps "$STEPS" \
        --switch_interval_s "$SWITCH_S" \
        --gait_pairs "$GAIT_PAIRS" \
        --seed "$SEED" \
        --headless \
        "$@"
}

# ── 1. Main result ──────────────────────────────────────────────────────────
run "v10  (main result)" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --checkpoint "$BASE/logs/phase2/phase2_v10/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/phase2_v10/playback.csv" \
    --save_plots  "$BASE/logs/phase2/phase2_v10/diag"

# ── 2. Hand-designed baselines (no checkpoint needed) ──────────────────────
run "baseline: smoothstep_ramp" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --baseline smoothstep_ramp \
    --save_csv    "$BASE/logs/phase2/baselines/smoothstep_ramp/playback.csv" \
    --save_plots  "$BASE/logs/phase2/baselines/smoothstep_ramp/diag"

run "baseline: linear_ramp" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --baseline linear_ramp \
    --save_csv    "$BASE/logs/phase2/baselines/linear_ramp/playback.csv" \
    --save_plots  "$BASE/logs/phase2/baselines/linear_ramp/diag"

run "baseline: discrete" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --baseline discrete \
    --save_csv    "$BASE/logs/phase2/baselines/discrete/playback.csv" \
    --save_plots  "$BASE/logs/phase2/baselines/discrete/diag"

# ── 3. Learned E2E baseline ─────────────────────────────────────────────────
run "baseline: e2e (phase2_e2e_v2)" \
    --task Isaac-B1-Phase2-E2E-v0 \
    --checkpoint "$BASE/logs/phase2/phase2_e2e_v2/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/baselines/e2e/playback.csv" \
    --save_plots  "$BASE/logs/phase2/baselines/e2e/diag"

# ── 4. Ablations ────────────────────────────────────────────────────────────
run "ablation: residual1d (scalar Δα)" \
    --task Isaac-B1-Phase2-Residual1D-v0 \
    --checkpoint "$BASE/logs/phase2/residual1d_v1/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/residual1d_v1/playback.csv" \
    --save_plots  "$BASE/logs/phase2/residual1d_v1/diag"

run "ablation: e2e_rate (integrated α)" \
    --task Isaac-B1-Phase2-E2E-Rate-v0 \
    --checkpoint "$BASE/logs/phase2/e2e_rate_v1/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/e2e_rate_v1/playback.csv" \
    --save_plots  "$BASE/logs/phase2/e2e_rate_v1/diag"

# ── 5. Seed robustness ──────────────────────────────────────────────────────
run "seed robustness: seed0" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --checkpoint "$BASE/logs/phase2/phase2_seed0/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/phase2_seed0/playback.csv" \
    --save_plots  "$BASE/logs/phase2/phase2_seed0/diag"

run "seed robustness: seed1" \
    --task Isaac-B1-Phase2-Transition-v0 \
    --checkpoint "$BASE/logs/phase2/phase2_seed1/model_final.pt" \
    --save_csv    "$BASE/logs/phase2/phase2_seed1/playback.csv" \
    --save_plots  "$BASE/logs/phase2/phase2_seed1/diag"

echo ""
echo "================================================================"
echo "  ALL DONE — CSVs and plots regenerated for all 9 runs"
echo "================================================================"
