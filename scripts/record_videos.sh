#!/usr/bin/env bash
# Record seed=42 transition videos for all 10 methods:
#   - 2 baselines  (Discrete, Smoothstep)
#   - 4 Experiment A variants (2×2: Sched-α/Action-q × 4D/12D)
#   - 4 Experiment B variants (2×2: Sched-α/Action-q × 4D/12D, contact-phase obs)
#
# Videos saved to logs/phase2_v3/videos/<method_name>/
# Each video: 2500 steps = 50 s, all 6 directed gait-pair transitions:
#   trot→bound, bound→pace, pace→trot, trot→pace, pace→bound, bound→trot
#
# Note on lag: Isaac Sim camera rendering runs ~0.1× real-time — each video takes
# ~10-15 min to generate. The resulting video plays back at correct speed.
#
# Usage:
#   conda activate env_isaaclab
#   cd ~/cpg-drl-transition
#   bash scripts/record_videos.sh 2>&1 | tee /tmp/record_videos.log
#
# To force re-record an existing video set FORCE=1:
#   FORCE=1 bash scripts/record_videos.sh

set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
PLAY="$BASE/scripts/play_b1_phase2.py"
OUT="$BASE/logs/phase2_v3/videos"

GAIT_PAIRS="trot,bound,pace,trot,pace,bound"
STEPS=2500        # canonical — same as playback_v3.sh and seed experiment
SWITCH_S=8.0      # canonical — same as playback_v3.sh and seed experiment
SEED=42
DUR=3.0           # pin transition duration for fair comparison

record() {
    local name="$1"
    local extra_args="$2"
    local vid_dir="$OUT/$name"

    # Skip if video already exists (any mp4 in the dir)
    if [[ "${FORCE:-0}" != "1" ]] && ls "$vid_dir"/*.mp4 2>/dev/null | grep -q .; then
        echo "  [skip] $name — video already exists"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  $name"
    echo "================================================================"

    python -u "$PLAY" \
        $extra_args \
        --num_envs 1 \
        --steps "$STEPS" \
        --seed "$SEED" \
        --gait_pairs "$GAIT_PAIRS" \
        --switch_interval_s "$SWITCH_S" \
        --transition_duration_s "$DUR" \
        --video "$vid_dir" \
        --headless

    echo "  Saved → $vid_dir/"
}

# ── Baselines ────────────────────────────────────────────────────────────────

record "discrete" \
    "--baseline discrete"

record "smoothstep" \
    "--baseline smoothstep_ramp"

# ── Experiment A — Direct Jerk Optimization ──────────────────────────────────

record "expA_sched_alpha_4d" \
    "--task Isaac-B1-Phase2-Transition-v0 \
     --checkpoint $BASE/logs/phase2/residual_alpha_4d_sp05_jw2/model_final.pt"

record "expA_sched_alpha_12d" \
    "--task Isaac-B1-Phase2-Alpha12D-v0 \
     --checkpoint $BASE/logs/phase2/residual_alpha_12d/model_final.pt"

record "expA_action_q_4d" \
    "--task Isaac-B1-Phase2-Joint4D-v0 \
     --checkpoint $BASE/logs/phase2/residual_q_4d_sp05_jw2/model_final.pt"

record "expA_action_q_12d" \
    "--task Isaac-B1-Phase2-ActionSpace-v0 \
     --checkpoint $BASE/logs/phase2/residual_q_12d_sp05_jw2/model_final.pt"

# ── Experiment B — Velocity Safety, contact-phase obs (V3) ───────────────────

record "expB_sched_alpha_4d" \
    "--task Isaac-B1-Phase2-V3-Alpha4D-v0 \
     --checkpoint $BASE/logs/phase2_new_approach/schedule_residual_4d_v3/model_final.pt"

record "expB_sched_alpha_12d" \
    "--task Isaac-B1-Phase2-V3-Alpha12D-v0 \
     --checkpoint $BASE/logs/phase2_new_approach/schedule_residual_12d_v3/model_final.pt"

record "expB_action_q_4d" \
    "--task Isaac-B1-Phase2-V3-Joint4D-v0 \
     --checkpoint $BASE/logs/phase2_new_approach/action_residual_4d_v3/model_final.pt"

record "expB_action_q_12d" \
    "--task Isaac-B1-Phase2-V3-Joint12D-v0 \
     --checkpoint $BASE/logs/phase2_new_approach/action_residual_12d_v3/model_final.pt"

echo ""
echo "================================================================"
echo "  ALL DONE — videos in $OUT/"
echo "================================================================"
