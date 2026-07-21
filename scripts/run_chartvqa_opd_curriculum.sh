#!/usr/bin/env bash

set -euo pipefail

# One-process convenience launcher for a smooth training-step budget schedule.
# It does not restart the trainer or load intermediate checkpoints.  By default
# the student keeps 50% initially, reaches 10% at 80% of training, and ends at
# 5%; all rollout requests and actor replay use the same global-step schedule.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
CURRICULUM_START_RATIO="${CURRICULUM_START_RATIO:-0.50}"
CURRICULUM_MID_RATIO="${CURRICULUM_MID_RATIO:-0.10}"
CURRICULUM_END_RATIO="${CURRICULUM_END_RATIO:-0.05}"
CURRICULUM_MID_FRACTION="${CURRICULUM_MID_FRACTION:-0.80}"

if [[ -z "${KEEP_RATIO_SCHEDULE:-}" ]]; then
    KEEP_RATIO_SCHEDULE="$(python3 - "${TOTAL_TRAINING_STEPS}" "${CURRICULUM_START_RATIO}" "${CURRICULUM_MID_RATIO}" "${CURRICULUM_END_RATIO}" "${CURRICULUM_MID_FRACTION}" <<'PY'
import json
import sys

total_steps = int(sys.argv[1])
start_ratio, mid_ratio, end_ratio, mid_fraction = map(float, sys.argv[2:])
if total_steps < 3:
    raise SystemExit("the three-milestone curriculum requires at least 3 training steps")
if not 0.0 < mid_fraction < 1.0:
    raise SystemExit("CURRICULUM_MID_FRACTION must be in (0, 1)")
if not all(0.0 < value < 1.0 for value in (start_ratio, mid_ratio, end_ratio)):
    raise SystemExit("curriculum ratios must be in (0, 1)")
mid_step = min(total_steps - 1, max(2, round(total_steps * mid_fraction)))
print(json.dumps({"milestones": [[1, start_ratio], [mid_step, mid_ratio], [total_steps, end_ratio]]}, separators=(",", ":")))
PY
)"
fi

export KEEP_RATIO_SCHEDULE
exec bash "${PROJECT_ROOT}/scripts/run_chartvqa_opd_training.sh"
