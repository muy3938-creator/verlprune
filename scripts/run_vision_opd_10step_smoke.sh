#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"

exec "${PROJECT_ROOT}/scripts/run_vision_pruning_experiment.sh" "$@"
