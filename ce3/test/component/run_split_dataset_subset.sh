#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DISTILLATION_INPUT="${DISTILLATION_INPUT:-ce3/test/output/llm_distillation_subset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/test/output/splits}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.20}"
TEST_RATIO="${TEST_RATIO:-0.20}"
PREFIX_MODE="${PREFIX_MODE:-bracket}"
PAIR_GENERATION_MODE="${PAIR_GENERATION_MODE:-controlled}"
PAIR_MIN_MARGIN="${PAIR_MIN_MARGIN:-0.01}"
PAIR_PER_QUERY_CAP="${PAIR_PER_QUERY_CAP:-0}"
PAIR_MAX_DISAGREEMENT_PER_QUERY="${PAIR_MAX_DISAGREEMENT_PER_QUERY:-6}"
PAIR_MAX_BOUNDARY_PER_QUERY="${PAIR_MAX_BOUNDARY_PER_QUERY:-6}"
PAIR_WEAK_MIN_PER_QUERY="${PAIR_WEAK_MIN_PER_QUERY:-10}"
PAIR_DISAGREE_PREFILTER_MIN="${PAIR_DISAGREE_PREFILTER_MIN:-0.70}"
PAIR_DISAGREE_TEACHER_MAX="${PAIR_DISAGREE_TEACHER_MAX:-0.30}"
PAIR_DISAGREE_MIN_MARGIN="${PAIR_DISAGREE_MIN_MARGIN:-0.15}"
PAIR_BOUNDARY_MIN_MARGIN="${PAIR_BOUNDARY_MIN_MARGIN:-0.05}"

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/split_distillation_dataset.py
  --distillation-input "${DISTILLATION_INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --prefix-mode "${PREFIX_MODE}"
  --pair-generation-mode "${PAIR_GENERATION_MODE}"
  --pair-min-margin "${PAIR_MIN_MARGIN}"
  --pair-per-query-cap "${PAIR_PER_QUERY_CAP}"
  --pair-max-disagreement-per-query "${PAIR_MAX_DISAGREEMENT_PER_QUERY}"
  --pair-max-boundary-per-query "${PAIR_MAX_BOUNDARY_PER_QUERY}"
  --pair-weak-min-per-query "${PAIR_WEAK_MIN_PER_QUERY}"
  --pair-disagree-prefilter-min "${PAIR_DISAGREE_PREFILTER_MIN}"
  --pair-disagree-teacher-max "${PAIR_DISAGREE_TEACHER_MAX}"
  --pair-disagree-min-margin "${PAIR_DISAGREE_MIN_MARGIN}"
  --pair-boundary-min-margin "${PAIR_BOUNDARY_MIN_MARGIN}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
