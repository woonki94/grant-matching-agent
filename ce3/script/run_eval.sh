#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-}"
MODEL_DIR="${MODEL_DIR:-ce3/models/aspect_reranker}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
TEST_LISTWISE="${TEST_LISTWISE:-}"
TEST_PAIRWISE="${TEST_PAIRWISE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/eval/results}"
BASE_MODEL="${BASE_MODEL:-dleemiller/ModernCE-base-sts}"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TOP_K="${TOP_K:-10}"
SCORE_FIELD="${SCORE_FIELD:-teacher_score}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.70}"
MID_THRESHOLD="${MID_THRESHOLD:-0.30}"
OOB_MARGIN="${OOB_MARGIN:-0.0}"
COMPARE_BASE="${COMPARE_BASE:-true}"
MULTIHEAD="${MULTIHEAD:-true}"
SAVE_ROWS="${SAVE_ROWS:-false}"

CMD=(
  "${PYTHON_BIN}" ce3/eval/eval_finetuned_model.py
  --model-dir "${MODEL_DIR}"
  --split-dir "${SPLIT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --base-model "${BASE_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --top-k "${TOP_K}"
  --score-field "${SCORE_FIELD}"
  --high-threshold "${HIGH_THRESHOLD}"
  --mid-threshold "${MID_THRESHOLD}"
  --oob-margin "${OOB_MARGIN}"
)

if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
fi
if [[ -n "${TEST_LISTWISE}" ]]; then
  CMD+=(--test-listwise "${TEST_LISTWISE}")
fi
if [[ -n "${TEST_PAIRWISE}" ]]; then
  CMD+=(--test-pairwise "${TEST_PAIRWISE}")
fi
if [[ "${COMPARE_BASE}" != "true" ]]; then
  CMD+=(--no-compare-base)
fi
if [[ "${MULTIHEAD}" != "true" ]]; then
  CMD+=(--no-multihead)
fi
if [[ "${SAVE_ROWS}" == "true" ]]; then
  CMD+=(--save-rows)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
