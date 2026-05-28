#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_combined.jsonl}"
PREFILTER_CACHE_BASE="${PREFILTER_CACHE_BASE:-ce3/dataset/source/prefilter_cache.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/dataset/distill}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-ce3/dataset/distill/llm_distillation.jsonl}"
TARGET_HIGH_PER_GRANT_ASPECT="${TARGET_HIGH_PER_GRANT_ASPECT:-4}"
TARGET_MID_PER_GRANT_ASPECT="${TARGET_MID_PER_GRANT_ASPECT:-8}"
TARGET_LOW_PER_GRANT_ASPECT="${TARGET_LOW_PER_GRANT_ASPECT:-4}"
PREFILTER_HIGH_MULTIPLIER="${PREFILTER_HIGH_MULTIPLIER:-4.0}"
PREFILTER_MID_MULTIPLIER="${PREFILTER_MID_MULTIPLIER:-2.0}"
PREFILTER_LOW_MULTIPLIER="${PREFILTER_LOW_MULTIPLIER:-1.25}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-24}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/llm_distillation.py
  --model-id "${MODEL_ID}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --prefilter-cache-base "${PREFILTER_CACHE_BASE}"
  --output-dir "${OUTPUT_DIR}"
  --distillation-output "${DISTILLATION_OUTPUT}"
  --target-high-per-grant-aspect "${TARGET_HIGH_PER_GRANT_ASPECT}"
  --target-mid-per-grant-aspect "${TARGET_MID_PER_GRANT_ASPECT}"
  --target-low-per-grant-aspect "${TARGET_LOW_PER_GRANT_ASPECT}"
  --prefilter-high-multiplier "${PREFILTER_HIGH_MULTIPLIER}"
  --prefilter-mid-multiplier "${PREFILTER_MID_MULTIPLIER}"
  --prefilter-low-multiplier "${PREFILTER_LOW_MULTIPLIER}"
  --distill-batch-size "${DISTILL_BATCH_SIZE}"
  --distill-max-new-tokens "${DISTILL_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if bool_true "${OVERWRITE}"; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
