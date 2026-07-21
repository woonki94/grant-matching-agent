#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
AUGMENTATION_INPUT="${AUGMENTATION_INPUT:-ce3/dataset/augmented/spec_augmentations_high.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/dataset/decomposed}"
AUGMENTED_DECOMPOSITION_OUTPUT="${AUGMENTED_DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_augmented.jsonl}"
DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-16}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/decompose_augmented_specializations.py
  --model-id "${MODEL_ID}"
  --augmentation-input "${AUGMENTATION_INPUT}"
  --output-dir "${OUTPUT_DIR}"
  --augmented-decomposition-output "${AUGMENTED_DECOMPOSITION_OUTPUT}"
  --decompose-batch-size "${DECOMPOSE_BATCH_SIZE}"
  --decompose-max-new-tokens "${DECOMPOSE_MAX_NEW_TOKENS}"
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
if bool_true "${REFRESH_FAILED_DECOMPOSITIONS}"; then
  CMD+=(--refresh-failed-decompositions)
else
  CMD+=(--no-refresh-failed-decompositions)
fi
if bool_true "${REFRESH_ALL_DECOMPOSITIONS}"; then
  CMD+=(--refresh-all-decompositions)
else
  CMD+=(--no-refresh-all-decompositions)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
