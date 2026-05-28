#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/dataset/augmented}"
AUGMENTATION_OUTPUT="${AUGMENTATION_OUTPUT:-ce3/dataset/augmented/spec_augmentations_high.jsonl}"
AUGMENTATIONS_PER_ASPECT="${AUGMENTATIONS_PER_ASPECT:-4}"
AUGMENT_BATCH_SIZE="${AUGMENT_BATCH_SIZE:-12}"
AUGMENT_MAX_NEW_TOKENS="${AUGMENT_MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/augment_specializations.py
  --model-id "${MODEL_ID}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output-dir "${OUTPUT_DIR}"
  --augmentation-output "${AUGMENTATION_OUTPUT}"
  --augmentations-per-aspect "${AUGMENTATIONS_PER_ASPECT}"
  --augment-batch-size "${AUGMENT_BATCH_SIZE}"
  --augment-max-new-tokens "${AUGMENT_MAX_NEW_TOKENS}"
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
