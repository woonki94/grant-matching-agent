#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-auto}"
DISTILLATION_INPUT="${DISTILLATION_INPUT:-ce2/test/output/llm_distillation_subset.jsonl}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce2/test/output/spec_decompositions_subset.jsonl}"
OUTPUT="${OUTPUT:-ce2/test/output/augmentation_subset.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-ce2/test/output/augmentation_subset_summary.json}"
SEED="${SEED:-42}"
TARGET_POLICY="${TARGET_POLICY:-explicit}"   # median | explicit
TARGET_HIGH="${TARGET_HIGH:-2}"
TARGET_MID="${TARGET_MID:-2}"
TARGET_LOW="${TARGET_LOW:-2}"
MAX_ADD_PER_BAND="${MAX_ADD_PER_BAND:-48}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-24}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
MAX_TRIES_PER_MISSING="${MAX_TRIES_PER_MISSING:-6}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-24}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"
ALLOW_NON_TEST_OUTPUT="${ALLOW_NON_TEST_OUTPUT:-false}"
PREVIEW_OUTPUT="${PREVIEW_OUTPUT:-ce2/test/output/augmentation_subset_preview.txt}"
PREVIEW_COUNT="${PREVIEW_COUNT:-30}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce2/test/augmentation_subset_smoke.py
  --model-id "${MODEL_ID}"
  --distillation-input "${DISTILLATION_INPUT}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output "${OUTPUT}"
  --summary-output "${SUMMARY_OUTPUT}"
  --seed "${SEED}"
  --target-policy "${TARGET_POLICY}"
  --target-high "${TARGET_HIGH}"
  --target-mid "${TARGET_MID}"
  --target-low "${TARGET_LOW}"
  --max-add-per-band "${MAX_ADD_PER_BAND}"
  --gen-batch-size "${GEN_BATCH_SIZE}"
  --gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}"
  --max-tries-per-missing "${MAX_TRIES_PER_MISSING}"
  --distill-batch-size "${DISTILL_BATCH_SIZE}"
  --distill-max-new-tokens "${DISTILL_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --preview-output "${PREVIEW_OUTPUT}"
  --preview-count "${PREVIEW_COUNT}"
)
if bool_true "${ALLOW_NON_TEST_OUTPUT}"; then
  CMD+=(--allow-non-test-output)
else
  CMD+=(--no-allow-non-test-output)
fi

if bool_true "${OVERWRITE}"; then
  CMD+=(--overwrite)
else
  CMD+=(--no-overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Preview: ${PREVIEW_OUTPUT}"
