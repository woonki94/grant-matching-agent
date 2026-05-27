#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-6}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-6}"
DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-16}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-true}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-false}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce2/test/output/spec_decompositions_subset.jsonl}"
PREVIEW_OUTPUT="${PREVIEW_OUTPUT:-ce2/test/output/spec_decompositions_subset_preview.txt}"
PREVIEW_COUNT="${PREVIEW_COUNT:-12}"
PREVIEW_KIND="${PREVIEW_KIND:-all}"  # all | grant | faculty

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce2/test/decomposition_subset_smoke.py
  --model-id "${MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --decompose-batch-size "${DECOMPOSE_BATCH_SIZE}"
  --decompose-max-new-tokens "${DECOMPOSE_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --preview-output "${PREVIEW_OUTPUT}"
  --preview-count "${PREVIEW_COUNT}"
  --preview-kind "${PREVIEW_KIND}"
)

if bool_true "${OVERWRITE}"; then
  CMD+=(--overwrite)
else
  CMD+=(--no-overwrite)
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
echo "Preview: ${PREVIEW_OUTPUT}"
