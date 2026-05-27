#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"
SUBSET_MODE="${SUBSET_MODE:-prefilter-debug}"   # auto | prefilter-debug | prefilter | random
PREFILTER_CACHE="${PREFILTER_CACHE:-ce2/dataset/source/prefilter_cache.jsonl}"
PREFILTER_DEBUG_ASPECTS="${PREFILTER_DEBUG_ASPECTS:-domain}"  # domain gives 10 grants x high/mid/low = 30 fac specs
PREFILTER_DEBUG_SELECTION_OUTPUT="${PREFILTER_DEBUG_SELECTION_OUTPUT:-ce2/test/output/prefilter_debug_selection.jsonl}"
PREFILTER_HIGH_PER_ASPECT="${PREFILTER_HIGH_PER_ASPECT:-4}"
PREFILTER_MID_PER_ASPECT="${PREFILTER_MID_PER_ASPECT:-4}"
PREFILTER_LOW_PER_ASPECT="${PREFILTER_LOW_PER_ASPECT:-4}"
PREFILTER_HIGH_THRESHOLD="${PREFILTER_HIGH_THRESHOLD:-0.70}"
PREFILTER_LOW_THRESHOLD="${PREFILTER_LOW_THRESHOLD:-0.30}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-10}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-24}"
DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-16}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"
ALLOW_NON_TEST_OUTPUT="${ALLOW_NON_TEST_OUTPUT:-false}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-false}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce2/test/output/spec_decompositions_subset.jsonl}"
PREVIEW_OUTPUT="${PREVIEW_OUTPUT:-ce2/test/output/spec_decompositions_subset_preview.txt}"
SUBSET_GRANT_DB_OUTPUT="${SUBSET_GRANT_DB_OUTPUT:-ce2/test/output/grant_keywords_spec_keywords_db_subset.json}"
SUBSET_FAC_DB_OUTPUT="${SUBSET_FAC_DB_OUTPUT:-ce2/test/output/fac_specs_db_subset.json}"
PREVIEW_COUNT="${PREVIEW_COUNT:-12}"
PREVIEW_KIND="${PREVIEW_KIND:-all}"  # all | grant | faculty
WRITE_PREVIEW="${WRITE_PREVIEW:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce2/test/decomposition_subset_smoke.py
  --model-id "${MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --subset-mode "${SUBSET_MODE}"
  --prefilter-cache "${PREFILTER_CACHE}"
  --prefilter-debug-aspects "${PREFILTER_DEBUG_ASPECTS}"
  --prefilter-debug-selection-output "${PREFILTER_DEBUG_SELECTION_OUTPUT}"
  --prefilter-high-per-aspect "${PREFILTER_HIGH_PER_ASPECT}"
  --prefilter-mid-per-aspect "${PREFILTER_MID_PER_ASPECT}"
  --prefilter-low-per-aspect "${PREFILTER_LOW_PER_ASPECT}"
  --prefilter-high-threshold "${PREFILTER_HIGH_THRESHOLD}"
  --prefilter-low-threshold "${PREFILTER_LOW_THRESHOLD}"
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
  --subset-grant-db-output "${SUBSET_GRANT_DB_OUTPUT}"
  --subset-fac-db-output "${SUBSET_FAC_DB_OUTPUT}"
  --preview-count "${PREVIEW_COUNT}"
  --preview-kind "${PREVIEW_KIND}"
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
if bool_true "${WRITE_PREVIEW}"; then
  CMD+=(--write-preview)
else
  CMD+=(--no-write-preview)
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
if bool_true "${WRITE_PREVIEW}"; then
  echo "Preview: ${PREVIEW_OUTPUT}"
fi
