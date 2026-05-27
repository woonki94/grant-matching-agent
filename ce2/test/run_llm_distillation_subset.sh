#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce2/test/output/spec_decompositions_subset.jsonl}"
PREFILTER_SOURCE="${PREFILTER_SOURCE:-auto}"   # auto | ce-cache | sts
PREFILTER_CACHE="${PREFILTER_CACHE:-ce2/dataset/source/prefilter_cache.jsonl}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-ce2/test/output/llm_distillation_subset.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-ce2/test/output/llm_distillation_subset_summary.json}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-6}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-6}"
TARGET_HIGH_PER_ASPECT="${TARGET_HIGH_PER_ASPECT:-1}"
TARGET_MID_PER_ASPECT="${TARGET_MID_PER_ASPECT:-1}"
TARGET_LOW_PER_ASPECT="${TARGET_LOW_PER_ASPECT:-1}"
PREFILTER_MULTIPLIER_HIGH="${PREFILTER_MULTIPLIER_HIGH:-8.0}"
PREFILTER_MULTIPLIER_MID="${PREFILTER_MULTIPLIER_MID:-8.0}"
PREFILTER_MULTIPLIER_LOW="${PREFILTER_MULTIPLIER_LOW:-4.0}"
PREFILTER_HIGH_THRESHOLD="${PREFILTER_HIGH_THRESHOLD:-0.70}"
PREFILTER_LOW_THRESHOLD="${PREFILTER_LOW_THRESHOLD:-0.30}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-24}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-false}"
ALLOW_NON_TEST_OUTPUT="${ALLOW_NON_TEST_OUTPUT:-false}"
PREVIEW_OUTPUT="${PREVIEW_OUTPUT:-ce2/test/output/llm_distillation_subset_preview.txt}"
PREVIEW_COUNT="${PREVIEW_COUNT:-36}"
WRITE_PREVIEW="${WRITE_PREVIEW:-false}"
WRITE_SUMMARY="${WRITE_SUMMARY:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce2/test/llm_distillation_subset_smoke.py
  --model-id "${MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --prefilter-source "${PREFILTER_SOURCE}"
  --prefilter-cache "${PREFILTER_CACHE}"
  --distillation-output "${DISTILLATION_OUTPUT}"
  --summary-output "${SUMMARY_OUTPUT}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --target-high-per-aspect "${TARGET_HIGH_PER_ASPECT}"
  --target-mid-per-aspect "${TARGET_MID_PER_ASPECT}"
  --target-low-per-aspect "${TARGET_LOW_PER_ASPECT}"
  --prefilter-multiplier-high "${PREFILTER_MULTIPLIER_HIGH}"
  --prefilter-multiplier-mid "${PREFILTER_MULTIPLIER_MID}"
  --prefilter-multiplier-low "${PREFILTER_MULTIPLIER_LOW}"
  --prefilter-high-threshold "${PREFILTER_HIGH_THRESHOLD}"
  --prefilter-low-threshold "${PREFILTER_LOW_THRESHOLD}"
  --distill-batch-size "${DISTILL_BATCH_SIZE}"
  --distill-max-new-tokens "${DISTILL_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
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
if bool_true "${WRITE_PREVIEW}"; then
  CMD+=(--write-preview)
else
  CMD+=(--no-write-preview)
fi
if bool_true "${WRITE_SUMMARY}"; then
  CMD+=(--write-summary)
else
  CMD+=(--no-write-summary)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
if bool_true "${WRITE_SUMMARY}"; then
  echo "Summary: ${SUMMARY_OUTPUT}"
fi
if bool_true "${WRITE_PREVIEW}"; then
  echo "Preview: ${PREVIEW_OUTPUT}"
fi
