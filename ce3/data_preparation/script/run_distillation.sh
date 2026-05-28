#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce3/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce3/dataset/source/fac_specs_db.json}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/dataset/distill}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-ce3/dataset/distill/llm_distillation.jsonl}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-0}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-0}"
PREFILTER_HIGH_PER_ASPECT="${PREFILTER_HIGH_PER_ASPECT:-3}"
PREFILTER_MID_PER_ASPECT="${PREFILTER_MID_PER_ASPECT:-3}"
PREFILTER_LOW_PER_ASPECT="${PREFILTER_LOW_PER_ASPECT:-3}"
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
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output-dir "${OUTPUT_DIR}"
  --distillation-output "${DISTILLATION_OUTPUT}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --prefilter-high-per-aspect "${PREFILTER_HIGH_PER_ASPECT}"
  --prefilter-mid-per-aspect "${PREFILTER_MID_PER_ASPECT}"
  --prefilter-low-per-aspect "${PREFILTER_LOW_PER_ASPECT}"
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
