#!/usr/bin/env bash
set -euo pipefail

# CE2 final data-preparation pass:
# 1) distill real cache-selected pairs
# 2) augment missing aspect-band clusters
# 3) export aspect-specific train/val/test split files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

DISTILL_MODEL_ID="${DISTILL_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"
AUGMENT_MODEL_ID="${AUGMENT_MODEL_ID:-Qwen/Qwen3-14B}"

GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"

SOURCE_DIR="${SOURCE_DIR:-ce2/dataset/source}"
DECOMPOSITION_DIR="${DECOMPOSITION_DIR:-ce2/dataset/decomposed}"
DISTILL_DIR="${DISTILL_DIR:-ce2/dataset/distill}"
SPLIT_DIR="${SPLIT_DIR:-ce2/dataset/splits}"

DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-${DECOMPOSITION_DIR}/spec_decompositions_3aspect_shortform.jsonl}"
PREFILTER_CACHE_OUTPUT="${PREFILTER_CACHE_OUTPUT:-${SOURCE_DIR}/prefilter_cache.jsonl}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-${DISTILL_DIR}/llm_distillation.jsonl}"
DISTILLATION_SUMMARY_OUTPUT="${DISTILLATION_SUMMARY_OUTPUT:-${DISTILL_DIR}/llm_distillation_summary.json}"
AUGMENTATION_OUTPUT="${AUGMENTATION_OUTPUT:-${DISTILL_DIR}/augmentation.jsonl}"
AUGMENTATION_SUMMARY_OUTPUT="${AUGMENTATION_SUMMARY_OUTPUT:-${DISTILL_DIR}/augmentation_summary.json}"

SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-0}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-0}"

TARGET_HIGH_PER_ASPECT="${TARGET_HIGH_PER_ASPECT:-2}"
TARGET_MID_PER_ASPECT="${TARGET_MID_PER_ASPECT:-2}"
TARGET_LOW_PER_ASPECT="${TARGET_LOW_PER_ASPECT:-2}"
PREFILTER_MULTIPLIER_HIGH="${PREFILTER_MULTIPLIER_HIGH:-8}"
PREFILTER_MULTIPLIER_MID="${PREFILTER_MULTIPLIER_MID:-8}"
PREFILTER_MULTIPLIER_LOW="${PREFILTER_MULTIPLIER_LOW:-4}"
PREFILTER_HIGH_THRESHOLD="${PREFILTER_HIGH_THRESHOLD:-0.70}"
PREFILTER_LOW_THRESHOLD="${PREFILTER_LOW_THRESHOLD:-0.30}"

DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-24}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-32}"
AUGMENT_GEN_BATCH_SIZE="${AUGMENT_GEN_BATCH_SIZE:-24}"
AUGMENT_GEN_MAX_NEW_TOKENS="${AUGMENT_GEN_MAX_NEW_TOKENS:-512}"
AUGMENT_MAX_TRIES_PER_MISSING="${AUGMENT_MAX_TRIES_PER_MISSING:-8}"
AUGMENT_MAX_ADD_PER_BAND="${AUGMENT_MAX_ADD_PER_BAND:-300}"
AUGMENT_TARGET_POLICY="${AUGMENT_TARGET_POLICY:-median}"
AUGMENT_TARGET_HIGH="${AUGMENT_TARGET_HIGH:-0}"
AUGMENT_TARGET_MID="${AUGMENT_TARGET_MID:-0}"
AUGMENT_TARGET_LOW="${AUGMENT_TARGET_LOW:-0}"

SPLIT_VAL_RATIO="${SPLIT_VAL_RATIO:-0.10}"
SPLIT_TEST_RATIO="${SPLIT_TEST_RATIO:-0.10}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-true}"

log "CE2 distill -> augment -> split"
log "distill_model_id=${DISTILL_MODEL_ID}"
log "augment_model_id=${AUGMENT_MODEL_ID}"
log "decomposition_output=${DECOMPOSITION_OUTPUT}"
log "prefilter_cache_output=${PREFILTER_CACHE_OUTPUT}"
log "distillation_output=${DISTILLATION_OUTPUT}"
log "augmentation_output=${AUGMENTATION_OUTPUT}"
log "split_dir=${SPLIT_DIR}"
log "target_per_aspect=high:${TARGET_HIGH_PER_ASPECT},mid:${TARGET_MID_PER_ASPECT},low:${TARGET_LOW_PER_ASPECT}"
log "prefilter_multiplier=high:${PREFILTER_MULTIPLIER_HIGH},mid:${PREFILTER_MULTIPLIER_MID},low:${PREFILTER_MULTIPLIER_LOW}"

DISTILL_CMD=(
  "${PYTHON_BIN}" ce2/data_preparation/llm_distillation.py
  --model-id "${DISTILL_MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --prefilter-source ce-cache
  --prefilter-cache "${PREFILTER_CACHE_OUTPUT}"
  --distillation-output "${DISTILLATION_OUTPUT}"
  --summary-output "${DISTILLATION_SUMMARY_OUTPUT}"
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
)
if [[ "${OVERWRITE}" == "true" ]]; then
  DISTILL_CMD+=(--overwrite)
fi
log "Running (distill): ${DISTILL_CMD[*]}"
"${DISTILL_CMD[@]}"

AUGMENT_CMD=(
  "${PYTHON_BIN}" ce2/data_preparation/augmentation.py
  --model-id "${AUGMENT_MODEL_ID}"
  --distillation-input "${DISTILLATION_OUTPUT}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output "${AUGMENTATION_OUTPUT}"
  --summary-output "${AUGMENTATION_SUMMARY_OUTPUT}"
  --seed "${SEED}"
  --target-policy "${AUGMENT_TARGET_POLICY}"
  --target-high "${AUGMENT_TARGET_HIGH}"
  --target-mid "${AUGMENT_TARGET_MID}"
  --target-low "${AUGMENT_TARGET_LOW}"
  --max-add-per-band "${AUGMENT_MAX_ADD_PER_BAND}"
  --gen-batch-size "${AUGMENT_GEN_BATCH_SIZE}"
  --gen-max-new-tokens "${AUGMENT_GEN_MAX_NEW_TOKENS}"
  --max-tries-per-missing "${AUGMENT_MAX_TRIES_PER_MISSING}"
  --distill-batch-size "${DISTILL_BATCH_SIZE}"
  --distill-max-new-tokens "${DISTILL_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  AUGMENT_CMD+=(--overwrite)
fi
log "Running (augment): ${AUGMENT_CMD[*]}"
"${AUGMENT_CMD[@]}"

SPLIT_CMD=(
  "${PYTHON_BIN}" ce2/data_preparation/split_distillation.py
  --distillation-input "${DISTILLATION_OUTPUT}"
  --augmentation-input "${AUGMENTATION_OUTPUT}"
  --output-dir "${SPLIT_DIR}"
  --seed "${SEED}"
  --val-ratio "${SPLIT_VAL_RATIO}"
  --test-ratio "${SPLIT_TEST_RATIO}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  SPLIT_CMD+=(--overwrite)
fi
log "Running (split): ${SPLIT_CMD[*]}"
"${SPLIT_CMD[@]}"

log "Done."
