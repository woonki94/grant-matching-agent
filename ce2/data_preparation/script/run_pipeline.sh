#!/usr/bin/env bash
set -euo pipefail

# CE2 pilot:
# 1) decompose grant/faculty specialization text into short-form 3 aspects
# 2) build a stable CE prefilter cache over decomposed aspect text
# 3) run LLM distillation over selected grant-faculty pairs
# 4) save analyzable JSONL + summary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"

SOURCE_DIR="${SOURCE_DIR:-ce2/dataset/source}"
DECOMPOSITION_DIR="${DECOMPOSITION_DIR:-ce2/dataset/decomposed}"
DISTILL_DIR="${DISTILL_DIR:-ce2/dataset/distill}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-${DECOMPOSITION_DIR}/spec_decompositions_3aspect_shortform.jsonl}"
PREFILTER_CACHE_OUTPUT="${PREFILTER_CACHE_OUTPUT:-${SOURCE_DIR}/prefilter_cache.jsonl}"
PREFILTER_CACHE_MANIFEST="${PREFILTER_CACHE_MANIFEST:-${SOURCE_DIR}/prefilter_cache.manifest.json}"
DISTILLATION_OUTPUT="${DISTILLATION_OUTPUT:-${DISTILL_DIR}/llm_distillation.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${DISTILL_DIR}/llm_distillation_summary.json}"

SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-0}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-0}"

DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-16}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-24}"
PREFILTER_CACHE_BATCH_SIZE="${PREFILTER_CACHE_BATCH_SIZE:-64}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-512}"
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-300}"
PREFILTER_CACHE_MAX_LENGTH="${PREFILTER_CACHE_MAX_LENGTH:-256}"
PREFILTER_CACHE_TOP_K_PER_ASPECT="${PREFILTER_CACHE_TOP_K_PER_ASPECT:-256}"
TARGET_HIGH_PER_ASPECT="${TARGET_HIGH_PER_ASPECT:-2}"
TARGET_MID_PER_ASPECT="${TARGET_MID_PER_ASPECT:-2}"
TARGET_LOW_PER_ASPECT="${TARGET_LOW_PER_ASPECT:-2}"
PREFILTER_MULTIPLIER_HIGH="${PREFILTER_MULTIPLIER_HIGH:-8}"
PREFILTER_MULTIPLIER_MID="${PREFILTER_MULTIPLIER_MID:-8}"
PREFILTER_MULTIPLIER_LOW="${PREFILTER_MULTIPLIER_LOW:-4}"
PREFILTER_HIGH_THRESHOLD="${PREFILTER_HIGH_THRESHOLD:-0.70}"
PREFILTER_LOW_THRESHOLD="${PREFILTER_LOW_THRESHOLD:-0.30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
PREFILTER_MODEL_ID="${PREFILTER_MODEL_ID:-dleemiller/ModernCE-base-sts}"
PREFILTER_SOURCE="${PREFILTER_SOURCE:-ce-cache}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

OVERWRITE="${OVERWRITE:-true}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-true}"
BUILD_PREFILTER_CACHE="${BUILD_PREFILTER_CACHE:-true}"
DECOMPOSE_ONLY="${DECOMPOSE_ONLY:-false}"
DISTILL_ONLY="${DISTILL_ONLY:-false}"

log "CE2 data-preparation pipeline"
log "model_id=${MODEL_ID}"
log "grant_db=${GRANT_DB}"
log "fac_db=${FAC_DB}"
log "max_grant_specs=${MAX_GRANT_SPECS} max_fac_specs=${MAX_FAC_SPECS}"
log "source_dir=${SOURCE_DIR}"
log "decomposition_dir=${DECOMPOSITION_DIR}"
log "distill_dir=${DISTILL_DIR}"
log "decomposition_output=${DECOMPOSITION_OUTPUT}"
log "prefilter_source=${PREFILTER_SOURCE}"
log "prefilter_cache_output=${PREFILTER_CACHE_OUTPUT}"
log "prefilter_cache_manifest=${PREFILTER_CACHE_MANIFEST}"
log "distillation_output=${DISTILLATION_OUTPUT}"
log "summary_output=${SUMMARY_OUTPUT}"
log "target_per_aspect=high:${TARGET_HIGH_PER_ASPECT},mid:${TARGET_MID_PER_ASPECT},low:${TARGET_LOW_PER_ASPECT}"
log "prefilter_multiplier=high:${PREFILTER_MULTIPLIER_HIGH},mid:${PREFILTER_MULTIPLIER_MID},low:${PREFILTER_MULTIPLIER_LOW}"
log "prefilter_thresholds=high:${PREFILTER_HIGH_THRESHOLD},low:${PREFILTER_LOW_THRESHOLD}"
log "overwrite=${OVERWRITE} refresh_failed_decompositions=${REFRESH_FAILED_DECOMPOSITIONS} refresh_all_decompositions=${REFRESH_ALL_DECOMPOSITIONS} build_prefilter_cache=${BUILD_PREFILTER_CACHE} decompose_only=${DECOMPOSE_ONLY} distill_only=${DISTILL_ONLY}"

run_decompose=true
run_cache=true
run_distill=true
if [[ "${DECOMPOSE_ONLY}" == "true" && "${DISTILL_ONLY}" != "true" ]]; then
  run_cache=false
  run_distill=false
elif [[ "${DISTILL_ONLY}" == "true" && "${DECOMPOSE_ONLY}" != "true" ]]; then
  run_decompose=false
fi
if [[ "${PREFILTER_SOURCE}" != "ce-cache" || "${BUILD_PREFILTER_CACHE}" != "true" ]]; then
  run_cache=false
fi

if [[ "${run_decompose}" == "true" ]]; then
  DECOMP_CMD=(
    "${PYTHON_BIN}" ce2/data_preparation/decompose_aspect_specs.py
    --model-id "${MODEL_ID}"
    --grant-db "${GRANT_DB}"
    --fac-db "${FAC_DB}"
    --output-dir "${DECOMPOSITION_DIR}"
    --decomposition-output "${DECOMPOSITION_OUTPUT}"
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
  )
  if [[ "${OVERWRITE}" == "true" ]]; then
    DECOMP_CMD+=(--overwrite)
  fi
  if [[ "${REFRESH_FAILED_DECOMPOSITIONS}" == "true" ]]; then
    DECOMP_CMD+=(--refresh-failed-decompositions)
  else
    DECOMP_CMD+=(--no-refresh-failed-decompositions)
  fi
  if [[ "${REFRESH_ALL_DECOMPOSITIONS}" == "true" ]]; then
    DECOMP_CMD+=(--refresh-all-decompositions)
  else
    DECOMP_CMD+=(--no-refresh-all-decompositions)
  fi
  log "Running (decompose): ${DECOMP_CMD[*]}"
  "${DECOMP_CMD[@]}"
fi

if [[ "${run_cache}" == "true" ]]; then
  CACHE_CMD=(
    "${PYTHON_BIN}" ce2/data_preparation/build_prefilter_cache.py
    --model-id "${PREFILTER_MODEL_ID}"
    --grant-db "${GRANT_DB}"
    --fac-db "${FAC_DB}"
    --decomposition-output "${DECOMPOSITION_OUTPUT}"
    --output "${PREFILTER_CACHE_OUTPUT}"
    --manifest "${PREFILTER_CACHE_MANIFEST}"
    --seed "${SEED}"
    --max-grant-specs "${MAX_GRANT_SPECS}"
    --max-fac-specs "${MAX_FAC_SPECS}"
    --top-k-per-aspect "${PREFILTER_CACHE_TOP_K_PER_ASPECT}"
    --batch-size "${PREFILTER_CACHE_BATCH_SIZE}"
    --max-length "${PREFILTER_CACHE_MAX_LENGTH}"
  )
  log "Running (cache): ${CACHE_CMD[*]}"
  "${CACHE_CMD[@]}"
fi

if [[ "${run_distill}" == "true" ]]; then
  DISTILL_CMD=(
    "${PYTHON_BIN}" ce2/data_preparation/llm_distillation.py
    --model-id "${MODEL_ID}"
    --grant-db "${GRANT_DB}"
    --fac-db "${FAC_DB}"
    --output-dir "${DISTILL_DIR}"
    --decomposition-output "${DECOMPOSITION_OUTPUT}"
    --prefilter-source "${PREFILTER_SOURCE}"
    --prefilter-cache "${PREFILTER_CACHE_OUTPUT}"
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
  )
  if [[ "${OVERWRITE}" == "true" ]]; then
    DISTILL_CMD+=(--overwrite)
  fi
  log "Running (distill): ${DISTILL_CMD[*]}"
  "${DISTILL_CMD[@]}"
fi
log "Done."
