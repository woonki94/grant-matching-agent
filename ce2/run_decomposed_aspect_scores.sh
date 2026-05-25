#!/usr/bin/env bash
set -euo pipefail

# CE2 pilot:
# 1) decompose grant/faculty specialization text into short-form 3 aspects
# 2) score selected grant-faculty pairs on each aspect
# 3) save analyzable JSONL + summary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-14B}"
GRANT_DB="${GRANT_DB:-ce/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce/dataset/source/fac_specs_db.json}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-ce2/dataset/distill/runs}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-${OUTPUT_DIR}/spec_decompositions_3aspect_shortform_${RUN_ID}.jsonl}"
SCORES_OUTPUT="${SCORES_OUTPUT:-${OUTPUT_DIR}/decomposed_3aspect_shortform_pair_scores_${RUN_ID}.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OUTPUT_DIR}/decomposed_3aspect_shortform_pair_scores_summary_${RUN_ID}.json}"

SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-60}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-1000}"
CANDIDATES_PER_GRANT_SPEC="${CANDIDATES_PER_GRANT_SPEC:-10}"
RANDOM_CANDIDATES_PER_GRANT_SPEC="${RANDOM_CANDIDATES_PER_GRANT_SPEC:-3}"

DECOMPOSE_BATCH_SIZE="${DECOMPOSE_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-24}"
DECOMPOSE_MAX_NEW_TOKENS="${DECOMPOSE_MAX_NEW_TOKENS:-512}"
SCORE_MAX_NEW_TOKENS="${SCORE_MAX_NEW_TOKENS:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

OVERWRITE="${OVERWRITE:-true}"
REFRESH_FAILED_DECOMPOSITIONS="${REFRESH_FAILED_DECOMPOSITIONS:-true}"
REFRESH_ALL_DECOMPOSITIONS="${REFRESH_ALL_DECOMPOSITIONS:-true}"
DECOMPOSE_ONLY="${DECOMPOSE_ONLY:-true}"
SCORE_ONLY="${SCORE_ONLY:-false}"

log "CE2 decomposed aspect score pilot"
log "model_id=${MODEL_ID}"
log "grant_db=${GRANT_DB}"
log "fac_db=${FAC_DB}"
log "max_grant_specs=${MAX_GRANT_SPECS} max_fac_specs=${MAX_FAC_SPECS}"
log "candidates_per_grant_spec=${CANDIDATES_PER_GRANT_SPEC} random=${RANDOM_CANDIDATES_PER_GRANT_SPEC}"
log "run_id=${RUN_ID}"
log "scores_output=${SCORES_OUTPUT}"
log "overwrite=${OVERWRITE} refresh_failed_decompositions=${REFRESH_FAILED_DECOMPOSITIONS} refresh_all_decompositions=${REFRESH_ALL_DECOMPOSITIONS} decompose_only=${DECOMPOSE_ONLY} score_only=${SCORE_ONLY}"

CMD=(
  "${PYTHON_BIN}" ce2/build_decomposed_aspect_scores.py
  --model-id "${MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --output-dir "${OUTPUT_DIR}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --scores-output "${SCORES_OUTPUT}"
  --summary-output "${SUMMARY_OUTPUT}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --candidates-per-grant-spec "${CANDIDATES_PER_GRANT_SPEC}"
  --random-candidates-per-grant-spec "${RANDOM_CANDIDATES_PER_GRANT_SPEC}"
  --decompose-batch-size "${DECOMPOSE_BATCH_SIZE}"
  --score-batch-size "${SCORE_BATCH_SIZE}"
  --decompose-max-new-tokens "${DECOMPOSE_MAX_NEW_TOKENS}"
  --score-max-new-tokens "${SCORE_MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-attempts "${MAX_ATTEMPTS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${REFRESH_FAILED_DECOMPOSITIONS}" == "true" ]]; then
  CMD+=(--refresh-failed-decompositions)
else
  CMD+=(--no-refresh-failed-decompositions)
fi
if [[ "${REFRESH_ALL_DECOMPOSITIONS}" == "true" ]]; then
  CMD+=(--refresh-all-decompositions)
else
  CMD+=(--no-refresh-all-decompositions)
fi
if [[ "${DECOMPOSE_ONLY}" == "true" ]]; then
  CMD+=(--decompose-only)
fi
if [[ "${SCORE_ONLY}" == "true" ]]; then
  CMD+=(--score-only)
fi

log "Running: ${CMD[*]}"
"${CMD[@]}"
log "Done."
