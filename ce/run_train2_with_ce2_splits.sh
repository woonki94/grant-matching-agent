#!/usr/bin/env bash
set -euo pipefail

# Convert CE2 split files into CE train2-compatible files, then run ce/train2.sh and eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
bool_true() { [[ "${1:-}" == "true" ]]; }

pick_eval_model_dir() {
  local run_dir="$1"
  local model_pick="$2"
  local out=""
  case "${model_pick}" in
    best_stage2_selected) out="${run_dir}/best_stage2_selected" ;;
    best_stage2_gated) out="${run_dir}/best_stage2_gated" ;;
    best_stage2_ranking) out="${run_dir}/best_stage2_ranking" ;;
    best_val_oob) out="${run_dir}/best_val_oob" ;;
    best) out="${run_dir}/best" ;;
    latest_stage2) out="$(ls -d "${run_dir}"/stage2_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)" ;;
    latest_stage1) out="$(ls -d "${run_dir}"/stage1_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)" ;;
    auto) ;;
    *)
      echo "Unsupported EVAL_MODEL_PICK: ${model_pick}" >&2
      return 1
      ;;
  esac

  if [[ "${model_pick}" == "auto" ]]; then
    local candidate
    for candidate in \
      "${run_dir}/best_stage2_selected" \
      "${run_dir}/best_stage2_gated" \
      "${run_dir}/best_stage2_ranking" \
      "${run_dir}/best_val_oob" \
      "${run_dir}/best"; do
      if [[ -d "${candidate}" ]]; then
        out="${candidate}"
        break
      fi
    done
    if [[ -z "${out}" ]]; then
      out="$(ls -d "${run_dir}"/stage2_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)"
    fi
    if [[ -z "${out}" ]]; then
      out="$(ls -d "${run_dir}"/stage1_epoch_* 2>/dev/null | sort -V | tail -n 1 || true)"
    fi
  fi

  if [[ -z "${out}" || ! -d "${out}" ]]; then
    out="${run_dir}"
  fi
  printf "%s" "${out}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"

# CE2 -> CE split conversion controls
CE2_SPLIT_INPUT_DIR="${CE2_SPLIT_INPUT_DIR:-ce2/dataset/splits}"
CE2_CE_SPLIT_DIR="${CE2_CE_SPLIT_DIR:-ce2/dataset/splits_ce_train2}"
CE2_SOURCE_ASPECTS="${CE2_SOURCE_ASPECTS:-domain,method,target}"
CE2_SPLITS="${CE2_SPLITS:-train,val,test}"
CE2_MIN_DOCS_PER_QUERY="${CE2_MIN_DOCS_PER_QUERY:-2}"
CE2_MAX_DOCS_PER_QUERY="${CE2_MAX_DOCS_PER_QUERY:-0}"
CE2_REQUIRE_PAIRWISE="${CE2_REQUIRE_PAIRWISE:-true}"
CE2_WRITE_AGGREGATE="${CE2_WRITE_AGGREGATE:-true}"
CE2_CONVERT_OVERWRITE="${CE2_CONVERT_OVERWRITE:-true}"

# Training defaults (can all be overridden through env)
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
OUTPUT_DIR="${OUTPUT_DIR:-ce/models/ce2_train2_compat}"
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-false}"
USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"
ASPECT_CONDITION_MODE="${ASPECT_CONDITION_MODE:-long_prefix}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"

# Eval defaults
RUN_EVAL="${RUN_EVAL:-true}"
EVAL_MODEL_PICK="${EVAL_MODEL_PICK:-auto}"  # auto | best_stage2_selected | best_stage2_gated | best_stage2_ranking | best_val_oob | best | latest_stage2 | latest_stage1
EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-${MODEL_ID}}"
EVAL_ASPECT_CONDITION_MODE="${EVAL_ASPECT_CONDITION_MODE:-${ASPECT_CONDITION_MODE}}"
EVAL_SCORE_FIELD="${EVAL_SCORE_FIELD:-teacher_score_raw}"
EVAL_ONLY_SELECTED="${EVAL_ONLY_SELECTED:-false}"
EVAL_INCLUDE_CONSTRAINT="${EVAL_INCLUDE_CONSTRAINT:-true}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-512}"
EVAL_HIGH_THRESHOLD="${EVAL_HIGH_THRESHOLD:-0.70}"
EVAL_MID_THRESHOLD="${EVAL_MID_THRESHOLD:-0.30}"
EVAL_OOB_MARGIN="${EVAL_OOB_MARGIN:-0.0}"
EVAL_ORDER_TOP_K="${EVAL_ORDER_TOP_K:-5}"
EVAL_PAIR_EPS="${EVAL_PAIR_EPS:-0.01}"
EVAL_HARD_GAP_MAX="${EVAL_HARD_GAP_MAX:-0.15}"
EVAL_MEDIUM_GAP_MAX="${EVAL_MEDIUM_GAP_MAX:-0.40}"
EVAL_SAVE="${EVAL_SAVE:-true}"
EVAL_PRINT="${EVAL_PRINT:-true}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-ce/eval/results}"
EVAL_SAVE_PREFIX="${EVAL_SAVE_PREFIX:-ce2_train2_compat_eval}"

# Converted CE-compatible file paths
SPLIT_DIR="${SPLIT_DIR:-${CE2_CE_SPLIT_DIR}}"
RAW_INPUT="${RAW_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_listwise.jsonl}"
METHOD_RAW_INPUT="${METHOD_RAW_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_listwise.jsonl}"
CONSTRAINT_RAW_INPUT="${CONSTRAINT_RAW_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_listwise.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_pairwise.jsonl}"
METHOD_PAIRWISE_INPUT="${METHOD_PAIRWISE_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_pairwise.jsonl}"
CONSTRAINT_PAIRWISE_INPUT="${CONSTRAINT_PAIRWISE_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_pairwise.jsonl}"

RAW_TRAIN_INPUT="${RAW_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_listwise_train.jsonl}"
RAW_VAL_INPUT="${RAW_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_listwise_val.jsonl}"
RAW_TEST_INPUT="${RAW_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_listwise_test.jsonl}"
METHOD_RAW_TRAIN_INPUT="${METHOD_RAW_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_listwise_train.jsonl}"
METHOD_RAW_VAL_INPUT="${METHOD_RAW_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_listwise_val.jsonl}"
METHOD_RAW_TEST_INPUT="${METHOD_RAW_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_listwise_test.jsonl}"
CONSTRAINT_RAW_TRAIN_INPUT="${CONSTRAINT_RAW_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_listwise_train.jsonl}"
CONSTRAINT_RAW_VAL_INPUT="${CONSTRAINT_RAW_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_listwise_val.jsonl}"
CONSTRAINT_RAW_TEST_INPUT="${CONSTRAINT_RAW_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_listwise_test.jsonl}"

PAIRWISE_TRAIN_INPUT="${PAIRWISE_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_pairwise_train.jsonl}"
PAIRWISE_VAL_INPUT="${PAIRWISE_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_pairwise_val.jsonl}"
PAIRWISE_TEST_INPUT="${PAIRWISE_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_domain_pairwise_test.jsonl}"
METHOD_PAIRWISE_TRAIN_INPUT="${METHOD_PAIRWISE_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_pairwise_train.jsonl}"
METHOD_PAIRWISE_VAL_INPUT="${METHOD_PAIRWISE_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_pairwise_val.jsonl}"
METHOD_PAIRWISE_TEST_INPUT="${METHOD_PAIRWISE_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_method_pairwise_test.jsonl}"
CONSTRAINT_PAIRWISE_TRAIN_INPUT="${CONSTRAINT_PAIRWISE_TRAIN_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_pairwise_train.jsonl}"
CONSTRAINT_PAIRWISE_VAL_INPUT="${CONSTRAINT_PAIRWISE_VAL_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_pairwise_val.jsonl}"
CONSTRAINT_PAIRWISE_TEST_INPUT="${CONSTRAINT_PAIRWISE_TEST_INPUT:-${CE2_CE_SPLIT_DIR}/llm_distill_constraint_pairwise_test.jsonl}"

EVAL_DOMAIN_INPUT="${EVAL_DOMAIN_INPUT:-${RAW_TEST_INPUT}}"
EVAL_METHOD_INPUT="${EVAL_METHOD_INPUT:-${METHOD_RAW_TEST_INPUT}}"
EVAL_CONSTRAINT_INPUT="${EVAL_CONSTRAINT_INPUT:-${CONSTRAINT_RAW_TEST_INPUT}}"

log "Convert CE2 splits -> CE train2-compatible splits"
log "ce2_input_split_dir=${CE2_SPLIT_INPUT_DIR}"
log "ce2_ce_compat_output_dir=${CE2_CE_SPLIT_DIR}"
log "source_aspects=${CE2_SOURCE_ASPECTS} splits=${CE2_SPLITS}"

CONVERT_CMD=(
  "${PYTHON_BIN}" ce2/data_preparation/split_for_ce_train2.py
  --input-split-dir "${CE2_SPLIT_INPUT_DIR}"
  --output-dir "${CE2_CE_SPLIT_DIR}"
  --source-aspects "${CE2_SOURCE_ASPECTS}"
  --splits "${CE2_SPLITS}"
  --min-docs-per-query "${CE2_MIN_DOCS_PER_QUERY}"
  --max-docs-per-query "${CE2_MAX_DOCS_PER_QUERY}"
)
if bool_true "${CE2_REQUIRE_PAIRWISE}"; then
  CONVERT_CMD+=(--require-pairwise)
else
  CONVERT_CMD+=(--no-require-pairwise)
fi
if bool_true "${CE2_WRITE_AGGREGATE}"; then
  CONVERT_CMD+=(--write-aggregate)
else
  CONVERT_CMD+=(--no-write-aggregate)
fi
if bool_true "${CE2_CONVERT_OVERWRITE}"; then
  CONVERT_CMD+=(--overwrite)
fi
log "Running: ${CONVERT_CMD[*]}"
"${CONVERT_CMD[@]}"

log "Train with ce/train2.sh using converted CE2 split files"
log "output_dir=${OUTPUT_DIR} append_args_to_output_dir=${APPEND_ARGS_TO_OUTPUT_DIR}"
log "eval_every_steps=${EVAL_EVERY_STEPS}"

env \
  RAW_INPUT="${RAW_INPUT}" \
  METHOD_RAW_INPUT="${METHOD_RAW_INPUT}" \
  CONSTRAINT_RAW_INPUT="${CONSTRAINT_RAW_INPUT}" \
  PAIRWISE_INPUT="${PAIRWISE_INPUT}" \
  METHOD_PAIRWISE_INPUT="${METHOD_PAIRWISE_INPUT}" \
  CONSTRAINT_PAIRWISE_INPUT="${CONSTRAINT_PAIRWISE_INPUT}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  RAW_TRAIN_INPUT="${RAW_TRAIN_INPUT}" \
  RAW_VAL_INPUT="${RAW_VAL_INPUT}" \
  RAW_TEST_INPUT="${RAW_TEST_INPUT}" \
  METHOD_RAW_TRAIN_INPUT="${METHOD_RAW_TRAIN_INPUT}" \
  METHOD_RAW_VAL_INPUT="${METHOD_RAW_VAL_INPUT}" \
  METHOD_RAW_TEST_INPUT="${METHOD_RAW_TEST_INPUT}" \
  CONSTRAINT_RAW_TRAIN_INPUT="${CONSTRAINT_RAW_TRAIN_INPUT}" \
  CONSTRAINT_RAW_VAL_INPUT="${CONSTRAINT_RAW_VAL_INPUT}" \
  CONSTRAINT_RAW_TEST_INPUT="${CONSTRAINT_RAW_TEST_INPUT}" \
  PAIRWISE_TRAIN_INPUT="${PAIRWISE_TRAIN_INPUT}" \
  PAIRWISE_VAL_INPUT="${PAIRWISE_VAL_INPUT}" \
  PAIRWISE_TEST_INPUT="${PAIRWISE_TEST_INPUT}" \
  METHOD_PAIRWISE_TRAIN_INPUT="${METHOD_PAIRWISE_TRAIN_INPUT}" \
  METHOD_PAIRWISE_VAL_INPUT="${METHOD_PAIRWISE_VAL_INPUT}" \
  METHOD_PAIRWISE_TEST_INPUT="${METHOD_PAIRWISE_TEST_INPUT}" \
  CONSTRAINT_PAIRWISE_TRAIN_INPUT="${CONSTRAINT_PAIRWISE_TRAIN_INPUT}" \
  CONSTRAINT_PAIRWISE_VAL_INPUT="${CONSTRAINT_PAIRWISE_VAL_INPUT}" \
  CONSTRAINT_PAIRWISE_TEST_INPUT="${CONSTRAINT_PAIRWISE_TEST_INPUT}" \
  USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS}" \
  REGENERATE_SPLITS="${REGENERATE_SPLITS}" \
  EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR}" \
  ASPECT_CONDITION_MODE="${ASPECT_CONDITION_MODE}" \
  MODEL_ID="${MODEL_ID}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash ce/train2.sh

TRAIN_RUN_DIR="${OUTPUT_DIR}"
if bool_true "${APPEND_ARGS_TO_OUTPUT_DIR}"; then
  LATEST_APPENDED_RUN="$(ls -td "${OUTPUT_DIR}"__* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${LATEST_APPENDED_RUN}" && -d "${LATEST_APPENDED_RUN}" ]]; then
    TRAIN_RUN_DIR="${LATEST_APPENDED_RUN}"
  fi
fi
if [[ ! -d "${TRAIN_RUN_DIR}" ]]; then
  echo "Training output directory not found: ${TRAIN_RUN_DIR}" >&2
  exit 1
fi

if bool_true "${RUN_EVAL}"; then
  EVAL_FINETUNED_MODEL="$(pick_eval_model_dir "${TRAIN_RUN_DIR}" "${EVAL_MODEL_PICK}")"
  log "Post-train eval with ce/eval/eval_finetuned_model.py"
  log "train_run_dir=${TRAIN_RUN_DIR}"
  log "eval_model_pick=${EVAL_MODEL_PICK} eval_finetuned_model=${EVAL_FINETUNED_MODEL}"

  EVAL_CMD=(
    "${PYTHON_BIN}" ce/eval/eval_finetuned_model.py
    --no-auto-resolve-finetuned
    --finetuned-model "${EVAL_FINETUNED_MODEL}"
    --aspect-condition-mode "${EVAL_ASPECT_CONDITION_MODE}"
    --base-model "${EVAL_BASE_MODEL}"
    --domain-input "${EVAL_DOMAIN_INPUT}"
    --method-input "${EVAL_METHOD_INPUT}"
    --constraint-input "${EVAL_CONSTRAINT_INPUT}"
    --score-field "${EVAL_SCORE_FIELD}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --max-length "${EVAL_MAX_LENGTH}"
    --high-threshold "${EVAL_HIGH_THRESHOLD}"
    --mid-threshold "${EVAL_MID_THRESHOLD}"
    --oob-margin "${EVAL_OOB_MARGIN}"
    --order-top-k "${EVAL_ORDER_TOP_K}"
    --pair-eps "${EVAL_PAIR_EPS}"
    --hard-gap-max "${EVAL_HARD_GAP_MAX}"
    --medium-gap-max "${EVAL_MEDIUM_GAP_MAX}"
    --output-dir "${EVAL_OUTPUT_DIR}"
    --save-prefix "${EVAL_SAVE_PREFIX}"
  )
  if bool_true "${EVAL_ONLY_SELECTED}"; then
    EVAL_CMD+=(--only-selected)
  else
    EVAL_CMD+=(--no-only-selected)
  fi
  if bool_true "${EVAL_INCLUDE_CONSTRAINT}"; then
    EVAL_CMD+=(--include-constraint)
  else
    EVAL_CMD+=(--no-include-constraint)
  fi
  if bool_true "${EVAL_SAVE}"; then
    EVAL_CMD+=(--save)
  else
    EVAL_CMD+=(--no-save)
  fi
  if bool_true "${EVAL_PRINT}"; then
    EVAL_CMD+=(--print)
  else
    EVAL_CMD+=(--no-print)
  fi

  log "Running: ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}"
fi

log "Done: converted CE2 splits -> ce/train2.sh -> eval."
