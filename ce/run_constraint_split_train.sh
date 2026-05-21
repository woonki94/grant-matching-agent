#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# CE pipeline: constraint distill only -> shared split -> train all aspects
# ==========================================================
# Usage:
#   bash ce/run_constraint_split_train.sh
#
# This keeps existing domain/method distill files as-is, generates only the
# constraint aspect, then rebuilds a shared domain+method+constraint split
# before launching ce/train2.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
bool_true() { [[ "$1" == "true" ]]; }

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------
# Stage switches
# ---------------------------
RUN_CONSTRAINT_DISTILL="${RUN_CONSTRAINT_DISTILL:-true}"  # true | false
RUN_SPLIT="${RUN_SPLIT:-true}"                            # true | false
RUN_TRAIN="${RUN_TRAIN:-true}"                            # true | false

# ---------------------------
# Existing domain/method inputs
# ---------------------------
RAW_INPUT="${RAW_INPUT:-ce/dataset/distill/llm_distill_domain_listwise.jsonl}"
METHOD_RAW_INPUT="${METHOD_RAW_INPUT:-ce/dataset/distill/llm_distill_method_listwise.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_domain_pairwise.jsonl}"
METHOD_PAIRWISE_INPUT="${METHOD_PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_method_pairwise.jsonl}"

# ---------------------------
# Constraint distill outputs
# ---------------------------
CONSTRAINT_ASPECT="${CONSTRAINT_ASPECT:-constraint}"
CONSTRAINT_RAW_INPUT="${CONSTRAINT_RAW_INPUT:-ce/dataset/distill/llm_distill_constraint_listwise.jsonl}"
CONSTRAINT_PAIRWISE_INPUT="${CONSTRAINT_PAIRWISE_INPUT:-ce/dataset/distill/llm_distill_constraint_pairwise.jsonl}"

TARGET_HIGH="${TARGET_HIGH:-4}"
TARGET_MID="${TARGET_MID:-8}"
TARGET_LOW="${TARGET_LOW:-4}"
CONSTRAINT_TARGET_HIGH="${CONSTRAINT_TARGET_HIGH:-${TARGET_HIGH}}"
CONSTRAINT_TARGET_MID="${CONSTRAINT_TARGET_MID:-${TARGET_MID}}"
CONSTRAINT_TARGET_LOW="${CONSTRAINT_TARGET_LOW:-${TARGET_LOW}}"

PREFILTER_MULTIPLIER="${PREFILTER_MULTIPLIER:-10}"
PREFILTER_MULTIPLIER_HIGH="${PREFILTER_MULTIPLIER_HIGH:-17}"
PREFILTER_MULTIPLIER_MID="${PREFILTER_MULTIPLIER_MID:-10}"
PREFILTER_MULTIPLIER_LOW="${PREFILTER_MULTIPLIER_LOW:-3}"
CONSTRAINT_PREFILTER_MULTIPLIER="${CONSTRAINT_PREFILTER_MULTIPLIER:-${PREFILTER_MULTIPLIER}}"
CONSTRAINT_PREFILTER_MULTIPLIER_HIGH="${CONSTRAINT_PREFILTER_MULTIPLIER_HIGH:-${PREFILTER_MULTIPLIER_HIGH}}"
CONSTRAINT_PREFILTER_MULTIPLIER_MID="${CONSTRAINT_PREFILTER_MULTIPLIER_MID:-${PREFILTER_MULTIPLIER_MID}}"
CONSTRAINT_PREFILTER_MULTIPLIER_LOW="${CONSTRAINT_PREFILTER_MULTIPLIER_LOW:-${PREFILTER_MULTIPLIER_LOW}}"
MAX_SPECS="${MAX_SPECS:-0}"  # 0 = all

# ---------------------------
# Shared split outputs
# ---------------------------
SPLIT_DIR="${SPLIT_DIR:-ce/dataset/splits}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"  # true | false

RAW_TRAIN_INPUT="${RAW_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_train.jsonl}"
RAW_VAL_INPUT="${RAW_VAL_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_val.jsonl}"
RAW_TEST_INPUT="${RAW_TEST_INPUT:-${SPLIT_DIR}/llm_distill_domain_listwise_test.jsonl}"
METHOD_RAW_TRAIN_INPUT="${METHOD_RAW_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_train.jsonl}"
METHOD_RAW_VAL_INPUT="${METHOD_RAW_VAL_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_val.jsonl}"
METHOD_RAW_TEST_INPUT="${METHOD_RAW_TEST_INPUT:-${SPLIT_DIR}/llm_distill_method_listwise_test.jsonl}"
CONSTRAINT_RAW_TRAIN_INPUT="${CONSTRAINT_RAW_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_constraint_listwise_train.jsonl}"
CONSTRAINT_RAW_VAL_INPUT="${CONSTRAINT_RAW_VAL_INPUT:-${SPLIT_DIR}/llm_distill_constraint_listwise_val.jsonl}"
CONSTRAINT_RAW_TEST_INPUT="${CONSTRAINT_RAW_TEST_INPUT:-${SPLIT_DIR}/llm_distill_constraint_listwise_test.jsonl}"

PAIRWISE_TRAIN_INPUT="${PAIRWISE_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_train.jsonl}"
PAIRWISE_VAL_INPUT="${PAIRWISE_VAL_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_val.jsonl}"
PAIRWISE_TEST_INPUT="${PAIRWISE_TEST_INPUT:-${SPLIT_DIR}/llm_distill_domain_pairwise_test.jsonl}"
METHOD_PAIRWISE_TRAIN_INPUT="${METHOD_PAIRWISE_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_train.jsonl}"
METHOD_PAIRWISE_VAL_INPUT="${METHOD_PAIRWISE_VAL_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_val.jsonl}"
METHOD_PAIRWISE_TEST_INPUT="${METHOD_PAIRWISE_TEST_INPUT:-${SPLIT_DIR}/llm_distill_method_pairwise_test.jsonl}"
CONSTRAINT_PAIRWISE_TRAIN_INPUT="${CONSTRAINT_PAIRWISE_TRAIN_INPUT:-${SPLIT_DIR}/llm_distill_constraint_pairwise_train.jsonl}"
CONSTRAINT_PAIRWISE_VAL_INPUT="${CONSTRAINT_PAIRWISE_VAL_INPUT:-${SPLIT_DIR}/llm_distill_constraint_pairwise_val.jsonl}"
CONSTRAINT_PAIRWISE_TEST_INPUT="${CONSTRAINT_PAIRWISE_TEST_INPUT:-${SPLIT_DIR}/llm_distill_constraint_pairwise_test.jsonl}"

# Defaults for train2.sh if the caller did not override them.
WANDB_TAGS="${WANDB_TAGS:-ce,domain,method,constraint,constraint-refresh}"

if bool_true "${RUN_CONSTRAINT_DISTILL}"; then
  log "Stage 1/3: Distill+augment (${CONSTRAINT_ASPECT}) with target ${CONSTRAINT_TARGET_HIGH}/${CONSTRAINT_TARGET_MID}/${CONSTRAINT_TARGET_LOW}"
  "${PYTHON_BIN}" ce/data_preparation/llm_distillation/llm_distillation.py \
    --run-mode full \
    --judge-aspect "${CONSTRAINT_ASPECT}" \
    --target-high "${CONSTRAINT_TARGET_HIGH}" \
    --target-mid "${CONSTRAINT_TARGET_MID}" \
    --target-low "${CONSTRAINT_TARGET_LOW}" \
    --prefilter-multiplier "${CONSTRAINT_PREFILTER_MULTIPLIER}" \
    --prefilter-multiplier-high "${CONSTRAINT_PREFILTER_MULTIPLIER_HIGH}" \
    --prefilter-multiplier-mid "${CONSTRAINT_PREFILTER_MULTIPLIER_MID}" \
    --prefilter-multiplier-low "${CONSTRAINT_PREFILTER_MULTIPLIER_LOW}" \
    --max-specs "${MAX_SPECS}"
else
  log "Stage 1/3: Skipping constraint distill (RUN_CONSTRAINT_DISTILL=false)"
fi

if bool_true "${RUN_SPLIT}"; then
  log "Stage 2/3: Shared query split for domain+method+constraint"
  SPLIT_ARGS=(
    --domain-input "${RAW_INPUT}"
    --method-input "${METHOD_RAW_INPUT}"
    --constraint-input "${CONSTRAINT_RAW_INPUT}"
    --domain-pairwise-input "${PAIRWISE_INPUT}"
    --method-pairwise-input "${METHOD_PAIRWISE_INPUT}"
    --constraint-pairwise-input "${CONSTRAINT_PAIRWISE_INPUT}"
    --output-dir "${SPLIT_DIR}"
    --seed "${SPLIT_SEED}"
    --val-ratio "${VAL_RATIO}"
    --test-ratio "${TEST_RATIO}"
  )
  if bool_true "${SPLIT_OVERWRITE}"; then
    SPLIT_ARGS+=(--overwrite)
  fi
  "${PYTHON_BIN}" ce/data_preparation/split_distill_listwise_domain_method.py "${SPLIT_ARGS[@]}"
else
  log "Stage 2/3: Skipping split (RUN_SPLIT=false)"
fi

if bool_true "${RUN_TRAIN}"; then
  log "Stage 3/3: Train all three aspects with ce/train2.sh"
  export PYTHON_BIN
  export RAW_INPUT METHOD_RAW_INPUT CONSTRAINT_RAW_INPUT
  export PAIRWISE_INPUT METHOD_PAIRWISE_INPUT CONSTRAINT_PAIRWISE_INPUT
  export SPLIT_DIR
  export RAW_TRAIN_INPUT RAW_VAL_INPUT RAW_TEST_INPUT
  export METHOD_RAW_TRAIN_INPUT METHOD_RAW_VAL_INPUT METHOD_RAW_TEST_INPUT
  export CONSTRAINT_RAW_TRAIN_INPUT CONSTRAINT_RAW_VAL_INPUT CONSTRAINT_RAW_TEST_INPUT
  export PAIRWISE_TRAIN_INPUT PAIRWISE_VAL_INPUT PAIRWISE_TEST_INPUT
  export METHOD_PAIRWISE_TRAIN_INPUT METHOD_PAIRWISE_VAL_INPUT METHOD_PAIRWISE_TEST_INPUT
  export CONSTRAINT_PAIRWISE_TRAIN_INPUT CONSTRAINT_PAIRWISE_VAL_INPUT CONSTRAINT_PAIRWISE_TEST_INPUT
  export WANDB_TAGS
  bash ce/train2.sh
else
  log "Stage 3/3: Skipping train (RUN_TRAIN=false)"
fi

log "Done: constraint distill -> shared split -> three-aspect training."
