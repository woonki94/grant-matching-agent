#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# CE pipeline: distill(+augment) for domain+method -> split -> train
# ==========================================================
# Usage:
#   bash ce/run_distill_split_train.sh
#
# Optional overrides:
#   TARGET_HIGH=3 TARGET_MID=6 TARGET_LOW=3 \
#   MODEL_ID=dleemiller/ModernCE-base-sts \
#   WANDB_MODE=offline \
#   bash ce/run_distill_split_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------
# Stage 1) Distill + Augment (both aspects)
# ---------------------------
TARGET_HIGH="${TARGET_HIGH:-3}"
TARGET_MID="${TARGET_MID:-6}"
TARGET_LOW="${TARGET_LOW:-3}"
PREFILTER_MULTIPLIER="${PREFILTER_MULTIPLIER:-10}"
MAX_SPECS="${MAX_SPECS:-0}"   # 0 = all

DOMAIN_ASPECT="${DOMAIN_ASPECT:-domain}"
METHOD_ASPECT="${METHOD_ASPECT:-method}"

# ---------------------------
# Stage 2) Shared split
# ---------------------------
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-ce/dataset/splits}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.05}"
TEST_RATIO="${TEST_RATIO:-0.05}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"   # true | false

# ---------------------------
# Stage 3) Train
# ---------------------------
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
SCORE_FIELD="${SCORE_FIELD:-teacher_score_raw}"   # teacher_score_raw | teacher_score
ONLY_SELECTED="${ONLY_SELECTED:-false}"           # true | false

MAX_LENGTH="${MAX_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
USE_BF16="${USE_BF16:-false}"                    # true | false
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-ce/models/mse_domain_method}"

WANDB_PROJECT="${WANDB_PROJECT:-ce_mse_distill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"               # online | offline | disabled

DOMAIN_TRAIN_FILE="${DOMAIN_TRAIN_FILE:-${SPLIT_OUTPUT_DIR}/llm_distill_domain_listwise_train.jsonl}"
DOMAIN_VAL_FILE="${DOMAIN_VAL_FILE:-${SPLIT_OUTPUT_DIR}/llm_distill_domain_listwise_val.jsonl}"
METHOD_TRAIN_FILE="${METHOD_TRAIN_FILE:-${SPLIT_OUTPUT_DIR}/llm_distill_method_listwise_train.jsonl}"
METHOD_VAL_FILE="${METHOD_VAL_FILE:-${SPLIT_OUTPUT_DIR}/llm_distill_method_listwise_val.jsonl}"

log "Stage 1/3: Distill+augment (${DOMAIN_ASPECT}) with target ${TARGET_HIGH}/${TARGET_MID}/${TARGET_LOW}"
"${PYTHON_BIN}" ce/data_preparation/llm_distillation/llm_distillation.py \
  --run-mode full \
  --judge-aspect "${DOMAIN_ASPECT}" \
  --target-high "${TARGET_HIGH}" \
  --target-mid "${TARGET_MID}" \
  --target-low "${TARGET_LOW}" \
  --prefilter-multiplier "${PREFILTER_MULTIPLIER}" \
  --max-specs "${MAX_SPECS}"

log "Stage 1/3: Distill+augment (${METHOD_ASPECT}) with target ${TARGET_HIGH}/${TARGET_MID}/${TARGET_LOW}"
"${PYTHON_BIN}" ce/data_preparation/llm_distillation/llm_distillation.py \
  --run-mode full \
  --judge-aspect "${METHOD_ASPECT}" \
  --target-high "${TARGET_HIGH}" \
  --target-mid "${TARGET_MID}" \
  --target-low "${TARGET_LOW}" \
  --prefilter-multiplier "${PREFILTER_MULTIPLIER}" \
  --max-specs "${MAX_SPECS}"

log "Stage 2/3: Shared query split for domain+method listwise"
SPLIT_ARGS=(
  --domain-input "ce/dataset/distill/llm_distill_domain_listwise.jsonl"
  --method-input "ce/dataset/distill/llm_distill_method_listwise.jsonl"
  --output-dir "${SPLIT_OUTPUT_DIR}"
  --seed "${SPLIT_SEED}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
)
if [[ "${SPLIT_OVERWRITE}" == "true" ]]; then
  SPLIT_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" ce/data_preparation/split_distill_listwise_domain_method.py "${SPLIT_ARGS[@]}"

log "Stage 3/3: Train MSE CE model"
TRAIN_ARGS=(
  --model-id "${MODEL_ID}"
  --domain-listwise "${DOMAIN_TRAIN_FILE}"
  --method-listwise "${METHOD_TRAIN_FILE}"
  --domain-val-listwise "${DOMAIN_VAL_FILE}"
  --method-val-listwise "${METHOD_VAL_FILE}"
  --score-field "${SCORE_FIELD}"
  --max-length "${MAX_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --seed "${TRAIN_SEED}"
  --output-dir "${TRAIN_OUTPUT_DIR}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
)
if [[ "${ONLY_SELECTED}" == "true" ]]; then
  TRAIN_ARGS+=(--only-selected)
fi
if [[ "${USE_BF16}" == "true" ]]; then
  TRAIN_ARGS+=(--use-bf16)
fi
"${PYTHON_BIN}" ce/train.py "${TRAIN_ARGS[@]}"

log "Done: domain+method distill/augment -> split -> train completed."
