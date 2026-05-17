#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# CE train-only runner (domain + method, MSE)
# ==========================================================
# Usage:
#   bash ce/train.sh
#
# Optional overrides:
#   MODEL_ID=dleemiller/ModernCE-base-sts EPOCHS=5 LR=1e-5 \
#   WANDB_MODE=offline OUTPUT_DIR=ce/models/mse_domain_method_exp1 \
#   bash ce/train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
bool_true() { [[ "$1" == "true" ]]; }

PYTHON_BIN="${PYTHON_BIN:-python}"

# Inputs
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
DOMAIN_LISTWISE="${DOMAIN_LISTWISE:-ce/dataset/splits/llm_distill_domain_listwise_train.jsonl}"
METHOD_LISTWISE="${METHOD_LISTWISE:-ce/dataset/splits/llm_distill_method_listwise_train.jsonl}"
DOMAIN_VAL_LISTWISE="${DOMAIN_VAL_LISTWISE:-ce/dataset/splits/llm_distill_domain_listwise_val.jsonl}"
METHOD_VAL_LISTWISE="${METHOD_VAL_LISTWISE:-ce/dataset/splits/llm_distill_method_listwise_val.jsonl}"
SCORE_FIELD="${SCORE_FIELD:-teacher_score_raw}"    # teacher_score_raw | teacher_score
ONLY_SELECTED="${ONLY_SELECTED:-false}"            # true | false
VAL_RATIO="${VAL_RATIO:-0.05}"                     # used only if val files are missing

# Optimization
MAX_LENGTH="${MAX_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-42}"
USE_BF16="${USE_BF16:-false}"                      # true | false
PREDICTION_SPACE="${PREDICTION_SPACE:-sigmoid}"    # sigmoid | logit
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.70}"
MID_THRESHOLD="${MID_THRESHOLD:-0.30}"
LOSS_HIGH_WEIGHT="${LOSS_HIGH_WEIGHT:-1.80}"
LOSS_MID_WEIGHT="${LOSS_MID_WEIGHT:-1.00}"
LOSS_LOW_WEIGHT="${LOSS_LOW_WEIGHT:-1.60}"

# Output / logging
OUTPUT_DIR="${OUTPUT_DIR:-ce/models/mse_domain_method}"
WANDB_PROJECT="${WANDB_PROJECT:-ce_mse_distill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"                 # online | offline | disabled

log "Starting CE train-only run"
log "model_id=${MODEL_ID}"
log "domain_train=${DOMAIN_LISTWISE}"
log "method_train=${METHOD_LISTWISE}"
log "domain_val=${DOMAIN_VAL_LISTWISE}"
log "method_val=${METHOD_VAL_LISTWISE}"
log "score_field=${SCORE_FIELD} only_selected=${ONLY_SELECTED} val_ratio=${VAL_RATIO}"
log "epochs=${EPOCHS} batch=${BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM}"
log "lr=${LR} wd=${WEIGHT_DECAY} warmup_ratio=${WARMUP_RATIO} max_grad_norm=${MAX_GRAD_NORM}"
log "max_length=${MAX_LENGTH} use_bf16=${USE_BF16} seed=${SEED}"
log "prediction_space=${PREDICTION_SPACE} thresholds=${HIGH_THRESHOLD}/${MID_THRESHOLD} loss_weights=${LOSS_HIGH_WEIGHT}/${LOSS_MID_WEIGHT}/${LOSS_LOW_WEIGHT}"
log "output_dir=${OUTPUT_DIR} wandb_mode=${WANDB_MODE} wandb_project=${WANDB_PROJECT}"

CMD=(
  "${PYTHON_BIN}" ce/train.py
  --model-id "${MODEL_ID}"
  --domain-listwise "${DOMAIN_LISTWISE}"
  --method-listwise "${METHOD_LISTWISE}"
  --domain-val-listwise "${DOMAIN_VAL_LISTWISE}"
  --method-val-listwise "${METHOD_VAL_LISTWISE}"
  --score-field "${SCORE_FIELD}"
  --val-ratio "${VAL_RATIO}"
  --max-length "${MAX_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --seed "${SEED}"
  --prediction-space "${PREDICTION_SPACE}"
  --high-threshold "${HIGH_THRESHOLD}"
  --mid-threshold "${MID_THRESHOLD}"
  --loss-high-weight "${LOSS_HIGH_WEIGHT}"
  --loss-mid-weight "${LOSS_MID_WEIGHT}"
  --loss-low-weight "${LOSS_LOW_WEIGHT}"
  --output-dir "${OUTPUT_DIR}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
)

if bool_true "${ONLY_SELECTED}"; then
  CMD+=(--only-selected)
fi
if bool_true "${USE_BF16}"; then
  CMD+=(--use-bf16)
fi

log "Running: ${CMD[*]}"
"${CMD[@]}"

log "Training done."
log "summary=${OUTPUT_DIR}/train_summary.json"
log "best=${OUTPUT_DIR}/best"
log "final=${OUTPUT_DIR}/final"
