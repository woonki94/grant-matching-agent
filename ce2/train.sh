#!/usr/bin/env bash
set -euo pipefail

# CE2 training-only launcher.
# Expects split_distillation.py outputs in SPLIT_DIR:
#   {domain,method,target}_{train,val,test}.jsonl
#   {domain,method,target}_pairwise_{train,val,test}.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

PYTHON_BIN="${PYTHON_BIN:-python}"

SPLIT_DIR="${SPLIT_DIR:-ce2/dataset/splits}"
MODEL_DIR="${MODEL_DIR:-ce2/models/basic_distill}"
TRAIN_MODEL_ID="${TRAIN_MODEL_ID:-dleemiller/ModernCE-base-sts}"
TRAIN_ASPECTS="${TRAIN_ASPECTS:-domain,method,target}"
SEED="${SEED:-42}"

# Starting-point training config: use every split row and every split pairwise row.
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-256}"
TRAIN_STAGE1_EPOCHS="${TRAIN_STAGE1_EPOCHS:-1}"
TRAIN_STAGE2_EPOCHS="${TRAIN_STAGE2_EPOCHS:-3}"
TRAIN_STAGE1_LR="${TRAIN_STAGE1_LR:-2e-5}"
TRAIN_STAGE2_LR="${TRAIN_STAGE2_LR:-2e-5}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.06}"
TRAIN_MAX_GRAD_NORM="${TRAIN_MAX_GRAD_NORM:-1.0}"
TRAIN_PAIR_BATCH_SIZE="${TRAIN_PAIR_BATCH_SIZE:-64}"
TRAIN_LIST_BATCH_SIZE="${TRAIN_LIST_BATCH_SIZE:-8}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-64}"
TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-1}"
TRAIN_PAIRWISE_SOURCE="${TRAIN_PAIRWISE_SOURCE:-files}"

# These only affect fallback pair derivation. With TRAIN_PAIRWISE_SOURCE=files,
# train.py reads all generated pairwise files directly.
TRAIN_PAIRS_PER_QUERY="${TRAIN_PAIRS_PER_QUERY:-16}"
TRAIN_MIN_PAIR_DELTA="${TRAIN_MIN_PAIR_DELTA:-0.15}"
TRAIN_MIN_PAIR_MARGIN="${TRAIN_MIN_PAIR_MARGIN:-0.05}"
TRAIN_MAX_PAIR_MARGIN="${TRAIN_MAX_PAIR_MARGIN:-0.75}"
TRAIN_PAIR_MARGIN_SCALE="${TRAIN_PAIR_MARGIN_SCALE:-1.0}"

TRAIN_TEACHER_TEMPERATURE="${TRAIN_TEACHER_TEMPERATURE:-1.0}"
TRAIN_LOSS_KL_WEIGHT="${TRAIN_LOSS_KL_WEIGHT:-1.0}"
TRAIN_LOSS_PAIR_WEIGHT="${TRAIN_LOSS_PAIR_WEIGHT:-0.5}"
TRAIN_LOSS_MSE_WEIGHT="${TRAIN_LOSS_MSE_WEIGHT:-0.05}"
TRAIN_USE_BF16="${TRAIN_USE_BF16:-true}"
TRAIN_NO_TQDM="${TRAIN_NO_TQDM:-false}"

TRAIN_WANDB_PROJECT="${TRAIN_WANDB_PROJECT:-ce2_distill}"
TRAIN_WANDB_ENTITY="${TRAIN_WANDB_ENTITY:-}"
TRAIN_WANDB_RUN_NAME="${TRAIN_WANDB_RUN_NAME:-ce2_basic_distill}"
TRAIN_WANDB_MODE="${TRAIN_WANDB_MODE:-online}"
TRAIN_WANDB_TAGS="${TRAIN_WANDB_TAGS:-ce2,basic_distill,all_listwise,all_pairwise}"

IFS=',' read -r -a ASPECT_ARRAY <<< "${TRAIN_ASPECTS}"
for aspect in "${ASPECT_ARRAY[@]}"; do
  aspect="$(echo "${aspect}" | xargs)"
  [[ -z "${aspect}" ]] && continue
  for split in train val test; do
    list_file="${SPLIT_DIR}/${aspect}_${split}.jsonl"
    pair_file="${SPLIT_DIR}/${aspect}_pairwise_${split}.jsonl"
    if [[ ! -f "${list_file}" ]]; then
      log "Missing listwise split file: ${list_file}"
      exit 1
    fi
    if [[ ! -f "${pair_file}" ]]; then
      log "Missing pairwise split file: ${pair_file}"
      exit 1
    fi
  done
done

log "CE2 train"
log "split_dir=${SPLIT_DIR}"
log "model_dir=${MODEL_DIR}"
log "model_id=${TRAIN_MODEL_ID}"
log "aspects=${TRAIN_ASPECTS}"
log "pairwise_source=${TRAIN_PAIRWISE_SOURCE}"
log "stage1_epochs=${TRAIN_STAGE1_EPOCHS}"
log "stage2_epochs=${TRAIN_STAGE2_EPOCHS}"
log "pair_batch_size=${TRAIN_PAIR_BATCH_SIZE}"
log "list_batch_size=${TRAIN_LIST_BATCH_SIZE}"
log "eval_batch_size=${TRAIN_EVAL_BATCH_SIZE}"
log "wandb_project=${TRAIN_WANDB_PROJECT}"
log "wandb_mode=${TRAIN_WANDB_MODE}"

CMD=(
  "${PYTHON_BIN}" ce2/train.py
  --model-id "${TRAIN_MODEL_ID}"
  --split-dir "${SPLIT_DIR}"
  --output-dir "${MODEL_DIR}"
  --aspects "${TRAIN_ASPECTS}"
  --max-length "${TRAIN_MAX_LENGTH}"
  --seed "${SEED}"
  --stage1-epochs "${TRAIN_STAGE1_EPOCHS}"
  --stage2-epochs "${TRAIN_STAGE2_EPOCHS}"
  --stage1-learning-rate "${TRAIN_STAGE1_LR}"
  --stage2-learning-rate "${TRAIN_STAGE2_LR}"
  --weight-decay "${TRAIN_WEIGHT_DECAY}"
  --warmup-ratio "${TRAIN_WARMUP_RATIO}"
  --max-grad-norm "${TRAIN_MAX_GRAD_NORM}"
  --pair-batch-size "${TRAIN_PAIR_BATCH_SIZE}"
  --list-batch-size "${TRAIN_LIST_BATCH_SIZE}"
  --eval-batch-size "${TRAIN_EVAL_BATCH_SIZE}"
  --grad-accum-steps "${TRAIN_GRAD_ACCUM_STEPS}"
  --pairwise-source "${TRAIN_PAIRWISE_SOURCE}"
  --pairs-per-query "${TRAIN_PAIRS_PER_QUERY}"
  --min-pair-delta "${TRAIN_MIN_PAIR_DELTA}"
  --min-pair-margin "${TRAIN_MIN_PAIR_MARGIN}"
  --max-pair-margin "${TRAIN_MAX_PAIR_MARGIN}"
  --pair-margin-scale "${TRAIN_PAIR_MARGIN_SCALE}"
  --teacher-temperature "${TRAIN_TEACHER_TEMPERATURE}"
  --loss-kl-weight "${TRAIN_LOSS_KL_WEIGHT}"
  --loss-pair-weight "${TRAIN_LOSS_PAIR_WEIGHT}"
  --loss-mse-weight "${TRAIN_LOSS_MSE_WEIGHT}"
  --wandb-project "${TRAIN_WANDB_PROJECT}"
  --wandb-run-name "${TRAIN_WANDB_RUN_NAME}"
  --wandb-mode "${TRAIN_WANDB_MODE}"
  --wandb-tags "${TRAIN_WANDB_TAGS}"
)

if [[ -n "${TRAIN_WANDB_ENTITY}" ]]; then
  CMD+=(--wandb-entity "${TRAIN_WANDB_ENTITY}")
fi
if [[ "${TRAIN_USE_BF16}" == "true" ]]; then
  CMD+=(--use-bf16)
fi
if [[ "${TRAIN_NO_TQDM}" == "true" ]]; then
  CMD+=(--no-tqdm)
fi

log "Running: ${CMD[*]}"
"${CMD[@]}"
