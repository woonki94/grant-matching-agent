#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Train cross-encoder from distilled dataset (safe defaults)
# ==========================================================
# Usage:
#   bash cross_encoder/train.sh
#
# Optional overrides (example):
#   TRAIN_BATCH_SIZE=1 MINI_LIST_SIZE=6 LOSS_MSE_WEIGHT=0.35 \
#   bash cross_encoder/train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===========================
# Config (env overrideable)
# ===========================
PYTHON_BIN="${PYTHON_BIN:-python}"

# Inputs / outputs
RAW_INPUT="${RAW_INPUT:-cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl}"
PAIRWISE_INPUT="${PAIRWISE_INPUT:-cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-cross_encoder/dataset/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-cross_encoder/models/bge_reranker_distill}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"

# Train schedule
SEED="${SEED:-42}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"

# Memory-safe knobs
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

# Loss calibration knobs (to improve absolute score behavior)
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-1.0}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.30}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.30}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.2}"

# Misc
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"

USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"       # true | false
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"          # true | false
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-true}"  # true | false
BF16="${BF16:-true}"                                     # true | false
FP16="${FP16:-false}"                                    # true | false
NO_WANDB="${NO_WANDB:-false}"                            # true | false
NO_TQDM="${NO_TQDM:-false}"                              # true | false

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

BOOL_TRUE() {
  [[ "$1" == "true" ]]
}

log "Starting training run"
log "raw_input=${RAW_INPUT}"
log "pairwise_input=${PAIRWISE_INPUT}"
log "split_dir=${SPLIT_DIR}"
log "output_dir=${OUTPUT_DIR}"
log "model_id=${MODEL_ID}"
log "stage1_epochs=${STAGE1_EPOCHS} stage2_epochs=${STAGE2_EPOCHS}"
log "train_batch_size=${TRAIN_BATCH_SIZE} eval_batch_size=${EVAL_BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS}"
log "max_length=${MAX_LENGTH} candidate_pool_size=${CANDIDATE_POOL_SIZE} mini_list_size=${MINI_LIST_SIZE}"
log "loss_kl=${LOSS_KL_WEIGHT} loss_pair=${LOSS_PAIR_WEIGHT} loss_mse=${LOSS_MSE_WEIGHT} teacher_temperature=${TEACHER_TEMPERATURE}"

CMD=(
  "${PYTHON_BIN}" cross_encoder/train_bge_distill.py
  --raw-input "${RAW_INPUT}"
  --pairwise-input "${PAIRWISE_INPUT}"
  --split-dir "${SPLIT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${MODEL_ID}"
  --seed "${SEED}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --max-length "${MAX_LENGTH}"
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
  --mini-list-size "${MINI_LIST_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --loss-kl-weight "${LOSS_KL_WEIGHT}"
  --loss-pair-weight "${LOSS_PAIR_WEIGHT}"
  --loss-mse-weight "${LOSS_MSE_WEIGHT}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --log-every-steps "${LOG_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
)

if BOOL_TRUE "${USE_PREPARED_SPLITS}"; then
  CMD+=(--use-prepared-splits)
else
  CMD+=(--no-use-prepared-splits)
fi

if BOOL_TRUE "${REGENERATE_SPLITS}"; then
  CMD+=(--regenerate-splits)
else
  CMD+=(--no-regenerate-splits)
fi

if BOOL_TRUE "${APPEND_ARGS_TO_OUTPUT_DIR}"; then
  CMD+=(--append-args-to-output-dir)
else
  CMD+=(--no-append-args-to-output-dir)
fi

if BOOL_TRUE "${BF16}"; then
  CMD+=(--bf16)
elif BOOL_TRUE "${FP16}"; then
  CMD+=(--fp16)
fi

if BOOL_TRUE "${NO_WANDB}"; then
  CMD+=(--no-wandb)
fi

if BOOL_TRUE "${NO_TQDM}"; then
  CMD+=(--no-tqdm)
fi

"${CMD[@]}"

log "Done: training completed."
