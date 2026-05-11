#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# llm_distill2(augment-only) -> split -> train
# ==========================================================
# Usage:
#   bash cross_encoder/run_llm_distill2_split_train.sh
#
# Optional: override any config below via env vars, e.g.
#   DISTILL_RAW_PATH=cross_encoder/dataset/distill/llm_distill2_distill_raw_scores.jsonl \
#   TARGET_HIGH=2 TARGET_MID=4 TARGET_LOW=2 \
#   bash cross_encoder/run_llm_distill2_split_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===========================
# Config
# ===========================
PYTHON_BIN="${PYTHON_BIN:-python}"

# Stage 1) Augment-only from existing distill raw file
DISTILL_RAW_PATH="${DISTILL_RAW_PATH:-cross_encoder/dataset/distill/llm_distill2_distill_raw_scores.jsonl}"
TARGET_HIGH="${TARGET_HIGH:-4}"
TARGET_MID="${TARGET_MID:-8}"
TARGET_LOW="${TARGET_LOW:-4}"

# Stage 2) Split
SPLIT_DIR="${SPLIT_DIR:-cross_encoder/dataset/splits}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"   # true | false

# Stage 3) Train
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP:-true}"                 # true | false
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE:-1}"  # epochs
STAGE2_START_FROM_BEST_STAGE1="${STAGE2_START_FROM_BEST_STAGE1:-true}"  # true | false
SEED="${SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-32}"
MINI_LIST_SIZE="${MINI_LIST_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-1.0}"
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.25}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.40}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS:-true}"             # true | false
REGENERATE_SPLITS="${REGENERATE_SPLITS:-false}"                # true | false
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR:-true}" # true | false
BF16="${BF16:-true}"   # true | false
FP16="${FP16:-false}"  # true | false
NO_WANDB="${NO_WANDB:-false}"   # true | false
NO_TQDM="${NO_TQDM:-false}"     # true | false
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
OUTPUT_DIR="${OUTPUT_DIR:-cross_encoder/models/bge_reranker_distill}"


RAW_INPUT="cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl"
PAIRWISE_INPUT="cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

SPLIT_OVERWRITE_FLAG="--overwrite"
if [[ "${SPLIT_OVERWRITE}" != "true" ]]; then
  SPLIT_OVERWRITE_FLAG="--no-overwrite"
fi

log "Stage 1/3: llm_distill2 (augment-only)"
"${PYTHON_BIN}" cross_encoder/data_preparation/llm_distillation/llm_distill2.py \
  --run-mode full \
  --distill-raw-path "${DISTILL_RAW_PATH}" \
  --target-high "${TARGET_HIGH}" \
  --target-mid "${TARGET_MID}" \
  --target-low "${TARGET_LOW}"

log "Stage 2/3: split_distill_dataset"
"${PYTHON_BIN}" cross_encoder/data_preparation/split_distill_dataset.py \
  --raw-input "${RAW_INPUT}" \
  --pairwise-input "${PAIRWISE_INPUT}" \
  --split-dir "${SPLIT_DIR}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --seed "${SPLIT_SEED}" \
  "${SPLIT_OVERWRITE_FLAG}"

log "Stage 3/3: train_bge_distill"
RAW_INPUT="${RAW_INPUT}" \
PAIRWISE_INPUT="${PAIRWISE_INPUT}" \
SPLIT_DIR="${SPLIT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_ID="${MODEL_ID}" \
SEED="${SEED}" \
STAGE1_EPOCHS="${STAGE1_EPOCHS}" \
STAGE2_EPOCHS="${STAGE2_EPOCHS}" \
STAGE1_EARLY_STOP="${STAGE1_EARLY_STOP}" \
STAGE1_EARLY_STOP_PATIENCE="${STAGE1_EARLY_STOP_PATIENCE}" \
STAGE2_START_FROM_BEST_STAGE1="${STAGE2_START_FROM_BEST_STAGE1}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
MAX_LENGTH="${MAX_LENGTH}" \
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE}" \
MINI_LIST_SIZE="${MINI_LIST_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
LEARNING_RATE="${LEARNING_RATE}" \
WEIGHT_DECAY="${WEIGHT_DECAY}" \
MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT}" \
LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT}" \
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT}" \
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE}" \
LOG_EVERY_STEPS="${LOG_EVERY_STEPS}" \
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \
USE_PREPARED_SPLITS="${USE_PREPARED_SPLITS}" \
REGENERATE_SPLITS="${REGENERATE_SPLITS}" \
APPEND_ARGS_TO_OUTPUT_DIR="${APPEND_ARGS_TO_OUTPUT_DIR}" \
BF16="${BF16}" \
FP16="${FP16}" \
NO_WANDB="${NO_WANDB}" \
NO_TQDM="${NO_TQDM}" \
bash cross_encoder/train.sh


log "Done: augment-only + split + train completed."
