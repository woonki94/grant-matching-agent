#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# llm_distill2 -> split -> train
# ==========================================================
# Usage:
#   bash cross_encoder/run_llm_distill2_split_train.sh
#
# Optional: override any config below via env vars, e.g.
#   MAX_SPECS=200 TARGET_HIGH=5 TARGET_MID=10 TARGET_LOW=5 \
#   bash cross_encoder/run_llm_distill2_split_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ===========================
# Config
# ===========================
PYTHON_BIN="${PYTHON_BIN:-python}"

# Stage 1) Distill
PREFILTER_MULTIPLIER="${PREFILTER_MULTIPLIER:-20}"
MAX_SPECS="${MAX_SPECS:-100}"
TARGET_HIGH="${TARGET_HIGH:-4}"
TARGET_MID="${TARGET_MID:-4}"
TARGET_LOW="${TARGET_LOW:-4}"

# Stage 2) Split
SPLIT_DIR="${SPLIT_DIR:-cross_encoder/dataset/splits}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"   # true | false

# Stage 3) Train
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
OUTPUT_DIR="${OUTPUT_DIR:-cross_encoder/models/bge_reranker_distill_llm2}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NO_WANDB="${NO_WANDB:-true}"                  # true | false

RAW_INPUT="cross_encoder/dataset/distill/llm_distill2_raw_scores.jsonl"
PAIRWISE_INPUT="cross_encoder/dataset/distill/llm_distill2_pairwise.jsonl"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

SPLIT_OVERWRITE_FLAG="--overwrite"
if [[ "${SPLIT_OVERWRITE}" != "true" ]]; then
  SPLIT_OVERWRITE_FLAG="--no-overwrite"
fi

WANDB_FLAG=""
if [[ "${NO_WANDB}" == "true" ]]; then
  WANDB_FLAG="--no-wandb"
fi

log "Stage 1/3: llm_distill2"
"${PYTHON_BIN}" cross_encoder/data_preparation/llm_distillation/llm_distill2.py \
  --prefilter-multiplier "${PREFILTER_MULTIPLIER}" \
  --max-specs "${MAX_SPECS}" \
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
"${PYTHON_BIN}" cross_encoder/train_bge_distill.py \
  --raw-input "${RAW_INPUT}" \
  --pairwise-input "${PAIRWISE_INPUT}" \
  --split-dir "${SPLIT_DIR}" \
  --use-prepared-splits \
  --model-id "${MODEL_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage1-epochs "${STAGE1_EPOCHS}" \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  ${WANDB_FLAG}

log "Done: distill + split + train completed."
