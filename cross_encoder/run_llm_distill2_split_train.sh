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
TARGET_HIGH="${TARGET_HIGH:-2}"
TARGET_MID="${TARGET_MID:-4}"
TARGET_LOW="${TARGET_LOW:-2}"

# Stage 2) Split
SPLIT_DIR="${SPLIT_DIR:-cross_encoder/dataset/splits}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_OVERWRITE="${SPLIT_OVERWRITE:-true}"   # true | false

# Stage 3) Train
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
NO_WANDB="${NO_WANDB:-false}"   # true | false


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

log "Stage 1/3: llm_distill2 (augment-only)"
"${PYTHON_BIN}" cross_encoder/data_preparation/llm_distillation/llm_distill2.py \
  --run-mode augment-only \
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
"${PYTHON_BIN}" cross_encoder/train_bge_distill.py \
  --stage1-epochs "${STAGE1_EPOCHS}" \
  --stage2-epochs "${STAGE2_EPOCHS}" \
  ${WANDB_FLAG}


log "Done: augment-only + split + train completed."
