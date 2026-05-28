#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/models/aspect_reranker}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-0}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-0}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TRAIN_LOG_EVERY_STEPS="${TRAIN_LOG_EVERY_STEPS:-1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"

LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.5}"
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-1.0}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-0.2}"
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-0.1}"
LOSS_CALIBRATION_WEIGHT="${LOSS_CALIBRATION_WEIGHT:-0.1}"
PAIR_TYPE_WEIGHT_MAP="${PAIR_TYPE_WEIGHT_MAP:-default=1.0,llm_disagreement=1.15,strong_vs_boundary=1.05,strong_vs_weak=0.95,strong_vs_hard=1.0}"
FP16="${FP16:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-ce3_distill}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-ce3,aspect-conditioned,multihead}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHDYNAMO_DISABLE

CMD=(
  "${PYTHON_BIN}" ce3/train.py
  --model-id "${MODEL_ID}"
  --split-dir "${SPLIT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --stage1-learning-rate "${STAGE1_LEARNING_RATE}"
  --stage2-learning-rate "${STAGE2_LEARNING_RATE}"
  --max-length "${MAX_LENGTH}"
  --train-log-every-steps "${TRAIN_LOG_EVERY_STEPS}"
  --eval-every-steps "${EVAL_EVERY_STEPS}"
  --loss-pair-weight "${LOSS_PAIR_WEIGHT}"
  --loss-kl-weight "${LOSS_KL_WEIGHT}"
  --loss-mse-weight "${LOSS_MSE_WEIGHT}"
  --loss-cluster-margin-weight "${LOSS_CLUSTER_MARGIN_WEIGHT}"
  --loss-calibration-weight "${LOSS_CALIBRATION_WEIGHT}"
  --pair-type-weight-map "${PAIR_TYPE_WEIGHT_MAP}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-tags "${WANDB_TAGS}"
)

if [[ "${FP16}" == "true" ]]; then
  CMD+=(--fp16)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
