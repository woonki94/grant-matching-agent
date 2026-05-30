#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-${PROJECT_ROOT}/ce3/models/aspect_reranker/stage1_epoch_1}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/models/aspect_reranker}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE2_FREEZE_BACKBONE="${STAGE2_FREEZE_BACKBONE:-true}"
STAGE2_TRAIN_LAST_LAYERS="${STAGE2_TRAIN_LAST_LAYERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-0}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TRAIN_LOG_EVERY_STEPS="${TRAIN_LOG_EVERY_STEPS:-1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"

LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.05}"
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.1}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-1.0}"
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-0.4}"
LOSS_CALIBRATION_WEIGHT="${LOSS_CALIBRATION_WEIGHT:-1.0}"
LOSS_ORDINAL_WEIGHT="${LOSS_ORDINAL_WEIGHT:-0.4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.8}"
CLUSTER_MARGIN_HM="${CLUSTER_MARGIN_HM:-0.18}"
CLUSTER_MARGIN_ML="${CLUSTER_MARGIN_ML:-0.18}"
CLUSTER_MARGIN_HL="${CLUSTER_MARGIN_HL:-0.45}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.70}"
MID_THRESHOLD="${MID_THRESHOLD:-0.30}"
MARGIN_MIN="${MARGIN_MIN:-0.02}"
MARGIN_MAX="${MARGIN_MAX:-0.60}"
PAIR_TYPE_WEIGHT_MAP="${PAIR_TYPE_WEIGHT_MAP:-default=1.0,llm_disagreement=1.15,strong_vs_boundary=1.05,strong_vs_weak=0.95,strong_vs_hard=1.0}"
STAGE1_PAIR_PRESET="${STAGE1_PAIR_PRESET:-balanced}"
case "${STAGE1_PAIR_PRESET}" in
  balanced)
    STAGE1_PAIR_TYPES="${STAGE1_PAIR_TYPES:-llm_disagreement,strong_vs_hard,strong_vs_weak,strong_vs_boundary}"
    STAGE1_PAIR_MIN_MARGIN="${STAGE1_PAIR_MIN_MARGIN:-0.10}"
    STAGE1_PAIR_MAX_PER_QUERY="${STAGE1_PAIR_MAX_PER_QUERY:-16}"
    ;;
  strong)
    STAGE1_PAIR_TYPES="${STAGE1_PAIR_TYPES:-llm_disagreement,strong_vs_hard}"
    STAGE1_PAIR_MIN_MARGIN="${STAGE1_PAIR_MIN_MARGIN:-0.15}"
    STAGE1_PAIR_MAX_PER_QUERY="${STAGE1_PAIR_MAX_PER_QUERY:-8}"
    ;;
  all)
    STAGE1_PAIR_TYPES="${STAGE1_PAIR_TYPES:-}"
    STAGE1_PAIR_MIN_MARGIN="${STAGE1_PAIR_MIN_MARGIN:-0.0}"
    STAGE1_PAIR_MAX_PER_QUERY="${STAGE1_PAIR_MAX_PER_QUERY:-0}"
    ;;
  *)
    echo "Unknown STAGE1_PAIR_PRESET='${STAGE1_PAIR_PRESET}'. Valid: balanced, strong, all." >&2
    exit 2
    ;;
esac
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
  --stage2-train-last-layers "${STAGE2_TRAIN_LAST_LAYERS}"
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
  --loss-ordinal-weight "${LOSS_ORDINAL_WEIGHT}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --cluster-margin-hm "${CLUSTER_MARGIN_HM}"
  --cluster-margin-ml "${CLUSTER_MARGIN_ML}"
  --cluster-margin-hl "${CLUSTER_MARGIN_HL}"
  --high-threshold "${HIGH_THRESHOLD}"
  --mid-threshold "${MID_THRESHOLD}"
  --margin-min "${MARGIN_MIN}"
  --margin-max "${MARGIN_MAX}"
  --pair-type-weight-map "${PAIR_TYPE_WEIGHT_MAP}"
  --stage1-pair-preset "${STAGE1_PAIR_PRESET}"
  --stage1-pair-types "${STAGE1_PAIR_TYPES}"
  --stage1-pair-min-margin "${STAGE1_PAIR_MIN_MARGIN}"
  --stage1-pair-max-per-query "${STAGE1_PAIR_MAX_PER_QUERY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-tags "${WANDB_TAGS}"
)

if [[ "${STAGE2_FREEZE_BACKBONE}" == "true" ]]; then
  CMD+=(--stage2-freeze-backbone)
fi

if [[ "${FP16}" == "true" ]]; then
  CMD+=(--fp16)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
