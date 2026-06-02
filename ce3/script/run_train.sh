#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-${PROJECT_ROOT}/ce3/models/aspect_reranker/stage1_epoch_1}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
OUTPUT_DIR="${OUTPUT_DIR:-ce3/models/aspect_reranker}"
AUTO_OUTPUT_HASH="${AUTO_OUTPUT_HASH:-true}"
RUN_NAME="${RUN_NAME:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-0}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_FREEZE_BACKBONE="${STAGE2_FREEZE_BACKBONE:-false}"
STAGE2_TRAIN_LAST_LAYERS="${STAGE2_TRAIN_LAST_LAYERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-0}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-384}"
TRAIN_LOG_EVERY_STEPS="${TRAIN_LOG_EVERY_STEPS:-1}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"

LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.04}"
LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.08}"
LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-1.0}"
LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-0.45}"
LOSS_CALIBRATION_WEIGHT="${LOSS_CALIBRATION_WEIGHT:-1.45}"
LOSS_ORDINAL_WEIGHT="${LOSS_ORDINAL_WEIGHT:-0.55}"
LOSS_COVERAGE_WEIGHT="${LOSS_COVERAGE_WEIGHT:-0.55}"
LOSS_ANY_COVERAGE_WEIGHT="${LOSS_ANY_COVERAGE_WEIGHT:-1.0}"
LOSS_HIGH_COVERAGE_WEIGHT="${LOSS_HIGH_COVERAGE_WEIGHT:-1.45}"
CALIBRATION_HIGH_WEIGHT="${CALIBRATION_HIGH_WEIGHT:-1.65}"
CALIBRATION_MID_WEIGHT="${CALIBRATION_MID_WEIGHT:-1.35}"
CALIBRATION_LOW_WEIGHT="${CALIBRATION_LOW_WEIGHT:-1.35}"
CALIBRATION_MID_LOW_WEIGHT="${CALIBRATION_MID_LOW_WEIGHT:-1.70}"
CALIBRATION_MID_HIGH_WEIGHT="${CALIBRATION_MID_HIGH_WEIGHT:-1.55}"
ORDINAL_MID_BOUNDARY_WEIGHT="${ORDINAL_MID_BOUNDARY_WEIGHT:-1.25}"
ORDINAL_HIGH_BOUNDARY_WEIGHT="${ORDINAL_HIGH_BOUNDARY_WEIGHT:-1.55}"
COVERAGE_BOUNDARY_MARGIN="${COVERAGE_BOUNDARY_MARGIN:-0.0}"
COVERAGE_ANY_POS_WEIGHT="${COVERAGE_ANY_POS_WEIGHT:-1.10}"
COVERAGE_ANY_NEG_WEIGHT="${COVERAGE_ANY_NEG_WEIGHT:-1.25}"
COVERAGE_HIGH_POS_WEIGHT="${COVERAGE_HIGH_POS_WEIGHT:-1.55}"
COVERAGE_HIGH_NEG_WEIGHT="${COVERAGE_HIGH_NEG_WEIGHT:-1.35}"
COVERAGE_ASPECT_WEIGHT_MAP="${COVERAGE_ASPECT_WEIGHT_MAP:-topic=1.15,approach=1.0,objective=1.45}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.75}"
CLUSTER_MARGIN_HM="${CLUSTER_MARGIN_HM:-0.20}"
CLUSTER_MARGIN_ML="${CLUSTER_MARGIN_ML:-0.20}"
CLUSTER_MARGIN_HL="${CLUSTER_MARGIN_HL:-0.50}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.70}"
MID_THRESHOLD="${MID_THRESHOLD:-0.30}"
MARGIN_MIN="${MARGIN_MIN:-0.02}"
MARGIN_MAX="${MARGIN_MAX:-0.60}"
PAIR_TYPE_WEIGHT_MAP="${PAIR_TYPE_WEIGHT_MAP:-default=1.0,llm_disagreement=1.2,strong_vs_boundary=1.15,strong_vs_weak=0.9,strong_vs_hard=1.0}"
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

OUTPUT_BASE_DIR="${OUTPUT_DIR}"
if [[ "${AUTO_OUTPUT_HASH}" == "true" ]]; then
  HASH_INPUT="$(printf '%s\n' \
    "model_id=${MODEL_ID}" \
    "split_dir=${SPLIT_DIR}" \
    "train_batch_size=${TRAIN_BATCH_SIZE}" \
    "eval_batch_size=${EVAL_BATCH_SIZE}" \
    "grad_accum_steps=${GRAD_ACCUM_STEPS}" \
    "stage1_epochs=${STAGE1_EPOCHS}" \
    "stage2_epochs=${STAGE2_EPOCHS}" \
    "stage2_freeze_backbone=${STAGE2_FREEZE_BACKBONE}" \
    "stage2_train_last_layers=${STAGE2_TRAIN_LAST_LAYERS}" \
    "learning_rate=${LEARNING_RATE}" \
    "stage1_learning_rate=${STAGE1_LEARNING_RATE}" \
    "stage2_learning_rate=${STAGE2_LEARNING_RATE}" \
    "max_length=${MAX_LENGTH}" \
    "loss_pair_weight=${LOSS_PAIR_WEIGHT}" \
    "loss_kl_weight=${LOSS_KL_WEIGHT}" \
    "loss_mse_weight=${LOSS_MSE_WEIGHT}" \
    "loss_cluster_margin_weight=${LOSS_CLUSTER_MARGIN_WEIGHT}" \
    "loss_calibration_weight=${LOSS_CALIBRATION_WEIGHT}" \
    "loss_ordinal_weight=${LOSS_ORDINAL_WEIGHT}" \
    "loss_coverage_weight=${LOSS_COVERAGE_WEIGHT}" \
    "loss_any_coverage_weight=${LOSS_ANY_COVERAGE_WEIGHT}" \
    "loss_high_coverage_weight=${LOSS_HIGH_COVERAGE_WEIGHT}" \
    "calibration_high_weight=${CALIBRATION_HIGH_WEIGHT}" \
    "calibration_mid_weight=${CALIBRATION_MID_WEIGHT}" \
    "calibration_low_weight=${CALIBRATION_LOW_WEIGHT}" \
    "calibration_mid_low_weight=${CALIBRATION_MID_LOW_WEIGHT}" \
    "calibration_mid_high_weight=${CALIBRATION_MID_HIGH_WEIGHT}" \
    "ordinal_mid_boundary_weight=${ORDINAL_MID_BOUNDARY_WEIGHT}" \
    "ordinal_high_boundary_weight=${ORDINAL_HIGH_BOUNDARY_WEIGHT}" \
    "coverage_boundary_margin=${COVERAGE_BOUNDARY_MARGIN}" \
    "coverage_any_pos_weight=${COVERAGE_ANY_POS_WEIGHT}" \
    "coverage_any_neg_weight=${COVERAGE_ANY_NEG_WEIGHT}" \
    "coverage_high_pos_weight=${COVERAGE_HIGH_POS_WEIGHT}" \
    "coverage_high_neg_weight=${COVERAGE_HIGH_NEG_WEIGHT}" \
    "coverage_aspect_weight_map=${COVERAGE_ASPECT_WEIGHT_MAP}" \
    "teacher_temperature=${TEACHER_TEMPERATURE}" \
    "cluster_margin_hm=${CLUSTER_MARGIN_HM}" \
    "cluster_margin_ml=${CLUSTER_MARGIN_ML}" \
    "cluster_margin_hl=${CLUSTER_MARGIN_HL}" \
    "high_threshold=${HIGH_THRESHOLD}" \
    "mid_threshold=${MID_THRESHOLD}" \
    "margin_min=${MARGIN_MIN}" \
    "margin_max=${MARGIN_MAX}" \
    "stage1_pair_preset=${STAGE1_PAIR_PRESET}" \
    "stage1_pair_types=${STAGE1_PAIR_TYPES}" \
    "stage1_pair_min_margin=${STAGE1_PAIR_MIN_MARGIN}" \
    "stage1_pair_max_per_query=${STAGE1_PAIR_MAX_PER_QUERY}" \
    "fp16=${FP16}")"
  RUN_HASH="$(printf '%s' "${HASH_INPUT}" | "${PYTHON_BIN}" -c 'import hashlib, sys; print(hashlib.sha1(sys.stdin.read().encode("utf-8")).hexdigest()[:10])')"
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="s2cov_l${STAGE2_TRAIN_LAST_LAYERS}_ord${LOSS_ORDINAL_WEIGHT}_cov${LOSS_COVERAGE_WEIGHT}_${RUN_HASH}"
  fi
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"
  if [[ "${ALLOW_OUTPUT_OVERWRITE:-false}" != "true" ]]; then
    OUTPUT_CANDIDATE="${OUTPUT_DIR}"
    OUTPUT_SUFFIX=2
    while [[ -e "${OUTPUT_CANDIDATE}" ]]; do
      OUTPUT_CANDIDATE="${OUTPUT_BASE_DIR}/${RUN_NAME}_r${OUTPUT_SUFFIX}"
      OUTPUT_SUFFIX=$((OUTPUT_SUFFIX + 1))
    done
    OUTPUT_DIR="${OUTPUT_CANDIDATE}"
  fi
fi
mkdir -p "${OUTPUT_BASE_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > "${OUTPUT_BASE_DIR}/.last_train_output_dir"
echo "Training output directory: ${OUTPUT_DIR}"

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
  --loss-coverage-weight "${LOSS_COVERAGE_WEIGHT}"
  --loss-any-coverage-weight "${LOSS_ANY_COVERAGE_WEIGHT}"
  --loss-high-coverage-weight "${LOSS_HIGH_COVERAGE_WEIGHT}"
  --calibration-high-weight "${CALIBRATION_HIGH_WEIGHT}"
  --calibration-mid-weight "${CALIBRATION_MID_WEIGHT}"
  --calibration-low-weight "${CALIBRATION_LOW_WEIGHT}"
  --calibration-mid-low-weight "${CALIBRATION_MID_LOW_WEIGHT}"
  --calibration-mid-high-weight "${CALIBRATION_MID_HIGH_WEIGHT}"
  --ordinal-mid-boundary-weight "${ORDINAL_MID_BOUNDARY_WEIGHT}"
  --ordinal-high-boundary-weight "${ORDINAL_HIGH_BOUNDARY_WEIGHT}"
  --coverage-boundary-margin "${COVERAGE_BOUNDARY_MARGIN}"
  --coverage-any-pos-weight "${COVERAGE_ANY_POS_WEIGHT}"
  --coverage-any-neg-weight "${COVERAGE_ANY_NEG_WEIGHT}"
  --coverage-high-pos-weight "${COVERAGE_HIGH_POS_WEIGHT}"
  --coverage-high-neg-weight "${COVERAGE_HIGH_NEG_WEIGHT}"
  --coverage-aspect-weight-map "${COVERAGE_ASPECT_WEIGHT_MAP}"
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
