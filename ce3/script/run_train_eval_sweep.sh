#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-ce3/models/aspect_reranker}"
EVAL_OUTPUT_BASE_DIR="${EVAL_OUTPUT_BASE_DIR:-ce3/eval/results/sweep}"
TRAIN_EVAL_SCRIPT="${TRAIN_EVAL_SCRIPT:-ce3/script/run_data_preparation_train_eval.sh}"

MODEL_ID="${MODEL_ID:-${PROJECT_ROOT}/ce3/models/aspect_reranker/stage1_epoch_1}"
COMPARE_BASE="${COMPARE_BASE:-true}"
SWEEP_CONTINUE_ON_ERROR="${SWEEP_CONTINUE_ON_ERROR:-false}"

run_combo() {
  local name="$1"
  shift

  echo
  echo "================================================================================"
  echo "CE3 sweep combo: ${name}"
  echo "Extra args: $*"
  echo "================================================================================"

  if [[ "${SWEEP_CONTINUE_ON_ERROR}" == "true" ]]; then
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      RUN_DATA_PREPARATION=false \
      RUN_TRAIN=true \
      RUN_EVAL=true \
      MODEL_ID="${MODEL_ID}" \
      SPLIT_DIR="${SPLIT_DIR}" \
      TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
      STAGE1_EPOCHS=0 \
      COMPARE_BASE="${COMPARE_BASE}" \
      RUN_NAME="${name}" \
      WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX:-ce3-sweep}-${name}" \
      EVAL_OUTPUT_DIR="${EVAL_OUTPUT_BASE_DIR}/${name}" \
      "$@" \
      bash "${TRAIN_EVAL_SCRIPT}" || {
        echo "Combo failed but continuing because SWEEP_CONTINUE_ON_ERROR=true: ${name}" >&2
        return 0
      }
  else
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      RUN_DATA_PREPARATION=false \
      RUN_TRAIN=true \
      RUN_EVAL=true \
      MODEL_ID="${MODEL_ID}" \
      SPLIT_DIR="${SPLIT_DIR}" \
      TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
      STAGE1_EPOCHS=0 \
      COMPARE_BASE="${COMPARE_BASE}" \
      RUN_NAME="${name}" \
      WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX:-ce3-sweep}-${name}" \
      EVAL_OUTPUT_DIR="${EVAL_OUTPUT_BASE_DIR}/${name}" \
      "$@" \
      bash "${TRAIN_EVAL_SCRIPT}"
  fi
}

# Balanced full fine-tune baseline: currently the best direction.
run_combo "s2_full_balanced_ord04" \
  TRAIN_BATCH_SIZE=4 \
  EVAL_BATCH_SIZE=8 \
  GRAD_ACCUM_STEPS=4 \
  STAGE2_EPOCHS=3 \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  STAGE2_LEARNING_RATE=8e-6 \
  TEACHER_TEMPERATURE=0.8 \
  LOSS_PAIR_WEIGHT=0.05 \
  LOSS_KL_WEIGHT=0.1 \
  LOSS_MSE_WEIGHT=1.0 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.4 \
  LOSS_CALIBRATION_WEIGHT=1.0 \
  LOSS_ORDINAL_WEIGHT=0.4 \
  CLUSTER_MARGIN_HM=0.18 \
  CLUSTER_MARGIN_ML=0.18 \
  CLUSTER_MARGIN_HL=0.45

# High-rescue probe: prioritize lowering HIGH_OUT, especially objective highs.
run_combo "s2_full_high_rescue" \
  TRAIN_BATCH_SIZE=4 \
  EVAL_BATCH_SIZE=8 \
  GRAD_ACCUM_STEPS=4 \
  STAGE2_EPOCHS=3 \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  STAGE2_LEARNING_RATE=6e-6 \
  TEACHER_TEMPERATURE=0.65 \
  LOSS_PAIR_WEIGHT=0.04 \
  LOSS_KL_WEIGHT=0.05 \
  LOSS_MSE_WEIGHT=0.8 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.45 \
  LOSS_CALIBRATION_WEIGHT=1.2 \
  LOSS_ORDINAL_WEIGHT=0.65 \
  CALIBRATION_HIGH_WEIGHT=2.0 \
  CALIBRATION_MID_WEIGHT=1.0 \
  CALIBRATION_LOW_WEIGHT=1.0 \
  CALIBRATION_MID_LOW_WEIGHT=1.0 \
  CALIBRATION_MID_HIGH_WEIGHT=1.5 \
  ORDINAL_HIGH_BOUNDARY_WEIGHT=2.2 \
  ORDINAL_MID_BOUNDARY_WEIGHT=0.7 \
  CLUSTER_MARGIN_HM=0.22 \
  CLUSTER_MARGIN_ML=0.16 \
  CLUSTER_MARGIN_HL=0.52

# Mid-protection probe: prioritize lowering MID_OUT while keeping highs/lows moderate.
run_combo "s2_full_mid_guard" \
  TRAIN_BATCH_SIZE=4 \
  EVAL_BATCH_SIZE=8 \
  GRAD_ACCUM_STEPS=4 \
  STAGE2_EPOCHS=3 \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  STAGE2_LEARNING_RATE=7e-6 \
  TEACHER_TEMPERATURE=1.0 \
  LOSS_PAIR_WEIGHT=0.06 \
  LOSS_KL_WEIGHT=0.15 \
  LOSS_MSE_WEIGHT=1.2 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.25 \
  LOSS_CALIBRATION_WEIGHT=1.4 \
  LOSS_ORDINAL_WEIGHT=0.25 \
  CALIBRATION_MID_WEIGHT=2.0 \
  CALIBRATION_MID_LOW_WEIGHT=1.6 \
  CALIBRATION_MID_HIGH_WEIGHT=2.2 \
  CALIBRATION_HIGH_WEIGHT=1.1 \
  CALIBRATION_LOW_WEIGHT=1.2 \
  ORDINAL_HIGH_BOUNDARY_WEIGHT=0.9 \
  ORDINAL_MID_BOUNDARY_WEIGHT=0.9 \
  CLUSTER_MARGIN_HM=0.14 \
  CLUSTER_MARGIN_ML=0.14 \
  CLUSTER_MARGIN_HL=0.38

# Extreme all-band shaping: strongest pressure to force all clusters into their bands.
run_combo "s2_full_extreme_bands" \
  TRAIN_BATCH_SIZE=4 \
  EVAL_BATCH_SIZE=8 \
  GRAD_ACCUM_STEPS=4 \
  STAGE2_EPOCHS=2 \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  STAGE2_LEARNING_RATE=4e-6 \
  TEACHER_TEMPERATURE=0.55 \
  LOSS_PAIR_WEIGHT=0.03 \
  LOSS_KL_WEIGHT=0.03 \
  LOSS_MSE_WEIGHT=0.7 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.8 \
  LOSS_CALIBRATION_WEIGHT=1.8 \
  LOSS_ORDINAL_WEIGHT=0.85 \
  CALIBRATION_HIGH_WEIGHT=1.8 \
  CALIBRATION_LOW_WEIGHT=1.8 \
  CALIBRATION_MID_WEIGHT=1.8 \
  CALIBRATION_MID_LOW_WEIGHT=1.6 \
  CALIBRATION_MID_HIGH_WEIGHT=1.8 \
  ORDINAL_HIGH_BOUNDARY_WEIGHT=1.7 \
  ORDINAL_MID_BOUNDARY_WEIGHT=1.3 \
  CLUSTER_MARGIN_HM=0.24 \
  CLUSTER_MARGIN_ML=0.24 \
  CLUSTER_MARGIN_HL=0.58

echo
echo "CE3 sweep complete."
echo "Models base directory: ${TRAIN_OUTPUT_DIR}"
echo "Eval results base directory: ${EVAL_OUTPUT_BASE_DIR}"
