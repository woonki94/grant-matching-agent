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
STAGE2_EPOCHS_DEFAULT="${STAGE2_EPOCHS_DEFAULT:-3}"
SWEEP_STAGE2_LR="${SWEEP_STAGE2_LR:-8e-6}"
COMPARE_BASE="${COMPARE_BASE:-false}"
SWEEP_CONTINUE_ON_ERROR="${SWEEP_CONTINUE_ON_ERROR:-false}"

COMMON_ENV=(
  PYTHON_BIN="${PYTHON_BIN}"
  RUN_DATA_PREPARATION=false
  RUN_TRAIN=true
  RUN_EVAL=true
  MODEL_ID="${MODEL_ID}"
  SPLIT_DIR="${SPLIT_DIR}"
  TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR}"
  STAGE1_EPOCHS=0
  STAGE2_LEARNING_RATE="${SWEEP_STAGE2_LR}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
  LOSS_PAIR_WEIGHT="${LOSS_PAIR_WEIGHT:-0.05}"
  LOSS_KL_WEIGHT="${LOSS_KL_WEIGHT:-0.1}"
  LOSS_MSE_WEIGHT="${LOSS_MSE_WEIGHT:-1.0}"
  LOSS_CLUSTER_MARGIN_WEIGHT="${LOSS_CLUSTER_MARGIN_WEIGHT:-0.4}"
  LOSS_CALIBRATION_WEIGHT="${LOSS_CALIBRATION_WEIGHT:-1.0}"
  TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.8}"
  CLUSTER_MARGIN_HM="${CLUSTER_MARGIN_HM:-0.18}"
  CLUSTER_MARGIN_ML="${CLUSTER_MARGIN_ML:-0.18}"
  CLUSTER_MARGIN_HL="${CLUSTER_MARGIN_HL:-0.45}"
  COMPARE_BASE="${COMPARE_BASE}"
)

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
      "${COMMON_ENV[@]}" \
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
      "${COMMON_ENV[@]}" \
      RUN_NAME="${name}" \
      WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX:-ce3-sweep}-${name}" \
      EVAL_OUTPUT_DIR="${EVAL_OUTPUT_BASE_DIR}/${name}" \
      "$@" \
      bash "${TRAIN_EVAL_SCRIPT}"
  fi
}

# Almost pure calibration head probe: no transformer layers, just heads + scale/bias.
run_combo "s2_heads_only_ord04" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=true \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  LOSS_ORDINAL_WEIGHT=0.4

# Current conservative direction: heads + last 2 layers.
run_combo "s2_l2_ord04" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=true \
  STAGE2_TRAIN_LAST_LAYERS=2 \
  LOSS_ORDINAL_WEIGHT=0.4

# Main next move: more capacity without changing loss pressure.
run_combo "s2_l4_ord04" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=true \
  STAGE2_TRAIN_LAST_LAYERS=4 \
  LOSS_ORDINAL_WEIGHT=0.4

# Strong capacity probe: many top layers unfrozen.
run_combo "s2_l8_ord04" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=true \
  STAGE2_TRAIN_LAST_LAYERS=8 \
  LOSS_ORDINAL_WEIGHT=0.4

# Full fine-tune: maximum representation movement.
run_combo "s2_full_ord04" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  LOSS_ORDINAL_WEIGHT=0.4

# Aggressive band-shaping probe: same last-4 capacity, stronger boundary/calibration pressure.
run_combo "s2_l4_aggressive_bands" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=true \
  STAGE2_TRAIN_LAST_LAYERS=4 \
  LOSS_ORDINAL_WEIGHT=0.8 \
  LOSS_CALIBRATION_WEIGHT=1.5 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.6

# Full-model version of the aggressive band-shaping probe.
run_combo "s2_full_aggressive_bands" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS_DEFAULT}" \
  STAGE2_FREEZE_BACKBONE=false \
  STAGE2_TRAIN_LAST_LAYERS=0 \
  LOSS_ORDINAL_WEIGHT=0.8 \
  LOSS_CALIBRATION_WEIGHT=1.5 \
  LOSS_CLUSTER_MARGIN_WEIGHT=0.6

echo
echo "CE3 sweep complete."
echo "Models base directory: ${TRAIN_OUTPUT_DIR}"
echo "Eval results base directory: ${EVAL_OUTPUT_BASE_DIR}"
