#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PREPARATION_SCRIPT="${DATA_PREPARATION_SCRIPT:-ce3/data_preparation/script/run_full_pipeline.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-ce3/script/run_train.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-ce3/script/run_eval.sh}"

RUN_DATA_PREPARATION="${RUN_DATA_PREPARATION:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"

SPLIT_DIR="${SPLIT_DIR:-${SPLIT_OUTPUT_DIR:-ce3/dataset/splits}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-ce3/models/aspect_reranker}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-ce3/eval/results}"

run_stage() {
  local name="$1"
  shift
  echo
  echo "========== CE3 stage: ${name} =========="
  echo "Running: $*"
  "$@"
}

if [[ "${RUN_DATA_PREPARATION}" == "true" ]]; then
  run_stage "data preparation" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      SPLIT_OUTPUT_DIR="${SPLIT_DIR}" \
      bash "${DATA_PREPARATION_SCRIPT}"
else
  echo "Skipping data preparation because RUN_DATA_PREPARATION=${RUN_DATA_PREPARATION}"
fi

if [[ "${RUN_TRAIN}" == "true" ]]; then
  run_stage "train" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      SPLIT_DIR="${SPLIT_DIR}" \
      OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
      bash "${TRAIN_SCRIPT}"
else
  echo "Skipping train because RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" == "true" ]]; then
  run_stage "held-out test eval" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      SPLIT_DIR="${SPLIT_DIR}" \
      MODEL_DIR="${TRAIN_OUTPUT_DIR}" \
      OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
      bash "${EVAL_SCRIPT}"
else
  echo "Skipping eval because RUN_EVAL=${RUN_EVAL}"
fi

echo
echo "CE3 data preparation + train + eval complete."
echo "Split directory: ${SPLIT_DIR}"
echo "Train output directory: ${TRAIN_OUTPUT_DIR}"
echo "Eval output directory: ${EVAL_OUTPUT_DIR}"
