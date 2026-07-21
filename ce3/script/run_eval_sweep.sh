#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT_DIR="${SPLIT_DIR:-ce3/dataset/splits}"
EVAL_SCRIPT="${EVAL_SCRIPT:-ce3/script/run_eval.sh}"
EVAL_OUTPUT_BASE_DIR="${EVAL_OUTPUT_BASE_DIR:-ce3/eval/results/model_compare}"
COMPARE_BASE="${COMPARE_BASE:-true}"
OOB_MARGINS="${OOB_MARGINS:-0,0.03,0.05}"

run_eval_model() {
  local name="$1"
  local model_path="$2"

  echo
  echo "================================================================================"
  echo "CE3 eval sweep model: ${name}"
  echo "Model: ${model_path}"
  echo "================================================================================"

  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    SPLIT_DIR="${SPLIT_DIR}" \
    MODEL="${model_path}" \
    OUTPUT_DIR="${EVAL_OUTPUT_BASE_DIR}/${name}" \
    COMPARE_BASE="${COMPARE_BASE}" \
    OOB_MARGINS="${OOB_MARGINS}" \
    bash "${EVAL_SCRIPT}"
}

run_eval_model "s2_l4_ord04" \
  "ce3/models/aspect_reranker/s2_l4_ord04/best_stage2_selected"

run_eval_model "s2_full_ord04" \
  "ce3/models/aspect_reranker/s2_full_ord04/best_stage2_selected"

run_eval_model "s2_full_extreme_bands" \
  "ce3/models/aspect_reranker/s2_full_extreme_bands/best_stage2_selected"

run_eval_model "s2_full_balanced_ord04" \
  "ce3/models/aspect_reranker/s2_full_balanced_ord04/best_stage2_selected"

run_eval_model "s2_full_high_rescue" \
  "ce3/models/aspect_reranker/s2_full_high_rescue/best_stage2_selected"

run_eval_model "s2_full_mid_guard" \
  "ce3/models/aspect_reranker/s2_full_mid_guard/best_stage2_selected"

echo
echo "CE3 eval sweep complete."
echo "Eval results base directory: ${EVAL_OUTPUT_BASE_DIR}"
