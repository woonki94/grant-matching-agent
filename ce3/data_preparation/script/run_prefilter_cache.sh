#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_combined.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-ce3/dataset/source/prefilter_cache.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"
INCLUDE_AUG_AUG="${INCLUDE_AUG_AUG:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/build_prefilter_cache.py
  --model-id "${MODEL_ID}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output-base "${OUTPUT_BASE}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
)

if bool_true "${INCLUDE_AUG_AUG}"; then
  CMD+=(--include-aug-aug)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
