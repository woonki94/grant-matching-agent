#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/test/output/spec_decompositions_combined_subset.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-ce3/test/output/prefilter_cache_subset.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"
INCLUDE_AUG_AUG="${INCLUDE_AUG_AUG:-false}"

bool_true() { [[ "${1:-}" == "true" ]]; }

if [[ ! -f "${DECOMPOSITION_OUTPUT}" ]]; then
  echo "Missing subset combined decomposition: ${DECOMPOSITION_OUTPUT}" >&2
  echo "Run ce3/test/component/run_combine_decompositions_subset.sh first." >&2
  exit 1
fi

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
echo "Subset prefilter cache base: ${OUTPUT_BASE}"
