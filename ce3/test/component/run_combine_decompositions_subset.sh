#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ORIGINAL_DECOMPOSITION="${ORIGINAL_DECOMPOSITION:-ce3/test/output/spec_decompositions_subset.jsonl}"
AUGMENTED_DECOMPOSITION="${AUGMENTED_DECOMPOSITION:-ce3/test/output/spec_decompositions_augmented_subset.jsonl}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-ce3/test/output/spec_decompositions_combined_subset.jsonl}"

if [[ ! -f "${ORIGINAL_DECOMPOSITION}" ]]; then
  echo "Missing subset original decomposition: ${ORIGINAL_DECOMPOSITION}" >&2
  echo "Run ce3/test/component/run_decomposition_subset.sh first." >&2
  exit 1
fi
if [[ ! -f "${AUGMENTED_DECOMPOSITION}" ]]; then
  echo "Missing subset augmented decomposition: ${AUGMENTED_DECOMPOSITION}" >&2
  echo "Run ce3/test/component/run_decompose_augmented_subset.sh first." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/combine_decomposition_files.py
  --original-decomposition "${ORIGINAL_DECOMPOSITION}"
  --augmented-decomposition "${AUGMENTED_DECOMPOSITION}"
  --combined-output "${COMBINED_OUTPUT}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Subset combined decomposition: ${COMBINED_OUTPUT}"
