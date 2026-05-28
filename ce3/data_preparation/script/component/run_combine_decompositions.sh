#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ORIGINAL_DECOMPOSITION="${ORIGINAL_DECOMPOSITION:-ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl}"
AUGMENTED_DECOMPOSITION="${AUGMENTED_DECOMPOSITION:-ce3/dataset/decomposed/spec_decompositions_augmented.jsonl}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_combined.jsonl}"

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/combine_decomposition_files.py
  --original-decomposition "${ORIGINAL_DECOMPOSITION}"
  --augmented-decomposition "${AUGMENTED_DECOMPOSITION}"
  --combined-output "${COMBINED_OUTPUT}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
