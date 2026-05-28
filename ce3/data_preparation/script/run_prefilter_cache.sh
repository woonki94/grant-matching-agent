#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
GRANT_DB="${GRANT_DB:-ce3/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce3/dataset/source/fac_specs_db.json}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/dataset/decomposed/spec_decompositions_topic_approach_objective.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-ce3/dataset/source/prefilter_cache.jsonl}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-0}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"

CMD=(
  "${PYTHON_BIN}" ce3/data_preparation/build_prefilter_cache.py
  --model-id "${MODEL_ID}"
  --grant-db "${GRANT_DB}"
  --fac-db "${FAC_DB}"
  --decomposition-output "${DECOMPOSITION_OUTPUT}"
  --output-base "${OUTPUT_BASE}"
  --seed "${SEED}"
  --max-grant-specs "${MAX_GRANT_SPECS}"
  --max-fac-specs "${MAX_FAC_SPECS}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
