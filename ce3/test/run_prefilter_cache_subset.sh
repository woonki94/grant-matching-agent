#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-dleemiller/ModernCE-base-sts}"
GRANT_DB="${GRANT_DB:-ce3/dataset/source/grant_keywords_spec_keywords_db.json}"
FAC_DB="${FAC_DB:-ce3/dataset/source/fac_specs_db.json}"
DECOMPOSITION_OUTPUT="${DECOMPOSITION_OUTPUT:-ce3/test/output/spec_decompositions_subset.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-ce3/test/output/prefilter_cache_subset.jsonl}"
SEED="${SEED:-42}"
MAX_GRANT_SPECS="${MAX_GRANT_SPECS:-10}"
MAX_FAC_SPECS="${MAX_FAC_SPECS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-256}"

if [[ ! -f "${DECOMPOSITION_OUTPUT}" ]]; then
  echo "Missing subset decomposition: ${DECOMPOSITION_OUTPUT}" >&2
  echo "Run ce3/test/run_decomposition_subset.sh first." >&2
  exit 1
fi

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
echo "Subset prefilter cache base: ${OUTPUT_BASE}"
