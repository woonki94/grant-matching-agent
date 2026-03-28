#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the opportunity import + extraction pipeline
#
# Usage:
#   ./scripts/import_opportunity.sh [page_size] [query] [agencies] [fetch_workers] [extract_workers]
#
# Examples:
#   ./scripts/import_opportunity.sh
#   ./scripts/import_opportunity.sh 200
#   ./scripts/import_opportunity.sh 200 "education"
#   ./scripts/import_opportunity.sh 200 "education" "HHS-NIH11,NSF"
#   ./scripts/import_opportunity.sh 200 "education" "HHS-NIH11,NSF" 10 6
# ───────────────────────────────────────────────

set -euo pipefail

# Resolve absolute project root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables if present
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

PAGE_SIZE="${1:-100}"
QUERY="${2:-}"
AGENCIES="${3:-}"
FETCH_WORKERS="${4:-8}"
EXTRACT_WORKERS="${5:-4}"

echo "Running opportunity import pipeline..."
echo "  Page size : $PAGE_SIZE"
echo "  Query     : ${QUERY:-<none>}"
echo "  Agencies  : ${AGENCIES:-<all>}"
echo "  Fetch wkr : $FETCH_WORKERS"
echo "  Xtract wkr: $EXTRACT_WORKERS"
echo

PYTHON_FILE="$PROJECT_ROOT/services/opportunity/import_opportunity.py"
PYTHON_BIN=""
if [[ -x "$PROJECT_ROOT/venv2/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv2/bin/python"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Error: no python interpreter found." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$QUERY" ]]; then
  EXTRA_ARGS+=(--query "$QUERY")
fi
if [[ -n "$AGENCIES" ]]; then
  EXTRA_ARGS+=(--agencies "$AGENCIES")
fi
EXTRA_ARGS+=(--fetch-workers "$FETCH_WORKERS")
EXTRA_ARGS+=(--extract-workers "$EXTRACT_WORKERS")

"$PYTHON_BIN" "$PYTHON_FILE" \
  --page-size "$PAGE_SIZE" \
  "${EXTRA_ARGS[@]}"
