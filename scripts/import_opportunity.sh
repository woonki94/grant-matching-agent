#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the opportunity import + extraction pipeline
#
# Usage:
#   ./scripts/import_opportunity.sh [page_size] [query]
#
# Examples:
#   ./scripts/import_opportunity.sh
#   ./scripts/import_opportunity.sh 200
#   ./scripts/import_opportunity.sh 200 "education"
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

echo "Running opportunity import pipeline..."
echo "  Page size : $PAGE_SIZE"
echo "  Query     : ${QUERY:-<none>}"
echo

PYTHON_FILE="$PROJECT_ROOT/services/opportunity/import_opportunity.py"

if [[ -n "$QUERY" ]]; then
  python "$PYTHON_FILE" \
    --page-size "$PAGE_SIZE" \
    --query "$QUERY"
else
  python "$PYTHON_FILE" \
    --page-size "$PAGE_SIZE"
fi