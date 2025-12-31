#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the opportunity import + commit pipeline
#
# Usage:
#   ./scripts/fetch_commit_opportunity.sh [page_offset] [page_size] [max_page] [query]
#
# Examples:
#   ./scripts/fetch_commit_opportunity.sh
#   ./scripts/fetch_commit_opportunity.sh 1 50 10 "education"
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

PAGE_OFFSET="${1:-1}"
PAGE_SIZE="${2:-50}"
MAX_PAGE="${3:-10}"
QUERY="${4:-}"

echo "Running import_opportunity.py ..."
echo "  Page offset : $PAGE_OFFSET"
echo "  Page size   : $PAGE_SIZE"
echo "  Max pages   : $MAX_PAGE"
echo "  Query       : ${QUERY:-<none>}"
echo

PYTHON_FILE="$PROJECT_ROOT/services/opportunity/import_opportunity.py"

python "$PYTHON_FILE" "$PAGE_OFFSET" "$PAGE_SIZE" "$MAX_PAGE" "$QUERY"