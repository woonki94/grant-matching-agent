#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run faculty crawl + publication enrichment
#
# Usage:
#   ./scripts/run_faculty_import.sh [max_faculty_pages] [publication_lookback_years]
#
# Examples:
#   ./scripts/run_faculty_import.sh
#   ./scripts/run_faculty_import.sh 10
#   ./scripts/run_faculty_import.sh 10 7
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

MAX_FACULTY_PAGES="${1:-0}"
PUBLICATION_LOOKBACK_YEARS="${2:-5}"

echo "Running faculty import"
echo "  Faculty pages to crawl : $MAX_FACULTY_PAGES"
echo "  Publication years back : $PUBLICATION_LOOKBACK_YEARS"
echo

PYTHON_FILE="$PROJECT_ROOT/services/faculty/import_faculty.py"

python "$PYTHON_FILE" "$MAX_FACULTY_PAGES" "$PUBLICATION_LOOKBACK_YEARS"