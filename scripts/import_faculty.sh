#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the faculty import + publication enrichment pipeline
#
# Usage:
#   ./scripts/import_faculty.sh [max_pages] [years_back]
#
# Examples:
#   ./scripts/import_faculty.sh
#   ./scripts/import_faculty.sh 1
#   ./scripts/import_faculty.sh 3 10
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

MAX_PAGES="${1:-0}"
YEARS_BACK="${2:-5}"

echo "Running faculty import pipeline..."
echo "  Max pages        : $MAX_PAGES"
echo "  Publications back: $YEARS_BACK years"
echo

PYTHON_FILE="$PROJECT_ROOT/services/faculty/import_faculty.py"

python "$PYTHON_FILE" \
  --max-pages "$MAX_PAGES" \
  --years-back "$YEARS_BACK"