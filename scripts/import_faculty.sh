#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the faculty import + publication enrichment pipeline
#
# Usage:
#   ./scripts/import_faculty.sh [max_pages] [years_back] [max_faculty] [workers] [extract_workers]
#
# Examples:
#   ./scripts/import_faculty.sh
#   ./scripts/import_faculty.sh 1
#   ./scripts/import_faculty.sh 3 10
#   ./scripts/import_faculty.sh 3 10 50
#   ./scripts/import_faculty.sh 3 10 50 8 6
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
MAX_FACULTY="${3:-0}"
WORKERS="${4:-8}"
EXTRACT_WORKERS="${5:-4}"


echo "Running faculty import pipeline..."
echo "  Max pages        : $MAX_PAGES"
echo "  Publications back: $YEARS_BACK years"
echo "  Max faculty links: $MAX_FACULTY"
echo "  Workers          : $WORKERS"
echo "  Extract workers  : $EXTRACT_WORKERS"
echo

PYTHON_FILE="$PROJECT_ROOT/services/faculty/import_faculty.py"
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

"$PYTHON_BIN" "$PYTHON_FILE" \
  --max-pages "$MAX_PAGES" \
  --years-back "$YEARS_BACK" \
  --max-faculty "$MAX_FACULTY" \
  --workers "$WORKERS" \
  --extract-workers "$EXTRACT_WORKERS"
