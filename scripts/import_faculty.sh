#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the faculty import + publication enrichment pipeline
#
# Usage:
#   ./scripts/import_faculty.sh [--max-pages <int>] [--years-back <int>] [--max-faculty <int>] [--workers <int>] [--extract-workers <int>]
#
# Examples:
#   ./scripts/import_faculty.sh
#   ./scripts/import_faculty.sh --max-pages 1
#   ./scripts/import_faculty.sh --max-pages 3 --years-back 10 --max-faculty 50
#   ./scripts/import_faculty.sh --max-pages 3 --years-back 10 --max-faculty 50 --workers 8 --extract-workers 6
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

print_usage() {
  cat <<'EOF'
Run the faculty import + publication enrichment pipeline.

Usage:
  ./scripts/import_faculty.sh [options]

Options:
  --max-pages <int>       Max faculty pages to crawl (default: 0)
  --years-back <int>      Publication lookback window in years (default: 5)
  --max-faculty <int>     Max faculty links to process (default: 0)
  --workers <int>         Crawl/import workers (default: 8)
  --extract-workers <int> Extraction workers (default: 4)
  -h, --help              Show this help message
EOF
}

MAX_PAGES="0"
YEARS_BACK="5"
MAX_FACULTY="0"
WORKERS="8"
EXTRACT_WORKERS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-pages)
      if [[ $# -lt 2 ]]; then
        echo "Error: --max-pages requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MAX_PAGES="$2"
      shift 2
      ;;
    --years-back)
      if [[ $# -lt 2 ]]; then
        echo "Error: --years-back requires a value." >&2
        print_usage >&2
        exit 1
      fi
      YEARS_BACK="$2"
      shift 2
      ;;
    --max-faculty)
      if [[ $# -lt 2 ]]; then
        echo "Error: --max-faculty requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MAX_FACULTY="$2"
      shift 2
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      WORKERS="$2"
      shift 2
      ;;
    --extract-workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --extract-workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      EXTRACT_WORKERS="$2"
      shift 2
      ;;
    -h|--help|help)
      print_usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

for pair in \
  "max-pages:$MAX_PAGES" \
  "years-back:$YEARS_BACK" \
  "max-faculty:$MAX_FACULTY" \
  "workers:$WORKERS" \
  "extract-workers:$EXTRACT_WORKERS"; do
  KEY="${pair%%:*}"
  VALUE="${pair#*:}"
  if ! [[ "$VALUE" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --${KEY} must be an integer (got: '$VALUE')." >&2
    print_usage >&2
    exit 1
  fi
done


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
