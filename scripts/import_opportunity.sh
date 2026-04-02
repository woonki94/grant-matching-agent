#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the opportunity import + extraction pipeline
#
# Usage:
#   ./scripts/import_opportunity.sh [--page-size <int>] [--query <text>] [--agencies <csv>] [--fetch-workers <int>] [--extract-workers <int>]
#
# Examples:
#   ./scripts/import_opportunity.sh
#   ./scripts/import_opportunity.sh --page-size 200
#   ./scripts/import_opportunity.sh --page-size 200 --query "education"
#   ./scripts/import_opportunity.sh --page-size 200 --query "education" --agencies "HHS-NIH11,NSF"
#   ./scripts/import_opportunity.sh --page-size 200 --query "education" --agencies "HHS-NIH11,NSF" --fetch-workers 10 --extract-workers 6
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
Run the opportunity import + extraction pipeline.

Usage:
  ./scripts/import_opportunity.sh [options]

Options:
  --page-size <int>       API page size (default: 100)
  --query <text>          Search query (optional)
  --agencies <csv>        Comma-separated agency ids (optional)
  --fetch-workers <int>   Detail fetch workers (default: 8)
  --extract-workers <int> Extraction workers (default: 4)
  -h, --help              Show this help message
EOF
}

PAGE_SIZE="100"
QUERY=""
AGENCIES=""
FETCH_WORKERS="8"
EXTRACT_WORKERS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --page-size)
      if [[ $# -lt 2 ]]; then
        echo "Error: --page-size requires a value." >&2
        print_usage >&2
        exit 1
      fi
      PAGE_SIZE="$2"
      shift 2
      ;;
    --query)
      if [[ $# -lt 2 ]]; then
        echo "Error: --query requires a value." >&2
        print_usage >&2
        exit 1
      fi
      QUERY="$2"
      shift 2
      ;;
    --agencies)
      if [[ $# -lt 2 ]]; then
        echo "Error: --agencies requires a value." >&2
        print_usage >&2
        exit 1
      fi
      AGENCIES="$2"
      shift 2
      ;;
    --fetch-workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --fetch-workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      FETCH_WORKERS="$2"
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
  "page-size:$PAGE_SIZE" \
  "fetch-workers:$FETCH_WORKERS" \
  "extract-workers:$EXTRACT_WORKERS"; do
  KEY="${pair%%:*}"
  VALUE="${pair#*:}"
  if ! [[ "$VALUE" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --${KEY} must be an integer (got: '$VALUE')." >&2
    print_usage >&2
    exit 1
  fi
done

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
