#!/usr/bin/env bash
set -euo pipefail

# Fetch grants -> generate keywords for newly inserted grants ->
# generate match rows for those new grants against all faculties
# where cosine similarity >= min-domain.
#
# Usage:
#   ./scripts/import_new_grants_keywords_matches.sh [options]
#
# Examples:
#   ./scripts/import_new_grants_keywords_matches.sh
#   ./scripts/import_new_grants_keywords_matches.sh --query "education"
#   ./scripts/import_new_grants_keywords_matches.sh --query "ai" --agencies "NSF,HHS-NIH11"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Fetch grants and process only newly inserted grants.

Pipeline:
  1) import opportunities (+ extraction)
  2) generate opportunity keywords for new grants
  3) generate matches for those new grants against all faculties
     (filtered by cosine similarity threshold)

Usage:
  ./scripts/import_new_grants_keywords_matches.sh [options]

Options:
  --page-size <int>            API page size (default: 100)
  --query <text>               Optional opportunity query
  --agencies <csv>             Optional agency codes CSV (e.g., NSF,HHS-NIH11)
  --fetch-workers <int>        Opportunity fetch workers (default: 8)
  --extract-workers <int>      Extraction workers (default: 4)
  --min-domain <float>         Cosine similarity threshold (default: 0.3)
  --rerank-workers <int>       LLM rerank workers (default: 4)
  --rerank-chunk-workers <int> Optional chunk-worker override
  -h, --help                   Show this help message
EOF
}

PAGE_SIZE="100"
QUERY=""
AGENCIES=""
FETCH_WORKERS="8"
EXTRACT_WORKERS="4"
MIN_DOMAIN="0.3"
RERANK_WORKERS="4"
RERANK_CHUNK_WORKERS=""

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
    --min-domain)
      if [[ $# -lt 2 ]]; then
        echo "Error: --min-domain requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MIN_DOMAIN="$2"
      shift 2
      ;;
    --rerank-workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --rerank-workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      RERANK_WORKERS="$2"
      shift 2
      ;;
    --rerank-chunk-workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --rerank-chunk-workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      RERANK_CHUNK_WORKERS="$2"
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
  "extract-workers:$EXTRACT_WORKERS" \
  "rerank-workers:$RERANK_WORKERS"; do
  KEY="${pair%%:*}"
  VALUE="${pair#*:}"
  if ! [[ "$VALUE" =~ ^-?[0-9]+$ ]]; then
    echo "Error: --${KEY} must be an integer (got: '$VALUE')." >&2
    print_usage >&2
    exit 1
  fi
done

if ! [[ "$MIN_DOMAIN" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: --min-domain must be numeric (got: '$MIN_DOMAIN')." >&2
  print_usage >&2
  exit 1
fi
if [[ -n "$RERANK_CHUNK_WORKERS" ]] && ! [[ "$RERANK_CHUNK_WORKERS" =~ ^-?[0-9]+$ ]]; then
  echo "Error: --rerank-chunk-workers must be an integer (got: '$RERANK_CHUNK_WORKERS')." >&2
  print_usage >&2
  exit 1
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

echo "Running new-grant pipeline..."
echo "  Page size           : $PAGE_SIZE"
echo "  Query               : ${QUERY:-<none>}"
echo "  Agencies            : ${AGENCIES:-<all>}"
echo "  Fetch workers       : $FETCH_WORKERS"
echo "  Extract workers     : $EXTRACT_WORKERS"
echo "  Min domain          : $MIN_DOMAIN"
echo "  Rerank workers      : $RERANK_WORKERS"
echo "  Rerank chunk workers: ${RERANK_CHUNK_WORKERS:-<default>}"
echo

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

PYTHON_FILE="$PROJECT_ROOT/services/opportunity/import_keywords_match_new.py"
EXTRA_ARGS=(
  --page-size "$PAGE_SIZE"
  --fetch-workers "$FETCH_WORKERS"
  --extract-workers "$EXTRACT_WORKERS"
  --min-domain "$MIN_DOMAIN"
  --rerank-workers "$RERANK_WORKERS"
)
if [[ -n "$QUERY" ]]; then
  EXTRA_ARGS+=(--query "$QUERY")
fi
if [[ -n "$AGENCIES" ]]; then
  EXTRA_ARGS+=(--agencies "$AGENCIES")
fi
if [[ -n "$RERANK_CHUNK_WORKERS" ]]; then
  EXTRA_ARGS+=(--rerank-chunk-workers "$RERANK_CHUNK_WORKERS")
fi

"$PYTHON_BIN" "$PYTHON_FILE" "${EXTRA_ARGS[@]}"

echo
echo "New-grant pipeline completed."
