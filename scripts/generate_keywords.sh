#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Usage:
  ./scripts/generate_keywords.sh [options]

Options:
  --mode <all|faculty|opp>  Run target (default: all)
  --limit <int>             Max rows to process; 0 or omitted means no limit
  --workers <int>           Thread workers per batch (default: 4)
  --force-regenerate        Regenerate rows that already have keywords
  -h, --help                Show this help message

Examples:
  ./scripts/generate_keywords.sh
  ./scripts/generate_keywords.sh --mode faculty --limit 50
  ./scripts/generate_keywords.sh --mode opp --limit 100 --force-regenerate
  ./scripts/generate_keywords.sh --mode all --workers 8
EOF
}

MODE="all"
LIMIT=""
FORCE_REGENERATE="0"
WORKERS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Error: --mode requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    --limit)
      if [[ $# -lt 2 ]]; then
        echo "Error: --limit requires a value." >&2
        print_usage >&2
        exit 1
      fi
      LIMIT="$2"
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
    --force-regenerate)
      FORCE_REGENERATE="1"
      shift
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

if [ -f "$PROJECT_ROOT/.env" ]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env"
  set +a
fi

echo "Running keyword generation pipeline..."
echo "  Mode  : $MODE"
echo "  Limit : ${LIMIT:-none}"
echo "  Force : ${FORCE_REGENERATE}"
echo "  Workers : ${WORKERS:-4}"
echo
echo "  LLM_PROVIDER                         : ${LLM_PROVIDER:-}"
echo "  EMBEDDING_PROVIDER                   : ${EMBEDDING_PROVIDER:-}"
echo "  AWS_REGION                           : ${AWS_REGION:-}"
echo "  EXTRACTED_CONTENT_BUCKET             : ${EXTRACTED_CONTENT_BUCKET:-}"
echo "  EXTRACTED_CONTENT_PREFIX_OPPORTUNITY : ${EXTRACTED_CONTENT_PREFIX_OPPORTUNITY:-}"
echo "  EXTRACTED_CONTENT_PREFIX_FACULTY     : ${EXTRACTED_CONTENT_PREFIX_FACULTY:-}"
echo

PYTHON_FILE="$PROJECT_ROOT/services/keywords/generate_keywords.py"

EXTRA_ARGS=()
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

case "$MODE" in
  all) ;;
  faculty) EXTRA_ARGS+=(--faculty-only) ;;
  opp) EXTRA_ARGS+=(--opp-only) ;;
  *)
    echo "Error: invalid mode '$MODE' (use: all | faculty | opp)" >&2
    echo >&2
    print_usage >&2
    exit 1
    ;;
esac

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]]; then
  echo "Error: workers must be a non-negative integer (got: '$WORKERS')." >&2
  echo >&2
  print_usage >&2
  exit 1
fi

if [[ -n "$LIMIT" ]] && ! [[ "$LIMIT" =~ ^-?[0-9]+$ ]]; then
  echo "Error: limit must be an integer (got: '$LIMIT')." >&2
  echo >&2
  print_usage >&2
  exit 1
fi

if [[ "$FORCE_REGENERATE" == "1" ]]; then
  EXTRA_ARGS+=(--force-regenerate)
fi

if [ "$WORKERS" -gt 0 ]; then
  EXTRA_ARGS+=(--workers "$WORKERS")
fi

PYTHON_BIN=""
if [ -x "${PROJECT_ROOT}/venv2/bin/python" ]; then
  PYTHON_BIN="${PROJECT_ROOT}/venv2/bin/python"
elif [ -x "${PROJECT_ROOT}/venv/bin/python" ]; then
  PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  RUN_ARGS=("$PYTHON_FILE" "${EXTRA_ARGS[@]}")
else
  RUN_ARGS=("$PYTHON_FILE")
fi

if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" "${RUN_ARGS[@]}"
else
  python3 "${RUN_ARGS[@]}"
fi

echo
echo "Keyword generation completed."
