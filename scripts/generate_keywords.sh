#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Usage:
  ./scripts/generate_keywords.sh [mode] [limit] [force] [workers]

Arguments:
  mode   : all | faculty | opp
           default: all
  limit  : max rows to process (integer)
           default: empty (no limit)
  force  : true/false flag for regenerating existing keywords
           accepted true values: true, 1, yes, force, --force-regenerate
           default: false
  workers: thread workers per batch
           default: 4

Examples:
  ./scripts/generate_keywords.sh
  ./scripts/generate_keywords.sh faculty 50
  ./scripts/generate_keywords.sh opp 100 true
  ./scripts/generate_keywords.sh all 0 --force-regenerate
  ./scripts/generate_keywords.sh all 0 false 8
EOF
}

case "${1:-}" in
  -h|--help|help)
    print_usage
    exit 0
    ;;
esac

if [ -f "$PROJECT_ROOT/.env" ]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env"
  set +a
fi

MODE="${1:-all}"
LIMIT="${2:-}"
FORCE_MODE="${3:-}"
WORKERS="${4:-4}"

echo "Running keyword generation pipeline..."
echo "  Mode  : $MODE"
echo "  Limit : ${LIMIT:-none}"
echo "  Force : ${FORCE_MODE:-false}"
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

case "$FORCE_MODE" in
  ""|false|0|no) ;;
  true|1|yes|force|--force-regenerate) EXTRA_ARGS+=(--force-regenerate) ;;
  *)
    echo "Error: invalid force flag '$FORCE_MODE' (use: true|false or force)" >&2
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
