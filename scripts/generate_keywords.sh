#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env"
  set +a
fi

MODE="${1:-all}"
LIMIT="${2:-}"

echo "Running keyword generation pipeline..."
echo "  Mode  : $MODE"
echo "  Limit : ${LIMIT:-none}"
echo
echo "  LLM_PROVIDER              : ${LLM_PROVIDER:-}"
echo "  EMBEDDING_PROVIDER        : ${EMBEDDING_PROVIDER:-}"
echo "  AWS_REGION                : ${AWS_REGION:-}"
echo "  EXTRACTED_CONTENT_BACKEND : ${EXTRACTED_CONTENT_BACKEND:-}"
echo "  EXTRACTED_CONTENT_BUCKET  : ${EXTRACTED_CONTENT_BUCKET:-}"
echo "  EXTRACTED_CONTENT_PREFIX  : ${EXTRACTED_CONTENT_PREFIX:-}"
echo

PYTHON_FILE="$PROJECT_ROOT/services/keywords/generate_keywords.py"

EXTRA_ARGS=""
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
fi

case "$MODE" in
  all) ;;
  faculty) EXTRA_ARGS="$EXTRA_ARGS --faculty-only" ;;
  opp) EXTRA_ARGS="$EXTRA_ARGS --opp-only" ;;
  *)
    echo "Error: invalid mode '$MODE' (use: all | faculty | opp)" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [ -x "$PYTHON_BIN" ]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$PYTHON_FILE" $EXTRA_ARGS
else
  # shellcheck disable=SC2086
  python3 "$PYTHON_FILE" $EXTRA_ARGS
fi

echo
echo "Keyword generation completed."
