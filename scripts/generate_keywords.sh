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
FORCE_MODE="${3:-}"

echo "Running keyword generation pipeline..."
echo "  Mode  : $MODE"
echo "  Limit : ${LIMIT:-none}"
echo "  Force : ${FORCE_MODE:-false}"
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
    exit 1
    ;;
esac

case "$FORCE_MODE" in
  ""|false|0|no) ;;
  true|1|yes|force|--force-regenerate) EXTRA_ARGS+=(--force-regenerate) ;;
  *)
    echo "Error: invalid force flag '$FORCE_MODE' (use: true|false or force)" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [ -x "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" "$PYTHON_FILE" "${EXTRA_ARGS[@]}"
else
  python3 "$PYTHON_FILE" "${EXTRA_ARGS[@]}"
fi

echo
echo "Keyword generation completed."
