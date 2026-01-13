#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the keyword generation pipeline
#
# Usage:
#   ./scripts/generate_keywords.sh [mode] [limit]
#
# Args:
#   mode  : all | faculty | opp
#   limit : max records per entity (omit/blank = no limit)
#
# Examples:
#   ./scripts/generate_keywords.sh
#   ./scripts/generate_keywords.sh all
#   ./scripts/generate_keywords.sh faculty
#   ./scripts/generate_keywords.sh faculty 50
#   ./scripts/generate_keywords.sh opp 100
# ───────────────────────────────────────────────
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

PYTHON_FILE="$PROJECT_ROOT/services/keywords/generate_keywords.py"

# Build args as plain strings
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

# shellcheck disable=SC2086
python "$PYTHON_FILE" $EXTRA_ARGS

echo
echo "Keyword generation completed."