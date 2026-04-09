#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Pre-generate and cache grant explanations for all grants missing one.

Usage:
  ./scripts/generate_grant_explanations.sh [--force]

Options:
  --force      Regenerate explanations even for grants that already have one
  -h, --help   Show this help message

Examples:
  ./scripts/generate_grant_explanations.sh
  ./scripts/generate_grant_explanations.sh --force
EOF
}

FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE="--force"
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

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

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

echo "Running grant explanation generation..."
echo "  Python : $PYTHON_BIN"
[[ -n "$FORCE" ]] && echo "  Mode   : force regenerate" || echo "  Mode   : missing only"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/services/justification/generate_grant_explanations.py" $FORCE
