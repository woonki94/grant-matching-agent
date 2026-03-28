#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Generate combined output (rerank + grant explanation + final justification) for one faculty.

Usage:
  ./scripts/generate_justification.sh [email] [k]

Arguments:
  email : faculty email in DB (required)
  k     : top-k opportunities to generate (default: 5)

Examples:
  ./scripts/generate_justification.sh dana.ainsworth@oregonstate.edu
  ./scripts/generate_justification.sh dana.ainsworth@oregonstate.edu 10
EOF
}

case "${1:-}" in
  -h|--help|help)
    print_usage
    exit 0
    ;;
esac

EMAIL="${1:-}"
K="${2:-5}"

if [[ -z "$EMAIL" ]]; then
  echo "Error: email is required." >&2
  echo >&2
  print_usage >&2
  exit 1
fi

if ! [[ "$K" =~ ^-?[0-9]+$ ]]; then
  echo "Error: k must be an integer (got: '$K')." >&2
  echo >&2
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

echo "Running justification generation..."
echo "  Python : $PYTHON_BIN"
echo "  Email  : $EMAIL"
echo "  k      : $K"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/services/justification/generate_justification.py" \
  --email "$EMAIL" \
  --k "$K"
