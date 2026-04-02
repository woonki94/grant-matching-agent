#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Generate combined output (grant explanation + final justification) for one faculty.

Usage:
  ./scripts/generate_justification.sh --email <email> [--k <k>]

Options:
  --email <value>  Faculty email in DB (required)
  --k <int>        Top-k opportunities to generate (default: 5)
  -h, --help       Show this help message

Examples:
  ./scripts/generate_justification.sh --email dana.ainsworth@oregonstate.edu
  ./scripts/generate_justification.sh --email dana.ainsworth@oregonstate.edu --k 10
EOF
}

EMAIL=""
K="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)
      if [[ $# -lt 2 ]]; then
        echo "Error: --email requires a value." >&2
        print_usage >&2
        exit 1
      fi
      EMAIL="$2"
      shift 2
      ;;
    --k)
      if [[ $# -lt 2 ]]; then
        echo "Error: --k requires a value." >&2
        print_usage >&2
        exit 1
      fi
      K="$2"
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

if [[ -z "$EMAIL" ]]; then
  echo "Error: --email is required." >&2
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
