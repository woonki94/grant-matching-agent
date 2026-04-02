#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Run group justification generation.

This is a shell wrapper for:
  services/justification/generate_group_justification.py

Usage:
  ./scripts/generate_group_justification.sh --email <email1> --email <email2> [options]

Required:
  --email <value>       Faculty email. Repeat for multiple members.

Common options:
  --team-size <int>     Team size (default: 3)
  --limit-rows <int>    Max match rows to scan (default: 200)
  --opp-id <value>      Target opportunity id (repeatable)
  --out-md <path>       Output markdown report path
  --include-trace       Include trace payload in output

Examples:
  ./scripts/generate_group_justification.sh \
    --email a@oregonstate.edu --email b@oregonstate.edu --email c@oregonstate.edu

  ./scripts/generate_group_justification.sh \
    --email a@oregonstate.edu --email b@oregonstate.edu \
    --team-size 2 --opp-id 599c5796-9954-4aa0-93d2-4a70f0c432ce
EOF
}

case "${1:-}" in
  -h|--help|help)
    print_usage
    exit 0
    ;;
esac

if [[ "$#" -eq 0 ]]; then
  echo "Error: arguments are required." >&2
  echo >&2
  print_usage >&2
  exit 1
fi

HAS_EMAIL="0"
for arg in "$@"; do
  if [[ "$arg" == "--email" ]] || [[ "$arg" == --email=* ]]; then
    HAS_EMAIL="1"
    break
  fi
done
if [[ "$HAS_EMAIL" != "1" ]]; then
  echo "Error: at least one --email is required." >&2
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

echo "Running group justification generation..."
echo "  Python : $PYTHON_BIN"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/services/justification/generate_group_justification.py" "$@"

