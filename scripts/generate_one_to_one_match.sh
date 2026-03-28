#!/usr/bin/env bash
set -euo pipefail

# Run one-to-one faculty <-> opportunity matching.
#
# Existing pairs are overwritten by upsert on (grant_id, faculty_id).
# Rows not touched in this run remain in the DB.
#
# Usage:
#   ./scripts/generate_one_to_one_match.sh [k] [min_domain] [limit_faculty] [commit_every]
#
# Arguments:
#   k             : top-k opportunities per faculty (default: 10)
#   min_domain    : domain similarity threshold (default: 0.30)
#   limit_faculty : max faculty rows to process; 0 = all (default: 100)
#   commit_every  : commit frequency in faculty loop (default: 30)
#
# Examples:
#   ./scripts/generate_one_to_one_match.sh
#   ./scripts/generate_one_to_one_match.sh 10 0.30 100 30
#   ./scripts/generate_one_to_one_match.sh 20 0.10 0 50

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Run one-to-one faculty <-> opportunity matching.

Existing pairs are overwritten by upsert on (grant_id, faculty_id).
Rows not touched in this run remain in the DB.

Usage:
  ./scripts/generate_one_to_one_match.sh [k] [min_domain] [limit_faculty] [commit_every]

Arguments:
  k             : top-k opportunities per faculty (default: 10)
  min_domain    : domain similarity threshold (default: 0.30)
  limit_faculty : max faculty rows to process; 0 = all (default: 100)
  commit_every  : commit frequency in faculty loop (default: 30)

Examples:
  ./scripts/generate_one_to_one_match.sh
  ./scripts/generate_one_to_one_match.sh 10 0.30 100 30
  ./scripts/generate_one_to_one_match.sh 20 0.10 0 50
EOF
}

case "${1:-}" in
  -h|--help|help)
    print_usage
    exit 0
    ;;
esac

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

K="${1:-100}"
MIN_DOMAIN="${2:-0.00}"
LIMIT_FACULTY="${3:-100}"
COMMIT_EVERY="${4:-30}"

if ! [[ "$K" =~ ^-?[0-9]+$ ]]; then
  echo "Error: k must be an integer (got: '$K')." >&2
  print_usage >&2
  exit 1
fi
if ! [[ "$MIN_DOMAIN" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: min_domain must be numeric (got: '$MIN_DOMAIN')." >&2
  print_usage >&2
  exit 1
fi
if ! [[ "$LIMIT_FACULTY" =~ ^-?[0-9]+$ ]]; then
  echo "Error: limit_faculty must be an integer (got: '$LIMIT_FACULTY')." >&2
  print_usage >&2
  exit 1
fi
if ! [[ "$COMMIT_EVERY" =~ ^-?[0-9]+$ ]]; then
  echo "Error: commit_every must be an integer (got: '$COMMIT_EVERY')." >&2
  print_usage >&2
  exit 1
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

echo "Running one-to-one matching..."
echo "  Python        : $PYTHON_BIN"
echo "  k             : $K"
echo "  min_domain    : $MIN_DOMAIN"
echo "  limit_faculty : $LIMIT_FACULTY"
echo "  commit_every  : $COMMIT_EVERY"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/services/matching/generate_one_to_one_match.py" \
  --k "$K" \
  --min-domain "$MIN_DOMAIN" \
  --limit-faculty "$LIMIT_FACULTY" \
  --commit-every "$COMMIT_EVERY"

echo
echo "One-to-one matching completed."
