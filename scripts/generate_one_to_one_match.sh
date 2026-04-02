#!/usr/bin/env bash
set -euo pipefail

# Run one-to-one faculty <-> opportunity matching.
#
# Existing pairs are overwritten by upsert on (grant_id, faculty_id).
# Rows not touched in this run remain in the DB.
#
# Usage:
#   ./scripts/generate_one_to_one_match.sh [--mode <match_and_rerank|rerank_only>] [--k <int>] [--min-domain <float>] [--limit-faculty <int>] [--commit-every <int>] [--rerank-workers <int>] [--max-context-chars <int>]
#
# Options:
#   --mode             : match_and_rerank | rerank_only (default: match_and_rerank)
#   --k                : top-k opportunities per faculty (default: 50)
#   --min-domain       : domain similarity threshold (default: 0.3)
#   --limit-faculty    : max faculty rows to process; 0 = all (default: 100)
#   --commit-every     : commit frequency in faculty loop (default: 30)
#   --rerank-workers   : workers for rerank-only mode (default: 4)
#   --max-context-chars: max payload chars per rerank chunk (default: 100000)
#   -h, --help      : show this help
#
# Examples:
#   ./scripts/generate_one_to_one_match.sh
#   ./scripts/generate_one_to_one_match.sh --mode match_and_rerank --k 10 --min-domain 0.30 --limit-faculty 100 --commit-every 30
#   ./scripts/generate_one_to_one_match.sh --mode rerank_only --limit-faculty 100 --rerank-workers 4 --max-context-chars 100000
#   ./scripts/generate_one_to_one_match.sh --k 20 --min-domain 0.10 --limit-faculty 0 --commit-every 50

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Run one-to-one faculty <-> opportunity matching.

Existing pairs are overwritten by upsert on (grant_id, faculty_id).
Rows not touched in this run remain in the DB.

Usage:
  ./scripts/generate_one_to_one_match.sh [--mode <match_and_rerank|rerank_only>] [--k <int>] [--min-domain <float>] [--limit-faculty <int>] [--commit-every <int>] [--rerank-workers <int>] [--max-context-chars <int>]

Options:
  --mode <str>              match_and_rerank | rerank_only (default: match_and_rerank)
  --k <int>                 Top-k opportunities per faculty (default: 50)
  --min-domain <float>      Domain similarity threshold (default: 0.30)
  --limit-faculty <int>     Max faculty rows to process; 0 = all (default: 600)
  --commit-every <int>      Commit frequency in faculty loop (default: 30)
  --rerank-workers <int>    Workers for rerank-only mode (default: 4)
  --max-context-chars <int> Max payload chars per rerank chunk (default: 100000)
  -h, --help            Show this help message

Examples:
  ./scripts/generate_one_to_one_match.sh
  ./scripts/generate_one_to_one_match.sh --mode match_and_rerank --k 10 --min-domain 0.30 --limit-faculty 100 --commit-every 30
  ./scripts/generate_one_to_one_match.sh --mode rerank_only --limit-faculty 100 --rerank-workers 4 --max-context-chars 100000
  ./scripts/generate_one_to_one_match.sh --k 20 --min-domain 0.10 --limit-faculty 0 --commit-every 50
EOF
}

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

MODE="match_and_rerank"
K="50"
MIN_DOMAIN="0.3"
LIMIT_FACULTY="600"
COMMIT_EVERY="30"
RERANK_WORKERS="4"
MAX_CONTEXT_CHARS="100000"

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
    --k)
      if [[ $# -lt 2 ]]; then
        echo "Error: --k requires a value." >&2
        print_usage >&2
        exit 1
      fi
      K="$2"
      shift 2
      ;;
    --min-domain)
      if [[ $# -lt 2 ]]; then
        echo "Error: --min-domain requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MIN_DOMAIN="$2"
      shift 2
      ;;
    --limit-faculty)
      if [[ $# -lt 2 ]]; then
        echo "Error: --limit-faculty requires a value." >&2
        print_usage >&2
        exit 1
      fi
      LIMIT_FACULTY="$2"
      shift 2
      ;;
    --commit-every)
      if [[ $# -lt 2 ]]; then
        echo "Error: --commit-every requires a value." >&2
        print_usage >&2
        exit 1
      fi
      COMMIT_EVERY="$2"
      shift 2
      ;;
    --rerank-workers)
      if [[ $# -lt 2 ]]; then
        echo "Error: --rerank-workers requires a value." >&2
        print_usage >&2
        exit 1
      fi
      RERANK_WORKERS="$2"
      shift 2
      ;;
    --max-context-chars)
      if [[ $# -lt 2 ]]; then
        echo "Error: --max-context-chars requires a value." >&2
        print_usage >&2
        exit 1
      fi
      MAX_CONTEXT_CHARS="$2"
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

if [[ "$MODE" != "match_and_rerank" && "$MODE" != "rerank_only" ]]; then
  echo "Error: mode must be match_and_rerank or rerank_only (got: '$MODE')." >&2
  print_usage >&2
  exit 1
fi
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
if ! [[ "$RERANK_WORKERS" =~ ^-?[0-9]+$ ]]; then
  echo "Error: rerank_workers must be an integer (got: '$RERANK_WORKERS')." >&2
  print_usage >&2
  exit 1
fi
if ! [[ "$MAX_CONTEXT_CHARS" =~ ^-?[0-9]+$ ]]; then
  echo "Error: max_context_chars must be an integer (got: '$MAX_CONTEXT_CHARS')." >&2
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
echo "  mode          : $MODE"
echo "  k             : $K"
echo "  min_domain    : $MIN_DOMAIN"
echo "  limit_faculty : $LIMIT_FACULTY"
echo "  commit_every  : $COMMIT_EVERY"
echo "  rerank_workers: $RERANK_WORKERS"
echo "  max_ctx_chars : $MAX_CONTEXT_CHARS"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/services/matching/generate_one_to_one_match.py" \
  --mode "$MODE" \
  --k "$K" \
  --min-domain "$MIN_DOMAIN" \
  --limit-faculty "$LIMIT_FACULTY" \
  --commit-every "$COMMIT_EVERY" \
  --rerank-workers "$RERANK_WORKERS" \
  --max-context-chars "$MAX_CONTEXT_CHARS"

echo
echo "One-to-one pipeline completed."
