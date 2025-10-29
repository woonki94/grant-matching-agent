#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the grant fetch + commit pipeline
# Usage:
#   ./scripts/fetch_commit_grant.sh [page_offset] [page_size] [query]


set -euo pipefail

cd "$(dirname "$0")/.."

# Optional: load .env for DB connection or API creds
if [ -f ".env" ]; then
  echo "Loading .env..."
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

PAGE_OFFSET=${1:-1}
PAGE_SIZE=${2:-5}
QUERY=${3:-""}   # optional text search term

echo " Running fetch_commit_grant.py ..."
echo "   Page offset : $PAGE_OFFSET"
echo "   Page size   : $PAGE_SIZE"
echo "   Query       : ${QUERY:-<none>}"
echo

python -m services.fetch_commit_grant "$PAGE_OFFSET" "$PAGE_SIZE" "$QUERY"