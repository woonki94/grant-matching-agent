#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the faculty keyword generation pipeline
# Usage:
#   ./scripts/generate_faculty_keywords.sh [batch_size] [max_keywords]
# ───────────────────────────────────────────────

set -euo pipefail

# go to repo root
cd "$(dirname "$0")/.."

# activate venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# load env if present (DB/url/API keys for the LLM, etc.)
if [ -f ".env" ]; then
  echo "Loading .env..."
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

BATCH_SIZE=${1:-50}
MAX_KEYWORDS=${2:-25}

echo " Generating faculty keywords ..."
echo "   Batch size   : $BATCH_SIZE"
echo "   Max keywords : $MAX_KEYWORDS"
echo

# call the actual module in your repo
python -m services.faculty.generate_keywords "$BATCH_SIZE" "$MAX_KEYWORDS"

echo "Faculty keyword generation complete"
