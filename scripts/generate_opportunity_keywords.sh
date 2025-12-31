#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Run the keyword mining pipeline
# Usage:
#   ./scripts/generate_opportunity_keywords.sh [batch_size] [max_keywords]
# ───────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".env" ]; then
  echo "Loading .env..."

  export $(grep -v '^#' .env | xargs)
fi

BATCH_SIZE=${1:-50}
MAX_KEYWORDS=${2:-25}

echo " Running generate_keyword.py ..."
echo "   Batch size   : $BATCH_SIZE"
echo "   Max keywords : $MAX_KEYWORDS"
echo


# Execute as module so imports always resolve
python -m services.opportunity.generate_keywords "$BATCH_SIZE" "$MAX_KEYWORDS"