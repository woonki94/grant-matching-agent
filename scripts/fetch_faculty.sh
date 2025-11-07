#!/usr/bin/env bash
# Run the faculty fetch + save pipeline

set -euo pipefail

# go to repo root
cd "$(dirname "$0")/.."

# activate venv if you want it to behave like your manual runs
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# load env if present (for DATABASE_URL / PG vars)
if [ -f ".env" ]; then
  echo "Loading .env..."
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

SRC_URL=${1:-"https://engineering.oregonstate.edu/people"}
LIMIT=${2:-0}

echo " Fetching / saving faculty ..."
echo "   Source : ${SRC_URL}"
echo "   Limit  : ${LIMIT}"

# 1) scrape (your real module path!)
if [ -n "$SRC_URL" ]; then
  python -m services.faculty.scrape_faculty "$SRC_URL" "$LIMIT"
else
  python -m services.faculty.scrape_faculty
fi

# 2) save to DB (assuming you also put this under services/faculty/)
python -m services.faculty.save_faculty

echo "Faculty fetch + save complete ✅"
