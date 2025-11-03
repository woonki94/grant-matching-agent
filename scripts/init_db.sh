#!/usr/bin/env bash
# ───────────────────────────────────────────────
# Initialize PostgreSQL tables via SQLAlchemy Base
# Usage:
#   ./scripts/init_db.sh
# ───────────────────────────────────────────────

set -e  # Exit immediately on error
set -u  # Treat unset vars as errors
set -o pipefail

cd "$(dirname "$0")/.."

# Optional: load .env variables if you have DATABASE_URL or POSTGRES_DSN
if [ -f ".env" ]; then
  echo "Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
fi

echo "Initializing database schema..."
python -m db.init_db   # or python -m data.init_db depending on venv setup
echo "Database initialization complete!"


