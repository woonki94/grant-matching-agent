#!/usr/bin/env bash
set -euo pipefail

# Run SQL migrations (in db/migrations) against the configured DATABASE_URL.
# Requires `psql` to be available.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables if present
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading .env..."
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  if [[ -n "${PGUSER:-}" && -n "${PGPASSWORD:-}" && -n "${PGHOST:-}" && -n "${PGPORT:-}" && -n "${PGDATABASE:-}" ]]; then
    DATABASE_URL="postgresql://${PGUSER}:${PGPASSWORD}@${PGHOST}:${PGPORT}/${PGDATABASE}"
    export DATABASE_URL
    echo "DATABASE_URL not set; constructed from PG* env vars."
  else
    echo "DATABASE_URL is not set, and PG* env vars are incomplete. Export DATABASE_URL or set PGUSER/PGPASSWORD/PGHOST/PGPORT/PGDATABASE in .env."
    exit 1
  fi
fi

# psql does not accept SQLAlchemy-style URLs (postgresql+psycopg2://)
if [[ "${DATABASE_URL}" == postgresql+psycopg2://* ]]; then
  DATABASE_URL="postgresql://${DATABASE_URL#postgresql+psycopg2://}"
  export DATABASE_URL
  echo "Normalized DATABASE_URL for psql."
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required to run migrations."
  exit 1
fi

echo "Ensuring migrations table exists..."
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
SQL

echo "Running SQL migrations..."
shopt -s nullglob
files=("$PROJECT_ROOT/db/migrations/"*.sql)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No SQL migrations found."
  exit 0
fi

for f in "${files[@]}"; do
  fname="$(basename "$f")"
  already_applied="$(psql "$DATABASE_URL" -tAc "SELECT 1 FROM schema_migrations WHERE filename = '$fname' LIMIT 1;")"
  if [[ "$already_applied" == "1" ]]; then
    echo "  skipping ${fname} (already applied)"
    continue
  fi
  echo "  applying ${fname}"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "INSERT INTO schema_migrations (filename) VALUES ('$fname');"
done

echo "Migrations complete."
