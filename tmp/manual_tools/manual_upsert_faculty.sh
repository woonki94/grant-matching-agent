#!/usr/bin/env bash
set -euo pipefail

# Temporary helper for manual faculty insertion/update from JSON payload(s).
#
# Usage:
#   ./tmp/manual_tools/manual_upsert_faculty.sh --payload-file /path/to/payload.json [--no-postprocess]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

print_usage() {
  cat <<'EOF'
Temporarily upsert faculty payload(s) from a JSON payload file.

Behavior:
  1) create faculty row(s) if missing (requires source_url/info_source_url)
  2) apply FacultyProfileService edit flow for basic_info/data_from
  3) run postprocess by default (keywords + matches)

Usage:
  ./tmp/manual_tools/manual_upsert_faculty.sh --payload-file /abs/path/payload.json [--no-postprocess]

Payload file shapes:
  1) single object: { "email": "...", ... }
  2) array: [ { "email": "...", ... }, { "email": "...", ... } ]
  3) wrapper object: { "faculties": [ { ... }, { ... } ] }

Options:
  --payload-file <path>   JSON payload file (required)
  --no-postprocess        Skip postprocess regeneration/matching
  -h, --help              Show this help message
EOF
}

PAYLOAD_FILE=""
NO_POSTPROCESS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --payload-file)
      if [[ $# -lt 2 ]]; then
        echo "Error: --payload-file requires a value." >&2
        print_usage >&2
        exit 1
      fi
      PAYLOAD_FILE="$2"
      shift 2
      ;;
    --no-postprocess)
      NO_POSTPROCESS="1"
      shift
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

if [[ -z "$PAYLOAD_FILE" ]]; then
  echo "Error: --payload-file is required." >&2
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

PYTHON_FILE="$PROJECT_ROOT/tmp/manual_tools/manual_upsert_faculty.py"
EXTRA_ARGS=(--payload-file "$PAYLOAD_FILE")
if [[ "$NO_POSTPROCESS" == "1" ]]; then
  EXTRA_ARGS+=(--no-postprocess)
fi

"$PYTHON_BIN" "$PYTHON_FILE" "${EXTRA_ARGS[@]}"
