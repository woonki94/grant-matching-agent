#!/usr/bin/env bash
set -euo pipefail

python -m services.agent.agent_cli "$@"
