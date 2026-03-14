#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

SERVER_ENV_FILE="${MPS_SERVER_ENV_FILE:-$MPS_REPO_ROOT/config/server.env}"
SERVER_ENV_FILE="$(mps_resolve_env_file "$SERVER_ENV_FILE" "$MPS_REPO_ROOT/config/server.env.example")"
DEFAULT_PYTHON_BIN="$MPS_REPO_ROOT/../.venv/bin/python"

mps_load_env_file "$SERVER_ENV_FILE"

PYTHON_BIN_RAW="${MPS_PYTHON_BIN:-${MPS_SERVER_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}}"
PYTHON_BIN="$(mps_resolve_repo_path "$PYTHON_BIN_RAW")"

exec "$PYTHON_BIN" "$MPS_REPO_ROOT/server/query_memory_index.py" "$@"
