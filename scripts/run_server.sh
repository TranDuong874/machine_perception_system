#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

SERVER_ENV_FILE="${MPS_SERVER_ENV_FILE:-$MPS_REPO_ROOT/config/server.env}"
SERVER_ENV_FILE="$(mps_resolve_env_file "$SERVER_ENV_FILE" "$MPS_REPO_ROOT/config/server.env.example")"
DEFAULT_SERVER_PYTHON_BIN="$MPS_REPO_ROOT/../.venv/bin/python"

mps_load_env_file "$SERVER_ENV_FILE"
mps_bridge_groq_key

server_python_bin_raw="${MPS_SERVER_PYTHON_BIN:-$DEFAULT_SERVER_PYTHON_BIN}"
server_python_bin="$(mps_resolve_repo_path "$server_python_bin_raw")"

if [[ ! -x "$server_python_bin" ]]; then
  echo "[run_server] python interpreter not found: $server_python_bin" >&2
  exit 1
fi

echo "[run_server] env_file=$SERVER_ENV_FILE"
echo "[run_server] depth_enabled=${MPS_SERVER_ENABLE_DEPTH:-0}"
echo "[run_server] python=$server_python_bin"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec "$server_python_bin" "$MPS_REPO_ROOT/server/main.py"
