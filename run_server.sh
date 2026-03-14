#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_ENV_FILE="${MPS_SERVER_ENV_FILE:-$REPO_ROOT/config/server.env}"
DEFAULT_SERVER_PYTHON_BIN="$REPO_ROOT/../.venv/bin/python"

declare -A FILE_ENV=()

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

strip_matching_quotes() {
  local value="$1"
  if [[ ${#value} -ge 2 ]]; then
    local first="${value:0:1}"
    local last="${value: -1}"
    if [[ "$first" == "$last" && ( "$first" == '"' || "$first" == "'" ) ]]; then
      printf '%s' "${value:1:${#value}-2}"
      return
    fi
  fi
  printf '%s' "$value"
}

load_env_file() {
  local path="$1"
  [[ -f "$path" ]] || return 0

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line
    line="$(trim "$raw_line")"
    [[ -n "$line" ]] || continue
    [[ "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="$(trim "$key")"
    value="$(strip_matching_quotes "$(trim "$value")")"
    FILE_ENV["$key"]="$value"
  done < "$path"

  for key in "${!FILE_ENV[@]}"; do
    if [[ -z "${!key+x}" ]]; then
      export "$key=${FILE_ENV[$key]}"
    fi
  done
}

resolve_repo_path() {
  local raw_path="$1"
  if [[ "$raw_path" = /* ]]; then
    printf '%s' "$raw_path"
  else
    printf '%s' "$REPO_ROOT/$raw_path"
  fi
}

load_env_file "$SERVER_ENV_FILE"

server_python_bin_raw="${MPS_SERVER_PYTHON_BIN:-$DEFAULT_SERVER_PYTHON_BIN}"
server_python_bin="$(resolve_repo_path "$server_python_bin_raw")"
python_bin="$server_python_bin"

if [[ ! -x "$python_bin" ]]; then
  echo "[run_server] python interpreter not found: $python_bin" >&2
  exit 1
fi

echo "[run_server] env_file=$SERVER_ENV_FILE"
echo "[run_server] depth_enabled=${MPS_SERVER_ENABLE_DEPTH:-0}"
echo "[run_server] python=$python_bin"

exec "$python_bin" "$REPO_ROOT/server/main.py"
