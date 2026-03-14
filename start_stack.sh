#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_ENV_FILE="${MPS_CLIENT_ENV_FILE:-$REPO_ROOT/config/client.env}"
SERVER_ENV_FILE="${MPS_SERVER_ENV_FILE:-$REPO_ROOT/config/server.env}"
DEFAULT_PYTHON_BIN="$REPO_ROOT/../.venv/bin/python"
LOG_DIR="${MPS_LOG_DIR:-/tmp/mps_stack_logs}"

declare -A FILE_ENV=()

ORB_PID=""
SERVER_PID=""
ORB_LOG=""
SERVER_LOG=""
VLM_ENABLED=""
VLM_HOST=""
VLM_PORT=""

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

resolve_repo_path() {
  local raw_path="$1"
  if [[ "$raw_path" = /* ]]; then
    printf '%s' "$raw_path"
  else
    printf '%s' "$REPO_ROOT/$raw_path"
  fi
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
}

export_loaded_env() {
  local key
  for key in "${!FILE_ENV[@]}"; do
    if [[ -z "${!key+x}" ]]; then
      export "$key=${FILE_ENV[$key]}"
    fi
  done
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local pid="$4"
  local attempts="${5:-120}"
  local sleep_s="${6:-0.25}"
  local i

  for ((i = 0; i < attempts; i += 1)); do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[$name] process exited before port ${host}:${port} became ready" >&2
      return 1
    fi
    if "$PYTHON_BIN" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.2)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
else:
    sock.close()
    raise SystemExit(0)
PY
    then
      return 0
    fi
    sleep "$sleep_s"
  done

  echo "[$name] timed out waiting for ${host}:${port}" >&2
  return 1
}

port_in_use() {
  local host="$1"
  local port="$2"
  if "$PYTHON_BIN" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.2)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
else:
    sock.close()
    raise SystemExit(0)
PY
  then
    return 0
  fi
  return 1
}

print_log_tail() {
  local log_path="$1"
  if [[ -f "$log_path" ]]; then
    echo "--- tail: $log_path ---" >&2
    tail -n 40 "$log_path" >&2 || true
  fi
}

stop_process() {
  local pid="$1"
  local name="$2"
  local i

  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  for ((i = 0; i < 20; i += 1)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done

  echo "[start_stack] forcing ${name} pid=${pid} to stop" >&2
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  stop_process "$SERVER_PID" "server"
  stop_process "$ORB_PID" "orb_adapter"
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

source "$REPO_ROOT/scripts/common_env.sh"
CLIENT_ENV_FILE="$(mps_resolve_env_file "$CLIENT_ENV_FILE" "$REPO_ROOT/config/client.env.example")"
SERVER_ENV_FILE="$(mps_resolve_env_file "$SERVER_ENV_FILE" "$REPO_ROOT/config/server.env.example")"

load_env_file "$CLIENT_ENV_FILE"
load_env_file "$SERVER_ENV_FILE"
export_loaded_env

if [[ -z "${GROQ_API_KEY:-}" && -n "${MPS_GROQ_API_KEY:-}" ]]; then
  export GROQ_API_KEY="$MPS_GROQ_API_KEY"
fi

if [[ "${MPS_HEADLESS:-0}" == "1" ]]; then
  export MPS_ENABLE_GUI=0
  export MPS_ENABLE_USER_GUI=0
  export MPS_SERVER_ENABLE_GUI=0
fi

if [[ -n "${MPS_DATASET_ROOT:-}" ]]; then
  export MPS_DATASET_ROOT
  MPS_DATASET_ROOT="$(resolve_repo_path "$MPS_DATASET_ROOT")"
  export MPS_DATASET_ROOT
fi

PYTHON_BIN_RAW="${MPS_PYTHON_BIN:-${MPS_SERVER_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}}"
PYTHON_BIN="$(resolve_repo_path "$PYTHON_BIN_RAW")"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[start_stack] python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

SLAM_HOST="${MPS_SLAM_HOST:-127.0.0.1}"
SLAM_PORT="${MPS_SLAM_PORT:-19090}"
SERVER_HOST="${MPS_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${MPS_SERVER_PORT:-19100}"
VLM_ENABLED="${MPS_SERVER_ENABLE_VLM_API:-0}"
VLM_HOST="${MPS_SERVER_VLM_BIND_HOST:-127.0.0.1}"
VLM_PORT="${MPS_SERVER_VLM_PORT:-19110}"

mkdir -p "$LOG_DIR"
ORB_LOG="$LOG_DIR/orb_adapter.log"
SERVER_LOG="$LOG_DIR/server.log"
: > "$ORB_LOG"
: > "$SERVER_LOG"

if port_in_use "$SLAM_HOST" "$SLAM_PORT"; then
  echo "[start_stack] port already in use: ${SLAM_HOST}:${SLAM_PORT}" >&2
  echo "Stop the existing ORB adapter or change MPS_SLAM_PORT." >&2
  exit 1
fi

if port_in_use "$SERVER_HOST" "$SERVER_PORT"; then
  echo "[start_stack] port already in use: ${SERVER_HOST}:${SERVER_PORT}" >&2
  echo "Stop the existing server or change MPS_SERVER_PORT." >&2
  exit 1
fi

if [[ "$VLM_ENABLED" == "1" ]] && port_in_use "$VLM_HOST" "$VLM_PORT"; then
  echo "[start_stack] port already in use: ${VLM_HOST}:${VLM_PORT}" >&2
  echo "Stop the existing VLM API process or change MPS_SERVER_VLM_PORT." >&2
  exit 1
fi

echo "[start_stack] repo=$REPO_ROOT"
echo "[start_stack] python=$PYTHON_BIN"
echo "[start_stack] client_env=$CLIENT_ENV_FILE"
echo "[start_stack] server_env=$SERVER_ENV_FILE"
echo "[start_stack] dataset=${MPS_DATASET_ROOT:-auto}"
echo "[start_stack] logs=$LOG_DIR"
echo "[start_stack] headless=${MPS_HEADLESS:-0}"
if [[ "$VLM_ENABLED" == "1" ]]; then
  echo "[start_stack] vlm_api=http://${VLM_HOST}:${VLM_PORT}"
fi

"$REPO_ROOT/integration/orb_native/run_native_adapter.sh" >"$ORB_LOG" 2>&1 &
ORB_PID="$!"
if ! wait_for_port "$SLAM_HOST" "$SLAM_PORT" "orb_adapter" "$ORB_PID"; then
  print_log_tail "$ORB_LOG"
  exit 1
fi
echo "[start_stack] ORB adapter ready pid=$ORB_PID log=$ORB_LOG"

"$REPO_ROOT/scripts/run_server.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
if ! wait_for_port "$SERVER_HOST" "$SERVER_PORT" "server" "$SERVER_PID"; then
  print_log_tail "$SERVER_LOG"
  exit 1
fi
echo "[start_stack] Server ready pid=$SERVER_PID log=$SERVER_LOG"

if [[ "$VLM_ENABLED" == "1" ]]; then
  if ! wait_for_port "$VLM_HOST" "$VLM_PORT" "vlm_api" "$SERVER_PID"; then
    print_log_tail "$SERVER_LOG"
    exit 1
  fi
  echo "[start_stack] VLM API ready http://${VLM_HOST}:${VLM_PORT}"
fi

if [[ "${MPS_SKIP_CLIENT:-0}" == "1" ]]; then
  echo "[start_stack] Backend is running without client. Press Ctrl+C to stop."
  while true; do
    sleep 3600
  done
fi

echo "[start_stack] Starting client"
"$PYTHON_BIN" "$REPO_ROOT/client/main.py"
