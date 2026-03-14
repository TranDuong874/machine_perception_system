#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

SERVER_ENV_FILE="${MPS_SERVER_ENV_FILE:-$MPS_REPO_ROOT/config/server.env}"
SERVER_ENV_FILE="$(mps_resolve_env_file "$SERVER_ENV_FILE" "$MPS_REPO_ROOT/config/server.env.example")"
DEFAULT_PYTHON_BIN="$MPS_REPO_ROOT/../.venv/bin/python"

mps_load_env_file "$SERVER_ENV_FILE"
mps_bridge_groq_key

MODE="${1:-current}"
QUESTION="${2:-What am I seeing right now?}"
TOP_K="${3:-3}"
PYTHON_BIN_RAW="${MPS_PYTHON_BIN:-${MPS_SERVER_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}}"
PYTHON_BIN="$(mps_resolve_repo_path "$PYTHON_BIN_RAW")"
VLM_HOST="${MPS_SERVER_VLM_BIND_HOST:-127.0.0.1}"
VLM_PORT="${MPS_SERVER_VLM_PORT:-19110}"

case "$MODE" in
  current)
    ENDPOINT="current-scene"
    ;;
  rag)
    ENDPOINT="video-rag"
    ;;
  *)
    echo "usage: $0 [current|rag] [question] [top_k]" >&2
    exit 1
    ;;
esac

exec "$PYTHON_BIN" - "$VLM_HOST" "$VLM_PORT" "$ENDPOINT" "$QUESTION" "$TOP_K" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host, port, endpoint, question, top_k = sys.argv[1:6]
url = f"http://{host}:{port}/query/{endpoint}"
payload = json.dumps({"question": question, "top_k": int(top_k)}).encode("utf-8")
request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        print(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode("utf-8"))
    raise
PY
