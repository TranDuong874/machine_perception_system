#!/usr/bin/env bash

MPS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPS_REPO_ROOT="$(cd "$MPS_SCRIPT_DIR/.." && pwd)"

mps_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

mps_strip_matching_quotes() {
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

mps_resolve_repo_path() {
  local raw_path="$1"
  if [[ "$raw_path" = /* ]]; then
    printf '%s' "$raw_path"
  else
    printf '%s' "$MPS_REPO_ROOT/$raw_path"
  fi
}

mps_load_env_file() {
  local path="$1"
  [[ -f "$path" ]] || return 0

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local line
    line="$(mps_trim "$raw_line")"
    [[ -n "$line" ]] || continue
    [[ "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="$(mps_trim "$key")"
    value="$(mps_strip_matching_quotes "$(mps_trim "$value")")"
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$path"
}

mps_bridge_groq_key() {
  if [[ -z "${GROQ_API_KEY:-}" && -n "${MPS_GROQ_API_KEY:-}" ]]; then
    export GROQ_API_KEY="$MPS_GROQ_API_KEY"
  fi
}

mps_resolve_env_file() {
  local primary="$1"
  local fallback="$2"
  if [[ -f "$primary" ]]; then
    printf '%s' "$primary"
    return
  fi
  printf '%s' "$fallback"
}
