#!/usr/bin/env bash
# Shared helpers: environment loading and pretty output.
# Sourced by the other scripts; not meant to be run directly.

# Used by the scripts that source this file, not here.
# shellcheck disable=SC2034
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[36m→\033[0m %s\n' "$*"; }
pass() { printf '\033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$*" >&2; }

# Load .env if present. Values already exported win, so a one-off
# `PROXY_API_KEY=x ./scripts/run.sh go` still overrides the file.
load_env() {
  local env_file="${ENV_FILE:-$REPO_ROOT/.env}"
  [ -f "$env_file" ] || return 0
  info "loading $(basename "$env_file")"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    local key="${line%%=*}"
    local value="${line#*=}"
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    [ -n "$key" ] || continue
    # Strip one layer of surrounding quotes, if any.
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    # Anything already exported wins over the file.
    if [ -n "${!key+x}" ]; then
      continue
    fi
    export "$key=$value"
  done < "$env_file"
}

require() {
  command -v "$1" >/dev/null 2>&1 || { fail "$1 is not installed"; exit 1; }
}
