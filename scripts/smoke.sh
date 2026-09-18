#!/usr/bin/env bash
# Exercise a running proxy end to end: model list, messages, streaming, chat.
#
#   ./scripts/run.sh go &
#   ./scripts/smoke.sh
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env
require curl

BASE="http://localhost:${SERVER_PORT:-8080}"
MODEL="${PROXY_MODEL_NAME:-claude-opus-5}"

# Expanding an empty array under `set -u` is an error on bash 3.2, so the
# auth header is carried as a plain string and split deliberately below.
AUTH_HEADER=""
if [ -n "${PROXY_API_KEY:-}" ]; then
  AUTH_HEADER="x-api-key: ${PROXY_API_KEY}"
fi

# curl_proxy <curl args...> — adds the auth header when one is configured.
curl_proxy() {
  if [ -n "$AUTH_HEADER" ]; then
    curl -H "$AUTH_HEADER" "$@"
  else
    curl "$@"
  fi
}

bold "GET $BASE/v1/models"
curl_proxy -sS "$BASE/v1/models"; echo; echo

bold "POST $BASE/v1/messages  (anthropic -> openai -> anthropic)"
curl_proxy -sS -H "Content-Type: application/json" "$BASE/v1/messages" -d "{
  \"model\": \"$MODEL\",
  \"max_tokens\": 64,
  \"system\": \"Answer in one short sentence.\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Which is larger, 9.11 or 9.8?\"}]
}"; echo; echo

bold "POST $BASE/v1/messages  (streaming)"
curl_proxy -sSN -H "Content-Type: application/json" "$BASE/v1/messages" -d "{
  \"model\": \"$MODEL\",
  \"max_tokens\": 64,
  \"stream\": true,
  \"messages\": [{\"role\": \"user\", \"content\": \"Count to three.\"}]
}" | head -n 24; echo

bold "POST $BASE/v1/chat/completions  (forwarded unchanged)"
curl_proxy -sS -H "Content-Type: application/json" "$BASE/v1/chat/completions" -d "{
  \"model\": \"$MODEL\",
  \"max_tokens\": 64,
  \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}]
}"; echo

pass "smoke complete"
