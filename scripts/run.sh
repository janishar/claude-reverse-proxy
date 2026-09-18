#!/usr/bin/env bash
# Start one implementation with .env loaded.
#
#   ./scripts/run.sh go        # or py | node
#   SERVER_PORT=9000 ./scripts/run.sh node
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_env

impl="${1:-go}"
case "$impl" in
  go|goproxy)
    require go
    info "building goproxy"
    (cd "$REPO_ROOT/goproxy" && go build -o goproxy .)
    bold "starting goproxy on :${SERVER_PORT:-8080}"
    exec "$REPO_ROOT/goproxy/goproxy"
    ;;
  py|python|pyproxy)
    require python3
    bold "starting pyproxy on :${SERVER_PORT:-8080}"
    exec python3 "$REPO_ROOT/pyproxy/main.py"
    ;;
  node|nodeproxy|js)
    require node
    [ -d "$REPO_ROOT/nodeproxy/node_modules" ] || (info "installing deps" && cd "$REPO_ROOT/nodeproxy" && npm install --silent)
    bold "starting nodeproxy on :${SERVER_PORT:-8080}"
    exec node "$REPO_ROOT/nodeproxy/main.js"
    ;;
  *)
    fail "unknown implementation '$impl' (expected: go | py | node)"
    exit 1
    ;;
esac
