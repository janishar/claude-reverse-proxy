#!/usr/bin/env bash
# Run the test suite for one implementation, or all three plus the parity check.
#
#   ./scripts/test.sh          # everything
#   ./scripts/test.sh go       # or py | node | parity
set -uo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# shellcheck disable=SC2329  # invoked indirectly by report()
run_go() {
  require go
  bold "goproxy"
  (
    cd "$REPO_ROOT/goproxy" || exit 1
    unformatted="$(gofmt -l .)"
    if [ -n "$unformatted" ]; then
      fail "gofmt: these files need formatting (run 'make fmt'):"
      # shellcheck disable=SC2086 # deliberate split: one line per file
      printf '  %s\n' $unformatted
      exit 1
    fi
    go vet ./... || exit 1
    go test ./... || exit 1
  )
}

# shellcheck disable=SC2329  # invoked indirectly by report()
run_py() {
  require python3
  bold "pyproxy"
  (
    cd "$REPO_ROOT/pyproxy" || exit 1
    python3 -m unittest discover -p "test_*.py" 2>&1 | grep -vE '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
    exit "${PIPESTATUS[0]}"
  )
}

# shellcheck disable=SC2329  # invoked indirectly by report()
run_node() {
  require node
  bold "nodeproxy"
  if [ ! -d "$REPO_ROOT/nodeproxy/node_modules" ]; then
    info "installing node dependencies"
    (cd "$REPO_ROOT/nodeproxy" && npm install --silent) || return 1
  fi
  (cd "$REPO_ROOT/nodeproxy" && npm test --silent)
}

# shellcheck disable=SC2329  # invoked indirectly by report()
run_parity() {
  require python3
  bold "parity"
  python3 "$REPO_ROOT/scripts/parity_check.py"
}

failed=0

# report <name> <function> — run a suite and record whether it passed.
report() {
  local name="$1" runner="$2"
  if "$runner"; then
    pass "$name"
  else
    fail "$name"
    failed=1
  fi
}

case "${1:-all}" in
  go)        report go     run_go ;;
  py|python) report python run_py ;;
  node|js)   report node   run_node ;;
  parity)    report parity run_parity ;;
  all)
    report go     run_go
    echo
    report python run_py
    echo
    report node   run_node
    echo
    report parity run_parity
    ;;
  *)
    fail "unknown target '${1}' (expected: go | py | node | parity | all)"
    exit 1
    ;;
esac

echo
if [ "$failed" -eq 0 ]; then
  pass "all suites green"
else
  fail "some suites failed"
fi
exit "$failed"
