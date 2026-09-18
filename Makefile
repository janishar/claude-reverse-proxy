# Claude Reverse Proxy — common tasks.
# Run `make` for the list.

.DEFAULT_GOAL := help
.PHONY: help setup env run run-go run-py run-node build test test-go test-py test-node \
        cover parity smoke lint fmt vet clean

GO      ?= go
PYTHON  ?= python3
NPM     ?= npm

help: ## Show this help
	@echo "Claude Reverse Proxy"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Run one implementation:  make run IMPL=go|py|node"

# ── Setup ────────────────────────────────────────────────────────────────────

setup: env ## Install dependencies and create .env
	@cd nodeproxy && $(NPM) install --silent
	@echo "Ready. Edit .env, then: make run IMPL=go"

env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "created .env — fill in PROVIDER_API_KEY")

# ── Run ──────────────────────────────────────────────────────────────────────

IMPL ?= go

run: ## Run one implementation (IMPL=go|py|node)
	@./scripts/run.sh $(IMPL)

run-go:   ## Run the Go proxy
	@./scripts/run.sh go
run-py:   ## Run the Python proxy
	@./scripts/run.sh py
run-node: ## Run the Node proxy
	@./scripts/run.sh node

build: ## Build the Go binary
	@cd goproxy && $(GO) build -o goproxy .
	@echo "built goproxy/goproxy"

# ── Test ─────────────────────────────────────────────────────────────────────

test: ## Run every test suite plus the parity check
	@./scripts/test.sh all

test-go:   ## Run the Go tests
	@cd goproxy && $(GO) test ./...
test-py:   ## Run the Python tests
	@cd pyproxy && $(PYTHON) -m unittest discover -p "test_*.py"
test-node: ## Run the Node tests
	@cd nodeproxy && $(NPM) test --silent

cover: ## Go coverage report
	@cd goproxy && $(GO) test -coverprofile=coverage.out ./... \
		&& $(GO) tool cover -func=coverage.out | tail -1 \
		&& echo "html: go tool cover -html=goproxy/coverage.out"

parity: build ## Check all three implementations behave identically
	@$(PYTHON) scripts/parity_check.py

smoke: ## Exercise a running proxy with curl
	@./scripts/smoke.sh

lint: ## Lint the shell scripts (same command CI runs)
	@for s in scripts/*.sh; do bash -n "$$s" || exit 1; done
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck --severity=warning --external-sources --source-path=SCRIPTDIR scripts/*.sh \
			&& echo "shellcheck: clean"; \
	else \
		echo "shellcheck not installed — syntax checked only"; \
		echo "  install: brew install shellcheck  |  pip install shellcheck-py"; \
	fi

# ── Housekeeping ─────────────────────────────────────────────────────────────

fmt: ## Format the Go sources
	@cd goproxy && gofmt -w .

vet: ## Vet the Go sources
	@cd goproxy && $(GO) vet ./...

clean: ## Remove build artefacts
	@rm -f goproxy/goproxy goproxy/coverage.out
	@rm -rf pyproxy/__pycache__ nodeproxy/node_modules
	@echo "cleaned"
