.PHONY: help build dev clean docker-build docker-up docker-down docker-logs deps tidy ui ui-dev ui-build ui-install tygo tygo-check

# Plugin output
PLUGIN_NAME = stakgraph-gateway
OUTPUT_DIR  = build
OUTPUT      = $(OUTPUT_DIR)/$(PLUGIN_NAME).so

# Bifrost upstream version this plugin is built against. Must match the
# bifrost-http binary it gets loaded into — Go plugins are strict.
BIFROST_VERSION ?= transports/v1.5.2
GO_VERSION      ?= 1.26.2

# Docker image we build (dynamic bifrost-http + this plugin).
IMAGE ?= stakgraph-gateway:dev

help:
	@echo "Stakgraph gateway plugin"
	@echo
	@echo "Local plugin build (matches host OS/arch):"
	@echo "  make dev               Build $(OUTPUT) for local testing"
	@echo "  make build             Same as dev, plus -trimpath / -ldflags=-w -s"
	@echo "  make clean             Remove $(OUTPUT_DIR)/"
	@echo
	@echo "Admin UI (Preact + Vite, served under /_plugin/ui/):"
	@echo "  make ui-install        npm install for the SPA"
	@echo "  make ui-build          Build SPA bundle into internal/adminapi/ui/dist"
	@echo "  make ui-dev            Vite dev server with HMR (proxies /_plugin/* to :8181)"
	@echo "  make tygo              Regenerate ui/src/api/types.ts from Go structs"
	@echo "  make tygo-check        CI: fail if generated types.ts is out of date"
	@echo
	@echo "Docker (the supported path — bifrost-latest can't load .so):"
	@echo "  make docker-build      Build $(IMAGE): dynamic bifrost-http + plugin + UI"
	@echo "  make docker-up         docker compose up -d"
	@echo "  make docker-down       docker compose down"
	@echo "  make docker-logs       Tail bifrost+plugin logs"
	@echo
	@echo "Pinned versions:"
	@echo "  BIFROST_VERSION=$(BIFROST_VERSION)"
	@echo "  GO_VERSION=$(GO_VERSION)"

deps tidy:
	go mod tidy

dev:
	@mkdir -p $(OUTPUT_DIR)
	CGO_ENABLED=1 go build -buildmode=plugin -o $(OUTPUT) .
	@echo "Built $(OUTPUT)"

build:
	@mkdir -p $(OUTPUT_DIR)
	CGO_ENABLED=1 go build -buildmode=plugin -ldflags="-w -s" -trimpath -o $(OUTPUT) .
	@echo "Built $(OUTPUT)"

clean:
	rm -rf $(OUTPUT_DIR)

docker-build:
	docker build \
		--build-arg BIFROST_VERSION=$(BIFROST_VERSION) \
		--build-arg GO_VERSION=$(GO_VERSION) \
		-t $(IMAGE) .

docker-up: docker-build
	docker compose up -d
	@echo "Bifrost up on http://localhost:8181"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f bifrost

# ─── UI ───────────────────────────────────────────────────────────────

UI_DIR = internal/adminapi/ui

ui-install:
	cd $(UI_DIR) && npm install --no-audit --no-fund

ui-build:
	cd $(UI_DIR) && npm run build

ui-dev:
	cd $(UI_DIR) && npm run dev

# Alias for the most common UI workflow: install then build. Run
# before `make dev` if iterating on the plugin and want a refreshed
# bundle in the embedded //go:embed dir.
ui: ui-install ui-build

# ─── tygo (Go -> TS struct codegen) ───────────────────────────────────
#
# Install once per dev machine:
#   go install github.com/gzuidhof/tygo@latest

tygo:
	tygo generate

# CI hook: regenerate, then fail the build if the diff is non-empty.
# Catches the case where someone added a Go struct field but forgot
# to commit the resulting TS change — the dashboard would then
# silently decode partial responses.
tygo-check: tygo
	@if ! git diff --quiet $(UI_DIR)/src/api/types.ts; then \
		echo "FAIL: $(UI_DIR)/src/api/types.ts is out of date. Run 'make tygo' and commit."; \
		git --no-pager diff $(UI_DIR)/src/api/types.ts; \
		exit 1; \
	fi
