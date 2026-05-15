.PHONY: help build dev clean docker-build docker-up docker-down docker-logs deps tidy

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
	@echo "Docker (the supported path — bifrost-latest can't load .so):"
	@echo "  make docker-build      Build $(IMAGE): dynamic bifrost-http + plugin"
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
