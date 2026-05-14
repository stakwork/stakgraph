# syntax=docker/dockerfile:1.7
#
# Build a Bifrost image that can load our Go plugin.
#
# Upstream `maximhq/bifrost:latest` is statically linked and therefore cannot
# load .so plugins (Go's plugin system requires dynamic linking — see
# https://docs.getbifrost.ai/plugins/building-dynamic-binary ).
#
# This Dockerfile does three things, in one image, in one CI step:
#   1. Clone bifrost @ ${BIFROST_VERSION} and build bifrost-http with CGO
#      enabled and *without* the static-linking flags.
#   2. Build our plugin .so against the same source tree, so every
#      transitive dep (sonic, fasthttp, mcp-go, …) is at the exact version
#      bifrost-http was built with — Go's plugin loader requires that.
#   3. Produce a minimal Alpine runtime image with both artifacts in place.

# Bifrost releases per-component tags (e.g. transports/v1.5.2, core/v1.5.8).
# We pin the transports tag because that's what produces bifrost-http; the
# core version comes along via go.work / replace directives in the tree.
ARG GO_VERSION=1.26.2
ARG BIFROST_VERSION=transports/v1.5.2

# ---------------------------------------------------------------------------
# Stage 1: build bifrost-http (dynamic) + our plugin from upstream sources.
# ---------------------------------------------------------------------------
FROM golang:${GO_VERSION}-alpine3.23 AS builder

ARG BIFROST_VERSION
ARG TARGETARCH

# git:            clone bifrost source.
# gcc, musl-dev:  CGO toolchain for bifrost-http + plugin.
# binutils-gold:  the linker gcc invokes via -fuse-ld=gold (Bifrost build).
# sqlite-dev:     headers for the embedded SQLite store.
# bash, make:     used by upstream Makefiles (kept for parity / future use).
RUN apk add --no-cache git gcc musl-dev binutils-gold sqlite-dev bash make

WORKDIR /src

# Clone bifrost at the exact tag we're targeting. Shallow clone keeps the
# image cache cheap.
RUN git clone --depth=1 --branch ${BIFROST_VERSION} \
        https://github.com/maximhq/bifrost.git . \
 && git rev-parse HEAD > /src/BIFROST_COMMIT

# --- 1a. Build bifrost-http dynamically linked. ---
#
# We deliberately do NOT use upstream's Makefile here because it pulls a
# whole UI build (Node) we don't need for an API-only governance gateway.
# bifrost-http has `//go:embed all:ui` so we stub the directory with a
# placeholder before compiling — the embed succeeds, the UI just serves
# a stub page (we hit Bifrost's JSON endpoints, not the UI).
WORKDIR /src/transports
RUN mkdir -p bifrost-http/ui \
 && printf 'Bifrost UI not bundled in this build (API-only).\n' \
        > bifrost-http/ui/index.html
RUN CGO_ENABLED=1 GOOS=linux \
    go build \
        -ldflags="-w -s -X main.Version=${BIFROST_VERSION}" \
        -trimpath \
        -o /out/bifrost-http \
        ./bifrost-http

# Sanity check: it must be dynamically linked or it won't load plugins.
RUN ldd /out/bifrost-http | grep -q 'musl' && echo "OK: dynamic" \
 || (echo "FAIL: bifrost-http is statically linked"; ldd /out/bifrost-http; exit 1)

# --- 1b. Build our plugin against this same source tree. ---
#
# By replacing the plugin's `github.com/maximhq/bifrost/core` with the
# local checkout, the plugin and bifrost-http compile against byte-identical
# sources and matching transitive deps. This is what avoids the dreaded
# "plugin was built with a different version of package …" errors.
WORKDIR /plugin
COPY go.mod ./
COPY main.go ./
RUN go mod edit -replace github.com/maximhq/bifrost/core=/src/core \
 && go mod tidy \
 && CGO_ENABLED=1 GOOS=linux \
    go build -buildmode=plugin \
        -ldflags="-w -s" -trimpath \
        -o /out/stakgraph-gateway.so .

# ---------------------------------------------------------------------------
# Stage 2: minimal runtime — Alpine + dynamic libc + ca-certs.
# ---------------------------------------------------------------------------
FROM alpine:3.23

# musl + libgcc: required by any CGO-enabled binary built on alpine.
# ca-certificates: TLS to upstream providers.
# wget: container healthcheck.
RUN apk add --no-cache musl libgcc ca-certificates wget tini \
 && addgroup -S app && adduser -S -G app -h /app app

WORKDIR /app

COPY --from=builder /out/bifrost-http        /app/bifrost-http
COPY --from=builder /out/stakgraph-gateway.so /app/plugins/stakgraph-gateway.so
COPY --from=builder /src/BIFROST_COMMIT       /app/BIFROST_COMMIT

# Data dir is where bifrost reads config.json and writes config.db / logs.db.
# Mount it as a volume in docker-compose to persist across restarts.
RUN mkdir -p /app/data/logs && chown -R app:app /app

USER app

# Bifrost defaults: listen on 0.0.0.0 inside the container. Compose maps to 8181.
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    APP_DIR=/app/data \
    LOG_LEVEL=info \
    LOG_STYLE=pretty

EXPOSE 8080
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:${APP_PORT}/metrics || exit 1

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["/app/bifrost-http", \
     "-app-dir", "/app/data", \
     "-host", "0.0.0.0", \
     "-port", "8080", \
     "-log-level", "info", \
     "-log-style", "pretty"]
