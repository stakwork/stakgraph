# syntax=docker/dockerfile:1.7
#
# Build a Bifrost image that can load our Go plugin AND owns the
# container's public HTTP port via a thin Go wrapper.
#
# Upstream `maximhq/bifrost:latest` is statically linked and therefore cannot
# load .so plugins (Go's plugin system requires dynamic linking — see
# https://docs.getbifrost.ai/plugins/building-dynamic-binary ).
#
# This Dockerfile does four things, in one image, in one CI step:
#   1. Clone bifrost @ ${BIFROST_VERSION} and build bifrost-http with CGO
#      enabled and *without* the static-linking flags.
#   2. Build our plugin .so against the same source tree, so every
#      transitive dep (sonic, fasthttp, mcp-go, …) is at the exact version
#      bifrost-http was built with — Go's plugin loader requires that.
#   3. Build the wrapper binary that fronts both bifrost-http and the
#      plugin's in-process HTTP server. See gateway/wrapper/main.go
#      for the rationale. This is the container's entrypoint.
#   4. Produce a minimal Alpine runtime image with all three artifacts.

# Bifrost releases per-component tags (e.g. transports/v1.5.2, core/v1.5.8).
# We pin the transports tag because that's what produces bifrost-http; the
# core version comes along via go.work / replace directives in the tree.
ARG GO_VERSION=1.26.2
ARG NODE_VERSION=25
ARG BIFROST_VERSION=transports/v1.5.2

# ---------------------------------------------------------------------------
# Stage 0: clone bifrost source once and share via subsequent stages.
# A separate small stage keeps the `git clone` layer cache stable across
# UI- and Go-build edits.
# ---------------------------------------------------------------------------
FROM alpine:3.23 AS source
ARG BIFROST_VERSION
RUN apk add --no-cache git
WORKDIR /src
RUN git clone --depth=1 --branch ${BIFROST_VERSION} \
        https://github.com/maximhq/bifrost.git . \
 && git rev-parse HEAD > /src/BIFROST_COMMIT

# ---------------------------------------------------------------------------
# Stage 1: build the React/Vite UI exactly the way upstream does.
# bifrost-http has `//go:embed all:ui` against transports/bifrost-http/ui,
# so we drop the compiled assets in there before compiling Go.
# ---------------------------------------------------------------------------
FROM node:${NODE_VERSION}-alpine3.23 AS ui-builder
WORKDIR /ui
COPY --from=source /src/ui/package.json /src/ui/package-lock.json ./
RUN npm ci
COPY --from=source /src/ui/ ./
# `build-enterprise` is upstream's prod build script — it outputs to ./out
# (Next.js static export). We then move ./out into the place bifrost-http
# expects to embed.
RUN npm run build-enterprise

# ---------------------------------------------------------------------------
# Stage 1b: build the gateway plugin's admin UI (Preact + Vite).
#
# The compiled bundle is dropped into gateway/internal/adminapi/ui/dist
# before the Go build below so `//go:embed all:ui/dist` picks it up at
# compile time. Cached separately from the rest of the Go build so
# changes to *.go files don't re-run `npm`.
#
# See gateway/plans/phases/phase-8-observability-dashboard.md.
# ---------------------------------------------------------------------------
FROM node:${NODE_VERSION}-alpine3.23 AS plugin-ui-builder
WORKDIR /pui
COPY internal/adminapi/ui/package.json internal/adminapi/ui/package-lock.json* ./
# `--no-audit --no-fund` keeps the output quiet and a hair faster.
# Use `npm install` (not `ci`) until the lockfile exists; once the
# first build commits package-lock.json this switches to `npm ci`
# for reproducibility. The conditional below covers both.
RUN if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi
COPY internal/adminapi/ui/ ./
RUN npm run build
# Output: /pui/dist

# ---------------------------------------------------------------------------
# Stage 2: build bifrost-http (dynamic) + our plugin from upstream sources.
# ---------------------------------------------------------------------------
FROM golang:${GO_VERSION}-alpine3.23 AS builder

ARG BIFROST_VERSION
ARG TARGETARCH

# git:            present in source stage already; kept for parity / safety.
# gcc, musl-dev:  CGO toolchain for bifrost-http + plugin.
# binutils-gold:  the linker gcc invokes via -fuse-ld=gold (Bifrost build).
# sqlite-dev:     headers for the embedded SQLite store.
# bash, make:     used by upstream Makefiles (kept for parity / future use).
RUN apk add --no-cache git gcc musl-dev binutils-gold sqlite-dev bash make

# Pull source from the dedicated clone stage and the built UI from the UI stage.
COPY --from=source    /src /src
COPY --from=ui-builder /ui/out /src/transports/bifrost-http/ui

# --- 2a. Build bifrost-http dynamically linked. ---
#
# bifrost-http has `//go:embed all:ui`; the UI assets were placed at
# /src/transports/bifrost-http/ui above. We intentionally do NOT use
# upstream's `-extldflags '-static'` flag because we need a dynamically
# linked binary that can load Go plugins (.so).
WORKDIR /src/transports
RUN CGO_ENABLED=1 GOOS=linux \
    go build \
        -ldflags="-w -s -X main.Version=${BIFROST_VERSION}" \
        -trimpath \
        -o /out/bifrost-http \
        ./bifrost-http

# Sanity check: it must be dynamically linked or it won't load plugins.
RUN ldd /out/bifrost-http | grep -q 'musl' && echo "OK: dynamic" \
 || (echo "FAIL: bifrost-http is statically linked"; ldd /out/bifrost-http; exit 1)

# --- 2b. Build our plugin against this same source tree. ---
#
# By replacing the plugin's `github.com/maximhq/bifrost/core` with the
# local checkout, the plugin and bifrost-http compile against byte-identical
# sources and matching transitive deps. This is what avoids the dreaded
# "plugin was built with a different version of package …" errors.
WORKDIR /plugin
COPY go.mod ./
COPY main.go ./
COPY internal/ ./internal/
# The pure-Go macaroon verifier lives under gateway/auth/go and is
# imported by gateway/internal/auth as github.com/stakwork/stakgraph/
# gateway/auth/go. Must be copied into the build context — without
# it, `go mod tidy` falls back to fetching the package from GOPROXY
# (which only has tagged releases of the parent module) and fails.
COPY auth/go/ ./auth/go/
# Drop the compiled UI in place BEFORE `go build` so the
# `//go:embed all:ui/dist` directive in
# gateway/internal/adminapi/ui.go finds the real bundle, not the
# placeholder index.html that lives in the repo for offline dev
# builds.
COPY --from=plugin-ui-builder /pui/dist /plugin/internal/adminapi/ui/dist
RUN go mod edit -replace github.com/maximhq/bifrost/core=/src/core \
 && go mod tidy \
 && CGO_ENABLED=1 GOOS=linux \
    go build -buildmode=plugin \
        -ldflags="-w -s" -trimpath \
        -o /out/stakgraph-gateway.so .

# ---------------------------------------------------------------------------
# Stage 2c: build the wrapper binary.
#
# The wrapper owns the container's public listener and reverse-proxies
# `/_plugin/*` to the plugin's in-process HTTP server (loopback:8189)
# and everything else to bifrost-http (loopback:8080). Why we need
# it: Bifrost plugins can't register arbitrary HTTP routes through
# Bifrost's own router, and we want both the dashboard UI and the
# plugin routes available on a single TLS-terminated host:port for
# Traefik / k8s ingress. See gateway/wrapper/main.go for the full
# story.
#
# Pure stdlib Go — no CGO, no bifrost dep tree, no plugin loader
# constraint. Static binary so the runtime image doesn't need any
# extra libs to run it.
# ---------------------------------------------------------------------------
FROM golang:${GO_VERSION}-alpine3.23 AS wrapper-builder
WORKDIR /wrapper
COPY wrapper/go.mod ./
# Copy every wrapper source file. Previously this was a single
# `COPY wrapper/main.go ./` which broke whenever a second .go file was
# added (silent "undefined: …" errors at build time). The wildcard
# pulls in main.go plus any sibling .go files (authsplit.go, etc).
# `*_test.go` are also matched but ignored by `go build .` and the
# whole stage is discarded — only /out/wrapper is copied forward.
COPY wrapper/*.go ./
RUN CGO_ENABLED=0 GOOS=linux \
    go build \
        -ldflags="-w -s" \
        -trimpath \
        -o /out/wrapper .

# ---------------------------------------------------------------------------
# Stage 3: minimal runtime — Alpine + dynamic libc + ca-certs.
# ---------------------------------------------------------------------------
FROM alpine:3.23

# musl + libgcc: required by any CGO-enabled binary built on alpine.
# ca-certificates: TLS to upstream providers.
# wget: container healthcheck.
RUN apk add --no-cache musl libgcc ca-certificates wget tini \
 && addgroup -S app && adduser -S -G app -h /app app

WORKDIR /app

COPY --from=builder         /out/bifrost-http             /app/bifrost-http
COPY --from=builder         /out/stakgraph-gateway.so     /app/plugins/stakgraph-gateway.so
COPY --from=builder         /src/BIFROST_COMMIT           /app/BIFROST_COMMIT
COPY --from=wrapper-builder /out/wrapper                  /app/wrapper

# Ship config.json as a SEED file outside the /app/data volume. The wrapper
# (see wrapper/main.go syncSeedConfig) copies it into /app/data/config.json on
# every container start, so a new image always wins over the volume's stale
# copy.
#
# Why not COPY directly to /app/data/config.json: that path is a named volume
# in docker-compose / swarm. Docker only copies image contents into a volume
# the first time the volume is empty; subsequent images get shadowed by the
# volume's existing files. Net effect: deploying a new image with an updated
# config.json had no effect on the running config, because the volume held the
# old version. See plans/bifrost-log-drop-debug.md for the broader storage
# investigation and the LLM governance docs for the original incident
# (config_store path was relative -> resolved to /app/config.db inside the
# writable layer -> VKs wiped on every container recreate).
#
# Bifrost itself still requires config.json to live in -app-dir (alongside
# config.db / logs.db), so the wrapper materialises it there on boot.
COPY data/config.json                                     /app/config.json.seed

# Data dir is where bifrost reads config.json and writes config.db / logs.db.
# Mount it as a named volume in docker-compose to persist across restarts.
RUN mkdir -p /app/data/logs && chown -R app:app /app

USER app

# Process layout inside the container:
#   tini (PID 1)
#     └─ wrapper      :8181 public — proxies / -> bifrost, /_plugin/* -> plugin
#         └─ bifrost-http   127.0.0.1:8080 (loopback only)
#             └─ stakgraph-gateway.so (loaded plugin)
#                 └─ plugin HTTP server  127.0.0.1:8189 (loopback only)
#
# APP_HOST is documented for clarity but the wrapper passes -host 127.0.0.1
# to bifrost-http explicitly, which overrides the env. We keep the env var
# for parity with upstream Bifrost docs / for anyone who inspects the image.
ENV APP_HOST=127.0.0.1 \
    APP_PORT=8080 \
    APP_DIR=/app/data \
    LOG_LEVEL=info \
    LOG_STYLE=pretty \
    BIFROST_PLUGIN_PORT=8189

EXPOSE 8181
VOLUME ["/app/data"]

# Healthcheck hits the wrapper (the public listener at :8181). The
# wrapper only starts accepting once bifrost is ready, so a 200 here
# confirms the whole chain is up. /health is unauthenticated even when
# auth_config is enabled — it's in Bifrost's systemWhitelistedRoutes.
#
# `wget -O- > /dev/null` does a GET (not HEAD). `wget --spider` issues
# HEAD which bifrost-http rejects with 405 → unhealthy container →
# traefik filters it out → no routing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O- http://127.0.0.1:8181/health > /dev/null || exit 1

ENTRYPOINT ["/sbin/tini", "--"]
# The wrapper runs with no flags: its defaults produce exactly the
# argv we used to pass directly to bifrost-http, plus -host 127.0.0.1
# so bifrost is loopback-only. To override (e.g. for debugging),
# `docker run ... /app/wrapper -listen :9090 -bifrost-arg -log-level
# -bifrost-arg debug` etc.
CMD ["/app/wrapper"]
