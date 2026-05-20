# stakgraph gateway plugin

A [Bifrost](https://github.com/maximhq/bifrost) custom plugin (Go, in-process)
for the stakgraph LLM governance layer.

This is **v0 boilerplate**: every hook logs a structured line and passes
through. No macaroon verification, Redis state, or budget enforcement yet.
The point is to prove the plugin loads cleanly into bifrost-http and that
all hook entry points fire in the order the [governance plan](./plans/llm-governance.md)
expects.

> **Observability is already free.** Bifrost extracts every `x-bf-dim-*`
> request header into the SQLite `logs.metadata` JSON column. So sending
> `x-bf-dim-run-id`, `x-bf-dim-realm-id`, `x-bf-dim-agent-name`,
> `x-bf-dim-user-id`, `x-bf-dim-session-id` on each call is enough to
> get fully-indexed per-call analytics without any plugin code. See the
> ["Observability via Bifrost dimensions"](./plans/llm-governance.md#observability-via-bifrost-dimensions)
> section of the plan.

## Layout

```
gateway/
├── main.go            # plugin entry points (Init, GetName, Cleanup, *Hook)
│                      #   one-line delegations into internal/ — auditable at a glance
├── go.mod             # pinned: bifrost/core v1.5.10, Go 1.26.2
├── Makefile           # local + docker + UI build targets
├── Dockerfile         # bifrost-http + plugin .so + wrapper + admin UI, all in one image
├── docker-compose.yml # host 8181 -> container 8181, named volume for /app/data
├── tygo.yaml          # Go -> TS struct codegen config (drives ui/src/api/types.ts)
├── data/
│   └── config.json    # seed config: providers, plugin entry, auth_config (baked in)
├── internal/          # plugin internals (no public API beyond what main.go re-exports)
│   ├── pluginlog/     # structured-ish stderr logging shim
│   ├── env/           # typed env-var readers + defaults
│   ├── pluginctx/     # typed wrappers around BifrostContext (dims, timing, request-id)
│   ├── hooks/         # bodies of every plugin entry point, one file per hook
│   ├── sessions/      # Redis-backed browser session store (phase 8)
│   ├── adminapi/      # /_plugin/* HTTP server (auth, credentials, health, observability, UI)
│   │   └── ui/        # Preact + Vite admin dashboard, //go:embed'd at compile time
│   ├── ratelimit/     # (stub) per-(agent|user|session) rate limits — coming soon
│   └── auth/          # macaroon verifier adapter wiring (phases 4–6)
└── wrapper/
    ├── go.mod         # stdlib-only Go module (separate from plugin's bifrost dep)
    └── main.go        # PID-1 binary: owns :8181, fronts bifrost + /_plugin/*
```

## Process layout inside the container

```
tini (PID 1)
 └─ wrapper                 :8181 public  (the only listener exposed by EXPOSE)
     ├─ proxies /_plugin/*  ─► 127.0.0.1:8189   (plugin's in-process HTTP server)
     └─ proxies everything  ─► 127.0.0.1:8080   (bifrost-http, loopback only)
         └─ stakgraph-gateway.so      (the .so loaded by bifrost as a Go plugin)
             └─ HTTP server on 127.0.0.1:8189   (started by Init)
```

The wrapper exists because Bifrost plugins can't register arbitrary HTTP
routes through Bifrost's own router (HTTPTransportPreHook only fires on
inference paths), and we need routes that _aren't_ behind Bifrost's
auth middleware so Hive can bootstrap on a fresh swarm. The wrapper
gives the plugin a clean `/_plugin/*` namespace on the same public port
as the dashboard, with one TLS terminator at the swarm/ingress edge.

> **Why a named volume instead of `./data:/app/data`?** macOS Docker
> Desktop's virtiofs bind mount silently drops a fraction of SQLite WAL
> writes, so log rows that gorm reported as successfully inserted never
> reach disk. Named volumes use the Linux VM's native filesystem and
> don't have this problem. `data/config.json` is `COPY`'d into the
> image at build time so first-boot seeding still works.
> Full investigation: [plans/bifrost-log-drop-debug.md](./plans/bifrost-log-drop-debug.md).

## Why a custom Dockerfile?

The upstream `maximhq/bifrost:latest` is statically linked and **cannot
load `.so` plugins** ([docs](https://docs.getbifrost.ai/plugins/building-dynamic-binary)).
The `Dockerfile` here clones `bifrost` at a pinned tag (`transports/v1.5.2`),
builds `bifrost-http` with CGO + dynamic linking, builds our plugin
against the same source tree (avoiding the "plugin was built with a
different version of package" trap), builds the (stdlib-only) wrapper
binary, and produces a single Alpine runtime image with all three.

## Authentication

The dashboard UI and Bifrost's `/api/*` admin endpoints (governance,
config, logs) require HTTP Basic auth. Inference endpoints (`/v1/*`,
`/openai/*`, `/anthropic/*`, etc.) stay open so existing agents work
unchanged.

Credentials come from two env vars that get resolved at boot by
Bifrost itself (config.json references `env.BIFROST_ADMIN_USER` and
`env.BIFROST_ADMIN_PASS` — bifrost expands these into `auth_config`
and bcrypt-hashes the password into config.db):

```bash
BIFROST_ADMIN_USER=admin                  # default in docker-compose.yml
BIFROST_ADMIN_PASS=bifrost-dev-password   # default in docker-compose.yml
```

In production (sphinx-swarm), these are auto-generated random values
per-swarm; the dev defaults above only exist so `make docker-up` works
out of the box.

To change the password later, edit the env, `make docker-build` to
rebuild the image, then `docker compose down && docker compose up -d`.
Bifrost detects the new `env.*` reference and re-hashes; existing
sessions get flushed (matches `loadAuthConfig` behaviour in
bifrost-http).

## The `/_plugin/*` namespace

The wrapper routes any path under `/_plugin/` to the plugin's
in-process HTTP server (loopback :8189). All `/_plugin/*` routes
require `Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>`. The
token is a shared secret with Hive (in swarm: same value as
`boltwall.stakwork_secret`).

Routes today:

| Method | Path                                       | Auth                | Description                                                                                                |
| ------ | ------------------------------------------ | ------------------- | ---------------------------------------------------------------------------------------------------------- |
| GET    | `/_plugin/health`                          | anon                | Plain `{"ok":true}`. Used by the wrapper itself.                                                           |
| GET    | `/_plugin/admin-credentials`               | bearer              | `{"admin_username":"…","admin_password":"…"}` so Hive can bootstrap on a fresh swarm.                      |
| POST   | `/_plugin/login`                           | Basic (in header)   | Sets the `bifrost_session` cookie. Rate-limited 10 attempts / 15min per-IP.                                |
| POST   | `/_plugin/logout`                          | cookie or bearer    | Clears session + cookie. Idempotent.                                                                       |
| GET    | `/_plugin/me`                              | cookie or bearer    | `{user, iat, last_seen}` for the SPA's boot probe.                                                         |
| GET    | `/_plugin/ui/*`                            | cookie or bearer    | The embedded Preact dashboard. SPA fallback serves index.html on deep links.                               |
| GET    | `/_plugin/spend/by-{agent,user}?window=…`  | cookie or bearer    | Phase-7 per-dim aggregations over `logs.db` via loopback to bifrost-http's `/api/logs`.                    |
| GET    | `/_plugin/histogram/cost?window=…&bucket=…&dimension=…` | cookie or bearer | Time-bucketed cost series, grouped by dim. agent-name / run-id / session-id / realm-id / user-id. |
| GET    | `/_plugin/runs/:run_id`                    | cookie or bearer    | Drill-down: every call recorded for one run_id, paginated.                                                 |
| GET    | `/_plugin/runs/:run_id/calls/:call_id`     | cookie or bearer    | Single call body — full input_history / output_message / params / tools / error_details / raw_response.    |
| GET/POST/DELETE | `/_plugin/trust/*`                | bearer              | Phase-5 trust registry CRUD.                                                                               |

### Admin UI (`/_plugin/ui/`)

A Preact + Vite + wouter + Tanstack Query + uPlot single-page app that
runs the operator dashboard. Source lives under
`internal/adminapi/ui/`; the bundle is compiled by `make ui-build` or
(in CI / docker) the `plugin-ui-builder` stage in the Dockerfile, then
embedded into the plugin .so via `//go:embed all:ui/dist`.

Pages in v1: Login, Dashboard, Agents, AgentDetail, RunDetail. All
read-only; phase 9 adds the kill switches, budget editors, and live
Redis-blended panels.

Local UI dev with HMR (proxies `/_plugin/*` to a running plugin on
`localhost:8181`):

```bash
make docker-up        # in one shell
make ui-dev           # in another, opens http://localhost:5173/_plugin/ui/
```

Default dev credentials: `admin` / `bifrost-dev-password`.

The credentials endpoint exists because there is otherwise no way for
Hive to learn the Bifrost admin password without an out-of-band
channel — and we don't want to expose either the admin password or
the bifrost API key to Hive's browser. Hive's backend hits this once,
encrypts the result into its DB, and never calls it again unless the
admin creds get lost. See `plans/phases/phase-3-swarm-handoff.md`
for the full handoff design.

Routes planned (drop into the same auth namespace as they're added):

| Path                                  | Why a plugin route vs Bifrost's `/api/logs`       |
| ------------------------------------- | ------------------------------------------------- |
| `/_plugin/metrics/agent-cost?since=…` | per-`x-bf-dim-agent-name` rollups, not in Bifrost |
| `/_plugin/metrics/run/{run_id}`       | per-run cost + tool history, not in Bifrost       |

## Quick start

```bash
# 1. Provider keys (read by docker-compose; same .env as mcp/).
cd gateway
ln -sf ../mcp/.env .env   # or: cp ../mcp/.env .env

# 2. Build the dynamic-bifrost+plugin+wrapper image and start it.
make docker-up

# 3. Tail the logs and watch the boilerplate hooks fire.
make docker-logs

# 4. Inference is open — same call as before.
curl -s http://localhost:8181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-macaroon: pretend.this.is.a.macaroon' \
  -H 'x-bf-dim-run-id: r_test_001' \
  -H 'x-bf-dim-agent-name: smoke-test' \
  -H 'x-bf-dim-realm-id: w1' \
  -H 'x-bf-dim-user-id: u1' \
  -H 'x-bf-dim-session-id: s_test_42' \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role":"user","content":"say hi in 3 words"}]
  }'

# 5. Dashboard / admin API now require Basic auth (defaults: admin /
#    bifrost-dev-password — see docker-compose.yml). Without creds:
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8181/api/governance/customers
# → 401

# With creds:
curl -s -u admin:bifrost-dev-password http://localhost:8181/api/governance/customers
# → {"customers":[...], "count":0, ...}

# 6. The /_plugin/admin-credentials route lets Hive bootstrap without
#    ever seeing the password until it makes this single call:
curl -s http://localhost:8181/_plugin/admin-credentials \
  -H 'Authorization: Bearer dev-provisioning-token'
# → {"admin_username":"admin","admin_password":"bifrost-dev-password"}

# Wrong token / missing header → 401.
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8181/_plugin/admin-credentials
# → 401

# 7. Confirm the dims landed in the SQLite logs table. The DB lives
#    in the `stakgraph-gateway-data` docker volume now, not the host path:
docker run --rm -v stakgraph-gateway-data:/data alpine:3.23 \
  sh -c 'apk add -q sqlite >/dev/null && sqlite3 /data/logs.db \
    "SELECT model, cost, metadata FROM logs ORDER BY created_at DESC LIMIT 1;"'
# → claude-3-5-haiku-latest|0.000043|{"run-id":"r_test_001","agent-name":"smoke-test",...}

# 8. (optional) end-to-end against the existing MCP harness — same port,
#    so the existing test agent just works.
cd ../mcp
npx tsx docs/gateway/test-agent.ts "hi, how are you?"
```

You should see lines like:

```
[stakgraph-gateway] 2026-... HTTPTransportPreHook method=POST path=/v1/chat/completions body_bytes=128 macaroon=set(pret…oon,len=27) run_id=r_test_001 ...
[stakgraph-gateway] 2026-... PreLLMHook provider=anthropic model=claude-3-5-haiku-latest request_type=chat_completion run_id=r_test_001
[stakgraph-gateway] 2026-... PostLLMHook run_id=r_test_001 had_resp=true had_err=false prompt_tokens=12 completion_tokens=8 total_tokens=20 elapsed_ms=812
[stakgraph-gateway] 2026-... HTTPTransportPostHook path=/v1/chat/completions status=200 body_bytes=441 run_id=r_test_001 elapsed_ms=813
```

## Local plugin build (without docker)

Useful while iterating on the Go code. You still need a dynamic
bifrost-http to actually _load_ the .so — typically the easiest path is
just to rebuild the docker image (`make docker-up` again).

```bash
make dev          # writes build/stakgraph-gateway.so
```

Cross-OS plugin builds are not portable (musl ≠ glibc, darwin ≠ linux,
arm64 ≠ amd64). Build inside the image when in doubt.

## Hook reference (what we log today)

| Hook                           | When                             | Logged fields                                                                     |
| ------------------------------ | -------------------------------- | --------------------------------------------------------------------------------- |
| `Init`                         | on plugin load                   | parsed config                                                                     |
| `HTTPTransportPreHook`         | before bifrost core              | method, path, body size, governance headers (`x-macaroon`, plus all `x-bf-dim-*`) |
| `PreLLMHook`                   | before provider call             | provider, model, request type, run_id                                             |
| `PostLLMHook`                  | after provider call (non-stream) | run_id, presence flags, usage tokens, elapsed                                     |
| `HTTPTransportPostHook`        | after bifrost core (non-stream)  | path, status, body size, run_id, elapsed                                          |
| `HTTPTransportStreamChunkHook` | per chunk (stream)               | only on error chunks (rate-limit risk if we log every chunk)                      |
| `Cleanup`                      | on shutdown                      | —                                                                                 |

The macaroon header is fingerprinted (`set(abcd…wxyz,len=N)`) rather than
logged in full.

## Next steps (from `llm-governance.md`)

This file is plugin v0. **Observability (rollout step 5) requires no
plugin code change** — just have callers send `x-bf-dim-*` headers and
query `logs.db`.

The plugin grows when we add hot enforcement (step 6 onwards):

1. **Per-run state in Redis (step 6)** — `cost:run:<run_id>`,
   `kill:<run_id>`, `tools:run:<run_id>`. Read in PreHook (reject on
   cap/kill), written in PostLLMHook. `run_id` still comes from the
   `x-bf-dim-run-id` header pre-macaroon.
2. **Macaroon verification (step 8)** in `HTTPTransportPreHook` —
   HMAC + caveats. Short-circuit with
   `&schemas.HTTPResponse{StatusCode: 401, …}`. Canonicalize the dim
   map from the caveats so `logs.metadata` always reflects the
   cryptographically verified identity.
3. **Tool-loop heuristic** — uses the tool history stored in step 1.

See the [governance plan](./plans/llm-governance.md) for the full
architecture.
