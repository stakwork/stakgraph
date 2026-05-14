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
> `x-bf-dim-run-id`, `x-bf-dim-workspace-id`, `x-bf-dim-agent-name`,
> `x-bf-dim-user-id`, `x-bf-dim-session-id` on each call is enough to
> get fully-indexed per-call analytics without any plugin code. See the
> ["Observability via Bifrost dimensions"](./plans/llm-governance.md#observability-via-bifrost-dimensions)
> section of the plan.

## Layout

```
gateway/
├── main.go            # plugin implementation (Init/GetName/*Hook/Cleanup)
├── go.mod             # pinned: bifrost/core v1.5.8, Go 1.26.2
├── Makefile           # local + docker build targets
├── Dockerfile         # dynamic bifrost-http + plugin, one image
├── docker-compose.yml # host 8181 -> container 8080, mounts ./data
└── data/
    └── config.json    # seed config: providers + plugin entry
```

## Why a custom Dockerfile?

The upstream `maximhq/bifrost:latest` is statically linked and **cannot
load `.so` plugins** ([docs](https://docs.getbifrost.ai/plugins/building-dynamic-binary)).
The `Dockerfile` here clones `bifrost` at a pinned tag (`v1.5.8`), builds
`bifrost-http` with CGO + dynamic linking, builds our plugin against the
same source tree (avoiding the "plugin was built with a different version
of package" trap), and produces a single Alpine runtime image with both.

## Quick start

```bash
# 1. Provider keys (read by docker-compose; same .env as mcp/).
cd gateway
ln -sf ../mcp/.env .env   # or: cp ../mcp/.env .env

# 2. Build the dynamic-bifrost+plugin image and start it.
make docker-up

# 3. Tail the logs and watch the boilerplate hooks fire.
make docker-logs

# 4. Sanity check: hit the bifrost openai-compatible endpoint and look
#    for the [stakgraph-gateway] lines in the log.
curl -s http://localhost:8181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-macaroon: pretend.this.is.a.macaroon' \
  -H 'x-bf-dim-run-id: r_test_001' \
  -H 'x-bf-dim-agent-name: smoke-test' \
  -H 'x-bf-dim-workspace-id: w1' \
  -H 'x-bf-dim-user-id: u1' \
  -H 'x-bf-dim-session-id: s_test_42' \
  -d '{
    "model": "anthropic/claude-3-5-haiku-latest",
    "messages": [{"role":"user","content":"say hi in 3 words"}]
  }'

# Then confirm the dims landed in the SQLite logs table:
sqlite3 data/logs.db \
  "SELECT model, cost, metadata FROM logs ORDER BY created_at DESC LIMIT 1;"
# → claude-3-5-haiku-latest|0.000043|{"run-id":"r_test_001","agent-name":"smoke-test","workspace-id":"w1","user-id":"u1","session-id":"s_test_42"}

# 5. (optional) end-to-end against the existing MCP harness — same port,
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
bifrost-http to actually *load* the .so — typically the easiest path is
just to rebuild the docker image (`make docker-up` again).

```bash
make dev          # writes build/stakgraph-gateway.so
```

Cross-OS plugin builds are not portable (musl ≠ glibc, darwin ≠ linux,
arm64 ≠ amd64). Build inside the image when in doubt.

## Hook reference (what we log today)

| Hook | When | Logged fields |
|---|---|---|
| `Init` | on plugin load | parsed config |
| `HTTPTransportPreHook` | before bifrost core | method, path, body size, governance headers (`x-macaroon`, plus all `x-bf-dim-*`) |
| `PreLLMHook` | before provider call | provider, model, request type, run_id |
| `PostLLMHook` | after provider call (non-stream) | run_id, presence flags, usage tokens, elapsed |
| `HTTPTransportPostHook` | after bifrost core (non-stream) | path, status, body size, run_id, elapsed |
| `HTTPTransportStreamChunkHook` | per chunk (stream) | only on error chunks (rate-limit risk if we log every chunk) |
| `Cleanup` | on shutdown | — |

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
