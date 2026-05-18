# Phase 7 — Observability: Per-Dim Analytics over `logs.db`

> Read-only HTTP surface that exposes per-agent / per-session /
> per-user / per-realm spend and usage analytics out of Bifrost's
> own log store, via the dim headers the plugin canonicalizes in
> `PreLLMHook`. Companion to `phase-6-plugin-enforcement.md` (which
> defines the kill/state admin endpoints over the same `/_plugin/*`
> namespace) and `llm-governance-v2.md` §"Observability" (which
> introduces the design at architectural level).
>
> Phase 6 made `logs.metadata` trustworthy by canonicalizing dim
> values from verified macaroon claims. This phase turns those
> stamped dims into queryable analytics that the Bifrost UI can't
> serve (its filter widget doesn't expose `metadata.*`).

## What this phase decides

- Where observability endpoints live in the repo
  (`gateway/internal/adminapi/`, same package as the kill/state
  handlers from phase 6 and the trust handlers from phase 5).
- How the plugin reads `logs.db` without coupling to Bifrost's
  storage schema (loopback HTTP to Bifrost's `/api/logs`).
- The v1 endpoint surface, response shapes, and pagination contract.
- What ships in v1 vs. what is explicitly deferred.

This phase produces no new wire format, no Redis schema changes,
no new crypto, and no new hook behaviour. It's a thin HTTP server
sitting on top of phase 6's already-canonicalized dims.

## Why a separate phase

Three reasons phase 7 was carved out of phase 6:

1. **Different repo home.** Phase 6 work lives in
   `gateway/internal/auth/` (`PreLLMHook` / `PostLLMHook` against
   Redis). Phase 7 work lives in `gateway/internal/adminapi/` (HTTP
   handlers calling out to Bifrost's log store over loopback). They
   share Redis state for kill/state ops, but observability is
   read-only and never touches Redis except for the live-state
   blends.
2. **Different data source.** Phase 6 enforces against Redis hot
   state. Phase 7 reads from Bifrost's persistent log store
   (SQLite or Postgres, transparent to the plugin) via Bifrost's
   own query API. The plugin never opens `logs.db` directly.
3. **Hard dependency on phase 6.** Without phase 6's dim
   canonicalization, `MetadataFilters` queries return junk — a
   caller could ship any `x-bf-dim-agent-name` they wanted.
   Observability is only trustworthy after phase 6 lands.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ wrapper (PID 1) :8181                                        │
│   ┌──────────────────────────┐  ┌────────────────────────┐   │
│   │ /_plugin/*  →  loopback   │  │ everything else  →     │   │
│   │  127.0.0.1:8189           │  │   127.0.0.1:8080       │   │
│   └────────────┬─────────────┘  └────────────┬───────────┘   │
└────────────────┼────────────────────────────┼───────────────┘
                 │                            │
                 ▼                            ▼
       gateway/internal/adminapi/      bifrost-http :8080
         (Go HTTP server)              ┌────────────────────┐
         ┌──────────────────┐          │ /api/logs          │
         │ /_plugin/spend/* │──────────►│  SearchFilters    │
         │ /_plugin/runs/:id│  loopback│  MetadataFilters   │
         │ /_plugin/sessions│   GET    │  Pagination        │
         │ ...              │          │ LogStore impl      │
         └──────────────────┘          │  (SQLite/Postgres) │
                                       └────────────────────┘
```

**Read path:** caller hits `/_plugin/spend/by-agent?window=24h` on
the wrapper's public port → wrapper proxies to the plugin server on
`127.0.0.1:8189` → handler builds a `SearchFilters` JSON body with
`MetadataFilters: {"agent-name": ...}` → handler GETs
`http://127.0.0.1:8080/api/logs?...` → JSON response → handler
shapes it for the caller and returns.

**No SQLite handle in the plugin.** All queries go through Bifrost's
HTTP API. This means:

- Schema migrations under Bifrost don't break us.
- `logs_store.type: postgres` swap works without plugin changes.
- Bifrost's matview optimizations (`framework/logstore/matviews.go`)
  apply transparently.
- One extra hop (~sub-millisecond against loopback) which is
  negligible vs. the work itself.

## Auth

Same per-swarm shared bearer secret as phase 5's `/_plugin/trust/*`
and phase 6's kill/state endpoints. Configured via
`BIFROST_PROVISIONING_TOKEN` env var, exposed to callers through
phase 3's swarm-handoff flow.

The plugin's loopback call to `/api/logs` is over `127.0.0.1` only;
Bifrost's `enforce_auth_on_inference` doesn't gate `/api/logs`, and
even if it did the plugin can authenticate against Bifrost's own
admin API using the configured admin credentials phase 3 already
manages.

## Endpoint inventory

### Aggregations

```
GET /_plugin/spend/by-user?window=24h
  MetadataFilters: {} (Bifrost's GetUserRankings groups by customer_id natively)
  Bifrost API: GET /api/logs/rankings/users
  returns: {
    window:  "24h",
    results: [ { user_id, user_name, total_cost, total_tokens, request_count }, ... ]
  }

GET /_plugin/spend/by-agent?window=24h
  MetadataFilters: none required; aggregate over metadata.agent-name client-side
  Bifrost API: GET /api/logs/histogram/dimension-cost?dimension=metadata.agent-name
              (or fall back to GetStats with each agent name once we have the list)
  returns: {
    window:  "24h",
    results: [ { agent_name, total_cost, total_tokens, request_count }, ... ]
  }

GET /_plugin/spend/by-realm?window=24h
  Same shape, dim = realm-id

GET /_plugin/spend/by-session?window=24h
  Same shape, dim = session-id

GET /_plugin/spend/by-model?window=24h
  Bifrost API: GetModelRankings — first-class on the LogStore interface
  returns: { window, results: [ { model, provider, total_cost, request_count }, ... ] }
```

### Histograms (time-series)

```
GET /_plugin/histogram/cost?window=24h&bucket=1h&dimension=agent-name
  Bifrost API: GetDimensionCostHistogram(dimension=metadata.agent-name, bucket=3600)
  returns: {
    bucket_size_seconds: 3600,
    series: [
      { dimension_value: "coder",       points: [ { ts, cost }, ... ] },
      { dimension_value: "web-search",  points: [ { ts, cost }, ... ] },
      ...
    ]
  }

GET /_plugin/histogram/tokens?window=24h&bucket=1h&dimension=user-id
  Bifrost API: GetDimensionTokenHistogram
  Same shape as above with `tokens` instead of `cost`.

GET /_plugin/histogram/latency?window=24h&bucket=1h&dimension=agent-name
  Bifrost API: GetDimensionLatencyHistogram
  Returns percentile series (p50, p95, p99) per dimension value.
```

### Drill-down

```
GET /_plugin/runs/:run_id
  MetadataFilters: {"run-id": <id>}
  Bifrost API: GET /api/logs?metadata_filters={"run-id":...}
  returns: { run_id, logs: [<Log>...], stats: <SearchStats> }
  Combine with phase-6 /_plugin/runs/:run_id/state for live numbers.

GET /_plugin/sessions/:session_id
  Bifrost API: GetSessionLogs(session_id) — first-class on LogStore;
              uses metadata.session-id under the hood.
  returns: <SessionDetailResult>

GET /_plugin/sessions/:session_id/summary
  Bifrost API: GetSessionSummary(session_id)
  returns: <SessionSummaryResult> (cost, tokens, started_at, latest_at, duration_ms)

GET /_plugin/users/:user_id/spend?window=24h
  MetadataFilters: none; SearchFilters.CustomerIDs = [user_id]
  Bifrost API: GET /api/logs with stats
  returns: { user_id, window, total_cost, total_tokens, request_count }

GET /_plugin/users/:user_id/quota
  Combines: Bifrost customer cap (GET /api/governance/customers/:id)
            + live Redis aggregate of in-flight runs the user owns
  returns: {
    user_id,
    daily_budget_usd, spent_today_usd, remaining_today_usd,
    inflight_runs: [ { run_id, current_spend, max_cost_usd, exp } ]
  }

GET /_plugin/agents/:name/spend?window=24h
  MetadataFilters: {"agent-name": <name>}
  Bifrost API: GET /api/logs with stats
  returns: { agent_name, window, total_cost, total_tokens, request_count }
```

## Query parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `window` | duration | `24h` | Accepts Bifrost duration vocabulary (`1h`, `24h`, `1d`, `1w`, `1M`, `1Y`). Translated to `StartTime` / `EndTime` server-side. |
| `bucket` | duration | required for histogram endpoints | Time-bucket granularity. Must be `≤ window`. |
| `dimension` | enum | required for histogram endpoints | One of `agent-name`, `user-id`, `realm-id`, `session-id`, `model`, `provider`. |
| `limit` | int | `100` | Pagination for drill-down endpoints. |
| `offset` | int | `0` | Pagination. |
| `sort_by` | enum | `timestamp` | For drill-down: `timestamp`, `latency`, `tokens`, `cost`. |
| `order` | enum | `desc` | `asc` or `desc`. |

The window→`StartTime`/`EndTime` translation uses the request's
arrival time as `now`. Bucket alignment follows Bifrost's
convention (the `1d` bucket is UTC midnight; `1M` is UTC
month-boundary; etc.) — see phase 6 "Duration vocabulary" for the
table.

## Response shape contract

All endpoints respond JSON with `Content-Type: application/json`.
Success is `200 OK`. Error shapes mirror Bifrost's own
`/api/logs` error envelope:

```json
{ "error": { "code": "<string>", "message": "<string>" } }
```

Error codes:

- `bad_request` (400) — invalid `window` / `bucket` / `dimension`
- `unauthorized` (401) — missing or wrong bearer
- `upstream_unavailable` (502) — Bifrost's `/api/logs` returned an
  error or was unreachable
- `internal` (500) — anything else

## Failure modes

| Failure | Behaviour |
|---|---|
| Bifrost `/api/logs` down | Return 502 `upstream_unavailable`; no caching, no stale-read fallback in v1 |
| Bifrost slow (>5s) | Hard timeout, return 502; client retries with its own policy |
| Redis down (only affects `runs/:id` blends and `users/:id/quota`) | Return the logs-derived portion; mark `inflight_runs: null` with `redis_available: false` |
| `logs.db` empty (new swarm) | Return empty results, not an error |
| Caller-supplied dim with too-many-distinct-values (cardinality bomb) | Bifrost's `MetadataFilters` is a single-value filter, so this is bounded — no risk of unbounded aggregation in v1 |

## What is NOT in this phase

- **Streaming / WebSocket live logs.** Bifrost itself ships
  `ws://...:8080/ws`. Hive can connect directly if it wants live
  log tail; the plugin doesn't proxy it.
- **Cross-workspace aggregation.** Each workspace's plugin only
  sees its own `logs.db`. "Top agents across all workspaces" is a
  Hive-side job that fans out to each workspace's `/_plugin/spend/*`
  and aggregates. v2 §"Cross-Bifrost aggregation" covers the
  long-term plan (shared Postgres backend).
- **Per-call drill-down on individual log rows.** Bifrost's own UI
  (and `/api/logs?id=...`) does this fine. The plugin doesn't add
  value by re-shaping a single row.
- **Caching.** All queries hit Bifrost live. Caching is a known
  optimization but not a v1 requirement — Bifrost's matviews and
  SQLite's read concurrency cover the expected load.
- **Write endpoints.** Phase 7 is read-only. The kill/state
  endpoints from phase 6 (`POST /_plugin/runs/:id/kill` etc.) cover
  every observability-adjacent mutation we need.
- **Authentication beyond per-swarm bearer.** Per-user scoping
  (e.g. "this VK can only see alice's logs") would belong on Hive
  if needed; v1 trusts whoever holds the bearer.

## Wire-up checklist

**Plugin observability HTTP (`gateway/internal/adminapi/`):**

- [ ] `logstore_client.go`: HTTP client to
      `http://127.0.0.1:8080/api/logs` (and rankings / histogram
      sub-paths). Owns `SearchFilters` serialization, retry,
      timeout. Reusable across all handlers below.
- [ ] `spend.go`: `/_plugin/spend/by-{user,agent,realm,session,model}`
      handlers.
- [ ] `histogram.go`: `/_plugin/histogram/{cost,tokens,latency}`
      handlers calling `GetDimensionCostHistogram` /
      `GetDimensionTokenHistogram` / `GetDimensionLatencyHistogram`
      through the client.
- [ ] `sessions.go`: `/_plugin/sessions/:id` and
      `/_plugin/sessions/:id/summary`.
- [ ] `users.go`: `/_plugin/users/:id/spend` and
      `/_plugin/users/:id/quota`. Quota blends Bifrost customer API
      + Redis live state.
- [ ] Extend `runs.go` (from phase 6) with `GET /_plugin/runs/:id`
      drill-down (logs.db slice). The phase-6 `state` and `kill`
      siblings already live here.
- [ ] Extend `agents.go` (from phase 6) with
      `GET /_plugin/agents/:name/spend`.
- [ ] Route registration in `server.go` — extend the existing
      `routeDeps` with a `logstore *LogstoreClient` and register
      the new route families behind the same bearer middleware.

**Tests:**

- [ ] `logstore_client_test.go`: hits a `miniredis`-equivalent
      fake Bifrost (`httptest.Server` mocking `/api/logs`).
- [ ] Per-handler tests: window/bucket/dimension parsing,
      MetadataFilters composition, upstream error mapping.
- [ ] One end-to-end test that stamps dims via PreHook → drives
      a real LLM call (Bifrost mocker plugin) → queries
      `/_plugin/spend/by-agent` → asserts the call shows up under
      the right agent.

**Documentation:**

- [ ] Update `gateway/internal/adminapi/server.go` doc to note the
      new route families (spend, histogram, sessions, users
      drill-down).
- [ ] Update `llm-governance-v2.md` §"Observability" forward-pointer
      to this phase if it's still sketchy at the time phase 7 ships.

**Gate:** phase 7 ships once the v1 endpoint surface returns
correct results for the canonical "alice spent $X on coder yesterday"
queries against a real Bifrost+plugin stack, and the loopback
client gracefully degrades when Bifrost is unreachable.

## What this phase buys

- **Per-dim analytics that Bifrost's UI can't serve.** Filter
  widgets in the Bifrost UI don't expose `metadata.*`; phase 7
  fills that gap with composed `MetadataFilters` queries.
- **No storage coupling.** The plugin doesn't open `logs.db`. A
  future swap to Postgres, ClickHouse, or a shared store is a
  Bifrost config change with zero plugin impact.
- **One namespace, one auth.** Observability shares
  `/_plugin/*` with phase 5's trust ops and phase 6's kill/state
  ops. Operators learn one URL prefix and one bearer.
- **Hive UI is the consumer.** Phase 7's job is to make the dim
  headers usable; Hive can build whatever dashboard it wants on
  top, knowing the shape contract above is stable.
- **Trustworthy results.** Because phase 6 canonicalizes dims from
  verified claims, every byte in `logs.metadata` was attested by
  the macaroon chain. Per-agent / per-user / per-run analytics
  aren't reporting what the caller claimed — they're reporting
  what cryptographically happened.
