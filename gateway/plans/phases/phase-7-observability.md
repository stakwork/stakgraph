# Phase 7 — Observability: Per-Dim Analytics over `logs.db`

> Read-only HTTP surface that exposes per-agent / per-session /
> per-user / per-model spend and usage analytics out of Bifrost's
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

GET /_plugin/spend/by-session?window=24h
  Same strategy, dim = session-id. A session spans runs, so each row
  also carries the user, a run count, and the first/last timestamps
  for a Sessions list page.
  returns: {
    window,
    results: [ { session_id, user_id, total_cost, total_tokens, request_count,
                 run_count, first_seen, last_seen }, ... ]
  }

GET /_plugin/spend/by-model?window=24h
  Keyed on (provider, model) — both are first-class columns, so unlike
  the dim rollups nothing is excluded; these totals reconcile with
  Bifrost's own dashboard. (Bifrost's GetModelRankings applies a
  trend calculation we don't want surfaced; same reasoning as
  by-user.)
  returns: { window, results: [ { model, provider, total_cost, total_tokens, request_count }, ... ] }

GET /_plugin/spend/by-agent-user?window=24h
  The (agent × user) crossing the Canvas page renders, with a
  per-provider breakdown per pairing. Not in the original sketch;
  added by phase 8.
```

Phase 11 removed `by-realm`: every row in a swarm's `logs.db` is
implicitly for that swarm's realm, so a per-swarm realm rollup is a
single number the central aggregator already has.

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
  Same shape; points are { ts, prompt_tokens, completion_tokens, total_tokens }.
  Rows without token usage (errors) contribute nothing.

GET /_plugin/histogram/latency?window=24h&bucket=1h&dimension=agent-name
  Same shape; points are { ts, p50, p95, p99, count } — nearest-rank
  percentiles (ms) over the bucket's successful calls, plus the
  sample size they came from. Rows with no latency (errored /
  in-flight) are excluded.
```

All three histograms bucket in Go. Bifrost's
`GetDimension{Cost,Token,Latency}Histogram` only group by its
column-bound dimensions (provider / team / customer / user /
business-unit), not `metadata.*`, so the plugin pages the window
out of `/api/logs` and folds rows into epoch-aligned buckets itself
(200k-row ceiling, same as the rollups).

### Drill-down

```
GET /_plugin/runs/:run_id
  MetadataFilters: {"run-id": <id>}
  Bifrost API: GET /api/logs?metadata_filters={"run-id":...}
  returns: { run_id, logs: [<Log>...], stats: <SearchStats> }
  Combine with phase-6 /_plugin/runs/:run_id/state for live numbers.

GET /_plugin/sessions/:session_id
  MetadataFilters: {"session-id": <id>}, paginated like /runs/:id
  returns: { session_id, logs: [<RunLogEntry>...], stats: <RunStats>, total_count }

GET /_plugin/sessions/:session_id/summary
  Scans the whole session (no window — a session is finite).
  returns: { session_id, user_id, total_cost, total_tokens, request_count,
             started_at, latest_at, duration_ms, agents: [...], runs: [...] }

  NOT Bifrost's native /api/logs/sessions/{id}: Bifrost's "session_id"
  aliases parent_request_id (its own multi-turn linkage), which is a
  different thing from the x-bf-dim-session-id dim Hive stamps. Both
  plugin routes filter on metadata.session-id, exactly like /runs/:id
  filters on metadata.run-id.

GET /_plugin/users/:user_id/spend?window=24h
  MetadataFilters: {"user-id": <id>}; Bifrost's SearchStats over the
  filtered set, so one limit=1 call — no row paging. Filters on the
  dim, not the customer_id column, for the reasons on spendByUser
  (customer_id is the VK's Hive UUID; metadata.user-id is what phase
  6 canonicalises from the verified claim).
  returns: { user_id, window, total_cost, total_tokens, request_count }

GET /_plugin/users/:user_id/quota
  Combines: Bifrost customer budget (GET /api/governance/customers/:id;
            Hive's reconciler provisions one per workspace × user)
            + the runs the accumulator indexed for the user in Redis
            (bifrost:runs:user:<id>, pruned by macaroon exp on read)
  returns: {
    user_id,
    customer_found,                       // false ⇒ budget fields null
    budget_usd, budget_window,            // Bifrost max_limit + reset_duration
    spent_usd, remaining_usd, budget_last_reset,
    redis_available,                      // false ⇒ inflight_runs null
    inflight_runs: [ { run_id, agent_name, cost_usd, steps,
                       max_cost_usd, max_steps, exp, killed } ]
  }
  The window is whatever Bifrost's reset_duration says; the sketch
  above assumed "daily", but that is the reconciler's decision.

GET /_plugin/agents/:name/spend?window=24h
  MetadataFilters: {"agent-name": <name>}; SearchStats, one limit=1 call
  returns: { agent_name, window, total_cost, total_tokens, request_count }
```

## Query parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `window` | duration | `24h` | Any Bifrost duration (`1h`, `6h`, `24h`, `1d`, `7d`, `1w`, `30d`, `1M`, `1Y`; see `internal/duration`), at most 1Y. Rolling: `end = now`, `start = now − window` — `1d` is the last 24 hours, not the calendar day. The SPA's picker offers a four-option subset. |
| `bucket` | duration | `1h` for histogram endpoints | Any Bifrost duration ≥ `1m` and `≤ window`. Buckets are aligned to the unix epoch. |
| `dimension` | enum | `agent-name` for histogram endpoints | One of `agent-name`, `user-id`, `session-id`, `run-id`. (`realm-id` removed by phase 11; `model` / `provider` are served by `/spend/by-model` instead.) |
| `limit` | int | `100` | Pagination for drill-down endpoints. |
| `offset` | int | `0` | Pagination. |
| `sort_by` | enum | `timestamp` | For drill-down: `timestamp`, `latency`, `tokens`, `cost`. |
| `order` | enum | `desc` | `asc` or `desc`. |

The window→`StartTime`/`EndTime` translation uses the request's
arrival time as `now`. Calendar alignment (`1d` = UTC midnight, `1M`
= month boundary — phase 6 "Duration vocabulary") is the agent
budget bucket's concern, not the analytics window's: an operator
asking for "1d" of spend wants the last day, and the histogram
buckets are epoch-aligned so the same window/bucket pair produces
identical bucket edges across polls.

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

- [x] `logstore_client.go`: HTTP client to
      `http://127.0.0.1:8080/api/logs` (+ `/api/logs/{id}` and
      `/api/governance/customers/{id}`). Owns the query-string
      composition, Basic auth, timeout, paging (`searchAll`) and
      the `upstreamError` mapping. Shared by every handler below.
- [x] `spend.go`: `by-session`, `by-model`, `agents/:name/spend`
      (`by-user`, `by-agent`, `by-agent-user` live in
      `observability.go` from phase 8).
- [x] `histogram.go`: `tokens` and `latency` (`cost` in
      `observability.go`). Bucketed in Go — see the note under
      "Histograms".
- [x] `sessions.go`: `/_plugin/sessions/:id` and
      `/_plugin/sessions/:id/summary`.
- [x] `users.go`: `/_plugin/users/:id/spend` and
      `/_plugin/users/:id/quota`. Quota blends Bifrost's customer
      budget with the accumulator's per-user run index in Redis.
- [x] `GET /_plugin/runs/:id` drill-down (`observability.go`) and
      `/runs/:id/calls/:call_id`; `state` / `kill` siblings in
      `hotstate.go`.
- [x] `GET /_plugin/agents/:name/spend` (dispatched from the shared
      `/_plugin/agents/` subtree in `server.go`).
- [x] Route registration in `server.go` — `routeDeps.logstore`;
      every read route is cookie-or-bearer.

**Tests:**

- [x] `logstore_client_test.go`: query composition, paging + row
      cap, 404 → nil, upstream error mapping, customer decode,
      against an `httptest.Server`.
- [x] Per-handler tests (`observability_test.go`,
      `observability_phase7_test.go`): window/bucket/dimension
      vocabulary, MetadataFilters scoping, percentile math, quota
      degradation with and without Redis, upstream 502 mapping.
- [ ] One end-to-end test that stamps dims via PreHook → drives
      a real LLM call (Bifrost mocker plugin) → queries
      `/_plugin/spend/by-agent` → asserts the call shows up under
      the right agent. `scripts/smoke-test.sh` does this by hand
      against a compose stack; it is not in CI.

**Documentation:**

- [x] `gateway/internal/adminapi/server.go` route table notes the
      phase-7 families.
- [x] `llm-governance-v2.md` §"Observability" points here.

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
