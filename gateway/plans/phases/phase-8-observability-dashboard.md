# Phase 8 — Observability Dashboard: Read-Only UI Over `logs.db`

> Slim, read-only operator dashboard served by the plugin at
> `/_plugin/ui/*`. Renders per-agent spend, per-run drill-downs, and
> cost histograms out of Bifrost's existing `logs.db` via the HTTP
> shape from `phase-7-observability.md`. **Ships before phase 6
> enforcement lands.**
>
> This phase is deliberately scoped to what can be built today,
> without macaroon verification, without Redis hot state, and without
> any destructive operator actions. It is the first user-visible UI
> on top of the gateway plugin, and the foundation that
> [`phase-9-operator-ui.md`](./phase-9-operator-ui.md) extends with
> kill switches, budget edits, and the rest of the full operator
> console.
>
> The frontend pipeline (Vite + Preact + wouter + Tanstack Query +
> uPlot + tygo) is set up in full here so phase 9 can layer on top
> without re-litigating the stack.

## Progress (as of 2026-05-19)

### Landed

**Backend (`gateway/internal/`):**

- `sessions/` — Redis-backed browser session store with miniredis
  tests covering create / get / refresh (write-skipping) / delete /
  kick-all-for-user / TTL expiry.
- `adminapi/session.go` — middleware combining cookie + bearer auth
  (`cookieOrBearer` + `bearerOnly` variants).
- `adminapi/login.go` — `POST /_plugin/login` (Basic header → cookie),
  `POST /_plugin/logout` (idempotent), `GET /_plugin/me`. Bearer
  fallback works for Hive / curl.
- `adminapi/ratelimit.go` — per-IP login attempt counter (10 fails /
  15 min → 429 with Retry-After).
- `adminapi/logstore_client.go` — HTTP client to Bifrost's loopback
  `/api/logs` with Basic auth, pagination, and `searchAll` for
  client-side aggregation.
- `adminapi/observability.go` — phase-7 subset:
  `GET /_plugin/spend/by-agent`, `GET /_plugin/spend/by-user`,
  `GET /_plugin/histogram/cost`, `GET /_plugin/runs/:id`. Go-side
  bucket/sum over `metadata.agent-name` (Bifrost's native
  dimension histogram doesn't support metadata columns).
- `adminapi/ui.go` — `//go:embed all:ui/dist` + SPA fallback so
  client-side routes survive a hard refresh.
- `env/env.go` — added `PRODUCTION` env var + `IsProduction()` so
  the session cookie's `Secure` flag is forced on behind a TLS
  ingress.

**Frontend (`gateway/internal/adminapi/ui/`):**

- Vite + Preact + wouter + Tanstack Query + uPlot scaffold; React
  aliased through `preact/compat`; SPA mounted at `/_plugin/ui/`;
  hashed asset filenames so browser caches bust on redeploy.
- Dark theme via CSS variables (`base.css` + `components.css`),
  desktop-only (≥1280px).
- `api/client.ts` — typed fetch, 401 → `UnauthorizedError` sentinel
  the QueryCache turns into a `/login?next=…` redirect; phase-7
  error envelope mapping for everything else.
- `api/queries.ts` — one hook per endpoint with the per-page
  polling cadence from the original phase 8 plan (KPIs/rankings
  30s, histograms 60s, run detail manual).
- Components: `Shell` / `Sidebar` / `Topbar`, `WindowPicker`,
  `DataTable` (sortable), `UplotChart` + `CostHistogram` (with
  pinned x-axis to the query window so sparse data renders sanely),
  `EmptyState`, `ErrorBoundary`.
- Pages: `Login`, `Dashboard` (KPIs + chart + top-5 agents/users),
  `Agents`, `AgentDetail` (per-agent histogram + recent runs),
  `RunDetail` (call log, paginated), `NotFound`.

**Build & docs:**

- Dockerfile gained a `plugin-ui-builder` stage that runs
  `npm ci && npm run build`, then copies `dist/` into the Go build
  context before `go build` so `//go:embed` picks it up. Final
  runtime image has no Node.
- Makefile: `ui-install`, `ui-build`, `ui-dev`, `tygo`,
  `tygo-check` targets.
- `tygo.yaml` for Go → TS struct codegen (`make tygo` regenerates
  `ui/src/api/types.ts`; CI hook fails on drift).
- `docker-compose.yml` reuses the existing Redis sidecar for
  sessions; added `PRODUCTION` env passthrough.
- README updated with the new route table + UI dev flow + default
  creds.

**Tests (all passing under `go test ./internal/...`):**

- `sessions/store_test.go` — 7 tests covering the full session
  lifecycle and TTL semantics.
- `adminapi/login_test.go` — login happy/sad paths, rate limit,
  /me cookie + bearer, logout, SPA gating, deep-link fallback.
- `adminapi/observability_test.go` — fake Bifrost upstream,
  end-to-end tests for all four observability endpoints,
  parameter validation, upstream-failure → 502 mapping.

### Verified end-to-end

`make docker-up` brings up the full stack; the dashboard is
reachable at <http://localhost:8181/_plugin/ui/> with default dev
creds `admin` / `bifrost-dev-password`. Fired a handful of
inference calls with `x-bf-dim-*` headers and confirmed the
Dashboard / Agents / AgentDetail / RunDetail pages all populate
from the same `logs.db` Bifrost already writes.

Branded as **"Agent Gateway"** in the UI for the demo (sidebar /
login / browser tab title). Package names, env vars, log prefixes,
and the `/_plugin/` URL prefix retain the `bifrost` / `stakgraph`
identifiers — those are internal contracts and stable across
renames.

### What's NOT in this phase

The original "explicitly does NOT do" list below remains accurate.
Repeating the highlights for skim-readers:

- No kill switches, no budget editors (mutations are phase 9).
- No live Redis state (cap meters, in-flight badges, current
  bucket spend) — all of that depends on phase 6 hot state.
- No macaroon-anchored attribution yet — `metadata.*` reflects
  what the caller sent until phase 6 canonicalizes dims from
  verified claims. The UI doesn't change when that lands.

### Demo gaps (need attention before the upcoming demo)

Two things the dashboard doesn't yet surface that are non-negotiable
for the demo narrative. Both are additive on top of phase 8's
scaffold (no new infra; just new endpoints + pages):

1. **Agent budgets.** Phase 6's plugin enforcement reads
   `agent_budgets.<name>.{cap_usd, window}` from the plugin config
   block, but the dashboard has no view of these caps or the
   current-period spend against them. For the demo we need at
   minimum a **read-only** display on the Agents list and Agent
   detail pages:
     - Column on `/agents`: "Budget" → e.g. `$5.00 / 1d`.
     - Card on `/agents/:name`: "Current period spend: $1.23 /
       $5.00 ▌▌▌▌▌░░░░░ 25%" (a progress bar against the cap).
   - Backend: a new
     `GET /_plugin/agents/:name/budget` endpoint that returns
     `{cap_usd, window, period_start, period_end, spent_usd,
     remaining_usd, ratio}`. Reads the cap from plugin config (or
     phase-9's `bifrost:config:agent_budgets:<name>` Redis hash
     once that lands) and the spent value from
     `bifrost:cost:agent:<name>:<bucket_key>` per phase 6
     "Redis schema". When neither source has a value, return
     `{cap_usd: null}` and the page renders "no budget set".
   - **Demo path:** even if phase 6's enforcement isn't fully
     wired, we can seed a config block with a couple of caps
     and Redis with mock cumulative spend so the UI renders
     real-looking data on the demo branch.

2. **Agent-run provenance.** The story "this run was authorized
   by alice in workspace w1" is core to the demo and the data is
   already on the wire — `logs.metadata` carries the `user-id` and
   `realm-id` dim values that the caller stamps on every call
   (and that phase 6 will canonicalize from the verified macaroon
   when enforcement lands). The Run detail page today shows them
   only inside the per-call metadata blob; the fix is purely a
   frontend rearrangement.
   - **Provenance card** above the call log on `/runs/:id`, read
     directly from the first log row's `metadata`:
       - `user-id` (rendered `mono`; Hive-side
         user-id → name resolution is out of scope here)
       - `realm-id` (workspace)
       - `agent-name`
       - `session-id` (collapses the run into its parent thread)
       - First-seen / last-seen timestamps, derived from the
         first and last log row in the response
       - Total cost + total calls, already in `stats`
   - **No backend change.** `GET /_plugin/runs/:id` already
     returns the per-call `metadata` map and aggregate stats.
   - When phase 6 lands and macaroon claims become the
     authoritative source for these dims, the UI is unchanged —
     the underlying values just become cryptographically attested.
     Richer provenance (org_id, caveats, attenuation chain,
     verification status) is a phase-9 concern; phase 8.5 just
     pulls what's already in `logs.metadata` up to the top of the
     page so an operator doesn't have to read it row-by-row.

### What's next (in priority order)

1. **Demo blockers above** — agent budgets and run provenance.
   Both are scoped enough that they could ship as a "phase 8.5"
   PR on top of the merged phase 8 without touching the
   scaffolding.
2. **tygo runs in CI** — currently `make tygo` works locally but
   no CI hook fails the build on drift. Once tygo is installed in
   the CI image, switch `make tygo-check` into the gate.
3. **Phase 9 scope items** (kill switches, budget editors, live
   state blends, users page, sessions page, config editor) — see
   [phase-9-operator-ui.md](./phase-9-operator-ui.md) for the full
   inventory and what each one needs from phases 6 / 7.

## What this phase decides

Three things you can deliver right now, without any new enforcement
work:

1. **The auth + UI scaffold.** Session cookies, login flow, Vite +
   Preact + wouter + Tanstack Query + uPlot, Dockerfile UI-builder
   stage, `//go:embed` of the built assets, tygo for Go→TS type
   sharing. Once this lands, every future UI page is "add a route +
   component" — no infrastructure work.
2. **A small read-only slice of the phase 7 endpoint surface.** Four
   handlers (spend-by-agent, spend-by-user, cost histogram, run
   drill-down) backed by a single `logstore_client.go` that talks
   to Bifrost's `/api/logs` over loopback. No Redis. No macaroon
   verification.
3. **Four pages that wrap them.** Dashboard, Agents list, Agent
   detail, Run detail. Read-only — no kill buttons, no budget
   editors, no in-flight badges, no Redis state blends.

After this phase: an operator clicks through
`https://swarm.example/_plugin/ui/`, logs in with their
`BIFROST_ADMIN_USER`/`PASS`, and sees real per-agent spend and cost
trends pulled from data the swarm is already generating.

## Relationship to phase 6 dim canonicalization

Phase 7's doc notes that observability becomes cryptographically
trustworthy only after phase 6 canonicalizes dims from verified
macaroon claims. Until then, the `metadata.agent-name` column in
`logs.db` reflects whatever `x-bf-dim-*` headers callers sent.

This doesn't change the UI in any way — the same endpoints and the
same components keep working when phase 6 lands. The only thing
that changes is the source-of-truth guarantee: pre-phase-6, callers
self-report dims; post-phase-6, the plugin overwrites the
`metadata.*` map from verified claims. Phase 8's UI code is
identical in either world.

## What this phase explicitly does NOT do

These are all in phase 9 or later. Listing them here so the scope
boundary is unambiguous:

- **No kill switches.** Phase 6's `/_plugin/runs/:id/kill` and
  `/_plugin/agents/:name/kill` endpoints depend on Redis state that
  doesn't exist yet.
- **No budget editors.** Phase 9's `/_plugin/config/*` endpoints
  depend on the in-memory effective-config layer that phase 6's
  enforcement hooks read from.
- **No `/runs/:id/state` calls.** Live Redis state (cost,
  steps, tools, TTL) doesn't exist until phase 6.
- **No "in-flight" badges on the runs list.** Same reason — the
  blend of logs-derived rows with Redis-derived live state is a
  phase 9 concern.
- **No current-bucket-vs-cap displays on agent detail.** Same.
- **No revocation viewer, no users-quota page, no sessions
  drill-down, no Config page.** All phase 9.
- **No macaroon inspector.** Speculative even for phase 9.
- **No WebSocket live tail.** Polling at the cadences in
  "Data-fetching cadence" below is sufficient.
- **No mobile / responsive layout.** Desktop ≥ 1280px only.

## Relation to prior phases

| Phase | Status for phase 8 |
|---|---|
| Phase 1 (reconciler) | Required: provides the workspace Bifrost with VKs and Customers. |
| Phase 2 (publish image) | Required: phase 8 ships in the same image. |
| Phase 3 (swarm handoff) | Required: provides `BIFROST_ADMIN_USER`/`PASS` that phase 8 authenticates against, and the wrapper proxying `/_plugin/*`. |
| Phase 4 (macaroon shape) | **Not required.** Phase 8 reads `logs.metadata` directly, regardless of whether macaroons are issued yet. |
| Phase 5 (trust registry) | **Not required.** No verification happens. |
| Phase 6 (enforcement) | **Not required.** No Redis hot state is read. |
| Phase 7 (observability HTTP) | Partially required: phase 8 implements the four endpoints it uses from phase 7's spec. The remaining phase 7 endpoints land alongside phase 9. |
| Phase 9 (operator UI) | Phase 8 is its scaffold. |

So the build order is: phases 1-3 are prerequisites (and already
done or in progress), phase 7's small four-endpoint slice lands as
part of phase 8, and phases 4-6 unblock phase 9.

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
       gateway/internal/adminapi/       bifrost-http :8080
         (Go HTTP server)               ┌────────────────────┐
         ┌──────────────────┐           │ /api/logs          │
         │ /_plugin/login   │           │  SearchFilters     │
         │ /_plugin/logout  │           │  MetadataFilters   │
         │ /_plugin/me      │           │  Pagination        │
         │ /_plugin/spend/* │ ──────────►│ LogStore impl     │
         │ /_plugin/histogram/cost      │ (SQLite)           │
         │ /_plugin/runs/:id│           └────────────────────┘
         │ /_plugin/ui/*    │ (//go:embed'd Preact SPA)
         └──────────────────┘
                 │
                 ▼
              Redis (sessions only)
```

**No new storage.** Sessions live in Redis alongside what will
become phase 6's hot state. The dashboard reads `logs.db`
exclusively via Bifrost's HTTP API — the plugin never opens a
SQLite file directly.

## Framework stack

Identical to phase 9's stack. Decided here so it's not relitigated:

| Concern | Choice | Why |
|---|---|---|
| UI framework | **Preact** (3 KB) via `@preact/preset-vite` | React-API-compatible, small bundle |
| Build | **Vite** | Fast HMR, clean prod bundles |
| Routing | **`wouter`** (1.5 KB) | Hooks-based, flat routes |
| Data | **`@tanstack/query`** via `preact/compat` | Caching, polling, mutations later |
| Charts | **`uPlot`** (40 KB) + thin Preact wrapper | Fast, small, no React runtime |
| Styling | **Vanilla CSS + CSS variables**, **dark mode by default** | No build complexity; CSS vars define a single dark palette (background, surface, text, accent, danger) consumed by every component, so the demo looks polished from first paint without a light-mode toggle to build |
| Type sharing | **`tygo`** (Go → TS struct codegen) | Single source of truth |

Explicit non-choices: no Tailwind, no styled-components, no SSR,
no react-router, no global state library beyond Tanstack Query.

## Repo layout

```
gateway/internal/adminapi/ui/
  package.json
  package-lock.json
  vite.config.ts
  tsconfig.json
  index.html
  src/
    main.tsx                      # entry, mounts <App/>
    app.tsx                       # wouter routes + QueryClient provider
    api/
      client.ts                   # fetch wrapper, credentials: 'include', 401 → /login
      types.ts                    # GENERATED by tygo from Go structs
      queries.ts                  # useQuery hook per GET endpoint
    components/
      layout/
        Shell.tsx                 # sidebar + topbar + outlet
        Sidebar.tsx
        Topbar.tsx                # user, logout
      charts/
        UplotChart.tsx            # generic uPlot wrapper
        CostHistogram.tsx
      tables/
        DataTable.tsx             # sortable, paginated; used by 3+ pages
      controls/
        WindowPicker.tsx          # window=1h|24h|7d|30d
      EmptyState.tsx
      ErrorBoundary.tsx
    pages/
      Login.tsx                   # basic-auth → session cookie
      Dashboard.tsx               # KPIs + headline chart + rankings
      Agents.tsx                  # observed agents table
      AgentDetail.tsx             # per-agent histogram + recent runs
      RunDetail.tsx               # per-run call log (read-only)
      NotFound.tsx
    styles/
      base.css                    # reset, CSS vars (palette, spacing, radii)
      components.css              # shared component utility classes
  dist/                           # build output, //go:embed target
```

Two phase-9-ready stubs intentionally absent from phase 8:
`mutations.ts` and the `controls/KillButton.tsx` /
`BudgetEditor.tsx` / `KillConfirmModal.tsx` components. They land
when phase 9 needs them; their absence in phase 8 keeps the scope
honest.

`dist/` is the `//go:embed` target and is **not** checked into git;
CI and the Dockerfile build it fresh.

## Build pipeline

A new build stage in `gateway/Dockerfile`, before the existing Go
build:

```dockerfile
FROM node:22-alpine AS ui-builder
WORKDIR /ui
COPY gateway/internal/adminapi/ui/package*.json ./
RUN npm ci
COPY gateway/internal/adminapi/ui/ ./
RUN npm run build
# Output: /ui/dist/
```

Then the Go build stage gains:

```dockerfile
COPY --from=ui-builder /ui/dist /src/gateway/internal/adminapi/ui/dist
```

`//go:embed ui/dist` in `adminapi/ui.go` picks it up at compile
time. Build cache invalidates the UI stage only when `package*.json`
or any file under `ui/` changes — Go-only changes don't re-run
`npm`.

For local development outside Docker:

```
cd gateway/internal/adminapi/ui
npm install
npm run dev     # Vite dev server on :5173 with HMR
                # Vite proxy forwards /_plugin/* → http://localhost:8181
```

`vite.config.ts` configures
`server.proxy['/_plugin'] = 'http://localhost:8181'` so the dev
server talks to a locally-running plugin (started via
`make docker-up`) for real API responses. No CORS dance.

`vite.config.ts` disables filename hashing (`entryFileNames:
'assets/[name].js'`). Long-term browser caching is meaningless for
an embedded admin UI behind session auth, and non-hashed names make
the Go `//go:embed` index predictable.

## Auth

Identical to phase 9's design — locked in here so phase 9 inherits
it without changes. Basic auth at `/_plugin/login` only; session
cookie issued by the plugin, stored in Redis. Bearer token continues
to work for Hive's machine-to-plugin calls.

### Login flow

```
1. Browser GET /_plugin/ui/   (no cookie)
   → SPA loads, calls /_plugin/me which 401s
   → SPA redirects to /_plugin/ui/login

2. User enters admin / password.
   SPA does:
     POST /_plugin/login
     Authorization: Basic <base64(user:pass)>

3. Plugin validates the Basic header against the hashed creds it
   already loaded from Bifrost's config.db (phase 3 mechanism).
   - If wrong: increment bifrost:login_attempts:<ip>; if >= 10 within
     15m, 429; otherwise 401.
   - If right: generate 32-byte random session ID, store in Redis,
     set cookie, return { user }.

4. Set-Cookie: bifrost_session=<43-char-base64url>;
                 HttpOnly;
                 Secure;        (set based on X-Forwarded-Proto or env)
                 SameSite=Strict;
                 Path=/_plugin;
                 Max-Age=28800

5. SPA redirects to /_plugin/ui/ (now authenticated).
```

Basic auth is **only** accepted on `POST /_plugin/login`. It is not
a fallback for other endpoints.

### Logout flow

```
POST /_plugin/logout                  (carries the cookie)
  DEL bifrost:session:<id>
  SREM bifrost:sessions:<user> <id>
  Set-Cookie: bifrost_session=; Max-Age=0
  → 204
```

### Cookie attributes

```
HttpOnly         JS can't read it (XSS protection)
Secure           HTTPS only — set when X-Forwarded-Proto=https OR env
                 PRODUCTION=1; off on dev localhost
SameSite=Strict  No cross-site usage — strong enough that explicit CSRF
                 tokens are not needed for state-changing endpoints
Path=/_plugin    Scoped to the plugin namespace, not Bifrost's /api
Max-Age=28800    8-hour session
```

`SameSite=Strict` is load-bearing: phase 8 ships no CSRF-token
layer. If a future requirement demands relaxing it, CSRF tokens
get added at the same time.

### Session storage (Redis)

```
bifrost:session:<random_id>     HASH    { user, iat, last_seen }   TTL 8h
bifrost:sessions:<user>         SET     <session_id, ...>          (no TTL)
bifrost:login_attempts:<ip>     STRING  <count>                    TTL 15m
```

Random ID is 32 bytes from `crypto/rand`, base64url-encoded. Stored
in Redis (this is the only Redis usage phase 8 introduces) rather
than as a stateless JWT because real logout, real admin kick, and
TTL on inactive sessions all need server-side state.

`last_seen` refreshes at most once per minute per session
(write-skipping) to avoid a hot key on chatty UIs. TTL resets on
each `last_seen` write.

### Session middleware

In `gateway/internal/adminapi/session.go`. Wraps the handler chain.
Tries the session cookie first, falls back to the existing bearer
token, rejects with 401 otherwise.

```go
allowAnon := []string{
    "/_plugin/health",
    "/_plugin/login",
}
```

Both auth schemes coexist under `/_plugin/*`:

| Caller | Path | Credential |
|---|---|---|
| Browser UI | `/_plugin/ui/*`, `/_plugin/spend/*`, `/_plugin/runs/*`, etc. | `Cookie: bifrost_session=<id>` |
| Hive | `/_plugin/admin-credentials`, `/_plugin/trust/*` | `Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>` |
| Anyone | `/_plugin/health`, `/_plugin/login` | none |

### Login rate limiting

`POST /_plugin/login` is the only brute-forceable endpoint. Per-IP
limit, not per-username (per-user lockout is an enumeration oracle):

```
bifrost:login_attempts:<ip>   INCR per failed attempt
                              EXPIRE 900   (15m)
On 10th failure → 429 too_many_login_attempts; Retry-After: 900
```

Successful login resets the counter via `DEL`. Client IP from
`X-Forwarded-For` set by the swarm ingress; running without an
ingress that sets XFF makes the limiter degenerate and is out of
phase 8's threat model.

## Endpoint surface (phase 8 subset)

Three categories of HTTP additions. Everything else from phases 6,
7, and 9 is deferred.

### Auth (new for phase 8)

```
POST   /_plugin/login           Basic-auth in, session cookie out, JSON { user }
POST   /_plugin/logout          Session-cookie in, cookie cleared, 204
GET    /_plugin/me              Session-cookie in, JSON { user, iat, last_seen }
GET    /_plugin/ui/*            //go:embed'd SPA assets
```

`/_plugin/me` is the SPA's "am I authenticated?" probe used at boot.

### Observability (phase 7 subset)

Four endpoints, all read-only, all backed by Bifrost's `/api/logs`
over loopback. They follow phase 7's response-shape contract.

```
GET /_plugin/spend/by-agent?window=24h
  Returns: { window, results: [ { agent_name, total_cost, total_tokens, request_count }, ... ] }

GET /_plugin/spend/by-user?window=24h
  Bifrost API: GET /api/logs/rankings/users
  Returns: { window, results: [ { user_id, user_name, total_cost, total_tokens, request_count }, ... ] }

GET /_plugin/histogram/cost?window=24h&bucket=1h&dimension=agent-name
  Bifrost API: GetDimensionCostHistogram(dimension=metadata.agent-name, bucket=3600)
  Returns: {
    bucket_size_seconds: 3600,
    series: [
      { dimension_value: "coder",       points: [ { ts, cost }, ... ] },
      { dimension_value: "web-search",  points: [ { ts, cost }, ... ] },
      ...
    ]
  }

GET /_plugin/runs/:run_id
  MetadataFilters: { "run-id": <id> }
  Bifrost API: GET /api/logs?metadata_filters={"run-id":...}
  Returns: { run_id, logs: [ <Log> ... ], stats: <SearchStats> }
```

Two endpoints from phase 7's full inventory are intentionally
**not** included in phase 8 even though they could technically
ship: `GET /_plugin/spend/by-realm` and
`GET /_plugin/spend/by-session`. They're easy to add when needed,
but the four pages phase 8 ships don't need them yet, and shipping
unused endpoints invites premature use.

### What's deferred

Phase 9 adds the rest of phase 7 plus the mutations:
- `/_plugin/spend/by-{realm,session,model}`
- `/_plugin/histogram/{tokens,latency}`
- `/_plugin/sessions/:id`, `/_plugin/sessions/:id/summary`
- `/_plugin/users/:id/spend`, `/_plugin/users/:id/quota`
- `/_plugin/agents/:name/spend`
- All phase 6 kill/state mutations
- All phase 9 config mutations

## View inventory

Four pages plus Login plus NotFound. Each is one file under
`src/pages/`.

### Login (`/login`)

User + password form, submits to `/_plugin/login`. On success,
redirects to wherever the user was trying to go (preserved in a
`?next=` query param) or to `/`.

### Dashboard (`/`)

The "what is this swarm doing right now" landing page.

- **Top KPIs (3 cards):** total spend today, total spend this
  month, total request count today.
- **Headline chart:** stacked-area cost histogram over the last
  24h bucketed by hour, grouped by `agent-name`. Hits
  `/_plugin/histogram/cost?window=24h&bucket=1h&dimension=agent-name`.
- **Two rankings tables:** top 5 agents by spend (last 24h), top 5
  users by spend (last 24h). Both clickable to their detail pages
  (agents → `/agents/:name`, users → `#` for now — Users page is
  phase 9).

Refresh: 30s for the KPIs and rankings, 60s for the histogram.

### Agents (`/agents`)

Observed agents (distinct `metadata.agent-name` values from
`logs.db`).

- **Table:** agent name, spend (selectable window), token count,
  call count, last seen.
- **No kill column, no budget editor.** Phase 9.
- **Window picker:** 1h, 24h, 7d, 30d.

Each row links to `/agents/:name`.

Refresh: 30s.

### Agent detail (`/agents/:name`)

- **Cost histogram:** filtered to this agent name, with the
  page-level window/bucket controls. Reuses the headline chart's
  uPlot wrapper.
- **Recent runs table:** distinct `metadata.run-id` values from
  this agent's logs over the selected window. Each row links to
  `/runs/:id`. Derived client-side from a `/_plugin/runs/:run_id`
  fan-out? No — derived from a single query that filters
  `/_plugin/spend/by-agent` upstream + groups by `run-id`. (If the
  cardinality is high, paginate; phase 8 caps at the first 100
  runs in the window.)

Refresh: 60s for the histogram, 30s for the runs table.

### Run detail (`/runs/:id`)

- **Run summary card:** total cost, total tokens, total calls,
  first-seen / last-seen timestamps, agent name, user.
- **Call log:** table of every call in this run, from
  `/_plugin/runs/:id`. Paginated; default 50 most recent.
- **No live state panel, no cap meters, no kill button.** All
  phase 9 (they need Redis).

Refresh: manual (data is historical; no polling needed in v1).

## Data-fetching contract

All API calls go through one `fetch` wrapper that:

- Prefixes `/_plugin`
- Sends `credentials: 'include'`
- Maps 401 → throws an `unauthorized` error → top-level handler
  redirects to `/login?next=<current>`
- Maps non-2xx → throws with the JSON `error.code` from phase 7's
  error envelope
- 5s default timeout (configurable per call)

Tanstack Query is configured with:

- `staleTime: 0` (always refetch on focus); fits an ops console
- `gcTime: 5 * 60_000` (5 min cache after unmount)
- `retry: 1` for GETs (no mutations in phase 8)
- 401 errors short-circuit retries via the queryCache `onError`
  handler

Per-query polling intervals at the hook level, not globally:

| Query | Refetch interval |
|---|---|
| Dashboard KPIs / rankings | 30s |
| Dashboard headline histogram | 60s |
| Agents list | 30s |
| Agent detail histogram | 60s |
| Agent detail runs table | 30s |
| Run detail | manual (no polling) |
| `/me` | once at app boot |

Polling pauses when the tab is hidden (Tanstack Query default).

## Type generation (tygo)

Go is the source of truth for response shapes. Hand-writing TS
equivalents drifts.

`tygo` configured in `gateway/tygo.yaml` to emit TS for the
adminapi response types:

```yaml
packages:
  - path: "github.com/sphinx-chat/stakgraph/gateway/internal/adminapi"
    output_path: "internal/adminapi/ui/src/api/types.ts"
    type_mappings:
      time.Time: "string"
```

`make tygo` runs the codegen; CI runs `make tygo && git diff
--exit-code` to catch drift.

Handlers declare named response types
(`type SpendByAgentResponse struct { ... }`) rather than ad-hoc
`map[string]any`, so the codegen has something to bind to. Small
discipline tax, large payoff — the UI never sees a server response
shape the compiler hasn't approved.

## Embedding and serving

```go
// gateway/internal/adminapi/ui.go

//go:embed ui/dist
var uiFS embed.FS

func uiHandler() http.Handler {
    sub, _ := fs.Sub(uiFS, "ui/dist")
    fsrv := http.FileServer(http.FS(sub))
    return http.StripPrefix("/_plugin/ui/", spaFallback(sub, fsrv))
}

// spaFallback serves index.html for any path that doesn't match a
// real asset, so wouter's client-side routes survive a hard refresh.
func spaFallback(root fs.FS, fileServer http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/")
        if _, err := fs.Stat(root, path); err == nil && path != "" {
            fileServer.ServeHTTP(w, r)
            return
        }
        index, err := fs.ReadFile(root, "index.html")
        if err != nil {
            http.Error(w, "ui not built", 500)
            return
        }
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        w.Header().Set("Cache-Control", "no-store")
        w.Write(index)
    })
}
```

`index.html` carries `Cache-Control: no-store` so updates roll out
on the next page load. Assets get default file-server caching.

The UI handler registers in `adminapi/server.go` *after* the
session middleware, so `/_plugin/ui/*` requires either a valid
session cookie or a valid bearer. The `/_plugin/ui/login` route is
served by the same SPA handler (wouter handles it client-side).

## Wire-up checklist

**Backend — sessions (`gateway/internal/sessions/`):**

- [ ] `store.go`: `SessionStore` interface; Redis implementation.
- [ ] `store_test.go`: miniredis-backed tests covering create, get,
      refresh, delete, kick-all-for-user, TTL expiry.

**Backend — adminapi (`gateway/internal/adminapi/`):**

- [ ] `session.go`: middleware combining session-cookie and
      bearer-token auth; `allowAnon` list.
- [ ] `login.go`: `POST /_plugin/login` (Basic-in, cookie-out),
      `POST /_plugin/logout`, `GET /_plugin/me`.
- [ ] `ratelimit.go`: per-IP login attempt counter.
- [ ] `logstore_client.go`: HTTP client to
      `http://127.0.0.1:8080/api/logs` (and rankings / histogram
      sub-paths). Owns `SearchFilters` serialization, retry,
      timeout. Reusable across all handlers below and any phase 9
      additions.
- [ ] `spend.go`: `/_plugin/spend/by-{agent,user}` handlers.
- [ ] `histogram.go`: `/_plugin/histogram/cost` handler calling
      `GetDimensionCostHistogram` through the client.
- [ ] `runs.go`: `/_plugin/runs/:run_id` drill-down (read-only;
      phase 9 extends with `/state` and `/kill`).
- [ ] `ui.go`: `//go:embed ui/dist` + SPA fallback handler.
- [ ] Route registration in `server.go`, with `/_plugin/ui/*` and
      every `/_plugin/spend/*`, `/_plugin/histogram/*`,
      `/_plugin/runs/:id` route behind the session middleware, and
      `/_plugin/health`, `/_plugin/login` in `allowAnon`.

**Backend — type codegen (`gateway/`):**

- [ ] `tygo.yaml` covering `internal/adminapi`.
- [ ] `make tygo` target invoking the codegen.
- [ ] CI step: `make tygo && git diff --exit-code internal/adminapi/ui/src/api/types.ts`.

**Frontend (`gateway/internal/adminapi/ui/`):**

- [ ] Vite + Preact + TS scaffold; `package.json`, `vite.config.ts`,
      `tsconfig.json`, `index.html`.
- [ ] `src/api/client.ts`: typed fetch wrapper, 401 redirect.
- [ ] `src/api/queries.ts`: one hook per endpoint.
- [ ] `src/app.tsx`: wouter routes, QueryClient provider, global
      auth-error handler.
- [ ] `src/components/layout/Shell.tsx` + `Sidebar` + `Topbar`.
- [ ] `src/components/charts/UplotChart.tsx`: generic wrapper.
- [ ] `src/components/charts/CostHistogram.tsx`: stacked-area
      cost-by-time-by-dimension chart.
- [ ] `src/components/tables/DataTable.tsx`: sortable, paginated.
- [ ] `src/components/controls/WindowPicker.tsx`.
- [ ] `src/components/EmptyState.tsx`, `ErrorBoundary.tsx`.
- [ ] `src/pages/Login.tsx`, `Dashboard.tsx`, `Agents.tsx`,
      `AgentDetail.tsx`, `RunDetail.tsx`, `NotFound.tsx`.
- [ ] `src/styles/base.css` + `components.css`.

**Dockerfile:**

- [ ] `ui-builder` stage running `npm ci && npm run build`.
- [ ] `COPY --from=ui-builder /ui/dist` into the Go build context
      before `go build`.

**Docs:**

- [ ] Update `gateway/README.md` with the observability-dashboard
      bullet (URL, default creds in dev, link to this phase).
- [ ] Update `phase-3-swarm-handoff.md` to note that admin creds
      now drive both `/api/*` and the observability dashboard.
- [ ] Update `phase-7-observability.md` to note that the four
      endpoints used by phase 8 ship in phase 8; the remainder
      ships with phase 9.
- [ ] Forward-pointer from `llm-governance-v2.md` §"Plugin" to this
      phase.

**Gate:** phase 8 ships when:

1. Login → dashboard → drill into an agent → drill into a run, all
   end-to-end against a real Bifrost+plugin stack with the existing
   test agent producing logs.
2. Tygo runs in CI and the generated `types.ts` is up to date.
3. The Docker image builds without a node runtime in the final
   image (UI is `//go:embed`'d).

## Phasing within phase 8

Three slices, each shippable independently:

1. **Slice 1 — Foundation.** Sessions store, login/logout/me
   endpoints, session middleware, Vite scaffold, Login page,
   placeholder Dashboard ("Hello, admin" + logout), Dockerfile
   build stage, tygo wired up. Proves the embed and auth pipeline
   end-to-end.
2. **Slice 2 — Dashboard + Agents.** `logstore_client.go`, the
   four observability handlers, Shell/Sidebar/Topbar landed
   properly, Dashboard with real data, Agents list, AgentDetail
   with histogram + runs table.
3. **Slice 3 — Run drill-down.** RunDetail page with call log
   from `/_plugin/runs/:id`. Mostly UI work; the backend handler
   exists from slice 2.

Slice 1 is one PR of plumbing; slices 2-3 are feature work on top.

## Forward path to phase 9

When phase 6 enforcement lands, phase 9 picks up where this leaves
off. The additions are purely additive:

| Phase 9 adds | What it needs from phase 6 |
|---|---|
| Run detail "live state" panel | `GET /_plugin/runs/:id/state` |
| Run detail kill button | `POST /_plugin/runs/:id/kill` |
| Agents list "current bucket spend" column | `GET /_plugin/agents/:name/state` |
| Agent detail budget editor | `PUT /_plugin/config/agent_budgets/:name` |
| Agent kill button (with typed confirmation) | `POST /_plugin/agents/:name/kill` |
| Users / User detail pages | phase 7 `/users/:id/spend`, `/users/:id/quota` |
| Sessions / Session detail | phase 7 `/sessions/:id`, `/sessions/:id/summary` |
| Config page | phase 9 config-overrides endpoints |
| Cryptographic attestation of dims | phase 6 dim canonicalization (no UI change) |

Nothing about the phase 8 stack changes. New routes, new
components, new queries — all on the same Vite/Preact/wouter/
Tanstack/uPlot foundation. The auth middleware, embed mechanism,
type-gen pipeline, and Docker build all carry over unchanged.

## What this phase buys

- **A useful operator dashboard, today.** Without waiting on
  phases 4-6, operators see live per-agent spend, per-run
  drill-downs, and cost trends from data the swarm is already
  generating.
- **The full UI infrastructure stack landed in one PR.** Vite,
  Preact, wouter, Tanstack Query, uPlot, tygo, embed, auth,
  Dockerfile — phase 9 adds features, not scaffolding.
- **Honest scope.** No kill switches built on Redis state that
  doesn't exist; no budget editors built on an effective-config
  layer that doesn't exist. The phase 8 surface area is exactly
  what the underlying systems can support.
- **No new SQL.** Sessions in Redis; logs read via Bifrost's
  `/api/logs`. Plugin opens no SQLite file.
- **Forward-compatible with phase 6.** When dim canonicalization
  lands, the same components keep working — the underlying data
  just becomes cryptographically attested. No UI change needed.
- **No coupling to Hive.** Works on a fresh swarm the moment
  phase 3's bootstrap completes.
