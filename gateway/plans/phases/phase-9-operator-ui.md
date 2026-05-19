# Phase 8 — Operator UI: Embedded Console for Kill, Budgets, and Analytics

> Operator-facing single-page app served by the plugin at
> `/_plugin/ui/*`. Companion to `phase-6-plugin-enforcement.md` (which
> defines the kill/state/budget mutation endpoints the UI drives) and
> `phase-7-observability.md` (which defines the read-only analytics
> endpoints the UI charts). This phase produces no new wire format,
> no new crypto, no new SQL tables, and no
> new hook behaviour. It is a UI on top of the HTTP surface phases 6
> and 7 already deliver.
>
> The UI ships inside the plugin binary via `//go:embed`. There is no
> separate frontend deployment, no CDN, no node runtime in the
> production image. One Docker build produces one image that contains
> Bifrost, the plugin, the wrapper, and the UI.

## What this phase decides

Phases 6 and 7 land roughly 20 HTTP endpoints under `/_plugin/*`
covering kill switches, run/agent state inspection, spend
aggregations, time-series histograms, and drill-downs. Without a UI,
the only consumers are `curl` and whatever Hive eventually builds.
But:

- Kill switches need confirmation modals and "evidence the kill
  worked" feedback. CLI workflows don't give operators that.
- Budget edits over `agent_budgets` are config mutations that benefit
  from inline validation against Bifrost's duration vocabulary.
- The "what is this swarm spending right now, and who is spending it"
  question is fundamentally visual — a chart, not a JSON blob.
- The plugin's `/_plugin/*` namespace is the natural home (one auth
  scheme, one binary, no cross-origin headache), and the wrapper
  already proxies it.

Phase 8 decides:

- The UI's framework stack, build pipeline, and embed mechanism.
- The auth model (basic-auth handoff to plugin-issued session cookies
  in Redis, coexisting with the existing bearer token for
  machine-to-plugin calls).
- The view inventory and how it maps to phase 6 / phase 7 endpoints.
- The data-fetching, caching, and live-refresh contract.
- The Go→TypeScript type generation pipeline so UI types stay in
  lock-step with the handler response shapes.
- The interaction patterns for destructive operations (kill, unkill,
  budget edits).

After this:

- An operator clicking through `https://swarm.example/_plugin/ui/`
  can see live spend, drill into a runaway run, kill it, kill its
  agent across the swarm, edit per-agent budgets, and rotate their
  own session — all without leaving the dashboard.
- Hive-side dashboards can keep talking to the same `/_plugin/*`
  endpoints; the UI is additive, not a re-platforming.
- New features that touch the same data (e.g. a future macaroon
  inspector view) drop into the existing routes/components without
  re-litigating the framework or build.

## Relation to v2 and prior phases

| Phase | Decides |
|---|---|
| Phase 4 | What bytes a macaroon contains |
| Phase 5 | Which orgs the swarm trusts |
| Phase 6 | How the plugin enforces caveats against Redis state |
| Phase 7 | Read-only per-dim analytics over `logs.db` |
| **Phase 8** | **Operator UI on top of phases 6 + 7** |

Phase 8 has a hard dependency on phases 6 and 7 because every UI
view it ships reads or writes one of their endpoints. If phase 6 or
7 changes a response shape, the typed UI client breaks at compile
time — by design.

Phase 8 does **not** depend on Hive shipping anything; the swarm
operator UI is fully self-contained inside the plugin. Hive
integration (single-sign-on from Hive's dashboard, cross-workspace
fan-out) is a follow-on phase.

## Out of scope for this phase

- **Org-admin CRUD over the agent registry.** That table lives in
  Hive (`plans/agent-registry.md`) and is edited from Hive's
  dashboard. The plugin UI shows *observed* agents and edits *plugin
  ceilings* (`agent_budgets:` in `plugin.yaml`), not org-scoped
  registry rows.
- **SSO from Hive.** Phase 8 authenticates with the same
  `BIFROST_ADMIN_USER`/`BIFROST_ADMIN_PASS` that gates `/api/*`
  today. SSO is a follow-on phase that piggybacks on the trust
  registry from phase 5.
- **Cross-workspace aggregation.** Each swarm's UI shows that swarm's
  state only. "What did `coder` spend across all workspaces" is a
  Hive-side job that fans out to each plugin's `/_plugin/spend/*`.
- **Macaroon inspector / debugger view.** Speculative; not in v1. The
  HTTP endpoints exist (`/api/logs` carries the raw `x-macaroon`
  header in metadata if logged) but the UI for it is not phase 8.
- **WebSocket live tail.** Phase 7 explicitly excluded this; Hive can
  connect to Bifrost's `ws://...:8080/ws` directly if it wants real
  log streams. Phase 8 polls at the cadences specified in
  "Data-fetching cadence" below.
- **Mobile / responsive layout.** Phase 8 targets desktop browsers
  ≥1280px wide. Operators use this from a workstation, not a phone.

## Framework stack

The stack below is chosen to keep the embedded bundle small
(~80 KB gzipped target), avoid runtime cost the UI doesn't need,
and not re-implement primitives that mature libraries already
solve.

| Concern | Choice | Why |
|---|---|---|
| UI framework | **Preact** (3 KB) via `@preact/preset-vite` | React-API-compatible, fits the embedded-binary use case |
| Build | **Vite** + esbuild + Rollup | Fast HMR for dev, clean prod bundles |
| Routing | **`wouter`** (1.5 KB) | Hooks-based, flat routes, Preact-compatible |
| Data | **`@tanstack/query`** via `preact/compat` | Caching, polling, optimistic mutations |
| Charts | **`uPlot`** (40 KB) + thin Preact wrapper | Fast, small, no React runtime overhead |
| Styling | **Vanilla CSS + CSS variables**; CSS modules where component-scoping helps | No build complexity; predictable output |
| Type sharing | **`tygo`** (Go → TS struct codegen) | Single source of truth for response shapes |

Explicit non-choices:

- No Tailwind / UnoCSS — atomic CSS overhead isn't worth it for ≤10
  pages.
- No styled-components / emotion — runtime CSS-in-JS fights the
  small-runtime story.
- No react-router — `wouter` is a better fit for flat routing under
  `preact/compat`.
- No SSR — embedded admin UI behind auth has no SEO concern and no
  hydration win.
- No state library beyond Tanstack Query — server state is the only
  meaningful state; URL is the second source of truth via wouter.
  Component-local `useState` covers the rest.

## Repo layout

The UI is one directory under the plugin's adminapi package:

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
      queries.ts                  # one useQuery hook per GET endpoint
      mutations.ts                # one useMutation hook per POST/DELETE
    components/
      layout/
        Shell.tsx                 # sidebar + topbar + outlet
        Sidebar.tsx
        Topbar.tsx                # user, logout, env indicator
      charts/
        UplotChart.tsx            # generic uPlot wrapper
        CostHistogram.tsx
        TokenHistogram.tsx
        LatencyPercentiles.tsx
      tables/
        DataTable.tsx             # sortable, paginated; used by 5+ pages
      controls/
        WindowPicker.tsx          # window=1h|24h|1d|1w|1M
        BucketPicker.tsx          # bucket aligned to window
        DimensionPicker.tsx       # agent-name | user-id | realm-id | session-id | model
        KillButton.tsx            # opens KillConfirmModal
        BudgetEditor.tsx          # cap_usd + window, validates duration vocab
      KillConfirmModal.tsx        # typed-confirmation for agent kills
      StatusBadge.tsx             # running / killed / exceeded / done / errored
      EmptyState.tsx
      ErrorBoundary.tsx
    pages/
      Login.tsx                   # basic-auth → session cookie
      Dashboard.tsx               # top-level rollups + headline chart
      Runs.tsx                    # list + filters; in-flight + recent
      RunDetail.tsx               # live state + history + kill
      Agents.tsx                  # observed agents + budget table
      AgentDetail.tsx             # per-agent histograms + budget editor
      Users.tsx
      UserDetail.tsx              # quota blend + per-user runs
      Sessions.tsx
      SessionDetail.tsx
      Config.tsx                  # agent_budgets, hard_ceiling, tool_loop
      NotFound.tsx
    styles/
      base.css                    # reset, CSS vars (palette, spacing, radii)
      components.css              # shared component utility classes
  dist/                           # build output, //go:embed target
```

`dist/` is the `//go:embed` target. It is *not* checked into git; CI
and the Dockerfile build it fresh. Local `npm run build` produces it
for dev-time embedding when needed.

## Build pipeline

A new build stage in `gateway/Dockerfile`, run before the Go build:

```dockerfile
FROM node:22-alpine AS ui-builder
WORKDIR /ui
COPY gateway/internal/adminapi/ui/package*.json ./
RUN npm ci
COPY gateway/internal/adminapi/ui/ ./
RUN npm run build
# Output: /ui/dist/ — copied into the Go build context below.
```

Then the existing Go build stage gains:

```dockerfile
COPY --from=ui-builder /ui/dist /src/gateway/internal/adminapi/ui/dist
```

`//go:embed ui/dist` in `adminapi/ui.go` picks it up at compile time.
Build cache invalidates the UI stage only when `package*.json` or any
file under `ui/` changes — Go-only changes don't re-run `npm`.

For local development outside Docker:

```
cd gateway/internal/adminapi/ui
npm install
npm run dev     # Vite dev server on :5173 with HMR
                # Vite proxy forwards /_plugin/* → http://localhost:8181
```

`vite.config.ts` configures `server.proxy['/_plugin'] = 'http://localhost:8181'`
so the dev server talks to a locally-running plugin (started via
`make docker-up`) for real API responses. No CORS dance.

`vite.config.ts` also disables filename hashing (`entryFileNames:
'assets/[name].js'`). Long-term browser caching is meaningless for
an embedded admin UI behind session auth, and non-hashed names make
the Go `//go:embed` index predictable.

## Auth model

Two auth schemes coexist under `/_plugin/*`, distinguished by which
header the request carries:

| Caller | Path | Credential |
|---|---|---|
| Browser UI | `/_plugin/ui/*`, `/_plugin/spend/*`, `/_plugin/runs/*`, `/_plugin/agents/*`, etc. | `Cookie: bifrost_session=<id>` |
| Hive / swarm tooling | `/_plugin/admin-credentials`, `/_plugin/trust/*`, `/_plugin/runs/*/kill`, etc. | `Authorization: Bearer <BIFROST_PROVISIONING_TOKEN>` |
| Anyone | `/_plugin/health`, `/_plugin/login` | none |

Both schemes route through one middleware on the plugin's adminapi
server. The middleware tries the session cookie first, falls back to
the bearer token, and rejects with 401 otherwise. This preserves
phase 3's swarm bootstrap flow unchanged (Hive still calls
`/_plugin/admin-credentials` with a bearer) while enabling cookies
for the human-facing UI.

### Login flow

```
1. Browser GET /_plugin/ui/   (no cookie)
   → SPA loads, calls /_plugin/me which 401s
   → SPA redirects to /_plugin/ui/login

2. User enters admin / password (the same BIFROST_ADMIN_USER /
   BIFROST_ADMIN_PASS that gates /api/*).
   SPA does:
     POST /_plugin/login
     Authorization: Basic <base64(user:pass)>

3. Plugin validates the Basic header against the hashed creds it
   already loaded from Bifrost's config.db (phase 3 mechanism).
   - If wrong: increment bifrost:login_attempts:<ip>; if >= 10 within
     15m, 429; otherwise 401.
   - If right: generate 32-byte random session ID, store in Redis,
     set cookie, return { user: "admin" }.

4. Set-Cookie: bifrost_session=<43-char-base64url>;
                HttpOnly;
                Secure;        (set based on X-Forwarded-Proto or env)
                SameSite=Strict;
                Path=/_plugin;
                Max-Age=28800

5. SPA redirects to /_plugin/ui/ (now authenticated).
```

Basic auth is **only** accepted on `POST /_plugin/login`. It is not
a fallback for other endpoints. Once the session-cookie mechanism
exists, allowing Basic anywhere else would make the cookie scheme
decorative and re-open the "password in every request" surface area.

### Logout flow

```
POST /_plugin/logout                  (carries the cookie)
  DEL bifrost:session:<id>
  SREM bifrost:sessions:<user> <id>
  Set-Cookie: bifrost_session=; Max-Age=0
  → 204
```

The SPA calls this on logout-button click; the cookie is invalidated
server-side immediately, not just dropped client-side.

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

`SameSite=Strict` is load-bearing: it's the reason phase 8 does not
ship a separate CSRF-token mechanism. If a future requirement
demands relaxing it (e.g. cross-domain admin tooling), CSRF tokens
get added at the same time.

### Session storage

```
bifrost:session:<random_id>     HASH    { user, iat, last_seen }   TTL 8h
bifrost:sessions:<user>         SET     <session_id, ...>          (no TTL)
bifrost:login_attempts:<ip>     STRING  <count>                    TTL 15m
```

Random ID is 32 bytes from `crypto/rand`, base64url-encoded → 43
chars. Stored in Redis (alongside phase 6's hot state) rather than
JWT-style stateless because real logout, real admin kick, and TTL
on inactive sessions all need server-side state. Cost: one extra
`GET` per request, often pipelined with phase 6 lookups.

`last_seen` is refreshed at most once per minute per session
(write-skipping) to avoid a hot key on chatty UIs. TTL is reset on
each `last_seen` write.

### Session middleware contract

Implemented in `gateway/internal/adminapi/session.go`. Wraps the
handler chain. Routes that opt out of auth declare so explicitly:

```go
allowAnon := []string{
    "/_plugin/health",
    "/_plugin/login",
}
```

Everything else requires either a valid session cookie or a valid
bearer. The middleware stamps `r.Context()` with a `user` value
(either the admin username or the literal `"_provisioning"`)
so handlers can audit-log who did what.

### Login rate limiting

`POST /_plugin/login` is the only brute-forceable endpoint. Limit per
client IP, not per username (per-user lockout is an enumeration
oracle):

```
bifrost:login_attempts:<ip>   INCR per failed attempt
                              EXPIRE 900   (15m)
On 10th failure → 429 too_many_login_attempts; Retry-After: 900
```

Successful login resets the counter via `DEL`.

The client IP is read from `X-Forwarded-For` (the swarm ingress sets
it). Without an X-Forwarded-For header, the remote-addr from the
wrapper is used, which in production is always `127.0.0.1` and
makes the limiter degenerate — operators running without an ingress
proxy in front are explicitly out of phase 8's threat model.

## Endpoint surface (additions in phase 8)

Phase 8 adds three endpoints to phase-6/7's surface. Everything else
is already specified.

```
POST   /_plugin/login           Basic-auth in, session cookie out, JSON { user }
POST   /_plugin/logout          Session-cookie in, cookie cleared, 204
GET    /_plugin/me              Session-cookie in, JSON { user, iat, last_seen }
                                Used by the SPA at boot to test "am I logged in?"
GET    /_plugin/ui/*            //go:embed'd SPA assets
```

`/_plugin/me` is the SPA's "am I authenticated?" probe. It deliberately
returns minimal data — no role, no permissions, because phase 8's
model is binary (the bearer of `BIFROST_ADMIN_PASS` has full access).

The full operator-relevant surface, by page:

| Page | GET | POST/DELETE |
|---|---|---|
| Dashboard | `/spend/by-agent`, `/spend/by-user`, `/histogram/cost?dimension=agent-name` | — |
| Runs list | `/spend/by-session?window=…` (filtered to in-flight via Redis blend) | — |
| Run detail | `/runs/:id`, `/runs/:id/state` | `/runs/:id/kill`, `/runs/:id/kill` (DELETE) |
| Agents list | `/spend/by-agent`, `/agents/:name/state` per row | — |
| Agent detail | `/agents/:name/spend`, `/histogram/cost?dimension=agent-name`, `/agents/:name/state` | `/agents/:name/kill`, `/agents/:name/kill` (DELETE), `/config/agent_budgets/:name` (PUT — *new endpoint, see below*) |
| Users | `/spend/by-user` | — |
| User detail | `/users/:id/spend`, `/users/:id/quota` | — |
| Sessions | `/spend/by-session` | — |
| Session detail | `/sessions/:id`, `/sessions/:id/summary` | — |
| Config | `/config` (new GET — see below) | `/config/agent_budgets/:name` (PUT), `/config/hard_ceiling` (PUT), `/config/tool_loop` (PUT) |

### Config endpoints (extension of phase 6)

The Config page edits `agent_budgets`, `hard_ceiling`, and
`tool_loop` from `plugin.yaml`. Phase 6 specified these as YAML-only
configuration. Phase 8 adds HTTP equivalents:

```
GET    /_plugin/config                          → full config (read-only fields redacted)
PUT    /_plugin/config/agent_budgets/:name      { cap_usd, window } → 204
DELETE /_plugin/config/agent_budgets/:name      → 204 (removes the cap)
PUT    /_plugin/config/hard_ceiling             { per_invocation_cost_usd, per_invocation_steps } → 204
PUT    /_plugin/config/tool_loop                { window, threshold } → 204
```

**Overrides are persisted in Redis, not in a new SQL table.** Phase 8
introduces no new SQL storage. The plugin already owns Redis (phase 6
hot state, phase 8 sessions); config overrides live there too:

```
bifrost:config:agent_budgets:<name>     HASH    { cap_usd, window }       no TTL
bifrost:config:hard_ceiling             HASH    { cost_usd, steps }       no TTL
bifrost:config:tool_loop                HASH    { window, threshold }     no TTL
```

These three key shapes extend phase 6's schema (the canonical schema
list in `phase-6-plugin-enforcement.md` "Redis schema" should be
updated to include them when phase 8 lands).

**Precedence at runtime:**

```
1. Plugin loads plugin.yaml at boot → YAML baseline in memory.
2. PreLLMHook reads from in-memory config, which is composed as:
     effective = overlay(yaml_baseline, redis_overrides)
   The plugin caches the composed config and invalidates the cache
   on any successful PUT/DELETE to /_plugin/config/*.
3. PUT/DELETE handlers write to bifrost:config:* and bump the cache.
4. On restart: YAML reloads, Redis overrides reapply on top. State
   survives restarts because Redis survives restarts.
```

**Operator-of-last-resort path:** if the UI is broken or the Redis
state is wrong, an operator can `DEL bifrost:config:agent_budgets:*`
and the YAML baseline takes over. Editing `plugin.yaml` and
restarting does **not** clear Redis overrides — they layer on top.
To return to a pure-YAML state, the operator deletes the
`bifrost:config:*` keys explicitly.

The PUT/DELETE flow is intentionally simple — no validation against
running state, no "this lowers the cap below current spend" warning
(that's a UI concern, the API just sets the value). The UI warns;
the API trusts.

Listing this here rather than in phase 6 because phase 6 has been
operational on YAML-only for as long as the plugin has existed.
Adding HTTP mutability is a UI-driven need; the Redis key shapes
above belong logically in phase 6's schema and should be folded
back in when phase 8 lands.

## View inventory

10 routes; each one is one file under `src/pages/`.

### Dashboard (`/`)

The "what is this swarm doing right now" landing page.

- **Top KPIs (4 cards):** total spend today, total spend this month,
  in-flight runs count, killed agents count.
- **Headline chart:** stacked-area cost histogram over the last 24h
  bucketed by hour, grouped by `agent-name`. Hits
  `/_plugin/histogram/cost?window=24h&bucket=1h&dimension=agent-name`.
- **Two rankings tables:** top 5 agents by spend (last 24h), top 5
  users by spend (last 24h). Both clickable to their detail pages.

Refresh: 30s for the KPIs and rankings, 60s for the histogram.

### Runs (`/runs`)

List of recent and in-flight runs.

- **Filters:** time window, agent name, user, status (running /
  killed / done / errored).
- **In-flight indicator:** for each run with `current_spend > 0` and
  no terminal log entry, the row shows a pulsing badge. This requires
  a Redis blend the UI does client-side: list rows come from
  `/_plugin/spend/by-session` (logs-derived), then enrich the first
  N rows via parallel `/_plugin/runs/:id/state` calls (Redis-derived).
- **Bulk actions:** none in v1. (Killing N runs at once is a
  footgun.)

Refresh: 10s for the list, 2s for in-flight badges on the visible
rows.

### Run detail (`/runs/:id`)

The "stop this thing" page.

- **Live state panel:** current cost, current steps, time-to-exp,
  TTL, last activity timestamp, recent tool calls. Hits
  `/_plugin/runs/:id/state` every 2s while the run is in-flight,
  every 30s once terminal.
- **Cap meters:** cost vs `max_cost_usd`, steps vs `max_steps`,
  rendered as filled bars. Walks the macaroon chain and shows one
  meter per ancestor.
- **Call log:** table of every call in this run, from
  `/_plugin/runs/:id`. Paginated; default 50 most recent.
- **Kill button:** opens KillConfirmModal. On confirm:
  `POST /_plugin/runs/:id/kill`. After 200, the UI flips polling on
  `/state` to 500ms for 30s to surface "no new activity" evidence.
- **Unkill button:** shown only when the kill is in effect. Same
  confirmation friction as kill.

### Agents (`/agents`)

Observed agents (from logs.metadata) + configured budgets.

- **Table:** agent name, spend (selectable window), call count,
  configured cap (or "—" if none), current bucket spend, kill state.
- **Inline budget editor:** click the cap cell → popover with
  `cap_usd` and `window` inputs. `window` validated against the
  duration vocabulary from phase 6.
- **Kill column:** kill button per row with the same modal as the
  run detail page, but typed-confirmation required (operator must
  type the agent name) because the blast radius is the whole swarm.

Refresh: 30s for the table.

### Agent detail (`/agents/:name`)

- **Three histograms (uPlot):** cost, tokens, p50/p95/p99 latency.
  All bucketed by the page-level window/bucket controls. Hits
  `/_plugin/histogram/{cost,tokens,latency}?dimension=agent-name`
  filtered server-side via `MetadataFilters`.
- **Budget editor + current bucket state:** shows
  `current_spend / configured_cap` in the active window.
- **Recent runs under this agent:** small table, links to run
  detail.

Refresh: 60s for histograms, 10s for current bucket state.

### Users (`/users`)

Same shape as Agents but keyed on `user-id`. Less interactive
because there's nothing to kill or edit at the user level — quotas
are governed by Bifrost's customer caps, not the plugin.

### User detail (`/users/:id`)

- **Quota panel:** daily budget, spent today, remaining today,
  list of in-flight runs the user owns. Hits
  `/_plugin/users/:id/quota`.
- **Recent runs:** small table.
- **Per-user histograms:** cost over time.

### Sessions / Session detail (`/sessions`, `/sessions/:id`)

Thin views over `/_plugin/spend/by-session` and
`/_plugin/sessions/:id`. Lower-priority page; tables and a summary
card, no charts in v1.

### Config (`/config`)

YAML-equivalent editor.

- **`hard_ceiling` section:** two number inputs.
- **`tool_loop` section:** two number inputs.
- **`agent_budgets` section:** table of (name, cap_usd, window) with
  add/remove/edit.

All edits POST individually (not as one big "save" button) so
partial-failure states are obvious. Each row shows a "saving" /
"saved" / "failed" pip after PUT.

### Login (`/login`)

User + password form, submits to `/_plugin/login`. On success,
redirects to wherever the user was trying to go (preserved in a
`?next=` query param) or to `/`.

## Data-fetching contract

All API calls go through one `fetch` wrapper that:

- Prefixes `/_plugin`
- Sends `credentials: 'include'`
- Maps 401 → throw an `unauthorized` error → top-level handler
  redirects to `/login?next=<current>`
- Maps non-2xx → throw with the JSON `error.code` from phase 7's
  error envelope
- 5s default timeout (configurable per call)

Tanstack Query is configured with:

- `staleTime: 0` (always refetch on focus); aggressive freshness fits
  an ops console.
- `gcTime: 5 * 60_000` (5 min cache after unmount).
- `retry: 1` for GETs, `retry: false` for mutations.
- 401 errors short-circuit retries via the queryCache `onError`
  handler.

Per-query polling intervals are set at the hook level, not globally:

| Query | Refetch interval |
|---|---|
| Dashboard KPIs / rankings | 30s |
| Dashboard headline histogram | 60s |
| Runs list | 10s |
| In-flight run state (Run detail, visible runs) | 2s, 500ms after a kill |
| Agents list | 30s |
| Agent detail histograms | 60s |
| Agent current bucket state | 10s |
| Config | manual (refetch on mount only) |
| `/me` | once at app boot |

Polling pauses when the tab is hidden (`document.visibilityState`,
which Tanstack Query honors by default).

## Mutation patterns

State-changing operations are either **destructive** (kill, unkill)
or **edits** (budget, ceiling, tool-loop). Both go through Tanstack
Query mutations, but the UX is different.

### Destructive (kill / unkill)

1. User clicks Kill.
2. KillConfirmModal opens. For run kills: "Type 'kill' to confirm."
   For agent kills: "Type the agent name to confirm." (Higher friction
   for higher blast radius.)
3. On confirm, mutation fires. The button shows a spinner.
4. On 200: close modal, invalidate the relevant queries (run state,
   agents list), bump the run-state polling cadence to 500ms for 30s
   so the operator sees the kill taking effect.
5. On non-200: keep the modal open, show the error, leave the
   button live for retry.

**No optimistic updates for kill/unkill.** The whole point of the
operator clicking Kill is to see the kill actually happen; faking it
client-side undermines the feedback loop.

### Edits (budgets, ceilings, tool-loop)

1. User edits a field, blurs (or presses enter).
2. Mutation fires immediately. The field shows a "saving" pip.
3. On 200: pip flips to "saved", fades after 2s. Underlying query
   refetches.
4. On 4xx (validation error from the API): pip flips to "failed",
   error tooltip shows the code; field reverts to last known good
   value.

Optimistic updates **are** used for edits — the field shows the new
value while the request is in flight. If the request fails, the
revert is jarring but correct.

## Type generation (tygo)

Go is the source of truth for response shapes. Hand-writing TS
equivalents drifts within a release.

`tygo` is configured in `gateway/tygo.yaml` to emit TS for the
adminapi response types:

```yaml
packages:
  - path: "github.com/sphinx-chat/stakgraph/gateway/internal/adminapi"
    output_path: "internal/adminapi/ui/src/api/types.ts"
    type_mappings:
      time.Time: "string"
  - path: "github.com/sphinx-chat/stakgraph/gateway/internal/auth"
    output_path: "internal/adminapi/ui/src/api/types.ts"
    type_mappings:
      time.Time: "string"
```

`make tygo` runs the codegen; CI runs `make tygo && git diff --exit-code`
to catch drift.

Types that are not response shapes (internal-only structs) are
either kept out of the codegen by living in non-exported packages or
explicitly excluded in `tygo.yaml`.

The handlers in `adminapi/` declare named response types
(`type RunStateResponse struct { ... }`) rather than returning
ad-hoc `map[string]any`, so the codegen has something to bind to.
This is a small discipline tax with a large payoff — the UI never
sees a server response shape the compiler hasn't already approved.

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
on the next page load. Assets (JS, CSS) get the default file-server
caching, which is fine since their filenames are stable within a
build and changes to them ride along with `index.html`.

The UI handler is registered in `adminapi/server.go` *after* the
session middleware, so `/_plugin/ui/*` requires either a valid
session cookie or a valid bearer. The `/_plugin/ui/login` route is
served by the same handler (it's an SPA route, the SPA itself
handles the login form on the client). The middleware's `allowAnon`
list grants `/_plugin/ui/login` through; the SPA then makes its
authenticated calls after submission.

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
- [ ] `config.go`: `GET /_plugin/config` and the PUT/DELETE
      endpoints for `agent_budgets`, `hard_ceiling`, `tool_loop`.
      Persists overrides to Redis under `bifrost:config:*` (no new
      SQL table). Composes YAML baseline + Redis overrides into an
      in-memory effective config that PreLLMHook reads.
- [ ] `ui.go`: `//go:embed ui/dist` + SPA fallback handler.
- [ ] Route registration in `server.go`, with `/_plugin/ui/*`
      behind the session middleware and `/_plugin/health`,
      `/_plugin/login` in `allowAnon`.

**Backend — type codegen (`gateway/`):**

- [ ] `tygo.yaml` covering `internal/adminapi` and any response
      types in `internal/auth`.
- [ ] `make tygo` target invoking the codegen.
- [ ] CI step: `make tygo && git diff --exit-code internal/adminapi/ui/src/api/types.ts`.

**Frontend (`gateway/internal/adminapi/ui/`):**

- [ ] Vite + Preact + TS scaffold; `package.json`, `vite.config.ts`,
      `tsconfig.json`, `index.html`.
- [ ] `src/api/client.ts`: typed fetch wrapper, 401 redirect.
- [ ] `src/api/queries.ts` + `mutations.ts`: one hook per endpoint.
- [ ] `src/app.tsx`: wouter routes, QueryClient provider, global
      auth-error handler.
- [ ] `src/components/layout/Shell.tsx` + `Sidebar` + `Topbar`.
- [ ] `src/components/charts/UplotChart.tsx`: generic wrapper.
- [ ] `src/components/tables/DataTable.tsx`: sortable, paginated.
- [ ] `src/components/controls/*`: WindowPicker, BucketPicker,
      DimensionPicker, KillButton, BudgetEditor.
- [ ] `src/pages/Login.tsx`, `Dashboard.tsx`, `Runs.tsx`,
      `RunDetail.tsx`, `Agents.tsx`, `AgentDetail.tsx`,
      `Users.tsx`, `UserDetail.tsx`, `Sessions.tsx`,
      `SessionDetail.tsx`, `Config.tsx`, `NotFound.tsx`.
- [ ] `src/styles/base.css` + `components.css`.

**Dockerfile:**

- [ ] `ui-builder` stage running `npm ci && npm run build`.
- [ ] `COPY --from=ui-builder /ui/dist` into the Go build context
      before `go build`.

**Docs:**

- [ ] Update `gateway/README.md` with the operator-UI bullet (URL,
      default creds in dev).
- [ ] Update `phase-3-swarm-handoff.md` to note that admin creds now
      drive both `/api/*` and the operator UI.
- [ ] Forward-pointer from `llm-governance-v2.md` §"Plugin" to this
      phase.

**Gate:** phase 8 ships when:

1. Login → dashboard → kill a test run → see the run state freeze
   end-to-end against a real Bifrost+plugin+Redis stack.
2. Editing an `agent_budgets` cap via the UI takes effect on the
   next call's PreHook (verified by hitting the same agent twice
   with a tightened cap and seeing 402 the second time).
3. Tygo runs in CI and the generated `types.ts` is up to date.
4. The Docker image builds without a node runtime in the final
   image (UI is `//go:embed`'d).

## Phasing of the UI itself

Don't try to ship all 10 pages in one PR. Order:

1. **Slice 1 — Auth + scaffold.** Sessions store, login/logout/me
   endpoints, session middleware, Vite scaffold, Login page,
   placeholder Dashboard ("Hello, admin" + logout button), Dockerfile
   build stage, tygo wired up. Proves the embed and auth pipeline.
2. **Slice 2 — Read-only Dashboard.** Real KPIs, real rankings, real
   headline chart. Layout/Sidebar/Topbar landed here. No mutations.
3. **Slice 3 — Runs + kill.** Runs list, Run detail, kill switch
   with confirmation modal and post-kill polling cadence. The
   minimum lovable operator console.
4. **Slice 4 — Agents + budgets.** Agents list, Agent detail,
   budget editor. Config endpoints added.
5. **Slice 5 — Drill-downs.** Users, User detail, Sessions, Session
   detail.
6. **Slice 6 — Config page.** Hard ceiling and tool-loop editing.

Each slice is independently shippable. Slice 1 lands the
infrastructure; slices 2-6 are feature work on top.

## What this design buys

- **One binary, one URL, one auth model** — operators don't learn
  three different ports or three different credential schemes. The
  same `BIFROST_ADMIN_PASS` that gates `/api/*` gates the operator UI.
- **Embedded UI** — no separate frontend deployment, no CORS, no CDN
  to keep in sync with the Go binary. `docker pull` ships the UI.
- **Cookie sessions with Redis backing** — real logout, real admin
  kick, no JWT-rotation problem, ~0.2 ms per request cost.
- **`SameSite=Strict` cookie** lets phase 8 skip a CSRF-token layer
  without weakening the security posture.
- **Tygo-generated types** — server response shapes and UI types are
  guaranteed in sync; a Go field rename surfaces in CI before it
  surfaces in production.
- **No new SQL tables anywhere** — observability reads `logs.db`
  via Bifrost's `/api/logs` (no plugin SQL access); enforcement
  state and config overrides both live in Redis alongside phase 6
  hot state and phase 8 sessions. The plugin never opens a SQLite
  file directly.
- **Polling, not WebSockets, in v1** — every freshness requirement
  in the view inventory is met by polling at the cadences specified.
  A WebSocket subsystem can land later without rewriting the data
  layer; Tanstack Query swaps polling for push transparently.
- **Phaseable** — six slices, each shippable independently. Slice 1
  is one PR of plumbing; slices 2-6 are feature work.
- **No coupling to Hive** — the operator UI works on a fresh swarm
  the moment phase 3's bootstrap completes. Hive integration is
  additive (SSO is a follow-on phase).
