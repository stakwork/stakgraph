# Phase 1 — Hive VK Reconciler: API Calls

> Concrete request/response shapes for the Hive reconciler that
> provisions Bifrost Customers and Virtual Keys per (workspace × user)
> pair. Derived from reading the actual Bifrost handler source
> (`transports/bifrost-http/handlers/governance.go`).
> See `llm-governance-v2.md` for the why; this doc is the how.

## Setup assumptions

- Each workspace has its own Bifrost reachable at a URL Hive can
  resolve, e.g. `http://<workspace-swarm-host>:8181`.
- Bifrost's admin auth is configured with Basic credentials. Hive's
  reconciler authenticates with:
  ```
  Authorization: Basic base64(<admin_user>:<admin_password>)
  ```
  Per-workspace admin creds live in Hive's secret store alongside the
  Bifrost URL. (If AuthConfig is disabled in a given Bifrost — e.g.
  dev mode — the Authorization header is ignored, calls still work.)
- Hive's user IDs are stable strings. We use that string directly as
  both the Customer `name` and the VK `name`. **This is the invariant
  that makes the same user identifiable across all workspace Bifrosts
  without any cross-Bifrost coordination.**

## What the reconciler does, in one sentence

For a `(workspace_id, user_id)` pair: ensure the workspace's Bifrost
has exactly one Customer with `name=user_id` (with a $1000/day budget
+ 1000 RPM / 5M TPM rate limit + `is_active=true`) and exactly one VK
with `name=user_id` attached to that Customer (with permissive
provider configs). Store the resulting VK `value` in Hive's secret
store. Idempotent — running twice does nothing the second time.

## Reconciliation algorithm

```
reconcile(workspace_id, user_id):
    1. customer_id = ensure_customer(workspace_id, user_id)
    2. vk_value    = ensure_vk(workspace_id, user_id, customer_id)
    3. store_vk(workspace_id, user_id, vk_value)

ensure_customer(workspace_id, user_id):
    customers = LIST customers WHERE name=user_id    # via search
    if exactly one found and looks correct:
        return found.id
    if none found:
        create customer; return new.id
    if multiple or malformed:
        log error, repair or bail

ensure_vk(workspace_id, user_id, customer_id):
    vks = LIST vks WHERE name=user_id, customer_id=customer_id
    if exactly one found and looks correct:
        return found.value     # already created previously
    if none found:
        create vk linked to customer_id; return new.value
    if multiple or malformed:
        log error, repair or bail
```

The "looks correct" check is up to Hive's policy. For phase 1, "exists
with the right name and customer_id" is enough; budget and rate-limit
drift can be reconciled by a separate `PUT` if Hive's source-of-truth
config changes.

---

## 1. Find existing Customer (GET, search by name)

**Request:**

```http
GET /api/governance/customers?search=u_alice&limit=10 HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
```

**Response — `200 OK`:**

```json
{
  "customers": [
    {
      "id": "9f8e7d6c-…",
      "name": "u_alice",
      "budget_id": "b1234…",
      "rate_limit_id": "rl1234…",
      "budget": {
        "id": "b1234…",
        "max_limit": 1000.00,
        "reset_duration": "1d",
        "current_usage": 12.47,
        "last_reset": "2026-05-14T00:00:00Z"
      },
      "rate_limit": {
        "id": "rl1234…",
        "request_max_limit": 1000,
        "request_reset_duration": "1m",
        "token_max_limit": 5000000,
        "token_reset_duration": "1m"
      },
      "teams": [],
      "virtual_keys": [],
      "config_hash": "…",
      "created_at": "2026-05-13T18:22:00Z",
      "updated_at": "2026-05-13T18:22:00Z"
    }
  ],
  "count": 1,
  "total_count": 1,
  "limit": 10,
  "offset": 0
}
```

**Hive's logic:**

- Filter `customers` to those with exact `name == user_id` (the
  `search` parameter does substring matching, not exact match — line
  1880-1887 of governance.go reads `Search` and passes to
  `GetCustomersPaginated`).
- If exactly one match → use its `id` as `customer_id`. Done.
- If zero matches → proceed to **§2 (create Customer)**.
- If multiple exact matches → ops issue; log + alert. Pick the oldest
  by `created_at`. Don't try to dedupe automatically in phase 1.

**Possible non-200s:**
- `401` — admin auth failed; misconfigured creds
- `500` — `"failed to retrieve customers"` — bifrost-internal; retry
  with backoff

---

## 2. Create Customer (POST, if not found)

**Request:**

```http
POST /api/governance/customers HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
Content-Type: application/json

{
  "name": "u_alice",
  "budget": {
    "max_limit": 1000.00,
    "reset_duration": "1d"
  },
  "rate_limit": {
    "request_max_limit": 1000,
    "request_reset_duration": "1m",
    "token_max_limit": 5000000,
    "token_reset_duration": "1m"
  }
}
```

Field notes:
- `name` (required) — Hive's `user_id`.
- `budget.max_limit` — USD float; `1000.00` per v2 plan.
- `budget.reset_duration` — see "Valid reset_duration values" below.
- `rate_limit` — all four fields are `*int64` (nullable in Go); set all
  four for v1.
- Bifrost generates `id` server-side (UUID), so don't send one.
- `is_active` defaults to `true` if not specified; don't send unless
  you want `false`.

**Response — `200 OK`:**

```json
{
  "message": "Customer created successfully",
  "customer": {
    "id": "9f8e7d6c-…",                ← capture this; pass to VK create
    "name": "u_alice",
    "budget_id": "b1234…",
    "rate_limit_id": "rl1234…",
    "budget": { … },
    "rate_limit": { … },
    "teams": [],
    "virtual_keys": [],
    "config_hash": "…",
    "created_at": "2026-05-14T09:00:00Z",
    "updated_at": "2026-05-14T09:00:00Z"
  }
}
```

The handler calls `governanceManager.ReloadCustomer` after the
transaction commits and returns the reloaded object. The `id` field is
what the next step needs.

**Possible non-200s:**

| Status | When | Body |
|---|---|---|
| `400` | `name` missing | `{"error":"Customer name is required"}` |
| `400` | `rate_limit` validation failed | `{"error":"Invalid rate limit: <detail>"}` |
| `400` | Bad JSON | `{"error":"Invalid JSON"}` |
| `500` | DB or anything else | `{"error":"failed to create customer"}` |

**Important: no built-in unique-name check.** Looking at
`createCustomer` (lines 1920-1999): it validates `Name != ""` and
nothing else. If Hive races and `POST`s the same name twice, you get
two Customer rows with the same name (different UUIDs). The reconciler
must guard against this with its own concurrency control (mutex per
`(workspace_id, user_id)` pair) or by tolerating the dup at read time
(pick oldest).

---

## 3. Find existing VK (GET, filter by customer_id)

**Request:**

```http
GET /api/governance/virtual-keys?customer_id=9f8e7d6c-…&search=u_alice&limit=10 HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
```

Filters available on this endpoint (lines 369-380 of governance.go):
`limit`, `offset`, `search`, `customer_id`, `team_id`, `sort_by`,
`order`, `export`. `search` is substring matching across the VK name.

**Response — `200 OK`:**

```json
{
  "virtual_keys": [
    {
      "id": "vk-uuid-…",
      "name": "u_alice",
      "description": "",
      "value": "sk-bf-Xy7…long…",        ← this is what Hive needs to stash
      "is_active": true,
      "team_id": null,
      "customer_id": "9f8e7d6c-…",
      "rate_limit_id": null,
      "calendar_aligned": false,
      "provider_configs": [
        {
          "id": 1,
          "virtual_key_id": "vk-uuid-…",
          "provider": "anthropic",
          "weight": null,
          "allowed_models": ["*"],
          "allow_all_keys": true,
          "keys": [],
          "budgets": [],
          "rate_limit": null
        }
        // … openai, openrouter, gemini
      ],
      "mcp_configs": [],
      "budgets": [],
      "rate_limit": null,
      "team": null,
      "customer": { "id": "9f8e7d6c-…", "name": "u_alice", … },
      "created_at": "2026-05-14T09:00:01Z",
      "updated_at": "2026-05-14T09:00:01Z"
    }
  ],
  "count": 1,
  "total_count": 1,
  "limit": 10,
  "offset": 0
}
```

**Hive's logic:**

- Filter results to those with exact `name == user_id` AND
  `customer_id == <expected customer id>`.
- If exactly one match → capture `value`, done.
- If zero → proceed to **§4 (create VK)**.
- If multiple → same handling as duplicate customers: pick oldest by
  `created_at`, log + alert.

**The `value` field is the bearer token Hive injects as `BIFROST_VK`
at spawn time.** Save it in Hive's secret store keyed by
`(workspace_id, user_id)`.

---

## 4. Create VK (POST, if not found)

**Request:**

```http
POST /api/governance/virtual-keys HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
Content-Type: application/json

{
  "name": "u_alice",
  "description": "Hive user u_alice — auto-provisioned",
  "customer_id": "9f8e7d6c-…",
  "provider_configs": [
    { "provider": "anthropic",  "allowed_models": ["*"], "key_ids": ["*"] },
    { "provider": "openai",     "allowed_models": ["*"], "key_ids": ["*"] },
    { "provider": "openrouter", "allowed_models": ["*"], "key_ids": ["*"] },
    { "provider": "gemini",     "allowed_models": ["*"], "key_ids": ["*"] }
  ]
}
```

Field notes:
- `name` (required) — Hive's `user_id`. Unique per Bifrost (DB unique
  index `idx_virtual_key_name`).
- `customer_id` — from §1 or §2. Mutually exclusive with `team_id`
  (line 466-469); send only one or neither, never both.
- `provider_configs` — empty array means deny-by-default; we want
  permissive, so list every provider we'll use. `allowed_models: ["*"]`
  permits all models for that provider.
- `key_ids: ["*"]` — **required for inference to actually work.** Tells
  Bifrost to set `allow_all_keys: true` on the resulting provider_config
  so the VK can use every provider-level API key. If you omit this,
  Bifrost defaults to "no keys attached" and every inference call fails
  with `no keys found for provider: <p> and model: <m>`, even though
  the provider key clearly exists in `/api/providers/<p>/keys`. The
  field name is `key_ids` on the **request** — the response-side
  `keys` array is hydrated/read-only and unrelated. Source:
  `transports/bifrost-http/handlers/governance.go` (`KeyIDs
  schemas.WhiteList json:"key_ids"`).
- `budgets` (optional, omitted here) — Customer's $1000/day already
  governs alice's spend. If you ever want a *separate* VK-level cap
  (e.g. one VK $100/day on top of $1000/day customer cap), put it
  here. **Phase 1: omit. Single source of truth for alice's budget is
  her Customer.**
- `rate_limit` (optional, omitted here) — same reasoning as `budgets`.
- `is_active` — omit; defaults to `true`.
- `value` — DO NOT SET. Bifrost generates the `sk-bf-…` value
  server-side (line 509: `Value: governance.GenerateVirtualKey()`).

**Response — `200 OK`:**

```json
{
  "message": "Virtual key created successfully",
  "virtual_key": {
    "id": "vk-uuid-…",
    "name": "u_alice",
    "value": "sk-bf-Xy7…long…",        ← stash this
    "is_active": true,
    "team_id": null,
    "customer_id": "9f8e7d6c-…",
    "provider_configs": [ … ],
    "mcp_configs": [],
    "budgets": [],
    "rate_limit": null,
    "team": null,
    "customer": { … },
    "created_at": "2026-05-14T09:00:01Z",
    "updated_at": "2026-05-14T09:00:01Z"
  }
}
```

The `value` field on the response is the same string the agent will
later pass as `Authorization: Bearer <value>`. Hive stashes it
immediately — there's no second-fetch path that returns the cleartext
value once it's been stored (the `getVirtualKeys` endpoint returns the
full row including `value`, so technically re-fetchable, but treat the
create response as canonical).

**Possible non-200s:**

| Status | When | Body |
|---|---|---|
| `400` | `name` missing | `{"error":"Virtual key name is required"}` |
| `400` | Both `team_id` and `customer_id` set | `{"error":"VirtualKey cannot be attached to both Team and Customer"}` |
| `400` | Bad provider name in `provider_configs` | `{"error":"invalid provider name: <p>"}` |
| `400` | Bad reset_duration | `{"error":"Invalid reset duration format: <d>"}` |
| `400` | Negative budget | `{"error":"Budget max_limit cannot be negative: <n>"}` |
| `400` | Duplicate `reset_duration` in budgets | `{"error":"Duplicate reset_duration in budgets: <d>"}` |
| `400` | Bad JSON | `{"error":"Invalid JSON"}` |
| `400` | DB-level dup `name` | `{"error":"Failed to ...: ... duplicate key ..."}` (treat as success: read back via §3) |
| `500` | Anything else | `{"error":"<detail>"}` |

---

## 5. (Optional) Update Customer — for budget/rate-limit drift repair

If Hive's source-of-truth budget for users changes (say, from $1000/day
to $500/day) and the reconciler wants to bring an existing Customer in
line:

**Request:**

```http
PUT /api/governance/customers/9f8e7d6c-… HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
Content-Type: application/json

{
  "budget": {
    "max_limit": 500.00,
    "reset_duration": "1d"
  }
}
```

Response: `200 OK` with the updated customer object (same shape as §1).

**Use sparingly.** Resetting `max_limit` does not change `current_usage`
— if alice has already spent $700 today, lowering the cap to $500
takes effect for the *next* request (which 402s). That's the correct
behavior but worth knowing.

For phase 1, recommend **skip this entirely**. Drift repair is a
later concern.

---

## 6. (Offboarding) Disable Customer

When alice is offboarded, Hive iterates over every workspace she had
access to and calls:

**Request:**

```http
PUT /api/governance/customers/9f8e7d6c-… HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
Content-Type: application/json

{
  "is_active": false
}
```

Wait — checking the handler.

```
UpdateCustomerRequest fields: Name, Budget, RateLimit (no is_active)
```

<!-- VERIFY: Customer doesn't have is_active per current schema -->

Let me cross-check: looking at `TableCustomer` (customer.go:5-24), it
has `Name`, `BudgetID`, `RateLimitID`, `ConfigHash`, timestamps —
**no `is_active` field**. The "disable Customer" path actually
disables at the VK level. So for offboarding, the API is:

```http
PUT /api/governance/virtual-keys/vk-uuid-… HTTP/1.1
Host: <workspace-swarm-host>:8181
Authorization: Basic <base64>
Content-Type: application/json

{
  "is_active": false
}
```

Response: `200 OK` with the updated VK. After this, any inference call
using that VK's bearer value returns 401 from Bifrost. To restore, set
`is_active: true`. To permanently delete, use `DELETE
/api/governance/virtual-keys/{vk_id}`.

**Phase 1: offboarding is out of scope** — note for future, not
required to ship. The mechanism is identified; deferred.

---

## Valid `reset_duration` values

From `configstoreTables.ParseDuration` (referenced at line 478 of
governance.go):

- `"30s"` — 30 seconds
- `"5m"` — 5 minutes
- `"1h"` — 1 hour
- `"1d"` — 1 day (our default for daily budgets)
- `"1w"` — 1 week
- `"1M"` — 1 month

Numeric prefix can vary (`"10m"`, `"2h"`, etc.). Case-sensitive: `"1M"`
is months, `"1m"` is minutes — keep them straight.

For phase 1, use:
- Customer budget: `"1d"` ($1000/day)
- Customer rate_limit request: `"1m"` (1000 RPM)
- Customer rate_limit token: `"1m"` (5M TPM)

---

## Concurrency safety

The reconciler can be called concurrently for the same `(workspace_id,
user_id)` from multiple Hive workers (e.g. workspace-create racing
with a user-grant-access). Bifrost has **no transactional
"create-if-not-exists" semantics** on the governance endpoints.

Two safe approaches:

**(a) Per-pair mutex in Hive.** Reconciler acquires a Redis or DB
advisory lock keyed by `(workspace_id, user_id)` before the
GET → maybe-create → GET → maybe-create sequence. Releases at end.
Simple, no Bifrost-side change.

**(b) Tolerate duplicates at read time.** Accept that races can create
two Customer rows or two VK rows. Both reads filter to "exact name
match, pick oldest." Hive emits a daily reconcile job that detects
duplicates and `DELETE`s the younger ones. Cheaper to write, eventual
consistency on cleanup.

Recommend **(a)** for phase 1 — it's a few lines and avoids ever
shipping two VKs to two agents that think they have alice's
credentials.

---

## Reconciler return contract

The reconciler is the only thing that talks to Bifrost's governance
API. Everywhere else in Hive that needs a VK does a key/value lookup
in Hive's local store, never calls Bifrost.

```go
type ReconcileResult struct {
    WorkspaceID string
    UserID      string
    CustomerID  string  // Bifrost's UUID
    VKID        string  // Bifrost's UUID
    VKValue     string  // "sk-bf-…", the bearer token
    Created     bool    // true if anything was created this call (audit signal)
}

func Reconcile(ctx context.Context, workspaceID, userID string) (*ReconcileResult, error)
```

Stash `VKValue` keyed by `(WorkspaceID, UserID)` in Hive's secret
store. The other fields are useful for audit/logging but Hive doesn't
need to look them up at spawn time.

---

## Trigger model: lazy-only for phase 1

The reconciler runs **only when the VK is actually needed**, not
proactively. Specifically, it's called from the agent-spawn / chat-LLM
code path that already does a `secret_store.get(workspace_id, user_id)`
lookup:

```
on agent spawn / chat LLM call for (workspace W, user U):
    vk = secret_store.get(W, U)
    if vk is None:
        # First-use for this pair — reconcile against W's Bifrost
        result, err = Reconcile(ctx, W, U)
        if err != nil:
            log.Error("VK provisioning failed: %v", err)
            return user_facing_error("LLM unavailable in this workspace right now")
        secret_store.put(W, U, result.VKValue)
        vk = result.VKValue
    proceed with LLM call using vk
```

**Why lazy-only for phase 1:**

- One trigger point in code; nothing to wire into workspace-create or
  grant-access handlers.
- If a workspace's Bifrost isn't deployed yet, nothing tries to
  reconcile against it. No log spam, no error queue. The first attempt
  happens when a user actually uses LLM in that workspace.
- Failure is immediately visible to the user, which makes ops issues
  obvious instead of buried in a background-job log.
- Bifrost outages don't block workspace-create or grant-access flows;
  they only block first-use of LLM in a workspace whose Bifrost is
  unreachable — which is the *correct* behavior, since LLM is
  unavailable in that workspace anyway.
- Subsequent LLM calls hit the cached VK in the secret store; no
  Bifrost round-trip after the first success.

**Concurrency:** if multiple agent processes are spawned for the same
`(W, U)` at the very first use, the per-pair mutex (see "Concurrency
safety" above) ensures only the first caller actually creates the
Customer + VK; the rest wait, then read the cached value.

**Failure handling:** if Bifrost is unreachable (connection refused,
timeout, 5xx), the reconciler returns an error. Hive logs it (loudly,
so ops sees it) and returns "LLM unavailable in this workspace right
now" to the user. **No retry queue, no background sweep, no
pending-reconciliation table in phase 1.** The next user attempt is
itself the retry. If a workspace's Bifrost stays down, every LLM call
in that workspace fails — which is correct: LLM *is* unavailable in
that workspace until ops fixes it.

**Out of scope for phase 1, planned for phase 2:**

- Eager reconciliation on workspace-create (for owner)
- Eager reconciliation on user-grant-access (for newly-granted user)
- Background sweep job (every N minutes; idempotent walk)
- Retry queue for transient Bifrost outages
- Drift repair (e.g. budget changed in Hive config; sync to existing
  Customers)
- Offboarding fan-out (`PUT virtual-keys/<id> {is_active: false}`
  across all workspaces a user had access to)

When phase 2 adds these, the lazy-only path stays as the safety net —
nothing about its design changes.

---

## Wire-up checklist for phase 1

- [ ] Hive has a registry: `workspace_id → (bifrost_url, admin_basic_creds)`
- [ ] Hive's secret store schema accepts `(workspace_id, user_id) → vk_value`
- [ ] `Reconcile()` function implemented per algorithm above
- [ ] **Per-pair mutex** wrapping `Reconcile()` calls, keyed by
  `(workspace_id, user_id)` — prevents racing dup creation on
  first-use
- [ ] Pilot agent's spawn path / chat-LLM path performs the
  lazy lookup-or-reconcile-then-cache sequence above
- [ ] Pilot agent's LLM client attaches the dim headers:
  `x-bf-dim-{run-id, workspace-id, agent-name, session-id, deployment}`
- [ ] Pilot agent's LLM client uses
  `Authorization: Bearer <vk_value>` (or the SDK's provider-specific
  equivalent — `x-api-key` for Anthropic, etc.)
- [ ] Verification SQL queries (from `llm-governance-v2.md` §"Observability")
  return non-empty results within ~1h of pilot deploy
- [ ] `enforce_auth_on_inference` stays `false` throughout phase 1

---

## Reference: file locations in Bifrost source

| Concern | File |
|---|---|
| Route registration | `transports/bifrost-http/handlers/governance.go:283-338` |
| Customer create handler | `transports/bifrost-http/handlers/governance.go:1919-1999` |
| Customer list handler | `transports/bifrost-http/handlers/governance.go:1857-1917` |
| VK create handler | `transports/bifrost-http/handlers/governance.go:453-693` |
| VK list handler | `transports/bifrost-http/handlers/governance.go:342-451` |
| Customer table schema | `framework/configstore/tables/customer.go` |
| VK table schema | `framework/configstore/tables/virtualkey.go:198-280` |
| Team table schema (not used in phase 1) | `framework/configstore/tables/team.go` |
| Admin auth middleware | `transports/bifrost-http/handlers/middlewares.go:800-1000` |
| VK→Customer resolver (how `customer_id` lands on log rows) | `plugins/governance/resolver.go:237-260` |

If anything in this doc seems off, check those files first — Bifrost
is evolving and the source is authoritative.
