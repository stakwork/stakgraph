# LLM Governance v2

> Supersedes `llm-governance.md`. Same goal, cleaner organizing
> principle. Read this one.
>
> v2 is the **architectural overview** of the governance stack —
> three-layer identity, Bifrost-per-workspace, VK provisioning,
> rollout. The concrete specs for each cryptographic and operational
> piece live in companion docs and supersede v2's pseudocode where
> they overlap:
>
> | Concern | Authoritative spec |
> |---|---|
> | Three-principal identity model, key custody, trust registration | [`cryptographic-identity.md`](./cryptographic-identity.md) |
> | Macaroon wire format, signing inputs, HMAC chain, verifier algorithm | [`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md) |
> | Trust registry storage, admin API, env-var seed | [`phases/phase-5-trust-registry.md`](./phases/phase-5-trust-registry.md) |
> | Plugin Redis schema, per-hook ops, TTL policy, failure modes | [`phases/phase-6-plugin-enforcement.md`](./phases/phase-6-plugin-enforcement.md) |
> | Agent defaults / registry | [`agent-registry.md`](./agent-registry.md) |
> | Bifrost Customer/VK reconciliation, duration vocabulary | [`phases/phase-1-reconciler.md`](./phases/phase-1-reconciler.md) |
>
> **Deltas already applied to this doc:**
>
> - The symmetric HMAC root key model is replaced by the asymmetric
>   trust chain (org → user → invocation). The plugin holds no
>   signing key.
> - "daily root macaroon" is deleted; each invocation gets its own
>   freshly-signed root.
> - "auth service" was renamed to "Hive macaroon issuer" and
>   rescoped. Its endpoint surface is `/macaroons/issue` and
>   `/macaroons/revoke` (cryptographic-identity §"Issuer endpoints").
>   v2's earlier `/macaroons/login` + `/macaroons/attenuate` are
>   gone.
> - The plugin's Redis schema, per-hook ops, and failure modes
>   originally sketched here as pseudocode now live in phase 6 with
>   concrete bucket-key formats, a TTL policy that works for
>   long-lived sessions, and the `kill:agent:<name>` per-agent kill
>   switch.
>
> Everything not listed above — Bifrost-per-workspace deployment,
> VK provisioning, dim header propagation, the rollout schedule —
> stands as written below.

## The organizing principle

**Every LLM call in this system traces to a specific human, end-to-end,
through any sub-agent chain.** Organizations are groups of people; agents
are tools those people invoke. Every dollar spent has a name attached.

That principle drives every design decision below. When something gets
optimized for, it's "make per-human attribution and per-human enforcement
the cheapest, most natural thing in the system." Everything else —
per-agent budgets, per-workspace dashboards, sub-agent containment — is
layered on top of that primary axis, not in competition with it.

## Three layers, three timescales

```
┌─────────────────────────────────────────────────────────────────┐
│  IDENTITY  ── Bifrost Customer = Hive user_id                   │
│   "alice exists, has a $1000/day org-wide backstop, account     │
│    is active." Indexed `customer_id` column on every log row.   │
│   Set: once at user provisioning.                               │
│   Carried as: VK (sk-bf-…) in the standard auth header.         │
│   Enforced by: Bifrost natively (budget + rate limit + active). │
├─────────────────────────────────────────────────────────────────┤
│  AUTHORIZATION  ── Macaroon chain                               │
│   "this specific run, in this workspace, with these agents,     │
│    for at most $5, expiring in 10 minutes, was authorized       │
│    by alice — and every sub-agent below it inherits from        │
│    those bounds, narrowed further at each step."                │
│   Set: per spawn / per sub-agent attenuation (very frequent).   │
│   Carried as: x-macaroon HTTP header.                           │
│   Enforced by: gateway plugin (HMAC chain + caveats + Redis     │
│   ancestor walk).                                               │
├─────────────────────────────────────────────────────────────────┤
│  HOT STATE  ── Plugin + Redis                                   │
│   Per-run accumulated cost & steps, tool-call history,          │
│   kill-switch flags, revocation list.                           │
│   Set: every LLM call (PostLLMHook accounting).                 │
│   Enforced by: gateway plugin against Redis state.              │
└─────────────────────────────────────────────────────────────────┘
```

Customer is slow-moving identity. Macaroon is fast-moving per-invocation
scope. Plugin/Redis is the real-time enforcement substrate.

Customer says **alice exists and has $1000/day.**
Macaroon says **this run in alice's tree is authorized for $5, expires
in 10 minutes, and any sub-agent inherits from those bounds.**
Plugin says **the current spend is $4.87, the kill switch hasn't been
flipped, the chain validates — pass.**

Each layer has one job. None can do the others' jobs. Together they
answer who / how much / how deep / kill-yes-or-no on every LLM call.

---

## Where Bifrost lives

**One Bifrost per workspace, inside the workspace's swarm.** Each
workspace has its own `logs.db`, its own Customer/VK store, its own
plugin process.

Hive (the central control plane / frontend / chat / sandbox spawner /
workflow triggerer) talks to many workspaces' Bifrosts. **Hive itself
does not run a Bifrost** — when Hive needs to make an LLM call (e.g.
serving alice's chat UI), it routes through the relevant workspace's
Bifrost.

**Org-level Hive chats** that span multiple workspaces route through
the org's **primary workspace** Bifrost. Primary workspace is a new
concept that needs to land in Hive's data model — every org designates
one. (See "Open questions.")

---

## Bifrost configuration

```
Bifrost Customer  = Hive user_id        (one per human; service identities → workspace owner)
Bifrost Team      = unused for v1       (reserved; see "Future" below)
Bifrost VK        = one per (workspace × user)
                     name  = user_id
                     value = sk-bf-… (generated by Bifrost)
                     customer_id = the user's Customer in this Bifrost
                     team_id     = nil
```

Bifrost natively enforces:

| Field | Where | v1 value |
|---|---|---|
| Daily $ budget | Customer (per user) | $1000/day |
| Rate limit RPM | Customer (per user) | 1000 RPM |
| Rate limit TPM | Customer (per user) | 5M TPM |
| Provider allowlist | VK | `[anthropic, openai, openrouter, gemini]`, all `["*"]` |
| Model allowlist | VK | `["*"]` initially |
| `is_active` | Customer (per user) | `true` by default; `false` disables account org-wide |

These are intentionally generous backstops. The real per-run governance
happens in the macaroon. Customer enforcement exists to catch
"someone's credentials are compromised and burning money at 3am" — not
to be the primary spending policy.

---

## Hive as credential broker

Hive holds the credential store: `vks[workspace_id][user_id] = "sk-bf-…"`.

### User lifecycle

**User creation (Hive issues a new user_id):**
No Bifrost work yet. The user has no LLM access until they're granted
to a workspace.

**Workspace creation (or user-granted-access-to-workspace):**
Hive calls the target workspace's Bifrost:
```
POST /api/governance/customers
  { name: <user_id>,
    budget:     { max_limit: 1000.00, reset_duration: "1d" },
    rate_limit: { request_max_limit: 1000, request_reset_duration: "1m",
                  token_max_limit: 5000000, token_reset_duration: "1m" } }

POST /api/governance/virtual-keys
  { name: <user_id>,
    customer_id: <returned customer id>,
    provider_configs: [
      { provider: "anthropic", allowed_models: ["*"] },
      { provider: "openai",    allowed_models: ["*"] },
      { provider: "openrouter",allowed_models: ["*"] },
      { provider: "gemini",    allowed_models: ["*"] },
    ]
  }
```
Stash the returned VK `value` in Hive's secret store keyed by
`(workspace_id, user_id)`.

This is **idempotent reconciliation** — Hive's reconciler walks
(workspaces × users) and ensures every pair has a Customer+VK. Run on
workspace creation, on user-grant-access, and as a background sweep to
catch any drift.

**User offboarding (revoked everywhere):**
```
for each workspace W the user had access to:
    PUT /api/governance/customers/<user_id> { is_active: false }   (on W's Bifrost)
    SET bifrost:revoke_user_before:<user_id> = <iso8601 now>       (in W's Redis)
```
The first call stops all *new* LLM activity from that user's VK in W.
The second call kills any in-flight macaroon chains rooted at that
user, at the next LLM call boundary. Both required: VK disable catches
attempts to start new work; macaroon revocation catches work that's
already running.

### Spawn / chat / workflow time

Every place Hive starts an LLM-using thing, the lookup is the same:

```
1. Determine the principal:
     - If a specific human triggered this → use their user_id
     - Otherwise (cron, webhook, scheduled workflow, autonomous agent)
       → use the workspace owner's user_id  ← default principal rule
2. Look up vks[workspace_id][principal]
3. Inject into the spawned env (or set on the HTTP client):
     LLM_GATEWAY_URL = http://<workspace-swarm>:8181
     BIFROST_VK      = <looked-up VK>
     # plus dim headers for run / agent / category / session
     # plus x-macaroon (once macaroons land)
```

**The default principal rule is total.** Every LLM call in the system
has a Customer; no nullable user, no `system:unknown` fallback. If an
agent isn't directly tied to a human, it's tied to the workspace owner,
and the workspace owner is the one whose budget pays for it.

---

## The wire protocol

Every LLM call arrives at Bifrost with:

```
# Credentials
Authorization: Bearer sk-bf-<user-VK>           ← Bifrost stamps customer_id = user_id on log
   or x-api-key / x-goog-api-key as the SDK requires
x-macaroon: <invocation macaroon>               ← per-run cryptographic scope (step 8+)

# Attribution dimensions → logs.metadata JSON (automatically by Bifrost)
x-bf-dim-workspace-id:  w1                      ← which workspace
x-bf-dim-run-id:        <uuid>                  ← this invocation
x-bf-dim-session-id:    <persistent thread id>  ← long-lived conversation grouping
x-bf-dim-agent-name:    coder                   ← which specific agent
x-bf-dim-category:      coding                  ← which category (optional)
x-bf-dim-deployment:    sandbox-goose           ← where this call physically ran
                                                   (hive-chat / swarm-mcp / sandbox-goose /
                                                    workflow-node / repo-agent / …)
```

`workspace-id` is technically redundant with "which Bifrost instance"
(one per workspace), but stamping it explicitly makes cross-Bifrost
aggregated queries trivial.

What ends up indexed in `logs` natively, free, with no plugin code:

| Column | Source | The question it answers cheaply |
|---|---|---|
| `customer_id` | Bifrost (from VK) = user_id | **"How much did alice spend?"** — primary axis |
| `virtual_key_id` / `virtual_key_name` | Bifrost (from VK) = user_id | Same as above; redundant index |
| `model`, `provider` | Bifrost | "Cost by model" |
| `latency` | Bifrost | p50/p95 |
| `cost` | Bifrost pricing manager | Per-call $ |
| `created_at` | Bifrost | Time-series |

What lives in `metadata` JSON (acceptable scan cost on small axes):

| Field | Used for |
|---|---|
| `workspace-id` | Per-workspace rollups (when aggregating across Bifrosts) |
| `agent-name` | Per-agent rollups; cardinality ~40, scan cheap |
| `category` | Per-category rollups; cardinality ~5–8, scan trivial |
| `run-id` | Drill into one run's calls |
| `session-id` | Group all calls in a chat thread |
| `deployment` | "Where did this call run" — useful for ops, not cost attribution |

---

## Observability: SQL today, custom endpoints when needed

The headline win: **the v0 boilerplate plugin gets you full per-user
attribution today.** No plugin code required for any of these queries:

```sql
-- Top spenders, this week
SELECT customer_id, SUM(cost) AS spend
FROM logs WHERE created_at >= datetime('now','-7 days')
GROUP BY customer_id ORDER BY spend DESC LIMIT 10;

-- Which agent has alice been using
SELECT json_extract(metadata,'$.agent-name') AS agent, SUM(cost), COUNT(*)
FROM logs WHERE customer_id = 'u_alice'
  AND created_at >= datetime('now','-24 hours')
GROUP BY agent ORDER BY 2 DESC;

-- Top spending agents across the org, last 24h
SELECT json_extract(metadata,'$.agent-name') AS agent, SUM(cost) AS spend
FROM logs WHERE created_at >= datetime('now','-24 hours')
GROUP BY agent ORDER BY spend DESC LIMIT 10;

-- All calls in one run (drill-down)
SELECT * FROM logs
WHERE json_extract(metadata,'$.run-id') = 'r_01H...'
ORDER BY created_at;
```

### Plugin HTTP endpoints (when SQL isn't enough)

Bifrost plugins can claim arbitrary URL prefixes by short-circuiting
`HTTPTransportPreHook`: if `req.Path` starts with `/api/stakgraph/`,
return an `*HTTPResponse` directly and Bifrost-core never sees the
request. This is how the plugin exposes purpose-built read endpoints
that mix `logs.db` SQL with real-time Redis state.

v1 endpoints (small set; grow as Hive's UI asks):

```
GET /api/stakgraph/spend/by-user?window=24h
GET /api/stakgraph/spend/by-agent?window=24h
GET /api/stakgraph/spend/by-workspace?window=24h
GET /api/stakgraph/runs/:run_id/state          ← live: Redis hot state
GET /api/stakgraph/users/:user_id/quota        ← Bifrost customer cap + live spend
```

The historical query path is SQL-only. Redis is queried only for
real-time / in-flight numbers (current run spend before the LLM call
completes, kill-switch state, active-run enumeration).

### Cross-Bifrost aggregation

Per-workspace queries answer themselves — each Bifrost's `logs.db` is
scoped to one workspace. For org-wide reporting across workspaces:

1. **Short-term:** scheduled job pulls each instance's daily slice into
   a central store (Postgres or DuckDB on a daily file dump). Hive
   serves dashboards from the central store.
2. **Long-term:** swap Bifrost's logging-store backend to a shared
   Postgres or ClickHouse — Bifrost's logging plugin supports pluggable
   stores (`framework/logstore/`). Config change, not code change.

---

## Macaroons: per-run scope and sub-agent containment

The Customer layer answers "is this alice and is alice allowed to spend
money today." The macaroon answers a **finer-grained** question that
Bifrost cannot answer: **"is this *particular invocation* in alice's
tree authorized — for how much, by which agent, until when, narrowing
how across sub-agents?"**

### Invocation macaroon

Issued by Hive's `/macaroons/issue` endpoint each time an agent is
launched (or a session begins, or a workflow starts — "invocation" is
whatever scope the caller asked the issuer to sign for; see
[`cryptographic-identity.md`](./cryptographic-identity.md) §"Per
invocation, not per day"). Held in the agent process for the run
lifetime. Carries caveats the user authorized:

```
user_id          = u_alice            ← from user_authorization, cross-checked against Customer
workspace        = w1                 ← must be in user_authorization.permissions.workspaces
agents           = [coder]            ← must be ⊆ user_authorization.permissions.agents
run_id           = r_01H...
max_cost_usd     = 5.00               ← THE per-run budget Bifrost cannot enforce
max_steps        = 100
exp              = now + 10m          ← absolute, computed from defaults + iat at issuance
nonce            = <random>
user_sig         = <Ed25519 over the above>
```

The full wire format including the org-signed `user_authorization`
envelope that wraps the invocation lives in
[`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md).

**Why redundant `user_id`?** It's the cross-check. The plugin's
`PreLLMHook` asserts `macaroon.user_id == ctx.customer_id`. A leaked
VK alone is useless (no macaroon → 401). A leaked macaroon alone is
useless (no matching VK → wrong Customer → assertion fails). Both must
come from the same principal.

### Sub-agent macaroon (the prettiest part of the design)

When a parent agent decides to spawn a sub-agent, the parent attenuates
its own macaroon **locally** — appending one HMAC link over the
previous signature. No signing keys involved, no network roundtrip,
no issuer call:

```
attenuations[i].caveats = {
  agents:        [coder, web-search],          ← grows; must include parent's agents
  max_cost_usd:  min(2.00, parent.remaining()), ← shrinks
  max_steps:     min(40,   parent.remaining_steps()),
  exp:           min(now + 120s, parent.exp),  ← shrinks
  run_id:        <child's run_id>,
  nonce:         <new random>,
}
attenuations[i].hmac = HMAC-SHA256(parent_sig_bytes, JCS(caveats))
```

The macaroon protocol mathematically guarantees a child cannot widen
any caveat — the HMAC chain breaks if you try. **Sub-agent
containment is physics, not policy.** Per-language attenuator
implementations need only JCS + HMAC-SHA256 + hex; full spec in
[`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md)
§"The attenuations chain".

### Why attenuation is the right primitive

Three properties, free, that no other layer can provide:

1. **The child cannot widen the parent.** Macaroon attenuation is a
   one-way valve — every new caveat narrows. Mathematically. There's no
   policy file to misconfigure, no exception to handle. The HMAC chain
   breaks if you try to remove or weaken a caveat.

2. **The child's choice of *its* sub-agent's budget is the child's call.**
   `coder` decides how to apportion its remaining $5 across however many
   sub-agents it spawns. Maybe it gives web-search $2 and review-changes
   $2 and keeps $1 for itself. Maybe it gives one sub-agent everything
   because it knows the call is expensive. **The parent decides the
   split, not the platform.** That's correct — only the parent knows
   what work it's about to delegate.

3. **The platform's job is just enforcement.** The plugin walks the chain
   on every LLM call from any descendant and accumulates spend into
   `bifrost:cost:run:<run_id>` for the leaf **and every ancestor**. If
   the parent's $5 cap is hit by combined parent+children spend, every
   descendant 402s on its next call, regardless of whether the
   descendant's own narrower cap has been hit. Belt and suspenders.
   Nothing escapes.

### The sub-agent budget split policy lives in the parent's code

The plan recommends a default convention (child gets at most
`parent.remaining * 0.5`, keep at least `parent.cap * 0.25` in the
parent), but doesn't enforce it centrally. Different agents will have
different sensible splits, and that's fine. **The platform guarantees
the *bound*; the parent picks the *distribution*.**

### What gets killed at every layer

| Action | Mechanism | Latency to effect |
|---|---|---|
| Disable alice org-wide | `PUT /customers/u_alice {is_active:false}` on every workspace's Bifrost | Next LLM call (Bifrost rejects at auth) |
| Kill alice's in-flight chains | `bifrost:revoke_user_before:u_alice = now` in every workspace's Redis | Next LLM call (plugin checks) |
| Kill one specific run | `bifrost:kill:<run_id>` in Redis | Next LLM call from any agent in chain |
| Revoke one macaroon chain | `bifrost:revoke:<nonce>` in Redis (with TTL = macaroon.exp) | Next LLM call in chain |
| Hit run budget naturally | Plugin reads `bifrost:cost:run:<run_id>` ≥ macaroon caveat | Next LLM call |
| Hit org-wide daily budget | Bifrost reads Customer budget | Immediate (at Bifrost auth) |

---

## Gateway plugin (Go, in-process with Bifrost)

> **Operational spec moved.** The Redis schema, per-hook ops, TTL
> policy, configuration block, and failure-mode contract for the
> plugin live in [`phases/phase-6-plugin-enforcement.md`](./phases/phase-6-plugin-enforcement.md).
> What this section sketched in pseudocode (Redis key shapes,
> PreLLMHook / PostLLMHook flow, per-agent budgets) was always
> intended as the contract; phase 6 pins it down with concrete
> bucket-key formats, a TTL policy that works for long-lived
> invocations, the `kill:agent:<name>` per-agent kill switch, and
> the explicit atomicity / failure-mode decisions v2 left implicit.
>
> Read phase 6 for the implementation contract. The summary below
> is the architectural framing only.

A Bifrost custom plugin implementing the standard hook surface. Single
binary, deployed alongside Bifrost in each workspace's swarm. Backed by
Redis for hot enforcement state (not for analytics — Bifrost's built-in
logging plugin handles per-call analytics into `logs.db`).

The plugin must be registered **before** Bifrost's built-in logging
plugin so dimensions stamped by macaroon canonicalization land in
`logs.metadata`.

### What the plugin does, at a glance

- **`HTTPTransportPreHook`** dispatches plugin-owned routes
  (`/api/stakgraph/*`) and otherwise lets the inference pipeline
  continue.
- **`PreLLMHook`** verifies the macaroon (via `gateway/auth/go`,
  per phase 4), checks revocations / kills / cost caps in Redis,
  canonicalizes log dimensions from caveats, and rejects with
  appropriate 401/402 errors when any check fails.
- **`PostLLMHook`** writes cost / step / tool accumulators back to
  Redis, walking the ancestor chain so every level of a sub-agent
  tree is charged for descendant spend.

The dimension-canonicalization rule (phase 6's PreHook step 6) is the
piece that makes per-user / per-agent / per-run attribution
trustworthy in `logs.db`:

```
dims["user-id"]      = macaroon.user_id
dims["workspace-id"] = macaroon.workspace
dims["run-id"]       = macaroon.run_id
dims["agent-name"]   = macaroon.agents[last]   ← most specific
```

Caller-supplied `x-bf-dim-*` values that disagree with the macaroon
are silently replaced. Anything the caller sent that wasn't in the
macaroon (e.g. `session-id`, `deployment`) is preserved as-is —
those are observability dimensions only and are not signature-bound.

### Failure modes (architectural posture)

| Failure | Behavior |
|---|---|
| Redis down | **Fail closed** for macaroon checks (revoke / kill / cap). **Fail open** for accounting (log loudly; spend goes uncounted; alerting fires). Auth-correctness wins over availability. |
| Hive macaroon issuer down | Existing invocation macaroons keep working until they expire. No new spawns possible. |
| Plugin panic | Fail-closed: reject the request. |

Concrete error codes, retry behavior, drift bounds, and crash-mid-call
recovery semantics are in phase 6's "Failure modes" section.

---

## Hive macaroon issuer

> **Endpoint spec moved.** The issuer endpoint surface
> (`/macaroons/issue`, `/macaroons/revoke`) is defined in
> [`cryptographic-identity.md`](./cryptographic-identity.md)
> §"Issuer endpoints". v2's earlier `/macaroons/login` +
> `/macaroons/attenuate` design is deleted (there is no daily root
> macaroon to log in against, and each invocation gets a freshly-
> signed root).

The Hive macaroon issuer is the Hive subsystem that signs macaroons
on behalf of an org (and, in custodial phase 1, on behalf of a
user). It's the only component besides user devices and org
key-holders that produces signatures — the plugin holds **no**
signing keys and is a pure verifier.

In phase 1 (custodial), the issuer signs both layers itself with
keys in Hive's secret store. In phase 2+ (user keys move to
Yubikey / mobile enclave), the issuer signs the `user_authorization`
layer with the org's key and orchestrates a signing request to the
user's device for each invocation's `user_sig`. In phase 3+ (org
keys move to multisig), the issuer orchestrates signing requests to
all org-key-holders. The wire format and verifier path are the
same across phases — only who holds which key changes.

Spawners (Hive's chat backend, MCP, workflow engine) call
`/macaroons/issue` when starting an agent. Parent agents
attenuating macaroons for their sub-agents do **not** call this
endpoint — they attenuate locally with their own parent macaroon's
signature, because that's all that's needed (the HMAC chain is
keyless, see [`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md)
§"The attenuations chain").

---

## How a request flows

End-to-end. Alice clicks "run coder on workspace w1" in the chat UI.

```
1. FRONTEND
   - Sends request to Hive's chat BFF with session cookie.

2. HIVE CHAT BFF (spawner)
   a. Looks up alice's session (Hive's own session token; not a macaroon).
   b. POST /macaroons/issue to the Hive macaroon issuer:
        { org_id:    "org_acme",
          user_id:   "u_alice",
          workspace: "w1",
          agent:     "coder",
          run_id:    "r_01H..",
          override?: { ... optional caller narrowing ... } }
      Issuer resolves agent defaults from the agent registry,
      applies any override as min() narrowing, signs the
      user_authorization (org key) and invocation (user key)
      layers, and returns a complete wire-format macaroon.
   c. Receives invocation macaroon M_inv.
   d. Looks up Hive's secret store:
        BIFROST_VK = vks[w1][u_alice]   = "sk-bf-…"
   e. Spawns coder agent process with:
        LLM_GATEWAY_URL = http://w1-swarm-bifrost:8181
        BIFROST_VK      = sk-bf-…
        AGENT_MACAROON  = M_inv
        RUN_ID          = r_01H..
        WORKSPACE_ID    = w1
        USER_ID         = u_alice
        AGENT_NAME      = coder

3. CODER AGENT (process)
   a. Constructs LLM client with:
        baseURL = $LLM_GATEWAY_URL/anthropic/v1
        apiKey  = $BIFROST_VK
        headers = {
          "x-macaroon":            $AGENT_MACAROON,
          "x-bf-dim-run-id":       $RUN_ID,
          "x-bf-dim-workspace-id": $WORKSPACE_ID,
          "x-bf-dim-agent-name":   $AGENT_NAME,
          "x-bf-dim-session-id":   <persistent chat id>,
          "x-bf-dim-deployment":   "sandbox-coder",
        }
   b. Runs agent loop. Each model call ships with all the above.

4. BIFROST TRANSPORT (in w1's swarm)
   a. Authenticates VK. Customer = u_alice attached to ctx.
   b. Extracts x-bf-dim-* into BifrostContextKeyDimensions.
   c. Invokes our plugin's HTTPTransportPreHook.

5. OUR PLUGIN PreHook
   - Path is /anthropic/v1/messages (not a stakgraph route) → continue.
   - Verifies macaroon: org ECDSA sig over user_authorization,
     user Ed25519 sig over invocation, HMAC chain over any
     attenuations. (Per phase 4.)
   - Asserts macaroon.user_id == customer_id (== u_alice). Match.
   - Canonicalizes the dim map from caveats.
   - bifrost:cost:run:r_01H.. currently $3.21, cap is $5.00. Passes.
   - bifrost:steps:run:r_01H.. currently 14, cap is 100. Passes.
   - No kill, no loop, no revocation.
   - Returns pass.

6. BIFROST CORE
   - Strips the VK, substitutes the real Anthropic key.
   - Forwards to api.anthropic.com.

7. ANTHROPIC
   - Returns model response.

8. BIFROST → our plugin's PostLLMHook:
   - Cost = $0.42.
   - HINCRBYFLOAT bifrost:cost:run:r_01H..  total 0.42 → now $3.63
   - HINCRBY      bifrost:steps:run:r_01H.. total 1     → now 15
   - LPUSH        bifrost:tools:run:r_01H.. "<tool>"

9. BIFROST → built-in logging plugin:
   - Writes a logs row with cost=0.42, latency, tokens, model,
     customer_id=u_alice (indexed), virtual_key_name=u_alice,
     metadata={"run-id":"r_01H..","workspace-id":"w1",
               "agent-name":"coder","session-id":"...",
               "deployment":"sandbox-coder"}.

10. RESPONSE to agent → next step in the loop.
```

**When the run runs away.** Accumulated cost crosses $5.00 mid-loop.
Next PreHook returns 402 `run_cost_exceeded`. AI SDK throws. Agent
loop terminates. Spawner notified by existing error path.

**When someone clicks "stop this agent."** Hive UI writes
`SET bifrost:kill:r_01H.. 1 EX 3600` to w1's Redis. Next PreHook returns
402 `run_killed`. Same termination path.

**When the agent spawns a sub-agent.**
```python
# inside the coder agent process
child_macaroon = attenuate(self.macaroon, {
    "agents":       self.agents + ["web-search"],
    "run_id":       new_run_id(),
    "max_cost_usd": min(2.00, self.remaining_budget()),
    "max_steps":    min(40, self.remaining_steps()),
    "exp":          min(now + 120s, self.macaroon.exp),
    "nonce":        random_nonce(),
})
spawn_subagent(env={
    "AGENT_MACAROON": child_macaroon,
    "AGENT_NAME":     "web-search",
    "RUN_ID":         new_run_id,
    # everything else inherited
})
```
Plugin walks the chain on every sub-agent LLM call: spend counts
against both `bifrost:cost:run:<child>` and `bifrost:cost:run:<parent>`.
If parent's $5 cap is exhausted by combined spend, the child is killed
even though its own $2 cap hasn't been hit.

---

## Caller integration

The work for each caller:

1. **Receive `BIFROST_VK` and `LLM_GATEWAY_URL`** from the spawner (env
   var, RPC arg, or workflow context). Caller does not pick its own VK.
2. **Receive `AGENT_MACAROON`** (env var, RPC arg, workflow payload).
3. **Attach headers** to the LLM client at construction:
   - `Authorization: Bearer $BIFROST_VK` (or appropriate SDK header)
   - `x-macaroon: $AGENT_MACAROON`
   - `x-bf-dim-*` for run/agent/workspace/session/deployment
4. **If you spawn sub-agents:** attenuate your macaroon *locally* before
   passing to the child process.

### TypeScript

```ts
const headers = {
  "x-macaroon":           process.env.AGENT_MACAROON,
  "x-bf-dim-run-id":       process.env.RUN_ID,
  "x-bf-dim-session-id":   sessionId,
  "x-bf-dim-agent-name":   process.env.AGENT_NAME,
  "x-bf-dim-workspace-id": process.env.WORKSPACE_ID,
  "x-bf-dim-deployment":   "sandbox-coder",
};

const anthropic = createAnthropic({
  apiKey:  process.env.BIFROST_VK,
  baseURL: process.env.LLM_GATEWAY_URL + "/anthropic/v1",
  headers,
});
```

### Python, Goose, Rust, etc.

Same pattern. Every modern LLM SDK supports default headers in one
line. Goose reads from its yaml/toml config; set `base_url`, the API
key env var to `BIFROST_VK`, and the headers map.

### Workflow engine (Stakwork)

Hive's workflow-trigger code injects `BIFROST_VK`, `LLM_GATEWAY_URL`,
`AGENT_MACAROON`, and dim headers into the workflow's runtime context.
The workflow runtime applies them as default headers on its LLM nodes.

---

## Spawner responsibilities

Each entity that spawns an agent is responsible for:

1. **Resolving the principal.** If a user triggered this, use their
   user_id. Otherwise, use the workspace owner's user_id. **No
   nullable principal, ever.**

2. **Looking up the VK.** `vks[workspace_id][principal]` from Hive's
   secret store.

3. **Calling `/macaroons/issue`** with
   `{org_id, user_id, workspace, agent, run_id, override?}`. The
   issuer reads agent defaults (max_cost_usd, max_steps,
   defaultMaxWallclockS) from the agent registry, applies any
   `override` as min() narrowing, computes a concrete `exp` from
   `iat + defaultMaxWallclockS`, and signs both layers. The spawner
   does not compute defaults itself — the registry is the source of
   truth and lives on the issuer.

4. **Spawning the process** with `BIFROST_VK`, `LLM_GATEWAY_URL`, the
   returned macaroon, and the dim values as env / args.

Service identities (cron, webhooks, scheduled workflows) are treated
exactly like users via the default-principal rule. There's no
`system:cron` user; cron's spawns are attributed to the workspace
owner whose schedule triggered them. Clean and uniform.

---

## Threat model

| Credential | Sensitivity | Where it lives | Blast radius if leaked |
|---|---|---|---|
| Org root key | **Crown jewel** (phase 1: custodial in Hive; phase 3: multisig) | Hive secret store or org-key-holders' devices | Authorize any user under that org. Custodial in phase 1; offline / multisig in phase 3. See [`cryptographic-identity.md`](./cryptographic-identity.md) §"Phase staging". |
| User key (Ed25519) | Medium (phase 1: custodial; phase 2: user device) | Hive secret store or user's Yubikey / Passkey / mobile enclave | Sign invocations as that user, within whatever permissions the org authorized. |
| Invocation signature | Low | Agent process, run lifetime | One invocation's worth of work, bounded by its caveats. |
| Sub-agent macaroon | Very low | Sub-process | Narrower scope, smaller cap, shorter exp. |
| User VK (`sk-bf-…`) | Low (without macaroon: useless) | Hive secret store + spawned agent env | After step 8: useless alone. Pre-step-8: that user's daily budget in that one workspace. |
| Bifrost admin key | High | Ops only | Reconfigure VKs, change Customer budgets. |
| Plugin Redis | Medium | Trusted swarm network | Manipulate run counters → bypass per-run caps; cannot forge macaroons (the plugin holds no signing keys). |

**The two-credential design.** A VK alone is useless after step 8 (the
plugin requires a matching macaroon). A macaroon alone is useless (no
VK → Bifrost auth rejects). Both must come from the same user, and the
plugin cross-checks via the `user_id == customer_id` assertion.

**No single platform key is a system-wide crown jewel.** A plugin
compromise affects accounting state in that one swarm; it cannot
forge identity. A Hive compromise affects custodial keys until phase
2/3 migrate them to user devices and org multisig. The
cryptographic-identity model is designed so that the highest-value
key material lives in the fewest places, can migrate off the platform
entirely over time, and is never replicated across workspaces.

---

## Rollout

Phased. Each step is independently verifiable. Through steps 1–5,
`enforce_auth_on_inference` stays `false` and missing macaroons are
warned-not-rejected. The system tightens monotonically.

**Step 1: Stand up per-workspace Bifrost.** Existing direct provider
calls keep working. Verify Bifrost is reachable and forwards correctly
to upstream providers using its own configured keys.

**Step 2: Route MCP through Bifrost.** Set `LLM_GATEWAY_URL` in MCP's
deploy. No VK yet. Pass-through mode. Verify requests show up in
Bifrost logs.

**Step 3: Hive provisioning reconciler.** Hive's reconciler walks
(workspaces × users) and ensures every pair has a Customer (with
$1000/day budget + rate limits) and a VK in that workspace's Bifrost.
Idempotent. Run on workspace-create, user-grant-access, and as a
background sweep. Hive's secret store now holds VK values. Stop
hardcoding VKs in deploy env vars; Hive injects per-spawn.

**Step 4: Observability via dimensions (no plugin code).** Spawners
ship `x-bf-dim-{run-id, workspace-id, agent-name, session-id,
deployment}` on every LLM call. Bifrost auto-stamps `customer_id` from
the VK. Verify the example SQL queries return useful org-level
dashboards. **No plugin code change needed for this step** — the v0
boilerplate is enough. The "spend per user" dashboard is now live.

**Step 5: Plugin v1 (HTTP endpoint dispatcher + per-run state).**
- Add the `/api/stakgraph/*` short-circuit in `HTTPTransportPreHook`.
- Ship 1–2 endpoints to prove the pattern: `spend/by-agent`,
  `runs/:id/state`.
- Add Redis.
- Enforce per-run cost cap and step cap from plugin config (hardcoded
  defaults per agent, keyed by `x-bf-dim-agent-name`). Macaroon
  verification is still a no-op.
- Tool-loop detection.
- `kill:<run-id>` switch (writable from Hive UI via direct Redis or via
  a new `/api/stakgraph/runs/:id/kill` endpoint).
- Watch for a week of real data; tune.

**Step 6: Build the Hive macaroon issuer.** Implement
`/macaroons/issue` and `/macaroons/revoke` per
[`cryptographic-identity.md`](./cryptographic-identity.md) §"Issuer
endpoints" using the wire format from
[`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md).
Phase 1 ships custodial signing (org + user keys in Hive's secret
store). Spawners call `/macaroons/issue` per spawn and stamp
`x-macaroon` on outgoing LLM calls. Plugin **logs but does not
enforce** macaroon presence — observability mode. Watch coverage
climb as callers participate.

**Step 6.5: Stand up the trust registry.** Per
[`phases/phase-5-trust-registry.md`](./phases/phase-5-trust-registry.md),
seed each workspace's plugin with the org's pubkey (or multisig
policy) and the admin API for ongoing trust changes. The plugin
needs this in place before it can verify anything.

**Step 7: Plugin enforcement.** Implement the adapter at
`gateway/internal/auth/` per
[`phases/phase-6-plugin-enforcement.md`](./phases/phase-6-plugin-enforcement.md):
- verifies macaroons via `gateway/auth/go`
  ([`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md));
- asserts `claims.user_id == ctx.customer_id`;
- canonicalizes log dims from the verified claims;
- enforces per-run / per-ancestor / per-agent budgets from Redis;
- enforces kill switches and revocations;
- **Hard reject** on missing or invalid macaroon.

Cutover from "observed" to "enforced." Per-workspace and per-user
daily caps remain native Bifrost (Customer). Per-agent budgets stay
unenforced (config block empty) — turn on selectively per agent as
needed.

**Step 8: Roll out remaining callers.** Goose, Python apps, workflow
engine, Rust callers, repo-agent, test-agent. Each picks up
`LLM_GATEWAY_URL`, `BIFROST_VK`, dim header propagation, and macaroon
receipt-and-forward in turn. Until a caller participates, its requests
401 at the plugin (post-step-7) — migration forcing function.

**Step 9: Flip `enforce_auth_on_inference: true`.** Bifrost itself now
also rejects calls without a VK. Belt-and-suspenders with the plugin's
macaroon check.

**Step 10: Observe and tune.** First month of real data drives:
- per-agent default budget tuning
- alerting thresholds for unusual user spend
- whether to turn on per-agent budgets in plugin config
- whether to fork Bifrost's logging-store backend to Postgres for
  cross-workspace aggregation

---

## Risks and mitigations

**Workflow engine custom headers.** If Stakwork (or the workflow
engine variant in play) doesn't natively support custom headers on its
LLM nodes, that's the schedule risk in the whole plan. Mitigation: a
thin per-engine sidecar that injects headers in front of Bifrost.
Doesn't block the rest of the rollout.

**Hive macaroon issuer is in the critical path.** Every agent spawn
calls `/macaroons/issue`. If the issuer is down, spawns break.
Mitigation: stateless service, multiple replicas, health-gated load
balancer. Plugin and Bifrost don't depend on it being up; only
spawning does.

**Org-key compromise (custodial phase 1).** During phase 1, Hive
holds the org root key and a leak would let an attacker authorize
any user under that org. Mitigation: harden Hive's secret store as a
high-value target; plan the phase-3 migration to multisig early
(adding additional signers is non-breaking — the wire format already
supports `multisig-v1` per
[`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md)).
A phase-3 org has no single key that can mint user authorizations
alone.

**Hive's secret store as a single point of failure.** Hive holds every
user's VK for every workspace. If compromised, attacker has all of
those bearer credentials — but each VK is only usable against one
workspace's Bifrost, and post-step-7 each VK is useless without a
matching macaroon. Still: harden Hive's secret store as a high-value
target.

**Sub-agent budget fragmentation.** A parent that attenuates too
eagerly may leave its own cap underused while children hit theirs.
Mitigation: parent's local accounting + plugin's chain enforcement
means *combined* spend is bounded by parent cap; how it's distributed
is the parent's design call.

**Redis as accounting SPOF.** Mitigation: Redis Sentinel / cluster,
daily backups, fail-closed posture on auth checks.

**Cross-Bifrost user offboarding fan-out.** Disabling alice requires
calling every workspace's Bifrost. If one workspace's Bifrost is
unreachable, the disable is partial. Mitigation: Hive's offboarding
job retries with exponential backoff; emits an alert if any workspace
is still pending after N minutes; also stamps
`bifrost:revoke_user_before:<user_id>` in every workspace's Redis
(which is a different failure surface).

---

## Open questions

1. **Org root key storage (custodial phase 1).** Plain env var for
   v1, secrets service for v2. Driven by infra readiness. Migration
   to user-held / multisig keys is phases 2-3 per
   [`cryptographic-identity.md`](./cryptographic-identity.md).

2. **Per-workspace agent defaults location.** **Resolved by
   [`agent-registry.md`](./agent-registry.md):** defaults live in the
   Hive `AgentDefinition` table, org-scoped, owned by the Hive
   macaroon issuer. Spawners pass an agent name to `/macaroons/issue`
   and the issuer reads defaults from the registry.

3. **Sub-agent budget split policy.** Convention: child gets at most
   `parent.remaining * 0.5`; keep at least `parent.cap * 0.25` in
   parent. Document; don't enforce centrally.

4. **"Primary workspace" for org-level Hive chat.** New concept Hive
   needs to add. Every org designates one workspace as primary; Hive's
   org-level chats route through that workspace's Bifrost using the
   triggering user's VK. **Out of scope of this doc, but a Hive-side
   prerequisite for org-level chat features.**

5. **Macaroon library choice.** **Resolved:** in-repo at
   [`gateway/auth/go/`](../auth/go/) (pure verifier + signer) and
   [`gateway/auth/ts/`](../auth/ts/) (published as `@stakwork/macaroon`).
   Wire format and verifier algorithm defined in
   [`phases/phase-4-macaroon-shape.md`](./phases/phase-4-macaroon-shape.md);
   cross-language byte-equivalence enforced by fixtures.

6. **Cross-Bifrost user-id consistency.** Each workspace's Bifrost has
   its own Customer table. We use `user_id` as both Customer ID and
   VK name in every workspace, so the same string represents alice
   everywhere. Hive enforces this — Bifrost doesn't know. Document the
   invariant so it's not accidentally broken.

7. **Per-agent budgets.** **Resolved by
   [`phases/phase-6-plugin-enforcement.md`](./phases/phase-6-plugin-enforcement.md).**
   Plugin supports `agent_budgets.<name>.{cap_usd, window}` with
   Bifrost's full duration vocabulary (`1d` / `1w` / `1M` / `1Y` /
   sub-day rolling). v1 ships with no agents configured.

8. **Session refresh semantics.** When an active session's invocation
   macaroon is near its `exp`, Hive's chat BFF silently re-mints
   via `/macaroons/issue` and continues. Log grouping by
   `x-bf-dim-session-id` keeps the analytics view intact across the
   refresh.

9. **Future xpub layer.** If non-repudiation or cross-org federation
   ever becomes a hard requirement beyond what `multisig-v1` already
   supports, add per-request secp256k1 signing on top of the macaroon
   layer. The plugin's verifier is already structured to admit
   additional checks alongside `auth.Verify`.

---

## What this design buys

- **Every dollar links to a human, end-to-end.** `customer_id = user_id`
  is indexed on every log row. "How much did alice spend" is the
  cheapest query in the system, available day one with no plugin code.

- **Sub-agent containment is mathematics, not policy.** The macaroon
  protocol guarantees a child can only narrow what its parent passed.
  A buggy or compromised sub-agent literally cannot spend more than
  its parent was authorized to spend.

- **Parent agents decide their children's budgets, locally.** No
  central policy server is in the loop on sub-agent spawn. The
  platform enforces the *bound*; the parent picks the *distribution*.

- **Two-credential safety.** A leaked VK without a matching macaroon
  is useless. A leaked macaroon without the matching VK is useless.
  No single platform-held key is a system-wide crown jewel: the org
  root key migrates to multisig (phase 3), user keys migrate to
  user devices (phase 2), and the plugin holds no signing keys at
  any phase.

- **Native per-user budget and rate limit.** Bifrost enforces alice's
  $1000/day backstop and her RPM/TPM caps in-process, before any
  plugin code runs. Account disable is one API call.

- **Default principal rule makes accounting total.** Cron, webhooks,
  scheduled workflows, autonomous agents — every LLM call has a
  Customer, attributed to the workspace owner if no human is in the
  loop. No nullable users, no `system:unknown` bucket.

- **Mid-loop kill, sub-second.** A "stop this agent" button anywhere
  in Hive's UI works via one Redis key. Works across every caller
  language uniformly.

- **Plugin owns its own HTTP endpoints.** Custom governance dashboards
  (top spenders, in-flight runs, per-agent live spend) live in the
  same process as enforcement. SQL for historical, Redis for
  real-time, one endpoint surface.

- **Reversible, additive rollout.** Every step is an upgrade, not a
  cutover. Bifrost without VKs works. VKs without macaroons work.
  Macaroons without per-agent budgets work. Each layer turns on when
  it's ready.

---

## Summary table

| Capability | Mechanism |
|---|---|
| Single gateway per workspace | Bifrost in each swarm; `LLM_GATEWAY_URL` per workspace |
| Per-user identity (primary axis) | Bifrost Customer = `user_id`; indexed `customer_id` column |
| Per-user daily $ backstop | Bifrost Customer budget ($1000/day) |
| Per-user rate limit | Bifrost Customer rate limit (1000 RPM, 5M TPM) |
| Per-user account disable | `PUT /customers/<u> { is_active: false }` (Hive fans out across workspaces) |
| Per-workspace attribution | One Bifrost per workspace + `x-bf-dim-workspace-id` for cross-instance aggregation |
| Per-agent attribution | `x-bf-dim-agent-name` → `logs.metadata` |
| Per-category attribution | `x-bf-dim-category` → `logs.metadata` |
| **Per-run $ budget (issuer-set)** | **Macaroon `max_cost_usd` caveat** |
| **Per-run step budget** | **Macaroon `max_steps` caveat** |
| **Per-run lifetime** | **Macaroon `exp` (absolute timestamp, stamped at issuance from agent-registry default)** |
| **Sub-agent budget bounded by parent** | **Local HMAC attenuation + plugin chain enforcement** |
| Per-agent budget (windowed) | Plugin + Redis `cost:agent:<n>:<bucket>` with Bifrost duration vocabulary (`1d` / `1w` / `1M` / `1Y` / sub-day rolling) per phase 6 |
| Per-agent kill switch | Plugin checks `bifrost:kill:agent:<name>` in Redis per phase 6 |
| Per-workspace daily spend alerts | Query `logs.db` on schedule; alert on threshold |
| Tool-loop detection | Plugin Redis tool history + threshold |
| Mid-loop kill switch | Plugin checks `bifrost:kill:<run-id>` in Redis |
| User-revocation kill | Plugin checks `bifrost:revoke_user_before:<user-id>` |
| Macaroon-chain revocation | Plugin checks `bifrost:revoke:<nonce>` for any ancestor |
| Cryptographic proof of who | Macaroon `user_id` caveat, HMAC-signed, cross-checked against `customer_id` |
| Custom governance endpoints | Plugin claims `/api/stakgraph/*` via `HTTPTransportPreHook` short-circuit |
| Default principal for unattributed agents | Workspace owner's `user_id` |
| Caller obligation | Set `BIFROST_VK` + `LLM_GATEWAY_URL` from spawner; ship `x-bf-dim-*` + `x-macaroon` |
| Spawner obligation | Resolve principal → lookup VK → call `/macaroons/issue` (identity doc) → inject env into spawn |
