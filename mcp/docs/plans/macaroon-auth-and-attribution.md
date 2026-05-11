# Macaroon-Based Auth + Per-Run Attribution

**Status:** Exploratory. Supersedes the identity layer of
[`llm-gateway-and-runaway-enforcement.md`](./llm-gateway-and-runaway-enforcement.md)
if adopted. Per-run plugin + Bifrost gateway design carries over unchanged.

## Why revisit

The original plan put workspace in the VK tree:

```
Customer = workspace
  Team   = environment
    VK   = agent
```

Two problems surfaced:

1. **Cross-workspace agents don't fit.** The org-chat agent reads from many
   workspaces. There is no single Customer it belongs under.
2. **No human in the chain.** A VK identifies an agent, not the person who
   triggered the run. We can't answer "which user authorized this LLM
   call" from anything Bifrost sees — VKs are shared across all users of
   an org.

We want every agent run tied cryptographically to the human who really
triggered it, with the user's actual permissions (which workspaces, which
agents, which pods) enforced at the gateway — not just in app code that
the agent process is free to bypass once it has a VK.

## Deployment shape: one Bifrost per org

The original plan assumed a single Bifrost serving all orgs, with the
Customer tier representing workspace. We now expect to run **a separate
Bifrost (and plugin, and Redis) per org**, for isolation, blast-radius
control, and per-org config.

That collapses the VK hierarchy:

- **Customer = org** → but there's only one org per Bifrost, so this tier
  is essentially fixed. Useful only if we ever consolidate.
- **Team = env** → prod / staging.
- **VK = agent** → one per agent type per env.

8 agents × 2 envs = **16 VKs per Bifrost.** Workspace isn't anywhere in
the Bifrost tree, which is fine — workspace lives in macaroon caveats
where it actually belongs.

## The security boundary

**Macaroons are the primary enforcement layer. VKs are routing/identity
tags.**

A leaked VK alone is **inert**. To call the gateway you need both:

1. A valid `sk-bf-<agent>` VK (proves the request comes from a known agent
   process — Bifrost handles this).
2. A valid, unexpired, unrevoked macaroon whose caveats authorize this
   specific call (the plugin handles this).

The plugin additionally asserts the two match: a `sk-bf-browser` VK can
only be paired with a macaroon whose `agent` caveat is `browser`. So
stealing the browser VK doesn't let you run org-chat, and stealing one
agent's invocation macaroon doesn't let you use a different agent's
budget.

### Credential sensitivity table

| Credential | Sensitivity | Where it lives | Blast radius if leaked |
|---|---|---|---|
| Macaroon root signing key | **Crown jewel** | Issuer service + plugin only | Full compromise — mint any macaroon |
| User's root macaroon (24h) | High | User browser/CLI session | All user's workspaces for up to 24h |
| Invocation macaroon (10m) | Low | Agent process for run lifetime | One agent, one workspace, ≤10 min |
| VK (`sk-bf-…`) | **Low** (inert alone) | Agent process env var | At worst: DOS against agent's RPM cap |
| Bifrost admin API key | High | Ops only | Reconfigure VKs, change budgets |

Compare to the original plan where the VK was the entire security
boundary. Here, VKs can leak without spend or data exposure, because
without a valid macaroon the plugin rejects the request before it ever
reaches an upstream provider.

### What happens with bad credentials

| Scenario | Plugin response |
|---|---|
| Valid VK, no macaroon | **401** `macaroon_required` |
| Valid VK, malformed macaroon | **401** `macaroon_invalid` |
| Valid VK, expired macaroon | **401** `macaroon_expired` |
| Valid VK, revoked macaroon | **401** `macaroon_revoked` |
| Valid VK, valid macaroon, agent mismatch (e.g. browser VK + org-chat macaroon) | **403** `agent_mismatch` |
| Valid VK, valid macaroon, header doesn't match caveat (e.g. `x-workspace-id=w2` but caveat says `[w1]`) | **403** `scope_violation` |
| No VK | **401** by Bifrost (before plugin runs) |

**No fail-open.** The original plan suggested fail-open if a caller
forgot to send `x-run-id`. That stance does not carry over for macaroons.
If macaroon enforcement is the primary auth layer, missing macaroon must
be a hard reject from day one of enforcement.

## The shape of the answer: macaroons

Macaroons are bearer tokens like JWTs, but with **attenuation**: a holder
can mint a strictly weaker token from a stronger one *without contacting
the issuer*. Verification is local — check the HMAC chain plus each
caveat.

Mapped to our world:

- Alice logs in → issuer mints **root macaroon**:
  `user=alice, workspaces=[w1,w2,w3], agents=[*], pods=[*], exp=24h`
- Alice triggers the `org-chat` agent → app backend attenuates:
  `+ agent=org-chat, + run=<uuid>, + exp=10m`
  (workspaces unchanged — org-chat legitimately needs all three)
- Alice triggers `browser` on w1 → app backend attenuates:
  `+ agent=browser, + workspaces=[w1], + run=<uuid>, + exp=10m`
- If `browser` spawns a sub-agent, it can **only** attenuate further
  (drop a workspace, never add one). The token mathematically cannot
  grow — adding caveats is unilateral, removing them isn't.

## The three layers

```
┌──────────────────────────────────────────────────────────┐
│  Macaroon layer  ── PRIMARY ENFORCEMENT                  │
│   answers: who (user), scope (workspaces, agents, pods)  │
│   lifetime: short (minutes for invocation, hours for     │
│             root)                                        │
│   verified by: gateway plugin (local HMAC + caveats)     │
├──────────────────────────────────────────────────────────┤
│  VK layer (Bifrost)  ── ROUTING / ORG-LEVEL GOVERNANCE   │
│   answers: which agent, org-wide $ caps, RPM, model      │
│            allowlist                                     │
│   lifetime: persistent (one per agent × env)             │
│   verified by: Bifrost natively                          │
├──────────────────────────────────────────────────────────┤
│  Run layer (plugin + Redis)  ── RUNAWAY KILL             │
│   answers: this invocation — cost cap, step cap, kill    │
│   lifetime: ephemeral                                    │
│   verified by: gateway plugin (Redis state)              │
└──────────────────────────────────────────────────────────┘
```

Each layer owns one axis. Macaroons answer *who and what scope*. VKs
answer *which agent and how much can it spend org-wide*. The run layer
answers *is this specific invocation off the rails*.

### Why keep VKs at all?

Even though macaroons are the security boundary, VKs still earn their
keep:

- **Org-wide visibility.** Bifrost's native analytics show cost per VK =
  cost per agent across the org. Useful for "what is `build` costing us
  this month?"
- **Org-wide rate limiting.** A runaway agent that somehow bypasses run
  caps still bumps into Bifrost's VK RPM/TPM ceiling.
- **Model allowlists.** Want to make sure the `browser` agent never
  accidentally calls Opus? VK `allowed_models` handles that natively, no
  plugin code needed.
- **Routing.** The VK is what tells Bifrost which upstream provider keys
  to use.

We don't lose anything by keeping VKs. They're free; we just stop
treating them as the auth layer.

## VK scope, revised

**VK = (agent, env).** Workspace is not in the VK tree.

```
vk-browser-prod        vk-browser-staging
vk-build-prod          vk-build-staging
vk-org-chat-prod       vk-org-chat-staging
vk-explore-prod        vk-explore-staging
vk-test-prod           vk-test-staging
vk-learn-prod          vk-learn-staging
vk-dream-prod          vk-dream-staging
vk-repair-prod         vk-repair-staging
```

16 VKs total per Bifrost instance per org. Reconcile script provisions
them once; rarely touched after.

The VK answers org-wide questions:

- "Has `browser` blown past its $200/day org-wide budget?"
- "Is `org-chat` exceeding 30 RPM across all callers?"
- "Is this agent allowed to use Claude Opus at all?"

It does **not** answer "is this workspace allowed to spend $X" or "is
alice allowed to use this agent." Those move to the plugin via macaroon
caveats and per-workspace counters.

## On every request

```
Authorization: Bearer sk-bf-<agent-vk>     ← agent identity (Bifrost auth)
x-macaroon: <signed token>                 ← user + scope (plugin auth)
x-run-id: <uuid>                           ← per-invocation
x-session-id: <persistent conversation>
x-agent-name: browser                      ← redundant w/ macaroon caveat
x-workspace-id: w1                         ← redundant w/ macaroon caveat
x-user-id: alice                           ← redundant w/ macaroon caveat
```

The redundant headers let the plugin stamp cost-attribution keys without
parsing the macaroon body on every write. The macaroon is the source of
truth; the plugin asserts the headers match its caveats during PreHook.
Tampering with a header without re-signing the macaroon is detected.

## Plugin responsibilities

### PreHook (auth gate)

```
1. VERIFY MACAROON
   - parse x-macaroon
   - check HMAC chain against root signing key
   - if invalid             → 401 macaroon_invalid
   - if missing             → 401 macaroon_required

2. CHECK MACAROON CAVEATS
   - if exp passed                          → 401 macaroon_expired
   - if revoke:<nonce> exists in Redis      → 401 macaroon_revoked

3. CHECK VK ↔ MACAROON BINDING
   - extract agent from VK metadata (Bifrost provides this)
   - if vk.agent != macaroon.agent          → 403 agent_mismatch

4. CHECK REQUEST HEADERS MATCH CAVEATS
   - x-workspace-id ∈ macaroon.workspaces?  → else 403 scope_violation
   - x-agent-name == macaroon.agent?        → else 403 scope_violation
   - x-user-id == macaroon.user?            → else 403 scope_violation

5. PER-RUN ENFORCEMENT
   - kill:<run-id> set?                     → 402 run_killed
   - cost > max_cost_for(agent)?            → 402 run_cost_exceeded
   - steps > max_steps_for(agent)?          → 402 run_step_exceeded
   - tool-loop detected?                    → 402 tool_loop_detected

6. PER-WORKSPACE DAILY CAP (replaces VK-per-workspace)
   - cost:workspace:<wid>:<yyyy-mm-dd> > cap_for(wid)?
                                            → 402 workspace_budget_exceeded
```

### PostHook (accounting)

```
1. PER-RUN STATE (unchanged from original plan)
   HINCRBYFLOAT run:<run-id> cost <cost>
   HINCRBY      run:<run-id> count 1
   LPUSH        run:<run-id>:tools <tool-name>
   LTRIM        run:<run-id>:tools 0 9
   EXPIRE       run:<run-id>          3600
   EXPIRE       run:<run-id>:tools    3600

2. ATTRIBUTION STAMPS (new — derives workspace/user dimensions)
   HINCRBYFLOAT cost:workspace:<wid>:<yyyy-mm-dd>  total      <cost>
   HINCRBYFLOAT cost:workspace:<wid>:agent:<name>  <yyyy-mm>  <cost>
   HINCRBYFLOAT cost:user:<uid>:<yyyy-mm>          total      <cost>
   HINCRBYFLOAT cost:user:<uid>:agent:<name>       <yyyy-mm>  <cost>
```

The redundant headers are trustworthy at this point because the PreHook
already verified them against the macaroon caveats. So PostHook can stamp
freely without re-parsing the token.

Total overhead: ~5 Redis ops per request, all pipelined, sub-millisecond.

## What the macaroon contains

First-party caveats (verified locally, no network):

```
user        = alice@acme.com
user_id     = u_01H...
workspaces  = [w1, w2, w3]          ← [] = none, [*] only on root
agents      = [browser, build]      ← [] = none, [*] only on root
pods        = [pod-a, pod-b]        ← optional, for pod-scoped agents
env         = prod                  ← prod | staging — must match VK env
parent_mac  = <hash of issuer>      ← attenuation chain
run_id      = <uuid>                ← present on invocation tokens
exp         = 2026-05-11T18:00:00Z
nonce       = <random>              ← for revocation by ID
```

### Attenuation chain example

```
root (from /macaroons/login):
  user=alice, user_id=u_01..., workspaces=[w1,w2,w3], agents=[*],
  env=prod, exp=24h, nonce=N1

invocation (from app backend when alice clicks "run browser on w1"):
  inherits above, adds:
    + agents=[browser]   (intersected with parent)
    + workspaces=[w1]    (intersected with parent)
    + run_id=r_01H...
    + exp=10m
    + nonce=N2

sub-agent spawn (browser spawns a sub-task):
  inherits above, adds:
    + agents=[browser-subtask]   (narrower — must be subset)
    + exp=2m                     (shorter — must be ≤ parent exp)
    + nonce=N3
  (cannot widen workspaces — adding more would not validate against the
   intersection rule at the plugin)
```

The plugin enforces monotonic narrowing: each layer of caveats can only
constrain, never relax, the previous layer's restrictions. This is the
core macaroon security property.

## Revocation

Macaroons don't natively revoke. Two layers:

1. **Short TTL by default.** Invocation tokens live 10 minutes. Re-mint
   freely; cheap.
2. **Revocation list in Redis** keyed by macaroon nonce.
   `SET revoke:<nonce> 1 EX 86400`. Plugin checks on every PreHook.

The plugin's existing `kill:<run-id>` mechanism is the same shape — same
Redis, same TTL pattern. Use one helper for both.

Bulk revocation ("revoke everything alice has"): the issuer keeps a
`user:<uid>:revoke_before` timestamp in Redis; plugin rejects macaroons
whose `iat` is older than the user's revoke_before. Same pattern as JWT
denylists. Implement when needed, not in v1.

## Issuer service

New service (or new endpoints in an existing auth service):

```
POST /macaroons/login        → root macaroon (24h, full user scope)
POST /macaroons/attenuate    → narrower child given a parent + caveats
POST /macaroons/revoke       → adds nonce to revocation list
```

The plugin verifies locally; no `/verify` endpoint needed in the hot
path. (Provide one for debugging only.)

The MCP server (and Stakwork, Goose, app backend, Rust callers) calls
`/macaroons/attenuate` at the start of each agent invocation, drops
workspace/agent/run-id caveats, hands the child token to the agent
process via env var or header. From there it rides on every LLM call.

### Root signing key handling

The macaroon root HMAC key is the crown jewel. Options:

1. **Shared secret in env** — simple, rotation = full deploy of issuer +
   plugin. Fine for v1.
2. **Secrets service** — issuer and plugin fetch at startup, periodic
   refresh. Cleaner ops.
3. **JWKS-style multi-key with `kid` caveat** — zero-downtime rotation.
   Overkill until we've felt the pain.

Start with (1). Move to (2) when we have a secrets service worth using.

## Caller obligations

Every caller from [the original plan](./llm-gateway-and-runaway-enforcement.md#phase-4--caller-header-propagation)
gains one extra responsibility: **receive a macaroon from the trigger,
attenuate it for the invocation, propagate the child as `x-macaroon` on
every LLM call.**

| Caller | Source of root macaroon | Attenuation point |
|---|---|---|
| TS / MCP | User session cookie / CLI token | Top of `get_context()` / `stream_context()` |
| App chat | Logged-in user session | On send |
| Rust agents | Macaroon passed by spawning service | At agent boot |
| Python agents | Macaroon passed by spawning service | At agent boot |
| Goose | Macaroon in session config | At `goose run` start |
| Stakwork | Macaroon in workflow trigger payload | Workflow entry node |

Sub-agent spawns: the parent attenuates and passes a child. **Never reuses
its own token.** If the parent's token leaked, the child can't undo the
caveats.

## Cost-attribution queries available after rollout

All answerable from Redis (`HGETALL`, sorted-set queries) without a
separate analytics pipeline:

- Top 10 most expensive workspaces this month
- Cost per agent within a workspace
- Cost per user across the org
- Cost per (user, agent) for chargeback / quota enforcement
- "How much did alice spend on `browser` this week"
- "How much is `org-chat` costing per workspace it touches"

Bifrost's native analytics give us (per-org, since one Bifrost per org):

- Cost per agent, org-wide
- Cost per env (Team budget)
- Rate-limit hits per agent

Combined, we cover every cost question without standing up Langfuse /
Helicone. The org-level rollup is exactly what the boss wants for "across
all workspaces, which agents are spending the most" — that's a VK-level
query, native to Bifrost.

## Open questions for the team chat

1. **Macaroon library.** [`libmacaroons`](https://github.com/rescrv/libmacaroons)
   is the reference C lib with bindings. Pure-Go option:
   `gopkg.in/macaroon.v2`. Plugin is Go, so Go-native preferred.

2. **Third-party caveats.** Macaroons can carry caveats discharged by
   external services ("workspace-access(w1) signed by workspaces-svc").
   Adds latency + complexity. Phase 1: first-party only. Revisit if we
   need cross-service delegation.

3. **Macaroon issuer service ownership.** New service, or endpoints in an
   existing auth service? Lives close to the user login flow either way.

4. **Per-workspace cap config source.** Where does the plugin read "w1's
   daily $ cap"? Env JSON, config endpoint, or a reconcile step. Same
   question as per-agent thresholds.

5. **Org-chat's blast radius.** A macaroon listing 50 workspaces is a
   juicy token even at 10-min TTL. Mitigations beyond TTL:
   - Bind to run-id so a stolen token can't outlive the run
   - Discharge caveat per workspace, requiring fresh signature from a
     workspace-service. Punts to "later."

6. **Cross-workspace agent cost attribution.** Org-chat reads from many
   workspaces. Options:
   - Attribute to the user's home workspace (simple, slightly misleading)
   - Attribute proportionally per workspace it reads from (accurate;
     requires the agent to stamp the "for" workspace per call)
   - Attribute to a dedicated "org" pseudo-workspace (clean bookkeeping)

7. **Stakwork integration.** Workflow trigger API needs to carry the
   macaroon. Confirm the trigger payload has room for an opaque token; if
   not, this is upstream work.

8. **Macaroon size on the wire.** A chain of 5–10 caveats is fine (~1 KB).
   If it grows, switch to compact serialization or push caveats into
   discharge tokens.

9. **Per-org-Bifrost ops cost.** N orgs = N Bifrost instances + N Redis +
   N plugin processes. Acceptable at our scale, but worth a back-of-the-
   envelope on infra. If the count gets large, multi-tenant Bifrost with
   the Customer tier = org becomes the alternative.

## Rollout sequence

The Bifrost gateway + plugin rollout from the original plan is unchanged.
Macaroon work slots in alongside:

1. **Bifrost stood up** per pilot org.
2. **TS/MCP routed through Bifrost** (`LLM_GATEWAY_URL` set).
3. **16 VKs provisioned** per Bifrost via reconcile script.
4. **Macaroon issuer service shipped** — login + attenuate endpoints.
   App backend wired up. Tokens minted but **not yet enforced** by
   plugin.
5. **Header propagation** — `x-run-id`, `x-session-id`, `x-agent-name`,
   `x-macaroon`, `x-workspace-id`, `x-user-id`. Visible in Bifrost logs.
6. **Plugin v1** — per-run cost + step caps only (no macaroon verification
   yet). Same as original plan.
7. **Plugin v2** — macaroon verification + caveat enforcement + VK ↔
   macaroon binding check + attribution stamps. **Hard reject** on
   missing or invalid macaroon.
8. **Roll out remaining callers** — Rust, Python, Goose, Stakwork. Each
   picks up `LLM_GATEWAY_URL`, `BIFROST_VK`, header propagation, *and*
   macaroon attenuation. Until a caller participates, its requests will
   401 once plugin v2 ships, so this is the migration forcing function.
9. **Enforce VK + macaroon org-wide.** `enforce_auth_on_inference: true`
   in Bifrost. Plugin requires macaroon. Both required for any LLM call.

## What this design buys

- **Cryptographic provenance.** Every LLM call traces back to a specific
  human via a verifiable signature chain. Audit logs are no longer
  "the app says alice did this" — they're "alice's macaroon signed this."
- **Cross-workspace agents are trivial.** Org-chat's macaroon just lists
  multiple workspaces. No special-case in the VK tree.
- **VK leakage is no longer game-over.** Without the macaroon root key,
  a leaked VK is inert.
- **Permission changes propagate without reconciliation.** Workspace
  access revoked → next macaroon mint reflects it. No VK shuffling.
- **Sub-agent containment.** A sub-agent literally cannot do more than
  its parent permitted. Attenuation is one-way.
- **Free org-level cost rollup.** Bifrost VK analytics show
  cost-per-agent across the org natively — exactly the "which agents are
  spending the most" report we want.
- **Free workspace/user cost rollup.** Plugin Redis stamps give every
  other slice we care about without a separate analytics pipeline.
