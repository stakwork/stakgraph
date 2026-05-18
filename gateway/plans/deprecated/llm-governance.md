# LLM Governance: Gateway, Identity, and Runaway Control

## What we're building

A single architecture that answers four questions about every LLM call
that originates anywhere in our stack:

1. **Where does it go?** Through one gateway, regardless of caller
   language, framework, or upstream provider.
2. **Who authorized it?** A specific human, with a verifiable scope
   (which workspaces, which agents, until when).
3. **How much can it spend?** Org-wide caps per agent. Per-run caps set
   by whoever spawned the run. Mathematically-bounded sub-agent caps.
4. **Can it be killed?** Yes, from any UI or service, mid-loop, in any
   language, with sub-second latency to the next model call boundary.

A fifth question — **"how do we see what happened?"** — gets answered for
free by Bifrost's built-in logging plugin, which writes a richly-indexed
row per LLM call into a SQLite `logs` table. Our governance dimensions
(`run_id`, `agent_name`, `workspace_id`, `user_id`, `session_id`) are
attached via `x-bf-dim-*` request headers, which Bifrost automatically
materializes into each row's `metadata` JSON column. **Observability is a
configuration concern, not a code concern.** See "Observability via
Bifrost dimensions" below.

Three layers, each with one clear responsibility:

```
┌─────────────────────────────────────────────────────────────┐
│  IDENTITY LAYER  ── macaroons                               │
│   Who: a specific human, via daily root + per-run child.    │
│   Scope: workspaces, agents, run_id, per-run $ cap, exp.    │
│   Carried as: x-macaroon HTTP header.                       │
│   Enforced by: the gateway plugin (HMAC + caveats).         │
├─────────────────────────────────────────────────────────────┤
│  ROUTING + ORG BUDGET LAYER  ── Bifrost Virtual Keys        │
│   Which caller deployment is this? (MCP, Goose, workflow…)  │
│   Org-wide $ and rate caps per agent. Model allowlists.     │
│   Carried as: the standard auth header (Bearer / x-api-key) │
│   with value = sk-bf-…                                      │
│   Enforced by: Bifrost natively.                            │
├─────────────────────────────────────────────────────────────┤
│  RUN STATE LAYER  ── plugin + Redis                         │
│   Per-run accumulated cost, step count, tool history.       │
│   Per-workspace and per-user attribution counters.          │
│   External kill switch.                                     │
│   Enforced by: the gateway plugin against Redis state.      │
└─────────────────────────────────────────────────────────────┘
```

Macaroons are the security boundary. VKs are routing identity and org
governance. The plugin enforces per-run state. No single layer is the
"one true auth" — they answer different questions and compose cleanly.

---

## Component overview

### Bifrost gateway

One Bifrost instance per **workspace** (revised — see "Open questions").
Sits between every LLM caller and every upstream provider. Provider-native
SDKs keep their native protocol; only the base URL changes.

Because there's one Bifrost per workspace, the Bifrost-native dimensions
map cleanly to our model:

| Bifrost concept                        | Our meaning                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| Bifrost instance                       | one workspace                                                               |
| `virtual_key_id`                       | caller deployment (e.g. `mcp-prod`, `goose-prod`)                           |
| `customer_id` (on the VK)              | **environment** (`prod` / `staging`) — _reserved, not used yet_             |
| `team_id` (on the VK)                  | unused for now; available if we ever want VK-level agent grouping           |
| `metadata` (from `x-bf-dim-*` headers) | run_id, agent_name, user_id, session_id — anything else we care to slice on |

`user_id`, `team_id`, `customer_id`, `business_unit_id` are first-class
indexed columns on the `logs` table, but they are populated by Bifrost's
**built-in governance plugin** from VK configuration (and `user_id` from
its enterprise auth middleware). They are explicitly marked
"DO NOT SET THIS MANUALLY" in Bifrost's source. We therefore do _not_
co-opt them from our plugin; we use `x-bf-dim-*` for everything that
varies per-request.

```
LLM_GATEWAY_URL=http://bifrost:8181

  Anthropic SDK  → http://bifrost:8181/anthropic/v1   (native path)
  OpenAI SDK     → http://bifrost:8181/openai/v1      (native path)
  Google GenAI   → http://bifrost:8181/genai/v1beta   (native path)
  OpenRouter     → http://bifrost:8181/openai/v1      (escape hatch —
                                                       any model
                                                       OpenRouter
                                                       supports)
```

Three native paths cover the providers we care about with full
provider-native features (Anthropic thinking blocks, Google
`thinkingBudget`, etc.). OpenRouter via the OpenAI path is the catch-all
for everything else (Kimi, Mistral, DeepSeek, new releases).

Bifrost holds the real upstream provider keys. Callers never see them.

### Virtual Keys (VKs)

Bifrost issues VKs (`sk-bf-…`) as bearer credentials carried in the
caller's existing auth header:

| SDK                                             | Header it sends           | Value we set it to |
| ----------------------------------------------- | ------------------------- | ------------------ |
| `@ai-sdk/anthropic`                             | `x-api-key`               | `sk-bf-…`          |
| `@ai-sdk/openai`, `@openrouter/ai-sdk-provider` | `Authorization: Bearer …` | `sk-bf-…`          |
| `@ai-sdk/google`                                | `x-goog-api-key`          | `sk-bf-…`          |
| Python `openai` client                          | `Authorization: Bearer …` | `sk-bf-…`          |
| Goose                                           | provider config env vars  | `sk-bf-…`          |

Per-VK governance Bifrost enforces natively:

- Daily / monthly `$` budget
- RPM / TPM rate limit
- Provider allowlist
- Model allowlist (e.g. "this caller may never invoke Opus")

Per-agent-type attribution comes from a separate `x-bf-dim-agent-name`
header, not from minting one VK per agent type. (See "Observability,"
below.)

#### How many VKs? — open for team discussion

The original plan was one VK per caller-deployment. Given the dim-based
observability layer and the macaroon-based enforcement layer that now
sit alongside VKs, a much simpler VK structure may be sufficient. The
table below lays out the spectrum; we'll commit to one option before
implementing step 3 of the rollout.

| Option                                                  | Examples (per workspace)                                                        | Pros                                                                                                                                                                                                                    | Cons                                                                                                                                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Single VK**                                        | `sk-bf-default`                                                                 | Trivially simple; one env var everywhere; all governance happens at macaroon + dim layer                                                                                                                                | No way to kill/cap/rate-limit one caller deployment independently of the rest; loses the indexed `virtual_key_id` log column as a slicing tool; can't restrict models per deployment at the gateway |
| **B. Per environment** _(recommended starting point)_   | `sk-bf-prod`, `sk-bf-staging`                                                   | Maps directly to the reserved `customer_id` slot; one env-wide budget per environment (staging gets a tiny cap, prod a generous one); two env vars to manage; still attributes per deployment via `x-bf-dim-deployment` | Buggy and well-behaved deployments inside the same environment share a budget and a rate-limit pool; can't surgically disable one deployment at the VK layer                                        |
| **C. Per (deployment × environment)** _(original plan)_ | `sk-bf-mcp-prod`, `sk-bf-mcp-staging`, `sk-bf-goose-prod`, `sk-bf-chat-prod`, … | Surgical: each deployment has its own budget, RPM cap, model allowlist, and on/off switch; `virtual_key_id` directly answers "what did goose-prod cost today"                                                           | N env vars to provision and rotate; mostly redundant once macaroons enforce per-run caps and the kill-switch table catches runaways                                                                 |
| **D. Per env, with deployment as a dim**                | `sk-bf-prod` + `x-bf-dim-deployment=mcp`                                        | Same VK count as B; preserves per-deployment attribution in logs                                                                                                                                                        | Attribution is self-reported (buggy deployment can mis-tag itself); no per-deployment enforcement                                                                                                   |

> **Note on Option C and `agent-name`.** A tempting variant of Option C
> is "VK per (agent-name × environment)" — minting one VK per agent
> type (`browser`, `coder`, `chat`, `reviewer`, …) so the agent name
> ends up in Bifrost's first-class `team_id` column and `GET
/api/logs/histogram/cost/by-dimension=team_id` answers "spend per
> agent" via HTTP for free. **Don't.** `deployment` is a small, mostly
> static set (`mcp`, `goose`, `chat-bff`, `workflow-engine` —
> single-digit count, changes rarely). `agent-name` is open-ended and
> grows with the product (every new agent type adds a row). Encoding
> it in the VK structure means every new agent ships with a
> provisioning task and a new env var, instead of just setting a
> header. Keep `agent-name` in `x-bf-dim-agent-name` →
> `metadata.agent-name` and answer per-agent rollups with SQL against
> `logs.db` or a small custom endpoint in our plugin. See
> `smoke-test-results.md` §"Next steps" for the reasoning.

**Recommendation: start with Option B (per environment), upgrade to C only when an incident or workflow demands surgical per-deployment control.** Rationale:

1. Steps 5–7 of the rollout don't depend on VK count — observability is by dims, enforcement is by macaroons.
2. Per-deployment VK budgets are most valuable as a stopgap _before_ macaroons are deployed. Once macaroons land at step 8, that protection is largely redundant.
3. Migrating from B to C is one-line: mint a new VK, update one env var in the offending deployment. No code change.
4. Maps cleanly to "customer_id = environment" — every log row gets indexed by env without extra config.
5. The plan's own statement at "VKs in detail" already says VKs are "the big-blast-radius cap that catches a buggy deployment, not the fine-grained policy layer." Per-environment VKs are the natural shape of that statement.

The _primary_ defense against runaway is the **macaroon layer**, not the
VK layer. VKs are a coarse big-blast-radius backstop (most useful in
steps 1–7 before macaroons) and an environment-isolation tool (forever).
Whichever option we pick, that framing holds.

### Macaroons

Bearer tokens with HMAC-chained caveats. Cryptographically narrowable
without contacting the issuer. Two tiers in practice:

**Daily root macaroon.** Minted at login (or session refresh). 24h
rolling TTL. Held server-side, keyed by alice's session. Contains:

```
user_id     = u_01H...
user        = alice@acme.com
workspaces  = [w1, w2, w3]
agents      = [*]              ← may invoke any agent her role allows
env         = prod
iat         = 2026-05-13T09:00:00Z
exp         = 2026-05-14T09:00:00Z
nonce       = <random 128-bit>
```

Re-minted silently when an active session's prior root nears expiry.

**Invocation macaroon.** Minted by the spawner (chat backend, MCP,
workflow engine, etc.) each time an agent is launched. 10-minute TTL.
Held in the agent process for the run lifetime. Strictly narrower than
the daily root:

```
parent_nonce     = <root's nonce>     ← lets the plugin revoke whole chains
user_id          = u_01H...           ← inherited, intersected
workspaces       = [w1]               ← narrowed from root
agents           = [browser]          ← narrowed from root
run_id           = r_01H...
max_cost_usd     = 5.00               ← spawner-set per-run budget
max_steps        = 100                ← spawner-set per-run step cap
max_wallclock_s  = 600
env              = prod
exp              = now + 10m
nonce            = <random>
```

**Sub-agent macaroon.** When a parent agent spawns a sub-agent, the
parent attenuates _locally_ — macaroon attenuation requires only the
parent's signature, not the HMAC root key. No network roundtrip:

```
parent_nonce     = <invocation macaroon's nonce>
... (inherited caveats) ...
agents           = [browser/web-search]    ← narrower
max_cost_usd     = 2.00                     ← parent's remaining budget
max_wallclock_s  = 120                      ← ≤ parent's remaining wallclock
exp              = min(now + 2m, parent.exp)
nonce            = <new random>
```

The macaroon protocol mathematically guarantees a child cannot widen any
caveat. Sub-agent containment is physics, not policy.

### Gateway plugin (Go, in-process with Bifrost)

A Bifrost custom plugin implementing the standard hook surface
(`HTTPTransportPreHook`, `PreLLMHook`, `PostLLMHook`,
`HTTPTransportStreamChunkHook`). Single binary, deployed alongside
Bifrost. Backed by Redis (for hot enforcement state only — _not_ for
analytics; Bifrost's built-in logging plugin handles per-call analytics).

The plugin must be registered **before** Bifrost's built-in logging
plugin so that any dimensions it derives from the macaroon land in
`logs.metadata` (Bifrost's logging plugin reads dimensions from
`BifrostContextKeyDimensions` once during its own `PreLLMHook`).

`HTTPTransportPreHook` / `PreLLMHook` (auth + spending gate, runs before
the upstream call):

```
1. RESOLVE CALLER IDENTITY (VK)
   - Bifrost has already auth'd the VK before the plugin runs (when
     enforce_auth_on_inference: true). Plugin reads the VK's metadata
     for caller-deployment label + env via Bifrost's context keys
     (BifrostContextKeyVirtualKey, BifrostContextKeyAPIKeyName, etc.).

2. VERIFY MACAROON
   - parse x-macaroon
   - check HMAC chain against root signing key
   - if missing  → 401 macaroon_required
   - if invalid  → 401 macaroon_invalid

3. CHECK MACAROON LIVENESS
   - exp passed?                              → 401 macaroon_expired
   - revoke:<leaf_nonce>  in Redis?           → 401 macaroon_revoked
   - revoke:<parent_nonce> in Redis?          → 401 macaroon_revoked
   - revoke_user_before:<user_id> > iat?      → 401 macaroon_revoked

4. CANONICALIZE DIMENSIONS FROM CAVEATS
   - read existing dim map: ctx.Value(BifrostContextKeyDimensions)
   - overwrite with macaroon-derived values:
       dims["run-id"]       = caveat.run_id
       dims["agent-name"]   = caveat.agents[0]   // (or sub-scope)
       dims["workspace-id"] = caveat.workspaces[0]
       dims["user-id"]      = caveat.user_id
   - ctx.SetValue(BifrostContextKeyDimensions, dims)
   (this is also our "headers match caveats" check — we don't compare,
    we *override*. caller-supplied dim values that disagree with the
    macaroon are silently replaced. anything they sent that wasn't in
    the macaroon is preserved.)

5. CHECK PER-RUN BUDGET (spawner-set)
   - caveat.max_cost_usd > hard_ceiling?      → 402 budget_unreasonable
     (hard_ceiling is a backstop, e.g. $1000)
   - cost:run:<run_id> >= caveat.max_cost?    → 402 run_cost_exceeded
   - for each ancestor run in the chain:
       cost:run:<ancestor> >= ancestor.cap?   → 402 run_cost_exceeded
   - steps:run:<run_id> >= caveat.max_steps?  → 402 run_step_exceeded

6. CHECK KILL SWITCH
   - kill:<run_id>  in Redis?                 → 402 run_killed
   - for each ancestor run_id:
       kill:<ancestor>?                       → 402 run_killed

7. CHECK PER-WORKSPACE / PER-USER DAILY CAPS (optional)
   - These can be derived on-demand from the Bifrost logs table
     (SUM(cost) WHERE day = ... GROUP BY user/workspace). For v1, skip
     this hot check entirely and rely on macaroon-set per-run caps
     plus VK-level deployment caps. Revisit if log-table aggregation
     proves too slow for inline checks.

8. CHECK TOOL-LOOP HEURISTIC
   - last 10 tool calls for run_id: same tool >= 8 of 10?
                                              → 402 tool_loop_detected
```

`PostLLMHook` (accounting, runs after a successful upstream call — and
also fires for streaming on the final chunk, distinguished by
`bifrost.IsFinalChunk(ctx)`):

```
1. Read cost from BifrostResponse.Usage / Cost (Bifrost's pricing
   manager has already computed it for non-streaming; for streams the
   final chunk carries the accumulator's cost).

2. Update per-run hot-state counters, walking the ancestor chain:
   for r in [run_id, ...ancestor_run_ids]:
     HINCRBYFLOAT cost:run:<r>   total $cost
     HINCRBY      steps:run:<r>  total 1
     EXPIRE       cost:run:<r>   3600
     EXPIRE       steps:run:<r>  3600

3. Update tool history for loop detection:
   LPUSH  tools:run:<run_id> <tool_name>
   LTRIM  tools:run:<run_id> 0 9
   EXPIRE tools:run:<run_id> 3600

   (No per-workspace / per-user counters in Redis. Those queries are
    answered from Bifrost's logs table — every row already has
    workspace_id and user_id in its metadata JSON. See "Observability"
    below.)
```

Total Redis overhead: ~3–5 ops per LLM call (down from ~6–10 in the
original plan, because workspace/user attribution moved to the logs
table). Sub-millisecond on a co-located Redis.

### Auth service (issuer)

A small new service (or new endpoints in an existing auth service)
exposing three endpoints:

```
POST /macaroons/login
  body: { user_id, scopes_from_user_record }
  returns: daily root macaroon
  stores: session_id → macaroon in server-side session store

POST /macaroons/attenuate
  body: { parent_macaroon, additional_caveats }
  returns: attenuated child macaroon
  validates: every new caveat is a strict narrowing of parent
  (this endpoint is called by spawners; never exposed to end users)

POST /macaroons/revoke
  body: { nonce | user_id }
  effects: writes revoke:<nonce> or revoke_user_before:<user_id> to Redis
```

The auth service is the only place the HMAC root signing key lives
besides the gateway plugin (which only needs it to _verify_, but in
practice HMAC verification requires the same key as minting — accept
this trade-off; see Open Questions).

---

## How a request flows

End-to-end, with a concrete example. Alice clicks "run browser on
workspace w1" in the chat UI.

```
1. FRONTEND
   - Sends request to chat BFF with session cookie.

2. CHAT BFF (spawner)
   a. Looks up alice's daily root macaroon from session store.
   b. Computes per-run budget:
      - agent default for browser:          $5.00 / 100 steps
      - alice's daily $ remaining:          $18.50
      - workspace w1 daily $ remaining:     $42.00
      - effective: $5.00 / 100 steps / 10m
   c. POST /macaroons/attenuate with caveats:
        agent=browser, workspaces=[w1], run_id=r_01H..,
        max_cost_usd=5.00, max_steps=100, exp=now+10m
   d. Receives invocation macaroon M_inv.
   e. Spawns browser agent process with:
        BIFROST_VK=sk-bf-prod          (env VK; deployment via dim header)
        AGENT_MACAROON=<M_inv>
        RUN_ID=r_01H..
        WORKSPACE_ID=w1
        USER_ID=u_01H..
        AGENT_NAME=browser

3. BROWSER AGENT (process)
   a. Reads macaroon and run context from env.
   b. Constructs LLM client with:
        baseURL: http://bifrost:8181/anthropic/v1
        apiKey:  process.env.BIFROST_VK
        headers: {
          "x-macaroon":            process.env.AGENT_MACAROON,
          "x-bf-dim-run-id":       process.env.RUN_ID,
          "x-bf-dim-realm-id": process.env.WORKSPACE_ID,
          "x-bf-dim-user-id":      process.env.USER_ID,
          "x-bf-dim-agent-name":   process.env.AGENT_NAME,
          "x-bf-dim-session-id":   <persistent conversation id>,
        }
   c. Runs agent loop. Each model call ships with all the above.

4. BIFROST TRANSPORT
   a. Authenticates VK (sk-bf-prod). customer_id=prod attached.
   b. Extracts x-bf-dim-* into BifrostContextKeyDimensions (the map
      that ends up in logs.metadata).
   c. Invokes our plugin's HTTPTransportPreHook.

5. OUR PLUGIN PreHook
   - Verifies macaroon (HMAC + caveats).
   - Canonicalizes the dim map: overwrites run-id/workspace-id/user-id/
     agent-name in BifrostContextKeyDimensions with macaroon-derived
     values. (Anything the caller sent that disagrees with the macaroon
     is silently replaced.)
   - Reads cost:run:r_01H.. → currently $3.21, cap is $5.00. Passes.
   - Reads steps:run:r_01H.. → currently 14, cap is 100. Passes.
   - No kill, no loop.
   - Returns pass.

6. BIFROST CORE
   - Strips the VK, substitutes the real Anthropic key from its config.
   - Forwards to api.anthropic.com.

7. ANTHROPIC
   - Returns model response.

8. BIFROST → our plugin's PostLLMHook (hot enforcement state):
   - Cost = $0.42 from BifrostResponse.
   - HINCRBYFLOAT cost:run:r_01H..  total 0.42 → now $3.63
   - HINCRBY      steps:run:r_01H.. total 1     → now 15
   - LPUSH        tools:run:r_01H.. "<tool>"

9. BIFROST → built-in logging plugin (analytics):
   - Writes a logs row with cost=0.42, latency, prompt_tokens,
     completion_tokens, model=claude-..., virtual_key=mcp-prod,
     metadata={"run-id":"r_01H..","workspace-id":"w1","user-id":"u_01H..",
     "agent-name":"browser","session-id":"..."}.
   - That row is now queryable by every dimension via SQL.

10. RESPONSE to agent → next step in the loop.
```

**When the run runs away.** Suppose the model gets stuck and the
accumulated cost crosses $5.00 mid-loop. The next PreHook returns 402
`run_cost_exceeded`. The AI SDK throws. The agent loop terminates. The
spawner is notified by the existing error-handling path. Kill latency =
"until the next model call boundary," which is the tightest kill
physically possible — once a request is on the wire to Anthropic, it
can't be recalled.

**When someone clicks "stop this agent" in the UI.** Backend writes
`SET kill:r_01H.. 1 EX 3600` to Redis. The next PreHook returns 402
`run_killed`. Same termination path. Works for any caller in any
language because all of them route through the plugin.

**When the agent spawns a sub-agent.** The parent locally attenuates
its macaroon:

```python
# inside the browser agent process
child_macaroon = attenuate(self.macaroon, {
    "agents": "browser/web-search",
    "run_id": new_run_id(),
    "max_cost_usd": min(2.00, self.remaining_budget()),
    "max_wallclock_s": 120,
    "exp": now + 120,
    "nonce": random_nonce(),
})
spawn_subagent(env={
    "AGENT_MACAROON": child_macaroon,
    "RUN_ID":         new_run_id,
    ...
})
```

The plugin enforces the chain on every sub-agent LLM call: the child's
spend counts against both `cost:run:<child>` and
`cost:run:<parent>`. If the parent's $5.00 cap is exhausted by the
combined parent+child spend, the child is killed even though its own
$2.00 cap hasn't been hit.

---

## Wire protocol (the canonical request)

Every LLM call from every caller arrives at Bifrost with two kinds of
headers: a small set of **credentials** (auth + future macaroon), plus a
set of **dimension headers** that Bifrost automatically materializes into
the log row's `metadata` JSON for observability.

```
# Credentials
Authorization: Bearer sk-bf-<env>                   ← VK (Bifrost auth)
   or x-api-key / x-goog-api-key as appropriate for the SDK

x-macaroon:           <invocation macaroon>         ← identity + scope (future)

# Attribution dimensions — written to logs.metadata automatically
x-bf-dim-run-id:       <uuid>                       ← this invocation
x-bf-dim-session-id:   <persistent conversation id> ← long-lived thread
x-bf-dim-agent-name:   browser                      ← which agent type
x-bf-dim-realm-id: w1                           ← which workspace
x-bf-dim-user-id:      u_01H...                     ← which human
x-bf-dim-deployment:   mcp                          ← which caller (mcp/goose/chat/…)
```

`x-bf-dim-deployment` is only required when we picked a VK structure
that doesn't already encode the deployment (Options A, B, D). Under
Option C (per-deployment VKs) it's redundant with `virtual_key_id` and
can be omitted — though setting it costs nothing and keeps the dim set
identical across VK options.

### Why `x-bf-dim-*`?

Bifrost's HTTP transport extracts any header matching `x-bf-dim-<key>`,
strips the prefix, lowercases the key, and stores the resulting map at
`BifrostContextKeyDimensions`. The built-in logging plugin then merges
that map into the SQLite `logs.metadata` JSON column on every call. The
result: any caller that sets these headers is **automatically observable
by every dimension, with no plugin code required**.

(Source: `transports/bifrost-http/lib/ctx.go` `ConvertToBifrostContext`
extracts the prefix into `dimensions[labelName]`;
`plugins/logging/main.go` `captureLoggingHeaders` merges them into
`MetadataParsed`, which is serialized to the `metadata` TEXT column.)

### Macaroon precedence (once macaroons land)

The macaroon is the **source of truth** for identity and scope. The
dimension headers are a convenience that:

1. let pre-macaroon callers be observable today, and
2. survive as an analytics-only signal once macaroons are enforced.

When the plugin verifies a macaroon in step 7+, it will **overwrite the
dim map** with values derived from the caveats — so `run_id`,
`agent_name`, `workspace_id`, `user_id` recorded in `metadata` always
reflect the macaroon, not whatever the caller sent. This makes the
attribution record cryptographically bound to the macaroon chain.

Until then, the dim headers are accepted as-is (fail-open observability,
fail-closed enforcement once macaroons exist).

### Required vs optional

**Required today (observability-only phase):** `Authorization` (VK),
`x-bf-dim-run-id`. Everything else is optional but recommended.

**Required after step 7 (enforcement phase):** `Authorization` (VK),
`x-macaroon`. `run_id` and the other identity fields move _into_ the
macaroon caveats; the dim headers become advisory only.

**Optional always:** `x-bf-dim-session-id`. Used for grouping in
analytics; no enforcement decisions depend on it.

---

## Caller integration

The work for each caller is:

1. **Set base URL** to `LLM_GATEWAY_URL`. (You already do this.)
2. **Set the auth header value** to the deployment's VK from env var.
3. **Receive the macaroon** at spawn time (env var, RPC arg, or
   workflow trigger payload).
4. **Attach headers** to the LLM client at construction time.
5. **If you spawn sub-agents:** attenuate your macaroon locally before
   passing to the child process.

The header-attaching step is what we mean by "speak the header
language." Every modern LLM SDK supports default headers in one line.

### TypeScript (MCP, AI SDK)

```ts
const headers = {
  // credentials
  "x-macaroon": process.env.AGENT_MACAROON, // optional pre-step-8

  // observability dimensions → logs.metadata
  "x-bf-dim-run-id": process.env.RUN_ID,
  "x-bf-dim-session-id": sessionId,
  "x-bf-dim-agent-name": process.env.AGENT_NAME,
  "x-bf-dim-realm-id": process.env.WORKSPACE_ID,
  "x-bf-dim-user-id": process.env.USER_ID,
  "x-bf-dim-deployment": "mcp", // this caller's deployment label
};

const anthropic = createAnthropic({
  apiKey: process.env.BIFROST_VK,
  baseURL: process.env.LLM_GATEWAY_URL + "/anthropic/v1",
  headers,
});
```

All four AI SDK provider factories (`createAnthropic`, `createOpenAI`,
`createGoogleGenerativeAI`, `createOpenRouter`) accept `headers`.

### Python (application code)

```python
client = OpenAI(
    base_url=os.environ["LLM_GATEWAY_URL"] + "/openai/v1",
    api_key=os.environ["BIFROST_VK"],
    default_headers={
        "x-macaroon":            os.environ["AGENT_MACAROON"],
        "x-bf-dim-run-id":       os.environ["RUN_ID"],
        "x-bf-dim-agent-name":   os.environ["AGENT_NAME"],
        "x-bf-dim-realm-id":     os.environ["REALM_ID"],
        "x-bf-dim-user-id":      os.environ["USER_ID"],
        "x-bf-dim-deployment":   "pyapp",
    },
)
```

### Goose

Goose reads provider config from a yaml/toml file with an `extra_headers`
(or equivalent) field. Set `base_url`, the API-key env var to
`BIFROST_VK`, and the headers map. Macaroon and run-id are injected by
the script that invokes `goose run`.

### Rust callers

One place per binary: the shared HTTP client builder. Set default
headers when constructing the client at agent start. Read from env.

### Workflow engine (Stakwork)

Workflow trigger payload carries the macaroon and run-id. Workflow
runtime sets them as default headers on the LLM-node HTTP client. If
the workflow engine doesn't natively support custom headers on LLM
nodes, this is the place upstream work may be required (see Risks).

### Frontend chat

Frontend never touches macaroons or VKs. Frontend sends a normal
authenticated request to the chat BFF. The BFF (a server) holds the
VK, looks up alice's macaroon from her session, attenuates per send,
and makes the LLM call with the right headers. Frontend stays trivial.

---

## Spawner responsibilities

Each entity that _spawns_ an agent is responsible for:

1. **Obtaining alice's daily root macaroon.** Either from her session
   store (the common case: chat BFF, MCP server invoked from web) or
   from the trigger payload (workflow engine, webhook receivers).

2. **Computing the per-run budget.** Single shared function:

   ```ts
   function budgetForRun(ctx: {
     agent: string;
     user: User;
     workspace: Workspace;
     parent_remaining?: { cost: number; wallclock: number };
   }): RunBudget {
     const def = AGENT_DEFAULTS[ctx.agent]; // { cost, steps, wallclock }
     return {
       max_cost_usd: Math.min(
         def.cost,
         ctx.user.daily_remaining_usd,
         ctx.workspace.daily_remaining_usd,
         ctx.parent_remaining?.cost ?? Infinity,
       ),
       max_steps: def.steps,
       max_wallclock_s: Math.min(
         def.wallclock,
         ctx.parent_remaining?.wallclock ?? Infinity,
       ),
     };
   }
   ```

3. **Calling `/macaroons/attenuate`** with the computed budget +
   workspace/agent/run_id caveats + 10m exp.

4. **Passing the resulting macaroon to the spawned process** as an env
   var (or workflow context, or RPC arg).

5. **Setting `BIFROST_VK`** in the spawned process's env to the
   deployment's VK.

Service identities (cron, webhooks, scheduled workflows) are treated as
"users" in this model. There's a `system:nightly-cron` user record with
its own daily root, scoped to whatever workspaces and agents it
legitimately needs. Keeps the model uniform.

---

## VKs in detail

### VK scope is configurable, not per-caller-deployment by default

See the open question at "How many VKs?" — the working assumption is
**Option B: one VK per environment** (`sk-bf-prod`, `sk-bf-staging` per
workspace), with `customer_id` set to the environment name.

In all options:

- VKs are set as one env var at deploy time (`BIFROST_VK`).
- Caller code never decides "which VK to use" — it's a single env-var
  lookup.
- Per-agent attribution comes from `x-bf-dim-agent-name`
  (auto-recorded in `logs.metadata`), not from VK structure.
- Per-deployment attribution comes from `x-bf-dim-deployment` (mcp,
  goose, chat, etc.), not from VK structure. (Optional with Option C,
  required with B/A/D since the deployment isn't encoded in the VK.)
- Per-agent caps come from spawner-set macaroon caveats, not VK.

### VK governance (set via reconcile script)

The shape of governance differs across the options; the numbers below
assume **Option B (per-environment VKs)**.

| Field                          | Typical value (Option B)                                 | Why                                                  |
| ------------------------------ | -------------------------------------------------------- | ---------------------------------------------------- |
| `budget`                       | $5000/day for `sk-bf-prod`, $200/day for `sk-bf-staging` | Env-wide ceiling against runaway anywhere in the env |
| `rate_limit.request_max_limit` | 1000 RPM (prod), 60 RPM (staging)                        | Env-wide backstop                                    |
| `rate_limit.token_max_limit`   | 5M TPM                                                   | Backstop                                             |
| `provider_configs`             | `[anthropic, openai, openrouter, gemini]` all `["*"]`    | Permissive by default                                |
| `allowed_models`               | `["*"]` initially                                        | Tighten per deployment when there's a reason         |
| `is_active`                    | `true`                                                   | Self-explanatory                                     |

The budgets here are intentionally generous. The _real_ governance
happens in the macaroon (per run, per workspace, per user). VKs are the
big-blast-radius cap that catches a buggy deployment, not the
fine-grained policy layer.

### Teams and Customers

Skip both initially. With one Bifrost per org, "Customer = org" is
fixed and trivial. Teams (`prod` / `staging` containing all of an
environment's VKs) become useful only when we want a single
environment-wide ceiling distinct from individual VK budgets. Add then.

### Bifrost config flags

```yaml
enforce_auth_on_inference: true # required: reject calls without a VK
# (default false during initial rollout; flipped to true once all
#  callers participate)
```

---

## Plugin in detail

### Distribution

One small Go repo containing the plugin source plus a Dockerfile that
builds Bifrost from upstream with the plugin compiled in. CI rebuilds
weekly against Bifrost `main`. We deploy our own Bifrost image.

### Configuration

Plugin reads on startup (env or config file):

```yaml
hmac_root_key: ${HMAC_ROOT_KEY} # for macaroon verification
hard_ceiling_cost_usd: 1000.00 # per-run cap backstop
hard_ceiling_steps: 10000 # per-run cap backstop
redis_url: redis://...
tool_loop:
  window: 10
  threshold: 8
```

Pricing data and per-call cost computation are owned by Bifrost's
pricing manager; we don't duplicate either. Per-workspace and per-user
daily caps are not in this config because they're not enforced inline
(see step 8) — they're alerting concerns, queried against the logs
table on whatever cadence makes sense.

### Redis schema

Redis holds **only hot enforcement state**. All historical attribution
queries (per-workspace, per-user, per-agent, per-day) go against
Bifrost's `logs` table — see "Observability" below.

```
cost:run:<run_id>                  HASH  { total: float }
steps:run:<run_id>                 HASH  { total: int   }
tools:run:<run_id>                 LIST  ["bash", "read", ...] (capped at 10)
kill:<run_id>                      STRING "1"
revoke:<nonce>                     STRING "1" with TTL
revoke_user_before:<uid>           STRING <iso8601 timestamp>
```

All run/tool keys expire after 1h. Revocation keys expire at the
referenced macaroon's `exp`.

### Failure modes

| Failure                  | Behavior                                                                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Redis down               | **Fail closed** for macaroon checks. **Fail open** for accounting (log loudly; spend goes uncounted; alerting fires). Auth-correctness wins over availability. |
| Auth service down        | Existing daily roots and invocation macaroons keep working until they expire. No new spawns possible.                                                          |
| HMAC key rotation needed | Hot rotation supported via `kid` caveat (multi-key verifier). Implement when first rotation is needed; not in v1.                                              |
| Plugin panic             | Bifrost falls back to whatever the panic-isolation policy is. Configure: fail-closed (reject the request).                                                     |

---

## Threat model: what each credential is worth if stolen

| Credential            | Sensitivity                     | Where it lives              | Blast radius                                                                                 |
| --------------------- | ------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------- |
| HMAC root signing key | **Crown jewel**                 | Auth service + plugin only  | Forge any macaroon, full impersonation                                                       |
| Daily root macaroon   | High                            | Server-side session store   | All of alice's permissions for ≤24h                                                          |
| Invocation macaroon   | Low                             | Agent process, run lifetime | One agent, one workspace, ≤ remaining run budget, ≤10m                                       |
| Sub-agent macaroon    | Very low                        | Sub-process                 | Narrower scope, ≤ shorter exp, ≤ smaller cap                                                 |
| VK (`sk-bf-…`)        | Low (without macaroon: useless) | Deployment env var          | With `enforce_auth_on_inference`, useless alone; otherwise: DOS against deployment's RPM cap |
| Bifrost admin key     | High                            | Ops only                    | Reconfigure VKs, change deployment budgets                                                   |
| Plugin Redis          | Medium                          | Trusted network             | Manipulate run counters → bypass per-run caps; cannot forge macaroons                        |

Without the macaroon, the VK is inert. Without the HMAC key, the
macaroon cannot be forged. The architecture deliberately makes the only
high-blast-radius credential (the HMAC key) the one that lives in the
fewest places (two: auth service and plugin).

---

## Observability (via Bifrost dimensions)

The cheapest, most flexible analytics layer in the system requires zero
custom code: Bifrost's built-in logging plugin already writes a fully
indexed row per LLM call into a SQLite `logs` table. We piggyback on
this by sending `x-bf-dim-*` headers, which Bifrost extracts and merges
into each row's `metadata` JSON column.

### Schema we get for free

| Column                                                                     | Source                                     | Use                                                   |
| -------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `timestamp`, `created_at`                                                  | Bifrost                                    | time-series slicing                                   |
| `provider`, `model`                                                        | Bifrost                                    | per-model cost rollups                                |
| `latency`                                                                  | Bifrost                                    | p50/p95 dashboards                                    |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_read_tokens` | Bifrost                                    | token accounting                                      |
| `cost`                                                                     | Bifrost (pricing manager)                  | per-call $ — no plugin code                           |
| `status`, `stop_reason`, `error_details`                                   | Bifrost                                    | reliability                                           |
| `virtual_key_id`, `virtual_key_name`                                       | Bifrost (from VK)                          | per-caller-deployment                                 |
| `customer_id`, `team_id`, `business_unit_id`                               | Bifrost governance plugin (from VK config) | env / reserved future dims                            |
| `metadata` (JSON)                                                          | **us, via `x-bf-dim-*`**                   | run_id, agent_name, workspace_id, user_id, session_id |
| `parent_request_id`                                                        | Bifrost                                    | request chain grouping                                |
| `stream`                                                                   | Bifrost                                    | streaming-vs-non                                      |

All columns above have indexes (`idx_logs_*`). The `metadata` JSON
column is queryable via SQLite's `json_extract()`.

### Example queries

```sql
-- Cost per workspace, today
SELECT json_extract(metadata, '$.workspace-id') AS ws,
       SUM(cost) AS spend
FROM logs
WHERE date(created_at) = date('now')
GROUP BY ws
ORDER BY spend DESC;

-- Cost per (user, agent) over a week
SELECT json_extract(metadata, '$.user-id')     AS user,
       json_extract(metadata, '$.agent-name')  AS agent,
       SUM(cost) AS spend, COUNT(*) AS calls
FROM logs
WHERE created_at >= datetime('now', '-7 days')
GROUP BY user, agent;

-- Latency percentiles per model for one workspace
SELECT model, AVG(latency), MAX(latency)
FROM logs
WHERE json_extract(metadata, '$.workspace-id') = 'w1'
  AND created_at >= datetime('now', '-1 day')
GROUP BY model;

-- All calls in one run
SELECT * FROM logs
WHERE json_extract(metadata, '$.run-id') = 'r_01H...'
ORDER BY created_at;
```

### Cross-Bifrost-instance reporting

Since there's one Bifrost per workspace, **per-workspace queries answer
themselves** (each instance's `logs.db` is naturally scoped to one
workspace). For org-wide reporting:

1. **Short-term:** scheduled job pulls each instance's daily slice into
   a central store (Postgres or DuckDB on a daily file dump).
2. **Long-term:** if/when this matters, swap Bifrost's logging-store
   backend to a shared Postgres or ClickHouse — Bifrost's logging plugin
   supports pluggable stores (`framework/logstore/`), so this is a
   config change, not a code change.

### When we'd need a custom plugin for observability

Never, for these basic queries. We only touch the plugin if:

- We want a derived metric written to a column Bifrost doesn't compute
  (e.g. "cost without cache reads" — write it to `metadata` ourselves).
- We need a real-time push (webhook, websocket) on certain events —
  add a `PostLLMHook` that fires the side-channel.

---

## Rollout

Phased rollout, each step independently verifiable. Throughout
steps 1–4, `enforce_auth_on_inference` stays `false`, so untagged
calls pass through. The system tightens monotonically.

**Step 1: Stand up Bifrost.** Deploy per pilot org. Existing direct
provider calls keep working (`LLM_GATEWAY_URL` unset for callers). No
behavior change for anyone. Verify Bifrost is reachable and forwards to
upstream providers correctly using its own configured keys.

**Step 2: Route MCP through Bifrost.** Set `LLM_GATEWAY_URL` in MCP's
deploy. No VK yet. Bifrost is in pass-through mode for MCP. Verify
behavior is unchanged. Look at Bifrost logs to confirm requests are
flowing through.

**Step 3: Provision VKs.** Pick a VK structure from the options table
(working assumption: Option B = per environment, so `sk-bf-prod` and
`sk-bf-staging` per workspace). Set `BIFROST_VK` in each calling
deployment. VKs are unlimited at this stage (no `budget` or
`rate_limit` set on creation). Confirm requests are tagged in Bifrost
analytics by `virtual_key_id`. If we chose B/A/D, also confirm
`x-bf-dim-deployment` is being set so per-deployment attribution still
works in logs.

**Step 4: Add VK-level governance.** Set per-VK daily budgets and RPM
rate limits via the reconcile script. Generous values appropriate to
the chosen option (e.g. $5000/day for `sk-bf-prod` under Option B,
$1000/day for `sk-bf-mcp-prod` under Option C). Watch the analytics
for a week to validate that production traffic comfortably fits inside
the caps.

**Step 5: Observability via dimensions (no plugin code).** Ship
`x-bf-dim-run-id`, `x-bf-dim-agent-name`, `x-bf-dim-realm-id`,
`x-bf-dim-user-id`, `x-bf-dim-session-id` propagation in MCP. Bifrost
auto-extracts these and writes them into `logs.metadata`. Verify with
the example SQL queries that we can slice cost/latency by every
dimension. **No plugin code change needed for this step** — the v0
boilerplate plugin is enough. This is the headline win and the place
to spend tuning time first.

**Step 6: Plugin v1 (per-run hot state, no macaroon yet).** Add Redis.
Plugin now enforces:

- per-run cost cap and step cap from plugin config (hardcoded defaults
  per agent, keyed by `x-bf-dim-agent-name`)
- tool-loop detection
- `kill:<run-id>` switch (writable from any UI/service via direct Redis)
  Macaroon verification is still a no-op. `run_id` is still read from
  the dim header (`x-bf-dim-run-id`). Watch for a week of real data;
  tune thresholds.

**Step 7: Build the auth service.** Implement `/macaroons/login`,
`/macaroons/attenuate`, `/macaroons/revoke`. Issue daily root macaroons
to logged-in sessions. Spawner code (chat BFF, MCP, etc.) calls
`/macaroons/attenuate` and stamps `x-macaroon` on outgoing LLM calls.
Plugin **logs but does not enforce** macaroon presence — observability
mode. Watch coverage climb as callers participate.

**Step 8: Plugin v2 (macaroon enforcement).** Plugin now:

- verifies macaroons
- canonicalizes the dim map from caveats (overwrites
  `BifrostContextKeyDimensions` with macaroon-derived values, so logs
  always reflect the cryptographically-verified identity)
- enforces per-run budgets _from the macaroon_ instead of plugin config
- enforces ancestor-chain budgets
- **Hard reject** on missing or invalid macaroon

This is the cutover from "observed" to "enforced". Per-workspace and
per-user daily caps are deliberately NOT enforced inline — those queries
are answered post-hoc from the `logs` table, with alerting (not inline
rejection) on breaches. Add inline checks only if a real incident shows
the alerting cadence is too slow.

**Step 9: Roll out remaining callers.** Goose, Python app, workflow
engine, Rust callers. Each picks up `LLM_GATEWAY_URL`, `BIFROST_VK`,
dim header propagation, and macaroon receipt-and-forward in turn. Until
a caller participates, its requests 401 at the plugin (post-step-8) —
this is the migration forcing function.

**Step 10: Flip `enforce_auth_on_inference: true`.** Bifrost itself now
also rejects calls without a VK. Belt-and-suspenders with the plugin's
macaroon check.

**Step 11: Observe and tune.** First month of real data drives:

- per-agent default budget tuning
- alerting thresholds for per-workspace / per-user daily spend
- whether to keep VK-level model allowlists or drop them
- whether to introduce per-Bifrost env-wide ceilings via the
  `customer_id`-as-environment slot we reserved

---

## Risks and mitigations

**Workflow engine custom headers.** If Stakwork (or whatever workflow
engine variant is in play) doesn't natively support custom headers on
its LLM nodes, that's the schedule risk in the whole plan. Mitigation:
a thin per-engine sidecar that injects headers in front of Bifrost. Not
elegant; doesn't block the rest of the rollout.

**Auth service is in the critical path.** Every agent spawn calls
`/macaroons/attenuate`. If auth service is down, spawns break.
Mitigation: stateless service, multiple replicas, health-gated load
balancer. Plugin and Bifrost don't depend on it being up; only spawning
does.

**HMAC key compromise.** Spawn fake macaroons as any user. Mitigation:
short root TTLs limit historical damage; rotation procedure ready;
secrets-service-backed delivery (not plain env) when we have one.

**Macaroon size on the wire.** A chain of 4–6 caveats is ~500 bytes;
three-level chains ~1.5 KB. No problem at HTTP scale. Monitor; switch
to compact serialization if it ever balloons.

**Spawner bugs setting absurd budgets.** Mitigation: plugin's
`hard_ceiling` backstop. A spawner that tries to authorize a $1M run
gets 402'd at the plugin regardless.

**Redis as a single point of failure for accounting.** Mitigation:
Redis Sentinel / cluster, daily backups, plus fail-closed posture on
auth checks (a stale view rejects rather than over-permits).

**Sub-agent budget accounting can fragment.** If a parent attenuates
too eagerly to children, the parent's cap may go unused while children
hit their own caps. Mitigation: parent's local accounting + plugin's
chain enforcement means the _combined_ spend is bounded by the parent
cap; how it's distributed across children is the parent's design call.

---

## Open questions

These are intentionally left open for the team chat; none of them are
blocking for steps 1–5 of the rollout.

1. **HMAC root key storage.** Plain env var for v1, secrets service for
   v2. When does v2 happen — driven by infra readiness, not this plan.

2. **Where the spawner's `AGENT_DEFAULTS` map lives.** Shared library?
   Auth service endpoint (`GET /agents/defaults`)? Per-spawner copy?
   Start with a shared TypeScript constant; centralize when the
   inconsistency starts to bite.

3. **Sub-agent budget split policy.** Parent agent decides how to
   apportion its remaining budget across children. Convention: give each
   sub-agent at most `parent.remaining * 0.5` until the parent itself
   has at least `parent.cap * 0.25` left. Document; don't enforce
   centrally.

4. **One Bifrost per workspace vs. per org.** Decision: **per workspace**
   for now. Rationale: each workspace gets its own `logs.db`, naturally
   scoped — per-workspace queries answer themselves with no JOIN. Cost:
   N Bifrost processes to run. Revisit if running >50 workspaces becomes
   operationally heavy; at that point we'd consolidate to per-org and
   add the (already-deployed) `workspace_id` dim filter to every query.

5. **Customer slot reserved for environment.** Bifrost's `customer_id`
   column will eventually be set per-VK to `prod` or `staging`. Not
   wired up yet because everything is currently `prod`. Set when the
   first staging deployment lands. (See "Bifrost gateway" table.)

   Tightly coupled to the next question.

5a. **How many VKs?** Per environment (Option B, recommended) vs per
deployment (Option C, original) vs single (Option A). See "How many
VKs?" under "Virtual Keys (VKs)". Decision pending team discussion;
parking at Option B as the working assumption. Whichever we pick
ships in step 3 of the rollout.

6. **Cross-workspace agents' cost attribution.** Agents like `org-chat`
   that legitimately span multiple workspaces. Options: (a) attribute
   to the user's home workspace, (b) attribute proportionally, (c)
   attribute to a dedicated `org` pseudo-workspace. Recommendation: (c)
   for clean bookkeeping. With one-Bifrost-per-workspace, `org-chat`
   either hits its own dedicated Bifrost (cleanest) or the user's home
   workspace's Bifrost (simpler). Decide before `org-chat` ships.

7. **Macaroon library choice.** `gopkg.in/macaroon.v2` for the Go
   plugin. TypeScript and Python need their own implementation — the
   wire format is small (a few hundred lines) and we'll have one
   library per language. Make sure all three encode identically.

8. **Service-identity macaroon issuance.** For cron, webhooks,
   schedules. Mint long-lived (e.g. 30d) daily roots for these,
   delivered to the service via the same secrets channel as everything
   else. Audit log when issued.

9. **Frontend chat refresh semantics.** When alice's daily root is 10
   minutes from expiring, the chat BFF should silently re-mint and
   update her session. Standard refresh-token pattern; just call it
   out so it's not forgotten.

10. **Future xpub layer.** If non-repudiation, cross-org federation, or
    "plugin can't mint" ever become hard requirements, we'd add
    per-request secp256k1 signing on top of (not in place of) the
    macaroon layer. Design the plugin's auth-verify as one function so
    adding `verifyXpubSig` alongside `verifyMacaroon` is a localized
    change.

---

## What this design buys

- **One gateway, many callers.** Each caller keeps its native LLM SDK.
  Only the base URL changes. Header injection is one-liner per SDK.

- **Single env var for governance enrollment.** A new caller deployment
  enrolls by setting `LLM_GATEWAY_URL` and `BIFROST_VK`. That's it.

- **Per-user, per-workspace, per-agent, per-run accounting.** All four
  dimensions, all in Bifrost's built-in `logs` SQLite table — populated
  for free via `x-bf-dim-*` headers. No custom analytics pipeline; SQL
  is the query interface. No plugin code required for observability.

- **Cryptographically scoped invocations.** Every LLM call traces to a
  specific human via the macaroon chain. Auditable, verifiable, with
  bounded blast radius if a token leaks.

- **Sub-agent containment is mathematics, not policy.** The macaroon
  protocol guarantees a child can only narrow, never widen, what the
  parent passed it. The plugin enforces the chain. A buggy or
  compromised sub-agent literally cannot spend more than its parent
  was authorized to spend.

- **Spawner-set per-run budgets.** The entity with the user, workspace,
  and agent context decides the cap. Plugin enforces what the macaroon
  says, with a generous backstop. Budgets adjust at the speed of
  spawner code, not platform redeploys.

- **Mid-loop kill, sub-second.** A "stop this agent" button anywhere
  in the UI works via one Redis key. Works across every caller language
  uniformly because all of them route through the plugin.

- **Reversible, additive rollout.** Every step is an upgrade, not a
  cutover. Bifrost without VKs works. VKs without macaroons work.
  Macaroons without per-workspace caps work. Each layer turns on when
  it's ready.

---

## Summary table

| Capability                                               | Mechanism                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Single gateway host; SDKs keep native protocols          | Bifrost + `LLM_GATEWAY_URL` + provider-specific drop-in paths                                           |
| Caller-deployment identification                         | Per-deployment Virtual Key (`sk-bf-…`)                                                                  |
| Org-wide $ / RPM caps per deployment                     | Bifrost VK budget + rate_limit                                                                          |
| Model allowlist per deployment                           | Bifrost VK `allowed_models` (optional)                                                                  |
| Identification of the human who triggered a run          | Macaroon (`x-macaroon`) — step 8+                                                                       |
| Scope: workspaces, agents, exp                           | Macaroon caveats                                                                                        |
| **Per-run $ budget (spawner-set)**                       | **Macaroon `max_cost_usd` caveat**                                                                      |
| **Per-run step / wallclock budget**                      | **Macaroon `max_steps` / `max_wallclock_s` caveats**                                                    |
| **Sub-agent budget bounded by parent**                   | **Local attenuation + plugin chain enforcement**                                                        |
| Per-workspace / per-user / per-agent / per-day analytics | Bifrost `logs` SQLite table; populated via `x-bf-dim-*` headers; queried with SQL                       |
| Per-workspace daily $ alert                              | Query the logs table on a schedule; alert on threshold                                                  |
| Tool-loop detection                                      | Plugin Redis tool history + threshold                                                                   |
| Mid-loop kill switch                                     | Plugin checks `kill:<run-id>` in Redis                                                                  |
| Caller obligation (today)                                | Set `BIFROST_VK`; send `x-bf-dim-{run-id,agent-name,workspace-id,user-id,session-id}` on every LLM call |
| Caller obligation (step 8+)                              | Above, plus `x-macaroon`                                                                                |
| Spawner obligation                                       | Attenuate macaroon with `max_cost_usd` + run scope; pass to spawned process                             |
