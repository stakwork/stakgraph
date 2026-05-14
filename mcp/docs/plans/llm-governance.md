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

One Bifrost instance per organization. Sits between every LLM caller and
every upstream provider. Provider-native SDKs keep their native
protocol; only the base URL changes.

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

Bifrost issues VKs (`sk-bf-…`) as caller-deployment credentials. Not
per-user, not per-agent-type — **per-caller-deployment**:

```
sk-bf-mcp-prod         → MCP server in prod
sk-bf-mcp-staging      → MCP server in staging
sk-bf-goose-prod       → Goose runners in prod
sk-bf-workflow-prod    → Workflow engine in prod
sk-bf-pyapp-prod       → Python application in prod
sk-bf-chat-prod        → BFF that backs the frontend chat in prod
```

Each VK is set as one env var in the corresponding deployment. The
caller's existing auth header carries the VK value:

| SDK | Header it sends | Value we set it to |
|---|---|---|
| `@ai-sdk/anthropic` | `x-api-key` | `sk-bf-…` |
| `@ai-sdk/openai`, `@openrouter/ai-sdk-provider` | `Authorization: Bearer …` | `sk-bf-…` |
| `@ai-sdk/google` | `x-goog-api-key` | `sk-bf-…` |
| Python `openai` client | `Authorization: Bearer …` | `sk-bf-…` |
| Goose | provider config env vars | `sk-bf-…` |

Per-VK governance Bifrost enforces natively:

- Daily / monthly `$` budget
- RPM / TPM rate limit
- Provider allowlist
- Model allowlist (e.g. "this caller may never invoke Opus")

Per-agent-type attribution comes from a separate `x-agent-name` header,
not from minting one VK per agent type. (See "Attribution," below.)

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
parent attenuates *locally* — macaroon attenuation requires only the
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

A Bifrost custom plugin implementing `PreHook` and `PostHook`. Single
binary, deployed alongside Bifrost. Backed by Redis.

`PreHook` (auth + spending gate, runs before the upstream call):

```
1. RESOLVE CALLER IDENTITY (VK)
   - Bifrost has already auth'd the VK before the plugin runs (when
     enforce_auth_on_inference: true). Plugin reads the VK's metadata
     for caller-deployment label + env.

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

4. CHECK HEADERS MATCH CAVEATS
   - x-workspace-id ∈ caveat.workspaces?      → else 403 scope_violation
   - x-agent-name == caveat.agents (or ∈)?    → else 403 scope_violation
   - x-user-id == caveat.user_id?             → else 403 scope_violation
   (headers are convenience for attribution; macaroon is source of truth)

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

7. CHECK PER-WORKSPACE / PER-USER DAILY CAPS
   - cost:workspace:<ws>:<yyyy-mm-dd> > cap?  → 402 workspace_budget_exceeded
   - cost:user:<uid>:<yyyy-mm-dd> > cap?      → 402 user_budget_exceeded
   (caps from plugin config, refreshed periodically from an admin store)

8. CHECK TOOL-LOOP HEURISTIC
   - last 10 tool calls for run_id: same tool >= 8 of 10?
                                              → 402 tool_loop_detected
```

`PostHook` (accounting, runs after a successful upstream call):

```
1. Compute cost from response usage + model pricing.

2. Update per-run counters, walking the ancestor chain:
   for r in [run_id, ...ancestor_run_ids]:
     HINCRBYFLOAT cost:run:<r>   total $cost
     HINCRBY      steps:run:<r>  total 1
     EXPIRE       cost:run:<r>   3600
     EXPIRE       steps:run:<r>  3600

3. Update tool history for loop detection:
   LPUSH  tools:run:<run_id> <tool_name>
   LTRIM  tools:run:<run_id> 0 9
   EXPIRE tools:run:<run_id> 3600

4. Update attribution counters (headers already verified in PreHook):
   HINCRBYFLOAT cost:workspace:<ws>:<yyyy-mm-dd>     total $cost
   HINCRBYFLOAT cost:workspace:<ws>:agent:<agent>    <yyyy-mm> $cost
   HINCRBYFLOAT cost:user:<uid>:<yyyy-mm-dd>         total $cost
   HINCRBYFLOAT cost:user:<uid>:agent:<agent>        <yyyy-mm> $cost
```

Total overhead: ~6–10 Redis ops per LLM call, all pipelined.
Sub-millisecond on a co-located Redis.

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
besides the gateway plugin (which only needs it to *verify*, but in
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
        BIFROST_VK=sk-bf-mcp-prod      (caller deployment VK)
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
          "x-macaroon":     process.env.AGENT_MACAROON,
          "x-run-id":       process.env.RUN_ID,
          "x-workspace-id": process.env.WORKSPACE_ID,
          "x-user-id":      process.env.USER_ID,
          "x-agent-name":   process.env.AGENT_NAME,
          "x-session-id":   <persistent conversation id>,
        }
   c. Runs agent loop. Each model call ships with all the above.

4. BIFROST GATEWAY
   a. Authenticates VK (sk-bf-mcp-prod). Looks up MCP-prod metadata.
   b. Invokes plugin PreHook.

5. PLUGIN PreHook
   - Verifies macaroon (HMAC + caveats).
   - Confirms headers match caveats.
   - Reads cost:run:r_01H.. → currently $3.21, cap is $5.00. Passes.
   - Reads steps:run:r_01H.. → currently 14, cap is 100. Passes.
   - No kill, no loop, no workspace cap blown.
   - Returns pass.

6. BIFROST
   - Strips the VK, substitutes the real Anthropic key from its config.
   - Forwards to api.anthropic.com.

7. ANTHROPIC
   - Returns model response.

8. BIFROST → plugin PostHook
   - Cost = $0.42 from response usage.
   - HINCRBYFLOAT cost:run:r_01H..  total 0.42 → now $3.63
   - HINCRBY      steps:run:r_01H.. total 1     → now 15
   - LPUSH        tools:run:r_01H.. "<tool>"
   - HINCRBYFLOAT cost:workspace:w1:2026-05-13 total 0.42
   - HINCRBYFLOAT cost:user:u_01H..:2026-05-13 total 0.42
   - (and per-agent breakdowns)

9. RESPONSE to agent → next step in the loop.
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

Every LLM call from every caller arrives at Bifrost with these headers.

```
Authorization: Bearer sk-bf-<caller-deployment>     ← VK (Bifrost auth)
   or x-api-key / x-goog-api-key as appropriate for the SDK

x-macaroon:     <invocation macaroon>               ← identity + scope
x-run-id:       <uuid>                              ← this invocation
x-session-id:   <persistent conversation id>        ← long-lived thread
x-agent-name:   browser                             ← attribution
x-workspace-id: w1                                  ← attribution
x-user-id:      u_01H...                            ← attribution
```

The macaroon is the source of truth for identity and scope. The
redundant attribution headers exist so the plugin's PostHook can stamp
counters without parsing the macaroon body on every write. The plugin
PreHook verifies that each redundant header matches the corresponding
caveat — tampering with a header without re-signing the macaroon is
detected.

**Required for any call past plugin v1:** `Authorization` (VK),
`x-macaroon`, `x-run-id`.

**Required-but-derivable:** `x-agent-name`, `x-workspace-id`,
`x-user-id`. The plugin can fall back to parsing the macaroon if these
are absent; setting them is just a perf optimization.

**Optional:** `x-session-id`. Used for grouping in analytics; no
enforcement decisions depend on it.

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
  "x-macaroon":     process.env.AGENT_MACAROON,
  "x-run-id":       process.env.RUN_ID,
  "x-session-id":   sessionId,
  "x-agent-name":   process.env.AGENT_NAME,
  "x-workspace-id": process.env.WORKSPACE_ID,
  "x-user-id":      process.env.USER_ID,
};

const anthropic = createAnthropic({
  apiKey:  process.env.BIFROST_VK,
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
        "x-macaroon":     os.environ["AGENT_MACAROON"],
        "x-run-id":       os.environ["RUN_ID"],
        "x-agent-name":   os.environ["AGENT_NAME"],
        "x-workspace-id": os.environ["WORKSPACE_ID"],
        "x-user-id":      os.environ["USER_ID"],
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

Each entity that *spawns* an agent is responsible for:

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
     const def = AGENT_DEFAULTS[ctx.agent];   // { cost, steps, wallclock }
     return {
       max_cost_usd:    Math.min(
                          def.cost,
                          ctx.user.daily_remaining_usd,
                          ctx.workspace.daily_remaining_usd,
                          ctx.parent_remaining?.cost ?? Infinity,
                        ),
       max_steps:       def.steps,
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

### VK is per-caller-deployment

Not per agent type, not per user, not per workspace. Per deployment.
This is the simplest assignment that supports the rest of the design:

- One VK per Bifrost-calling process.
- Set as one env var at deploy time.
- Caller code never decides "which VK to use."

Per-agent attribution comes from `x-agent-name` and per-agent caps come
from spawner-set macaroon caveats. Neither needs a separate VK.

### VK governance (set via reconcile script)

| Field | Typical value | Why |
|---|---|---|
| `budget` | $1000/day for prod-MCP, $100/day for prod-Goose | Org-wide ceiling against runaway deployment |
| `rate_limit.request_max_limit` | 60 RPM | Backstop against deployment-wide stampede |
| `rate_limit.token_max_limit` | 5M TPM | Backstop |
| `provider_configs` | `[anthropic, openai, openrouter, gemini]` all `["*"]` | Permissive by default |
| `allowed_models` | `["*"]` initially | Tighten per deployment when there's a reason |
| `is_active` | `true` | Self-explanatory |

The budgets here are intentionally generous. The *real* governance
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
enforce_auth_on_inference: true     # required: reject calls without a VK
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
hmac_root_key:           ${HMAC_ROOT_KEY}     # for macaroon verification
hard_ceiling_cost_usd:   1000.00              # per-run cap backstop
hard_ceiling_steps:      10000                # per-run cap backstop
workspace_daily_caps:    { default: 100.00 }  # refreshed every 60s
user_daily_caps:         { default:  50.00 }  # refreshed every 60s
redis_url:               redis://...
pricing_table:           ${PRICING_JSON}      # model → $/1k tokens
tool_loop:
  window:   10
  threshold: 8
```

`workspace_daily_caps` and `user_daily_caps` are pulled from an admin
store (initially: an env-var JSON blob; later: a small admin API).

### Redis schema

```
cost:run:<run_id>                  HASH  { total: float }
steps:run:<run_id>                 HASH  { total: int   }
tools:run:<run_id>                 LIST  ["bash", "read", ...] (capped at 10)
kill:<run_id>                      STRING "1"
cost:workspace:<ws>:<yyyy-mm-dd>   HASH  { total, agent:<name>: float }
cost:user:<uid>:<yyyy-mm-dd>       HASH  { total, agent:<name>: float }
cost:user:<uid>:agent:<name>:<yyyy-mm>  STRING float
revoke:<nonce>                     STRING "1" with TTL
revoke_user_before:<uid>           STRING <iso8601 timestamp>
```

All run/tool keys expire after 1h. Daily attribution keys expire after
40 days (room for monthly reporting).

### Failure modes

| Failure | Behavior |
|---|---|
| Redis down | **Fail closed** for macaroon checks. **Fail open** for accounting (log loudly; spend goes uncounted; alerting fires). Auth-correctness wins over availability. |
| Auth service down | Existing daily roots and invocation macaroons keep working until they expire. No new spawns possible. |
| HMAC key rotation needed | Hot rotation supported via `kid` caveat (multi-key verifier). Implement when first rotation is needed; not in v1. |
| Plugin panic | Bifrost falls back to whatever the panic-isolation policy is. Configure: fail-closed (reject the request). |

---

## Threat model: what each credential is worth if stolen

| Credential | Sensitivity | Where it lives | Blast radius |
|---|---|---|---|
| HMAC root signing key | **Crown jewel** | Auth service + plugin only | Forge any macaroon, full impersonation |
| Daily root macaroon | High | Server-side session store | All of alice's permissions for ≤24h |
| Invocation macaroon | Low | Agent process, run lifetime | One agent, one workspace, ≤ remaining run budget, ≤10m |
| Sub-agent macaroon | Very low | Sub-process | Narrower scope, ≤ shorter exp, ≤ smaller cap |
| VK (`sk-bf-…`) | Low (without macaroon: useless) | Deployment env var | With `enforce_auth_on_inference`, useless alone; otherwise: DOS against deployment's RPM cap |
| Bifrost admin key | High | Ops only | Reconfigure VKs, change deployment budgets |
| Plugin Redis | Medium | Trusted network | Manipulate run counters → bypass per-run caps; cannot forge macaroons |

Without the macaroon, the VK is inert. Without the HMAC key, the
macaroon cannot be forged. The architecture deliberately makes the only
high-blast-radius credential (the HMAC key) the one that lives in the
fewest places (two: auth service and plugin).

---

## Attribution and reporting

All cost-attribution queries are answerable from Redis without a
separate analytics pipeline:

**From plugin Redis counters:**

- "Top 10 most expensive workspaces this month"
- "Cost per agent within workspace w1"
- "How much did alice spend on `browser` this week"
- "How much is `org-chat` costing across all workspaces it touches"
- "Cost per (user, agent) for chargeback or quota enforcement"

**From Bifrost native VK analytics:**

- "How much did the prod-MCP deployment spend today"
- "RPM hits per caller deployment"
- "Cost split across providers per deployment"

**From cross-joining the two (offline, daily roll-up):**

- "Which agent types are costing the most org-wide" — sum the
  `agent:<name>` slice across workspaces.
- "Spend per user per agent per workspace per day" — full
  four-dimensional cube, all already in Redis.

If we outgrow Redis-as-analytics, the daily roll-up dumps to Postgres
or a warehouse. Plugin code doesn't change.

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

**Step 3: Provision the default deployment VKs.** Use the reconcile
script to create `sk-bf-mcp-prod`, `sk-bf-mcp-staging`, etc. Set
`BIFROST_VK` for the MCP deployment. VKs are unlimited at this stage
(no `budget` or `rate_limit` set on creation). Confirm requests are
tagged in Bifrost analytics.

**Step 4: Add VK-level governance.** Set per-VK daily budgets and RPM
rate limits via the reconcile script. Generous values (e.g. $1000/day
for prod-MCP). Watch the analytics for a week to validate that
production traffic comfortably fits inside the caps.

**Step 5: Build the plugin v1 (per-run state only, no macaroon yet).**
Ship `x-run-id` header propagation in MCP. Plugin v1 enforces per-run
cost cap and step cap from plugin config (hardcoded defaults per
agent), plus tool-loop detection and the `kill:<run-id>` switch.
Macaroon verification is a no-op for now. Watch for a week of real
data; tune thresholds.

**Step 6: Build the auth service.** Implement `/macaroons/login`,
`/macaroons/attenuate`, `/macaroons/revoke`. Issue daily root macaroons
to logged-in sessions. Spawner code (chat BFF, MCP, etc.) calls
`/macaroons/attenuate` and stamps `x-macaroon` on outgoing LLM calls.
Plugin **logs but does not enforce** macaroon presence — observability
mode. Watch coverage climb as callers participate.

**Step 7: Plugin v2 (macaroon enforcement).** Plugin now verifies
macaroons, walks caveats, enforces per-run budgets *from the macaroon*
instead of plugin config, enforces ancestor-chain budgets, enforces
per-workspace and per-user daily caps. **Hard reject** on missing or
invalid macaroon. This is the cutover from "observed" to "enforced."

**Step 8: Roll out remaining callers.** Goose, Python app, workflow
engine, Rust callers. Each picks up `LLM_GATEWAY_URL`, `BIFROST_VK`,
header propagation, and macaroon receipt-and-forward in turn. Until a
caller participates, its requests 401 at the plugin — this is the
migration forcing function.

**Step 9: Flip `enforce_auth_on_inference: true`.** Bifrost itself now
also rejects calls without a VK. Belt-and-suspenders with the plugin's
macaroon check.

**Step 10: Observe and tune.** First month of real data drives:

- per-workspace and per-user daily cap tuning
- per-agent default budget tuning
- whether to keep VK-level model allowlists or drop them
- whether to introduce Teams for env-wide ceilings

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
chain enforcement means the *combined* spend is bounded by the parent
cap; how it's distributed across children is the parent's design call.

---

## Open questions

These are intentionally left open for the team chat; none of them are
blocking for steps 1–4 of the rollout.

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

4. **Per-workspace cap source.** Plugin reads from env JSON for v1.
   Move to a small admin API when there are enough workspaces that
   editing JSON is annoying.

5. **Cross-workspace agents' cost attribution.** Agents like `org-chat`
   that legitimately span multiple workspaces. Options: (a) attribute
   to the user's home workspace, (b) attribute proportionally, (c)
   attribute to a dedicated `org` pseudo-workspace. Recommendation: (c)
   for clean bookkeeping. Decide before `org-chat` ships.

6. **Macaroon library choice.** `gopkg.in/macaroon.v2` for the Go
   plugin. TypeScript and Python need their own implementation — the
   wire format is small (a few hundred lines) and we'll have one
   library per language. Make sure all three encode identically.

7. **Service-identity macaroon issuance.** For cron, webhooks,
   schedules. Mint long-lived (e.g. 30d) daily roots for these,
   delivered to the service via the same secrets channel as everything
   else. Audit log when issued.

8. **Frontend chat refresh semantics.** When alice's daily root is 10
   minutes from expiring, the chat BFF should silently re-mint and
   update her session. Standard refresh-token pattern; just call it
   out so it's not forgotten.

9. **Future xpub layer.** If non-repudiation, cross-org federation, or
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
  dimensions, all in one Redis store the plugin already writes for
  other reasons. No analytics pipeline needed for v1.

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

| Capability | Mechanism |
|---|---|
| Single gateway host; SDKs keep native protocols | Bifrost + `LLM_GATEWAY_URL` + provider-specific drop-in paths |
| Caller-deployment identification | Per-deployment Virtual Key (`sk-bf-…`) |
| Org-wide $ / RPM caps per deployment | Bifrost VK budget + rate_limit |
| Model allowlist per deployment | Bifrost VK `allowed_models` (optional) |
| Identification of the human who triggered a run | Macaroon (`x-macaroon`) |
| Scope: workspaces, agents, exp | Macaroon caveats |
| **Per-run $ budget (spawner-set)** | **Macaroon `max_cost_usd` caveat** |
| **Per-run step / wallclock budget** | **Macaroon `max_steps` / `max_wallclock_s` caveats** |
| **Sub-agent budget bounded by parent** | **Local attenuation + plugin chain enforcement** |
| Per-workspace daily $ cap | Plugin Redis counter + config-driven cap |
| Per-user daily $ cap | Plugin Redis counter + config-driven cap |
| Tool-loop detection | Plugin Redis tool history + threshold |
| Mid-loop kill switch | Plugin checks `kill:<run-id>` in Redis |
| Per-agent / per-workspace / per-user analytics | Plugin Redis attribution counters |
| Caller obligation | Send `x-macaroon`, `x-run-id` (+ optional attribution headers) on every LLM call |
| Spawner obligation | Attenuate macaroon with `max_cost_usd` + run scope; pass to spawned process |
