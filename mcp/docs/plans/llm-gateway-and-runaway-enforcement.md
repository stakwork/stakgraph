# LLM Gateway + Per-Run Enforcement

## Goal

Route every LLM call from every caller (TS, Rust, Python, Stakwork, Goose,
in-app chat) through a single gateway that can:

1. Enforce per-agent daily/monthly spend caps (governance).
2. Enforce per-agent rate limits (RPM, TPM).
3. Restrict each agent to specific providers/models.
4. **Kill a single agent run mid-loop** if it goes off the rails — cost
   runaway, infinite tool loop, stuck process.

Bifrost is the chosen gateway. Enforcement is the goal, not observability.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Callers: TS (MCP), Rust, Python, Stakwork, Goose, app    │
│  Each speaks its NATIVE provider protocol — unchanged.     │
│  Only the base_url is swapped to a Bifrost drop-in path:   │
│    Anthropic SDK → http://bifrost/anthropic                │
│    OpenAI SDK    → http://bifrost/openai                   │
│    Google GenAI  → http://bifrost/genai                    │
│    OpenRouter    → http://bifrost/openai  (escape hatch    │
│                     for any model not natively supported)  │
│  Auth header is the same one the native SDK already sends, │
│  but its value is the Bifrost VK (sk-bf-…):                │
│    Anthropic → x-api-key: sk-bf-…                          │
│    OpenAI    → Authorization: Bearer sk-bf-…               │
│    Google    → x-goog-api-key: sk-bf-…                     │
│  Plus, on every request:                                   │
│    x-session-id: <persistent conversation id>              │
│    x-run-id:     <fresh uuid per agent invocation>         │
└──────────────────────┬─────────────────────────────────────┘
                       ▼
              ┌────────────────────────────┐
              │       BIFROST GATEWAY      │
              │                            │
              │  Identity layer (built-in):│
              │   - Virtual Keys           │
              │   - VK budget / rate limit │
              │   - Team / Customer budgets│
              │                            │
              │  Run layer (custom plugin):│
              │   - per-run cost cap       │
              │   - per-run step cap       │
              │   - tool-loop detection    │
              │   - external kill switch   │
              │   - backed by Redis        │
              └────────────────┬───────────┘
                               ▼
              ┌────────────────────────────┐
              │   Provider APIs (direct)   │
              │ Anthropic / OpenAI / etc.  │
              └────────────────────────────┘
```

Two enforcement planes, one gateway:

| Plane | Scope | Tool |
|---|---|---|
| Identity | per agent × env × workspace (long-lived) | Bifrost Virtual Keys |
| Event | per single agent invocation (ephemeral) | Bifrost Go plugin + Redis |

---

## Provider strategy & the OpenRouter escape hatch

Bifrost has first-class drop-in paths for the providers we actually care
about — Anthropic, OpenAI, Google GenAI. Callers using one of these native
SDKs just swap their `base_url` and keep all provider-native features
(Anthropic thinking blocks, cache control, Google `thinkingBudget`, etc.).

For everything else — Kimi, Mistral, DeepSeek, Llama variants, anything
new that drops next month — we use **OpenRouter as the escape hatch**.
OpenRouter speaks the OpenAI protocol and exposes hundreds of models
behind it. The flow:

```
caller (AI SDK / Python / Goose / Rust)
   ↓  speaks OpenAI protocol
   ↓  base_url = http://bifrost/openai
Bifrost (governance, plugin, VK)
   ↓  forwards to OpenRouter
OpenRouter
   ↓  routes to the actual underlying provider
Moonshot / Mistral / DeepSeek / Together / etc.
```

Our existing `mcp/src/aieo/src/provider.ts` already wires `openrouter`
through the same `/openai/v1` Bifrost path for this reason.

Implications for the rest of the plan:

- **Three native paths + one catch-all** is the full surface. We do not
  need to add a Bifrost drop-in path per niche provider.
- Anything routed via OpenRouter loses provider-native fields (no
  Anthropic thinking, no Google `thinkingConfig`). That's an acceptable
  tradeoff for "long tail" models; mainline agents stick to the native
  paths.
- For governance: a VK whose `allowed_models` includes
  `moonshotai/kimi-k2.6` will accept that model name on the
  `/openai/v1` path and Bifrost will route it via OpenRouter
  transparently.
- For the plugin: it sees every request regardless of which upstream
  path is used, so per-run enforcement works uniformly for native and
  OpenRouter traffic.

---

## Phase 1 — Gateway integration

Already partially done. `mcp/src/aieo/src/provider.ts` reads `LLM_GATEWAY_URL`
and routes each provider through Bifrost's drop-in paths when set:

```
LLM_GATEWAY_URL=http://bifrost:8080
  → anthropic SDK  uses http://bifrost:8080/anthropic/v1   (native)
  → openai    SDK  uses http://bifrost:8080/openai/v1      (native)
  → google    SDK  uses http://bifrost:8080/genai/v1beta   (native)
  → openrouter SDK uses http://bifrost:8080/openai/v1      (escape hatch:
                                                            any model
                                                            OpenRouter
                                                            supports)
```

### Remaining work

- Apply the same `LLM_GATEWAY_URL` pattern to:
  - Rust callers (HTTP client base URL)
  - Python agent callers (OpenAI client base URL)
  - Goose (config: model host override)
  - Stakwork workflow LLM nodes
- Set `enforce_auth_on_inference: true` in Bifrost so requests without a VK
  are rejected.

---

## Phase 2 — Identity / governance via Virtual Keys

Three-tier hierarchy mapped to our axes:

```
Customer = workspace        ← workspace-wide ceiling
  Team   = environment      ← prod vs staging
    VK   = agent            ← per-agent daily $ + RPM + model allowlist
```

Why this assignment:

- **Agent = VK** because VKs are the only tier with rate limits (needed
  for RPM caps) and the only tier that can optionally restrict models
  or providers when we want to.
- **Env = Team** so prod and staging have separate shared ceilings.
- **Workspace = Customer** because workspace is the billing boundary.

### Policy as code

Single source of truth in the repo, applied by a reconcile script:

```ts
// agents-policy.ts
//
// Each agent gets a daily $ cap per environment and an RPM sanity cap.
// Model choice is the agent's runtime decision — Bifrost VKs accept any
// configured model by default. Add `models: [...]` only when we have a
// specific reason to pin (e.g. cost ceiling per call, compliance, a
// time-boxed experiment).
export const AGENT_POLICIES = {
  test:    { daily: { prod: 50,  staging: 10 }, rpm: 20 },
  build:   { daily: { prod: 200, staging: 30 }, rpm: 10 },
  browser: { daily: { prod: 30,  staging: 5  }, rpm: 5  },
  learn:   { daily: { prod: 75,  staging: 15 }, rpm: 15 },
  dream:   { daily: { prod: 100, staging: 20 }, rpm: 10 },
  explore: { daily: { prod: 300, staging: 50 }, rpm: 30 },
  repair:  { daily: { prod: 150, staging: 25 }, rpm: 15 },
  // ...
};
```

Reconcile script reads this file and `POST/PUT /api/governance/virtual-keys`
to Bifrost. Run on:

- New workspace created (provision full set of VKs)
- Policy change (push deltas)
- New agent type added (create VK in every workspace)

VKs are persistent — one per `(workspace, env, agent)` tuple. Not created
per run.

### Caller integration

Each caller process is launched with its VK as an env var:

```bash
BIFROST_VK=sk-bf-prod-acme-browser-xyz
```

In `mcp/src/aieo/src/provider.ts` the VK is passed as the `apiKey` to each
provider factory. The AI SDK then puts that value into whatever native
auth header the underlying SDK uses — Bifrost accepts it on all three:

| Native SDK | Header sent | Value |
|---|---|---|
| `@ai-sdk/anthropic` | `x-api-key` | `sk-bf-…` |
| `@ai-sdk/openai`, `@openrouter/ai-sdk-provider` | `Authorization: Bearer …` | `sk-bf-…` |
| `@ai-sdk/google` | `x-goog-api-key` | `sk-bf-…` |

The code change is one line:

```ts
const apiKey = process.env.BIFROST_VK ?? apiKeyIn ?? getApiKeyForProvider(provider);
```

Hard cutoffs gained from this phase alone:

- Per-agent daily $ → VK budget `reset_duration: "1d"`, `calendar_aligned: true`
- Per-agent RPM → VK rate_limit `request_max_limit` + `request_reset_duration: "1m"`
- Per-agent token cap → VK rate_limit `token_max_limit`
- Env-wide monthly $ → Team budget
- Workspace-wide monthly $ → Customer budget
- (Optional) Agent restricted to specific models or providers →
  VK `allowed_models` / `provider_configs`. Off by default; turn on
  per agent when there's a real reason.

On limit hit, Bifrost returns HTTP 402 `budget_exceeded` or 429
`rate_limited`. `mcp/src/repo/agent.ts` already catches and marks the
session `status: "error" | "aborted"` — just surface a friendlier message.

---

## Phase 3 — Per-run enforcement via Bifrost plugin

**The crux of the design.** VKs catch aggregate misbehavior. They don't
catch "this single invocation is in a tool loop." For that we need
per-run state, keyed by an ID every caller includes on every request.

### Run vs session

| Concept | Lifespan | Header |
|---|---|---|
| Session | Persistent conversation thread (can resume days later) | `x-session-id` |
| Run | One call to `get_context()` / one user "send" / one agent invocation | `x-run-id` |

A session contains 1..N runs. **Enforcement is per-run** because
runaway-loop semantics only make sense within a single invocation. Reusing
`sessionId` as `runId` would mean today's runaway-detection state poisons
tomorrow's follow-up message.

### What the plugin does

Bifrost custom Go plugin with `PreHook` and `PostHook`:

```
PreHook(req):
    run_id = req.headers["x-run-id"]
    if not run_id:
        return pass  # fail-open if caller doesn't participate

    state = redis.HGETALL("run:" + run_id)
    if redis.EXISTS("kill:" + run_id):
        return error(402, "run_killed", "Run terminated by external request")
    if float(state.cost) > config.max_cost_for_agent(req):
        return error(402, "run_cost_exceeded")
    if int(state.count) > config.max_steps_for_agent(req):
        return error(402, "run_step_exceeded")
    if loop_detected(state.recent_tools):
        return error(402, "tool_loop_detected")

    return pass

PostHook(req, resp):
    run_id = req.headers["x-run-id"]
    if not run_id:
        return

    cost = compute_cost(resp.usage, req.model)
    tool = first_tool_name(resp)
    redis.pipeline:
        HINCRBYFLOAT  "run:"+run_id  "cost"   cost
        HINCRBY       "run:"+run_id  "count"  1
        LPUSH         "run:"+run_id+":tools"  tool
        LTRIM         "run:"+run_id+":tools"  0 9
        EXPIRE        "run:"+run_id           3600
        EXPIRE        "run:"+run_id+":tools"  3600
```

### How the kill happens mid-loop

The plugin **cannot interrupt a request in flight**. What it does:

1. Agent makes tool call → tool returns
2. Agent calls model (next step in loop)
3. Request hits Bifrost → plugin `PreHook` checks state → returns 402
4. AI SDK in agent throws on 402
5. Agent loop terminates (`agent.ts:489-502` already catches and persists)

Kill latency = "until next model call boundary." That's the tightest
possible kill — once a request is in flight to Anthropic, you can't recall
it. The plugin blocks **before** any further spend.

### Bonus: external kill switch

Once per-run state lives in Redis, app code can kill any run by setting a
Redis key:

```
SET kill:<run-id> 1 EX 3600
```

The plugin checks this on every PreHook. Result: a working "Stop this
agent" button in the UI that works across every language and runtime,
because all roads lead through Bifrost.

### Detection thresholds

Start dumb. Two hardcoded thresholds, ship it, tune from a week of data:

```
max_run_cost      = $5.00      (overridable per-agent via x-agent-name)
max_run_steps     = 100
tool_loop_window  = last 10 tool calls
tool_loop_trigger = same tool name >= 8 of last 10
```

Later additions:

- Hash recent tool *arguments*, not just names ("called `bash` with `ls /` 14×")
- Per-agent overrides via config map keyed on `x-agent-name`
- Wall-clock TTL ("run alive > 5 minutes")

### Plugin trade-offs

| Pro | Con |
|---|---|
| Sees every caller uniformly (TS, Rust, Python, Goose, Stakwork) | New Go code to maintain |
| Sub-millisecond overhead (two Redis ops per request) | Rebuild against new Bifrost releases |
| Catches semantic signals VKs can't (cost, steps, tool repetition) | Plugin only sees what Bifrost sees (no tool arguments yet) |
| Free kill-switch API via Redis | Redis becomes a soft dependency (fail-open) |
| Auto-cleanup via Redis TTL | — |

---

## Phase 4 — Caller header propagation

The plugin only works if every caller sends `x-run-id`. This is the only
piece of work that touches each codebase.

### TS / MCP (`mcp/src/repo/agent.ts` + `mcp/src/aieo/src/provider.ts`)

- Mint `runId = crypto.randomUUID()` at the top of `get_context()` and
  `stream_context()`. Already have `sessionId`.
- Pass both into `getModelDetails()` / `getModel()`.
- In `provider.ts`, add `headers` to each `createX({...})` call:

```ts
const headers: Record<string, string> = {};
if (runId)     headers["x-run-id"] = runId;
if (sessionId) headers["x-session-id"] = sessionId;
if (agentName) headers["x-agent-name"] = agentName;

const anthropic = createAnthropic({
  apiKey,
  ...(baseURL && { baseURL }),
  ...(Object.keys(headers).length && { headers }),
});
```

All four AI SDK provider factories (`createAnthropic`, `createOpenAI`,
`createGoogleGenerativeAI`, `createOpenRouter`) accept a `headers` option
that ships on every request.

### Rust callers

One place — the shared HTTP client wrapper. Mint UUID at start of agent
invocation, set as default header on the client for that invocation.

### Python agents

```python
run_id = str(uuid4())
client = OpenAI(
    base_url=os.environ["LLM_GATEWAY_URL"] + "/openai/v1",
    api_key=os.environ["BIFROST_VK"],
    default_headers={
        "x-run-id": run_id,
        "x-session-id": session_id,
        "x-agent-name": agent_name,
    },
)
```

### Goose

Goose supports custom headers in config. One UUID per `goose run`
invocation, written to the session config at startup.

### Stakwork workflows

Mint UUID at workflow start. Propagate through the workflow execution
context to every LLM call node as a header.

### In-app chat

Existing `sessionId` as `x-session-id`. New `runId` minted each time the
user clicks send (one user turn = one run).

### Fail-open

If a caller forgets to send `x-run-id`, the plugin passes through with no
per-run enforcement. VK-level caps still apply. We tighten over time.

---

## Rollout sequence

Throughout steps 1–7, Bifrost runs with `enforce_auth_on_inference: false`,
so requests without a VK pass through (no governance applied). Hard
enforcement of VKs is the final step.

1. **Bifrost stood up** in our infra. Existing direct provider calls keep
   working (`LLM_GATEWAY_URL` unset).
2. **TS/MCP routed through Bifrost** (`LLM_GATEWAY_URL` set in deploy).
   No VK yet — requests pass through Bifrost unauthenticated. Verify
   behavior unchanged.
3. **VKs provisioned** for a single pilot workspace via reconcile script.
   Set `BIFROST_VK` for MCP server. VK budgets now active for MCP traffic;
   other callers still pass through unauthenticated.
4. **Header propagation** added in TS — `x-run-id`, `x-session-id`,
   `x-agent-name`. Visible in Bifrost logs.
5. **Plugin v1 shipped** — hardcoded thresholds, cost + step cap only.
   Watch for kill events for a week.
6. **Plugin v2** — tool-loop detection, per-agent threshold overrides,
   external kill-switch endpoint.
7. **Roll out to remaining callers** — Rust, Python, Goose, Stakwork —
   one at a time, each picking up `LLM_GATEWAY_URL`, a `BIFROST_VK`, and
   header propagation.
8. **Enforce VK** — set `enforce_auth_on_inference: true` once all
   callers participate.

---

## What we explicitly are NOT building

- A separate Postgres budget table. Bifrost owns budget state.
- A Langfuse/Helicone observability stack. Bifrost analytics + our session
  records are enough for now; revisit if outgrown.
- LLM-as-judge policies. Hard caps only. Per the requirements: enforcement
  is the goal.
- Per-run virtual keys. VKs are identities, runs are events. Ephemeral VKs
  are the wrong abstraction (provisioning latency, cleanup burden, hierarchy
  collision with daily caps). Per-run state lives in Redis via the plugin.
- A custom Bifrost plugin for governance. VK + budget + rate_limit cover
  the identity layer natively.

---

## Open questions

- **Plugin distribution.** Do we fork Bifrost and add the plugin to its
  build, or ship as a sidecar? Bifrost's plugin docs prefer in-tree.
  Cleanest path: small repo with the plugin source + Dockerfile that
  builds Bifrost from upstream with our plugin compiled in. Weekly CI
  rebuild against `main`.
- **Redis topology.** Reuse Bifrost's existing Redis (single instance,
  shared) or a dedicated one for per-run state? Start shared.
- **Failure semantics.** If Redis is down: plugin fails open, logs loud,
  Slack alert. VK caps still enforced.
- **Cost computation.** Plugin needs a model→pricing table. Either hardcode
  in plugin, sync from `mcp/src/aieo/src/provider.ts:TOKEN_PRICING`, or read
  from Bifrost's own pricing config. Probably the third.
- **Per-agent threshold config.** Where does the plugin read agent-specific
  caps from? Env var with JSON blob? Config endpoint? Start with a flat env
  var, add a config endpoint when it gets unwieldy.

---

## Summary

| Capability | Mechanism |
|---|---|
| Single gateway host for every caller; each SDK keeps its native protocol | Bifrost gateway, `LLM_GATEWAY_URL` + provider-specific drop-in paths |
| Per-agent daily/monthly $ caps | Bifrost VK budget |
| Per-agent RPM/TPM caps | Bifrost VK rate_limit |
| Per-agent model/provider allowlist (optional) | Bifrost VK `allowed_models` / `provider_configs` |
| Workspace and env ceilings | Bifrost Team + Customer budgets |
| **Per-run cost cap** | **Bifrost plugin + Redis** |
| **Per-run step cap** | **Bifrost plugin + Redis** |
| **Tool-loop detection** | **Bifrost plugin + Redis** |
| **External kill switch** | **Bifrost plugin + Redis (`kill:<run-id>`)** |
| Policy as code | `agents-policy.ts` + reconcile script |
| Caller obligation | Send `x-run-id`, `x-session-id`, `x-agent-name` on every call |
