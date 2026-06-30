# Agent Catalog

> Companion to `agent-registry.md`. That doc describes a Hive/Postgres
> `AgentDefinition` table whose job is to feed **budget caveats** to the
> macaroon issuer. This doc describes a separate, complementary thing:
> a **neo4j catalog of what each agent _is_** — its prompts, tools, and
> skills — surfaced in the gateway admin UI alongside cost/budget data.
>
> The two never conflict. The registry answers "what is this agent
> _allowed_ to spend?"; the catalog answers "what is this agent _made
> of_?". As of this writing the Hive `AgentDefinition` registry is **not
> implemented** — the macaroon issuer treats `agentName` as a free-form
> string (`hive/src/services/bifrost/macaroon-issuer.ts:80-82`). So the
> catalog is greenfield with nothing to coordinate against.

## Why this exists

The `/agents` page today is derived purely from Bifrost log traffic:
spend, call count, budget. An agent is just a string — the
`x-bf-dim-agent-name` header on each LLM call. There is no record of
**what the agent actually is**: its system prompt, the tools it can
call, the skills it loads.

That information today is scattered across systems:

- **Hive** records per-run telemetry in `agent_logs.config`, which
  already carries `{ model, provider, source, repos, tools,
  toolsConfig }` (`hive/prisma/schema.prisma:1116-1148`).
- **Stakwork prompt-manager** owns versioned prompts.
- **ai-sdk agents in this repo** define prompts/tools in code.
- **Goose CLI agents on remote pods** carry their own skills/prompts.

We want one place that answers "show me everything about this agent"
without the gateway having to learn how to scrape each heterogeneous
source.

## Approach: push, don't pull

Every source **pushes** a manifest of the agents it knows about into
the gateway. The gateway is the single write front-door; it writes the
manifest into the shared neo4j graph. The gateway UI reads it back.

```
  hive ─┐
  prompt-manager ─┤
  ai-sdk agents ──┤  POST /_plugin/agents  ──▶  gateway  ──▶  neo4j
  goose pods ─────┘   (whole fleet object)         │            (catalog)
                                                    │
  gateway UI  ◀── GET /_plugin/agents/:name/catalog ┘
```

The join key everywhere is the agent **name** — the same string the
gateway already keys cost/budget on (`x-bf-dim-agent-name`). No new
identifier is introduced.

Sources never talk to neo4j directly. Only the gateway holds the neo4j
credential and owns the upsert. This keeps the gateway thin (pure
stdlib HTTP, no bolt driver) while making it the one place to reason
about how the catalog is written.

## Neo4j access — the native HTTP Cypher endpoint

The gateway speaks to neo4j over its **native transactional Cypher HTTP
API**, not over bolt and not through mcp:

```
POST http://neo4j.sphinx:7474/db/neo4j/tx/commit
Authorization: Basic base64(NEO4J_USER:NEO4J_PASSWORD)
Content-Type: application/json

{ "statements": [ { "statement": "<cypher>", "parameters": { ... } } ] }
```

This is authenticated with the **neo4j password** (Basic auth), which
already lives in the swarm env — distinct from mcp's `API_TOKEN` (which
gates mcp's `:3355` API) and distinct from the gateway's own
`BIFROST_PROVISIONING_TOKEN`. Using neo4j's own endpoint means the
gateway needs neither the Go bolt driver nor a hop through mcp.

### New env vars (gateway `internal/env`)

| Var | Default | Purpose |
| --- | --- | --- |
| `NEO4J_HTTP_URL` | `http://neo4j.sphinx:7474` | base URL of neo4j's HTTP API. Note the **7474** HTTP port, not 7687 (bolt). |
| `NEO4J_USER` | `neo4j` | Basic-auth user (reused name from mcp/standalone). |
| `NEO4J_PASSWORD` | _(unset)_ | Basic-auth password. |
| `NEO4J_DATABASE` | `neo4j` | path segment in `/db/<db>/tx/commit`. |

Follow the existing `env.go` contract: a `Neo4jHTTPConfig() (cfg, ok)`
getter where `ok=false` when `NEO4J_PASSWORD` is unset. When not
configured, the catalog endpoints return `503` and the UI hides the
prompts/tools/skills tabs — exactly the graceful-degradation pattern
Redis uses for "observability mode".

A thin `internal/graphclient` package wraps this: `Query(cypher,
params)` and `Run(statements...)`, plus the label constants (see
below). ~100 lines, no third-party deps.

## Data model

All labels are **PascalCase with a `Hive` prefix** so they never
collide with the generic `Agent`/`Prompt`/`Tool`/`Skill` nodes other
systems may push into the shared graph. Hive nodes deliberately do
**not** carry the `Data_Bank` label (that label drives mcp's code
full-text + vector indexes; the catalog must not pollute code search).

```
(:HiveAgent  { name, display_name, description, updated_at })
   -[:HAS_PROMPT]-> (:HivePrompt { node_key, name, role, body, source, version, updated_at })
   -[:HAS_TOOL]->   (:HiveTool   { node_key, name, description, schema, source, version, updated_at })
   -[:HAS_SKILL]->  (:HiveSkill  { node_key, name, description, source, version, updated_at })

(:HiveAgent)-[:DEFINED_BY]->(:HiveSource { name, kind, updated_at })
```

Label/edge names live in one place — `graphclient/schema.go`:

```go
const (
    LabelAgent   = "HiveAgent"
    LabelPrompt  = "HivePrompt"
    LabelTool    = "HiveTool"
    LabelSkill   = "HiveSkill"
    LabelSource  = "HiveSource"

    RelHasPrompt = "HAS_PROMPT"
    RelHasTool   = "HAS_TOOL"
    RelHasSkill  = "HAS_SKILL"
    RelDefinedBy = "DEFINED_BY"
)
```

### Keys & properties

- **`HiveAgent.name`** is the merge key and the join to cost/budget
  data. Stable, URL-safe, lowercase-with-hyphens — same string as the
  `x-bf-dim-agent-name` header.
- **`node_key`** on prompt/tool/skill nodes is a deterministic hash of
  `(agent_name, source, kind, name)` so re-pushing is idempotent and a
  child belongs to exactly one (agent, source) pairing.
- **`source`** records which system contributed the node (`hive`,
  `prompt-manager`, `ai-sdk`, `goose`). Multiple sources can describe
  the same agent without clobbering each other — see reconciliation.
- **`version`** is the contributing source's own version stamp (git
  sha, ISO timestamp, prompt-manager revision id). Surfaced in the UI;
  also lets a source no-op a push that hasn't changed.
- **`schema`** on a tool is the JSON-encoded parameter schema (stored
  as a string; neo4j has no nested-object property type).

### Indexes (created on first write)

```cypher
CREATE CONSTRAINT hive_agent_name IF NOT EXISTS
  FOR (a:HiveAgent) REQUIRE a.name IS UNIQUE;
CREATE INDEX hive_prompt_key IF NOT EXISTS FOR (p:HivePrompt) ON (p.node_key);
CREATE INDEX hive_tool_key   IF NOT EXISTS FOR (t:HiveTool)   ON (t.node_key);
CREATE INDEX hive_skill_key  IF NOT EXISTS FOR (s:HiveSkill)  ON (s.node_key);
```

## Write path — `POST /_plugin/agents`

Accepts the **whole fleet as one object** (your call from the design
discussion). Bearer-auth only, against `BIFROST_PROVISIONING_TOKEN` —
same posture as the trust-registry mutations (cookie auth is for the
read-only UI; writes are server-to-server).

### Request

```jsonc
{
  "source": "hive",                  // REQUIRED: the contributing system
  "agents": [
    {
      "name": "coder-agent",         // REQUIRED: matches x-bf-dim-agent-name
      "display_name": "Coder",       // optional
      "description": "Writes and modifies code.",
      "version": "git:9f3a1c2",      // optional; source's own stamp
      "prompts": [
        { "name": "system", "role": "system", "body": "You are..." }
      ],
      "tools": [
        { "name": "read_file", "description": "Read a file",
          "schema": { "type": "object", "properties": { "path": { "type": "string" } } } }
      ],
      "skills": [
        { "name": "security-review", "description": "OWASP review",
          "source": "goose-pod-3" }   // optional per-item source override
      ]
    }
  ]
}
```

### Semantics

Per `(agent, source)` the write is **replace-by-source**: the push is
the complete picture of what *this source* knows about *this agent*, so
the gateway deletes that source's existing children for the agent and
re-creates them in one transaction. Other sources' contributions to the
same agent are untouched.

```
For each agent in body.agents:
  MERGE (:HiveAgent {name})            -- create-or-update scalar props
  MERGE (:HiveSource {name: body.source})
  MERGE (agent)-[:DEFINED_BY]->(source)
  DETACH DELETE this source's existing HAS_PROMPT/TOOL/SKILL children for this agent
  CREATE the prompt/tool/skill children from the payload, wiring node_key + source + version
```

All statements for one request go in a single `tx/commit` batch.

### Response

```jsonc
{ "written": { "agents": 6, "prompts": 11, "tools": 34, "skills": 8 } }
```

### Why "whole fleet object" and not one-agent-per-call

A source typically knows its entire fleet at once (hive enumerates its
agents; a goose pod knows its loaded skills). One call per deploy is
simpler to operate and lets the gateway batch the whole write. A source
that only knows one agent just sends a one-element `agents` array.

## Read path — `GET /_plugin/agents/:name/catalog`

Cookie-or-bearer (read-only, like the other observability endpoints).
Returns the merged view across all sources for one agent.

```jsonc
{
  "name": "coder-agent",
  "display_name": "Coder",
  "description": "Writes and modifies code.",
  "sources": ["hive", "prompt-manager"],
  "prompts": [ { "name": "system", "role": "system", "body": "...",
                 "source": "prompt-manager", "version": "rev-42",
                 "updated_at": "..." } ],
  "tools":   [ { "name": "read_file", "description": "...", "schema": {...},
                 "source": "hive", "version": "git:9f3a1c2", "updated_at": "..." } ],
  "skills":  [ { "name": "security-review", "description": "...",
                 "source": "goose", "updated_at": "..." } ]
}
```

Go response structs live in `internal/adminapi/` and are mirrored into
`internal/adminapi/ui/src/api/types.ts` (eventually tygo-generated),
per the UI's "add a new page" checklist in
`internal/adminapi/ui/AGENTS.md`.

## UI changes

`AgentDetail.tsx` (`/agents/:name`) gains a tabbed section below the
existing budget card:

- **Overview** (current) — cost histogram + budget + recent runs.
- **Prompts** — each prompt as a collapsible card: role badge, source +
  version chip, body in a `mono` block.
- **Tools** — table: name, description, source; row expands to the JSON
  schema.
- **Skills** — table: name, description, source.

Wiring per `internal/adminapi/ui/AGENTS.md`: one `useAgentCatalog(name)`
hook in `api/queries.ts` (slow cadence — catalog changes on deploy, not per
second), types in `api/types.ts`, no new route (same page). The
`/agents` list can later show small count badges (📄 prompts, 🔧 tools,
✦ skills) sourced from a lightweight `GET /_plugin/agents/catalog/summary`,
but that's optional follow-up.

When `Neo4jHTTPConfig()` reports not-configured, the read endpoint
returns `503` and the tabs render an empty-state ("catalog not wired on
this swarm") instead of erroring.

## Push sources (initial)

| Source | Pushes | How |
| --- | --- | --- |
| **hive** | tools (and model/provider) it already records | reshape `agent_logs.config.{tools,toolsConfig}` into a manifest; POST on agent deploy / config change |
| **stakwork prompt-manager** | prompts (versioned) | POST on prompt publish; `version` = prompt revision id |
| **ai-sdk agents (this repo)** | prompts + tools defined in code | a small `pushManifest()` helper called at process start |
| **goose pods** | skills + prompts loaded on the pod | pod posts its loaded skill/prompt set at boot |

None of these require schema changes in the source systems for v1 — hive
already has the data in `agent_logs.config`; the others enumerate what
they load at startup.

## Addendum: the catalog as authoritative registry (implemented)

The original framing above treats the catalog as *descriptive*. In
practice it has become the **registry of record** for which agents
exist on a swarm and what model each defaults to. Two concrete
deltas landed:

### `default_model` on the agent node

Each `HiveAgent` now carries a `default_model` scalar — the default
LLM used for that agent (a model id or shortcut string, e.g.
`claude-sonnet-4-6`). It flows through `POST /_plugin/agents`
(`agents[].default_model`), is stored on the node, and is returned by
`GET /:name/catalog` (`default_model`) and surfaced as a chip on the
agent page. Like `display_name`/`description`, it is set only when the
push supplies a non-empty value (a blank push never clobbers it).

### Hive seeds the default agent set

Hive's `BIFROST_AGENT_NAMES` stays the compile-time source of truth for
*which* agents exist (every LLM call site is typed against it). On the
first LLM call per swarm, Hive's orchestrator pushes those names — with
display name, description, and `default_model` — to the swarm's gateway
via `POST /_plugin/agents` (`source="hive"`). The two join by name; the
catalog owns the *capabilities* (prompts/tools/skills), authored later.

The push is **content-addressed**: Hive hashes the default-agent
manifest and stamps it on `Swarm.bifrostAgentsSeedHash`. A cache hit is
a single DB read with no HTTP; the manifest re-seeds only when the set
or a default model changes. It rides the same lazy, best-effort,
per-workspace-locked path as the trust reconciler
(`ensureBifrostAgentCatalog`, mirroring `ensureBifrostTrust`); a `503`
(neo4j unset on the swarm) is a benign skip, and any failure is logged
and swallowed so a stale catalog never blocks an LLM call.

User-authored agents and SPA-side editing remain future work — for now
Hive is the only writer and the default set is the whole catalog.

## Reconciliation & staleness

- **Replace-by-source** (above) keeps a redeployed source from leaving
  orphaned children.
- A source that disappears leaves stale nodes. v1 tolerates this; the
  `updated_at` per node lets the UI grey out anything older than N days.
  A later sweep job (or a `DELETE /_plugin/agents/:name?source=...`)
  can prune. Not needed for v1.
- No versioning/history of catalog nodes in v1 — nodes are mutable,
  like the trust registry. History can become a sibling label later.

## Phasing

1. **`graphclient` + write endpoint.** `POST /_plugin/agents`, neo4j
   HTTP client, label constants, indexes-on-first-write. Verifiable with
   curl + a Cypher read.
2. **Read endpoint + UI tabs.** `GET /_plugin/agents/:name/catalog`,
   types, hook, the three tabs.
3. **First real source.** Hive pushes `agent_logs.config` tools. Then
   prompt-manager, then goose pods.
4. **(Optional) list-page badges** and a staleness sweep.

## Out of scope for v1

- Editing prompts/tools/skills from the UI (read-only view).
- Versioning/history of catalog entries.
- Per-workspace or per-org scoping of the catalog (the catalog is
  swarm-global, keyed on agent name; org scoping is the *registry's*
  concern, not the catalog's).
- Wiring the catalog into the macaroon issuer or any enforcement path —
  the catalog is descriptive, never authoritative.
- Direct source→neo4j writes (always go through the gateway).
