# Agent Registry

> Companion to `llm-governance-v2.md` and `cryptographic-identity.md`.
> The agent registry is what the Hive macaroon issuer reads to pick
> caveat values when minting an invocation macaroon. The plugin
> doesn't read from it directly; the plugin only sees the resulting
> caveats on each request.

## Why this exists

A macaroon's invocation caveats include `agents=[…]`, `max_cost_usd`,
`max_steps`, `max_wallclock_s`. The issuer has to fill those in with
something. For each agent name (`coder`, `browser`, `repair-agent`, …)
there has to be a known default budget. That known set of agents and
their defaults is the agent registry.

It also doubles as the answer to "what agents are available?" for any
UI or tool that needs to list or describe them.

## Scope

- **Org-scoped.** Agents are defined per organization. Every workspace
  in an org sees the same agent fleet. This matches the cryptographic
  identity model (org is the principal that authorizes work) and the
  cross-swarm story (an agent name should mean the same thing on every
  swarm that trusts the org's key).
- **Custodial in phase 1.** Hive admins seed and edit the registry.
  Per-workspace customization is not in phase 1.

This table is the home for the agent defaults that
`llm-governance-v2.md` refers to as `AGENT_DEFAULTS[…]` (a TypeScript
constant in spawner code) and lists as open question #2 ("per-workspace
`AGENT_DEFAULTS` map location"). **That open question is resolved by
this doc:** the defaults live in the `AgentDefinition` table, owned by
the Hive macaroon issuer, scoped per org. Spawners no longer need to
ship their own copy; they pass the agent name to `/macaroons/issue`
and the issuer reads the defaults from the registry.

## Data model

One new Hive table.

```prisma
model AgentDefinition {
  id                   String   @id @default(cuid())
  orgId                String   @map("org_id")
  name                 String                                // e.g. "coder", "browser"
  displayName          String   @map("display_name")
  description          String

  // Defaults the macaroon issuer stamps onto invocations of this agent.
  // The issuer can override per-invocation (e.g. user explicitly asks
  // for a smaller budget); these are the starting points.
  defaultMaxCostUsd    Float    @map("default_max_cost_usd")
  defaultMaxSteps      Int      @map("default_max_steps")
  defaultMaxWallclockS Int      @map("default_max_wallclock_s")

  // Optional model allowlist. Empty = inherit org/workspace default
  // (i.e. all models the user's VK permits).
  allowedModels        String[] @map("allowed_models")

  enabled              Boolean  @default(true)

  createdAt            DateTime @default(now()) @map("created_at")
  updatedAt            DateTime @updatedAt      @map("updated_at")

  org                  Org      @relation(fields: [orgId], references: [id], onDelete: Cascade)

  @@unique([orgId, name])
  @@index([orgId])
  @@map("agent_definitions")
}
```

`name` is the string that lands in the macaroon's `agents=[…]` caveat
and in the `x-bf-dim-agent-name` Bifrost log header. It's an
identifier, not a label — stable, URL-safe, lowercase-with-hyphens.

The `agents` caveat in the macaroon is an **array** so sub-agents
can extend it (`["coder", "web-search"]`, etc.) — see v2's plugin
canonicalization, which keys `agent-name` on the last (most-specific)
entry. The issuer always writes a single-element array on first
issuance; later entries are added by parent agents during local
attenuation.

`displayName` and `description` exist for UIs that need to render
human-readable agent listings.

### `allowedModels`

`allowedModels` is a per-agent model allowlist that **does not**
flow into the macaroon as a caveat in phase 1. It is enforced by the
Hive macaroon issuer at issuance time: if the spawner asks for an
agent and a specific model that isn't in `allowedModels`, the issuer
rejects the spawn. Empty means "inherit whatever the user's VK
permits" — which today is `["*"]` (every provider, every model) per
v2's VK provisioning.

The plugin and Bifrost continue to enforce VK-level model allowlists
(v2, line 109) independently. The two layers compose: a model must be
permitted by **both** the VK (org/workspace-wide) and the agent
registry (per-agent) to be callable. If the layers disagree, the
intersection wins.

Promoting `allowedModels` to a macaroon caveat is a future option —
it would make the per-agent constraint verifiable downstream on
cross-swarm calls — but it's not needed for phase 1 since the issuer
and the workspace's plugin are in the same trust domain.

## Resolution at issuance time

When the Hive macaroon issuer mints an invocation macaroon for
`(org, user, workspace, agent_name)`:

```
1. Look up AgentDefinition by (orgId, name=agent_name).
   - If not found → reject the spawn: "unknown agent for this org".
   - If enabled=false → reject the spawn: "agent disabled".

2. Use the row's default* fields as the invocation caveats:
     max_cost_usd     = row.defaultMaxCostUsd
     max_steps        = row.defaultMaxSteps
     max_wallclock_s  = row.defaultMaxWallclockS

3. If the caller (e.g. a chat BFF, a workflow engine) wants a smaller
   budget, it passes a narrower override. The issuer takes the min of
   the override and the registry default. The issuer never widens.

4. Sign the resulting caveats. Done.
```

The plugin never sees the registry. It only sees the signed caveats.

## Seeding

The phase-1 set of agents is a fixed list seeded into the table at
org-create time. It can also be re-seeded by a migration script if
new well-known agents are added to the platform.

```ts
// src/services/agents/seed.ts
export const SEED_AGENTS = [
  {
    name: "coder",
    displayName: "Coder",
    description: "Writes and modifies code in the workspace's repositories.",
    defaultMaxCostUsd: 5.0,
    defaultMaxSteps: 100,
    defaultMaxWallclockS: 600,
    allowedModels: [],
  },
  {
    name: "browser",
    displayName: "Browser",
    description: "Operates a headless browser to eval the product.",
    defaultMaxCostUsd: 3.0,
    defaultMaxSteps: 50,
    defaultMaxWallclockS: 300,
    allowedModels: [],
  },
  {
    name: "repair-agent",
    displayName: "Repair Agent",
    description:
      "Diagnoses and fixes failing CI / failing tests / failing pods.",
    defaultMaxCostUsd: 10.0,
    defaultMaxSteps: 200,
    defaultMaxWallclockS: 1800,
    allowedModels: [],
  },
] as const;
```

On org create: insert one row per `SEED_AGENTS` entry with the new
`orgId`. On platform-add-new-agent: a migration script inserts the
new row into every existing org's `agent_definitions`.

## Admin API

Phase 1 exposes read-only listing for users + tools, and full CRUD
for org admins. All paths are scoped to the caller's org.

```
GET    /api/orgs/:orgId/agents               → list (any member)
GET    /api/orgs/:orgId/agents/:name         → one (any member)
POST   /api/orgs/:orgId/agents               → create (org admin only)
PATCH  /api/orgs/:orgId/agents/:name         → update (org admin only)
DELETE /api/orgs/:orgId/agents/:name         → delete (org admin only)
```

`name` is immutable after creation (it's the identifier the macaroon
issuer keys on). Budgets, descriptions, model allowlists, and
`enabled` are mutable.

## Relationship to the plugin's `agent_budgets` config

The plugin has a separate `agent_budgets:` block in its YAML
(`llm-governance-v2.md` line 561). That block is an operator-level
**hard ceiling** per agent name, enforced regardless of what the
issuer asked for. It is distinct from this registry:

| Layer                | Where                        | Mutable by          | Enforces                                             |
| -------------------- | ---------------------------- | ------------------- | ---------------------------------------------------- |
| **Registry default** | Hive `AgentDefinition` table | Org admins via UI   | What the issuer asks for on this agent's invocations |
| **Plugin ceiling**   | Plugin's YAML config         | Swarm operator only | Hard cap regardless of issuer request                |

The plugin enforces `min(macaroon caveat, plugin ceiling)`. If they
disagree, the lower wins. If the plugin has no entry for an agent
name, the macaroon caveat is the binding limit.

## Relationship to the cryptographic identity stack

The registry sits between the user and the agent in the principal
chain:

```
ORG          ← signs user_authorization
USER         ← signs invocation
REALM        ← caveat in invocation (which workspace this runs in)
AGENT        ← caveat in invocation; defaults sourced from this registry
SUB-AGENT    ← HMAC attenuation, bounded by parent's caveats
```

The registry doesn't introduce a new principal — agents aren't
identities. It's just the lookup table the issuer uses to translate
"alice wants to run `coder` in `w1`" into concrete caveat values
before signing.

## Out of scope for phase 1

- Per-workspace overrides of an org's agent definitions.
- User-defined custom agents.
- Per-org cascading from a platform default registry.
- Agent versioning (registry rows are mutable; there's no history).
- Per-agent rate limits (the macaroon caveats are cost + steps +
  wallclock; RPM/TPM is not a registry concern in phase 1).
- Marketplace / third-party-published agents.

When any of these arrive, the table grows columns or a sibling table
appears; nothing about the signing chain or the plugin changes.
