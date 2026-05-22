# Phase 11 — Symmetric Recursive Authorization

> Every signed layer (org → user, user → agent, agent → sub-agent)
> carries the same shape: a set of permitted realms and a per-realm
> budget. Children narrow what parents granted, uniformly, at every
> layer. The HMAC chain — not the Hive issuer — is what authorizes
> cross-realm sub-agents.
>
> Companion to `phase-4-macaroon-shape.md` (the wire format this
> doc extends) and `phase-6-plugin-enforcement.md` (the enforcement
> layer this doc adds per-realm buckets to). Supersedes the
> singular-`realm` field on `Invocation` defined in phase 4.

## The idea

Today, `user_authorization` is multi-realm but `invocation` is
pinned to one realm and `attenuation` can't change it. That
asymmetry is why cross-realm sub-agents need a Hive issuer
round-trip and why a UA with N permitted realms can leak up to
N × cap of spending (each swarm has its own Redis; no swarm sees
the others).

The fix: **make every signed layer look the same.** Each layer
grants a set of realms and a budget per realm; children narrow
what parents granted; sub-agents on a different realm are just
narrower attenuations, signed locally by the parent's HMAC.

```
┌─────────────────────────────────────────────────────────────┐
│  ORG  ─signs→  USER                                         │
│    permissions.realms      = [w1, w2, w3]                   │
│    budget.realm_budgets    = {w1: $500, w2: $200, w3: $300} │
└──────────┬──────────────────────────────────────────────────┘
           ↓ user narrows
┌─────────────────────────────────────────────────────────────┐
│  USER  ─signs→  AGENT (root invocation)                     │
│    permissions.realms      = [w1, w2]            ⊆ org's    │
│    budget.realm_budgets    = {w1: $5, w2: $2}    ≤ org's    │
└──────────┬──────────────────────────────────────────────────┘
           ↓ agent narrows (HMAC, local, no network)
┌─────────────────────────────────────────────────────────────┐
│  AGENT  ─HMAC→  SUB-AGENT                                   │
│    permissions.realms      = [w2]                ⊆ parent's │
│    budget.realm_budgets    = {w2: $1}            ≤ parent's │
└─────────────────────────────────────────────────────────────┘
```

## What this buys

**One mental model at every layer.** Org-to-user is the same
shape as user-to-agent is the same shape as agent-to-sub-agent.
"A signed layer is a permitted set of realms plus a per-realm
budget; children narrow." That's the whole protocol.

**Cross-realm sub-agents are HMAC-chained.** A parent on swarm
w1 wanting a sub-agent on w2 attenuates locally, hands the
narrowed macaroon to the sub-agent, the sub-agent presents it on
w2, w2's plugin verifies the chain. No Hive issuer round-trip on
the spawn path. No network dependency for cross-realm work.

**Per-realm spend caps are exact, signed.** The parent's $5 cap
in w2 bounds every descendant's spend in w2 via the chain walk
on w2's Redis. The "$N × realms = $N·realms total leakage"
problem disappears.

**No more single-`realm` special case.** Today the wire format
has UAs with `realms[]` and invocations with `realm` (singular)
— two different shapes for the same concept. Symmetric design
collapses to one.

## What changes on the wire

`Invocation` and `AttenuationCaveats` gain the same
`(permissions, budget)` substructure that `UserAuthorization`
already has. The singular `invocation.realm` field is replaced.

Every signed layer (UA, invocation, attenuation) carries the
same two substructures:

- `permissions: { realms: [...], agents: [...] }` — what's allowed
- `budget: { realm_budgets: {...}, max_per_invocation_usd: ... }` — how much

```jsonc
"user_authorization": {
  "permissions": { "realms": ["w1","w2","w3"], "agents": ["coder","browser","web"] },
  "budget": {
    "max_per_invocation_usd": 25.0,
    "realm_budgets": { "w1": { "max_total_usd": 500 },
                       "w2": { "max_total_usd": 200 },
                       "w3": { "max_total_usd": 300 } }
  },
  ...
  "org_sig": { ... }
}

"invocation": {
  "permissions": { "realms": ["w1","w2"], "agents": ["coder"] },
  "budget": {
    "max_per_invocation_usd": 5.0,
    "realm_budgets": { "w1": { "max_total_usd": 5.0 },
                       "w2": { "max_total_usd": 2.0 } }
  },
  "run_id": "...", "iat": "...", "exp": "...", "nonce": "...",
  "user_sig": { ... }
}

"attenuations": [{
  "caveats": {
    "permissions": { "realms": ["w2"], "agents": ["coder","web"] },
    "budget": { "realm_budgets": { "w2": { "max_total_usd": 1.0 } } },
    "run_id": "...", "exp": "...", "nonce": "..."
  },
  "hmac": "..."
}]
```

The singular `invocation.realm` field and the top-level
`agents` field on invocations and attenuations are **removed**;
both move under `permissions`. Same shape at every layer.

## Verifier rule (uniform at every layer)

For every parent → child boundary (org→user, user→invocation,
invocation→attenuation, attenuation→attenuation):

1. `child.permissions.realms ⊆ parent.permissions.realms`
   (non-empty).
2. `child.permissions.agents ⊆ parent.permissions.agents`
   (non-empty).
3. For each `r` in `child.permissions.realms`:
   `child.budget.realm_budgets[r].max_total_usd ≤
    parent.budget.realm_budgets[r].max_total_usd`
   (absent parent entry means 0; absent child entry means 0).
4. `child.budget.max_per_invocation_usd ≤
    parent.budget.max_per_invocation_usd` if parent set it.
5. `child.exp ≤ parent.exp`.
6. Signature (or HMAC) verifies.

One rule, applied identically at every layer boundary. The
verifier code is `narrow(parent, child) -> effective | err`
called in a loop down the chain.

## What it costs

- **Wire format breaking change.** `invocation.realm` → removed;
  `invocation.permissions.realms` added. Every fixture rewrites;
  every consumer of `claims.realm` updates to either
  `claims.permitted_realms` or the per-call "which realm did this
  call actually run in" dimension.
- **More verifier work.** Per-realm narrowing checks at every
  attenuation link. Still O(chain depth × realms-per-link),
  still microseconds.
- **`Claims` shape changes.** No single `realm` scalar; instead
  `permitted_realms: []` plus a per-call dimension that the
  plugin stamps from "the realm this swarm actually represents."
- **The user signature now authorizes more.** Today: one run, one
  realm. Tomorrow: many realms, with budgets, for the lifetime of
  the invocation. Bounded entirely by what the user signed; no
  expansion of trust beyond the user's stated `realm_budgets`.

## Implementation hints (for the next agent)

### Wire format (`gateway/auth/`)

- **`types.go` / `types.ts`:** rename `UserBudget` → `Budget` (it's
  the shape every layer uses now). Add
  `RealmBudgets map[string]RealmBudget` to it where
  `RealmBudget { MaxTotalUSD float64 }`. Add
  `Permissions { Realms []string; Agents []string }` and
  `Budget *Budget` to `Invocation` and `AttenuationCaveats`.
  **Remove** the singular `Realm string` and the top-level
  `Agents []string` from `Invocation` and `AttenuationCaveats`
  — both move under `Permissions`.
- **`Claims`:** add `PermittedRealms []string` (the union/walk
  result of the chain) and remove `Realm` (the plugin stamps the
  per-call realm separately; see `phase-6-plugin-enforcement.md`
  below). Keep `EffectiveCaveats` shape and add per-realm budget
  rollup if useful for adapter pipelining.
- **Narrowing in `verify.go` / `verify.ts`:** factor the layer-to-layer
  narrowing rule into one function `narrow(parent, child) ->
  effective | err`, call it for every boundary (UA → inv, inv →
  att, att → att). The rule is described in "Verifier rule" above.
- **Fixtures:** the existing fixtures stay valid in their existing
  semantics (one realm, no realm_budgets). Add `06-multi-realm.json`
  (UA + invocation both multi-realm with realm_budgets) and
  `07-cross-realm-attenuation.json` (parent macaroon allows two
  realms, child attenuates to one realm with a smaller per-realm
  cap). Regenerate via `regenerate-fixtures.ts`; both Go and TS
  fixture tests must pass byte-identically.

### Hot-path enforcement (`gateway/internal/auth/`, phase 6)

- **Per-realm cumulative spend bucket per layer:**
  - UA layer: `bifrost:cost:ua:<ua_nonce>:realm:<realm_id>` (already
    sketched as a phase-6 extension).
  - Invocation layer: `bifrost:cost:run:<run_id>:realm:<realm_id>`.
  - Attenuation layers contribute to their own `run_id` bucket via
    the existing chain-walk, **scoped by the realm the actual call
    ran in.**
- **The chain walk in PreLLMHook** reads, for the call's actual
  realm: this realm's `cost:ua` bucket, this realm's `cost:run`
  bucket for the leaf and every ancestor in the macaroon chain.
  All in one Redis pipeline (same shape as phase 6 today; just
  every key gets a `:realm:<r>` suffix and the comparison reads
  the parent's `realm_budgets[r]` cap).
- **PostLLMHook** increments the same set of `:realm:<r>` keys
  (leaf run, every ancestor's run, UA), one HINCRBYFLOAT each, in
  one pipeline. The "which realm" is the swarm's own realm — each
  swarm knows what it is at boot.
- **Dimension stamping:** `x-bf-dim-realm-id` is the swarm's own
  realm, **not** anything from the macaroon (the macaroon permits
  a set, but the call physically ran in this one swarm's realm).
  Update phase-6's "dimension canonicalization" rule.

### Issuer (`/macaroons/issue`)

- Accept `realms: []` and `budget.realm_budgets` in the request
  body. Defaults narrow the UA's `realm_budgets` proportionally
  (or via an agent-registry policy — TBD) when the caller doesn't
  specify amounts. Reject any realm not in the UA's
  `permissions.realms`, or any per-realm cap exceeding the UA's.
- **No cross-realm spawn path on the issuer.** Cross-realm
  sub-agents are produced by parent agents locally (HMAC
  attenuation); the issuer is only called for top-of-tree
  invocations. Document this clearly so spawners on existing
  paths don't accidentally re-issue per cross-realm spawn.

### Agent SDK / attenuator

- `attenuate(parentSigBytes, caveats)` works unchanged — the
  caveats just now include `permissions` and `budget`
  substructures. Polyglot agents need no new primitives (still
  JCS + HMAC-SHA256 + hex).
- **Recommended convention** for parents spawning cross-realm
  sub-agents: pick the realms the child needs, allocate from
  remaining per-realm budget, leave at least 25% headroom in the
  parent's own per-realm bucket. Documented as a convention only;
  enforced as a hard upper bound by the verifier.

### Docs to update

- `phase-4-macaroon-shape.md`: rewrite the `Invocation` and
  `AttenuationCaveats` sections; rename `UserBudget` →
  `Budget`; describe the symmetric narrowing rule.
- `phase-6-plugin-enforcement.md`: extend Redis schema with
  `:realm:<r>` suffixed keys; update PreLLMHook/PostLLMHook
  pseudocode to chain-walk per realm; update dimension stamping
  to use the swarm's own realm.
- `cryptographic-identity.md`: the three-principal model is
  unchanged; the protocol description should be reframed as "every
  signed layer is a Budget that narrows."
- `llm-governance-v2.md`: the request-flow walk and threat model
  pick up a cross-realm sub-agent example.

### Backward compatibility

Phase 11 is **a breaking wire-format change.** There is no clean
way to support both shapes (singular `realm` vs.
`permissions.realms`) in one verifier without ambiguity, and the
hot path can't afford a per-call branch on which shape is in
play. Ship as `v=2` on the macaroon's top-level `v` field; the
verifier dispatches on version. Phase-1 macaroons (`v=1`) keep
working under the existing single-realm rules during migration;
new issuances mint `v=2`. Plan the cutover with the same
"observability → enforcement" gating pattern that v2's rollout
uses for the original macaroon adoption.

## Open questions

1. **Issuer-side default allocation policy.** When a spawner
   calls `/macaroons/issue` with `realms: [w1, w2]` but doesn't
   specify per-realm caps, what does the issuer fill in? Options:
   even split of the agent registry's default cap; proportional
   to the UA's `realm_budgets` ratios; require explicit caps.

2. **Per-realm rate limits in addition to per-realm budgets.**
   `Budget.realm_budgets[r]` currently only carries
   `max_total_usd`. Should it also carry `max_steps` and rate
   info? Probably yes for completeness; defer if not needed for
   v1 of the symmetric design.

3. **Cross-realm dimension on logs.** Today `x-bf-dim-realm-id`
   is one value per call (the call ran in one realm). That stays
   true. But for analytics queries like "how much did this
   invocation spend across all the realms it touched," the
   dashboard needs to join across swarms' `logs.db`. Defer to
   phase-7-observability; no protocol change required.
