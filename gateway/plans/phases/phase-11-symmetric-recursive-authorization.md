# Phase 11 — Symmetric Recursive Authorization

> Every signed layer (org → user, user → agent, agent → sub-agent)
> carries the same shape and follows the same narrowing rule. Realms
> become an optional, additive feature for multi-swarm deployments;
> single-deployment users never have to think about them.
>
> Companion to `phase-4-macaroon-shape.md` (the wire format this
> doc extends) and `phase-6-plugin-enforcement.md` (the enforcement
> layer). Supersedes the singular-`realm` field on `Invocation`
> defined in phase 4.

## The idea

Today's wire format has two awkward asymmetries:

1. `UserAuthorization` carries `permissions.realms[]` (a set) and
   `permissions.agents[]` (a set), but `Invocation` carries
   `realm` (a single string) and `agents[]` (a set). Different
   shapes for the same concept.
2. `Invocation.realm` is pinned at issue time, and
   `AttenuationCaveats` has no `realm` field — so a parent agent on
   swarm w1 cannot spawn a sub-agent on swarm w2 by local HMAC
   attenuation. Cross-swarm spawn requires a network round-trip to
   the Hive issuer for every cross-realm hop.

The fix is twofold:

- **Make every layer look the same.** Each signed layer has a flat
  set of allowed agents and an optional budget block. Narrowing is
  one uniform rule at every layer boundary.
- **Make realms an opt-in, multi-swarm feature.** If you only run
  one swarm you don't configure realms anywhere; if you run many
  swarms and want exact cross-swarm budget enforcement, you opt
  into per-realm caps in the macaroon.

```
┌─────────────────────────────────────────────────────────────┐
│  ORG  ─signs→  USER  (user_authorization)                   │
│    agents = [coder, browser, web]                           │
│    budget.max_per_invocation_usd = $25                      │
│    budget.max_total_usd          = $1000  (UA-cumulative)   │
│    budget.realm_budgets          = {w1: $500, w2: $200}     │
│                                    ← optional, multi-swarm  │
└──────────┬──────────────────────────────────────────────────┘
           ↓ user narrows
┌─────────────────────────────────────────────────────────────┐
│  USER  ─signs→  AGENT  (invocation, the root of one run)    │
│    agents = [coder]                          ⊆ org's        │
│    max_cost_usd = $5                         (per-run cap)  │
│    budget.realm_budgets = {w1: $5, w2: $2}   ≤ org's        │
└──────────┬──────────────────────────────────────────────────┘
           ↓ agent narrows (HMAC, local, no network)
┌─────────────────────────────────────────────────────────────┐
│  AGENT  ─HMAC→  SUB-AGENT  (attenuation)                    │
│    agents = [coder, web]                     ⊇ parent       │
│    max_cost_usd = $2                         ≤ parent       │
│    budget.realm_budgets = {w2: $1}           ≤ parent       │
└─────────────────────────────────────────────────────────────┘
```

## What this buys

**One mental model at every layer.** "A signed layer is a set of
allowed agents plus an optional budget; children narrow." That's
the whole protocol. The verifier code is one `narrow(parent, child)`
function called at every boundary.

**Cross-swarm sub-agents are HMAC-chained.** A parent on swarm w1
that wants a sub-agent on w2 attenuates locally with the realm
budgets it wants the child to have on w2. No Hive issuer round-trip
on the spawn path.

**Per-swarm spend caps are exact, signed.** When the org sets
`realm_budgets`, each swarm enforces its own entry against its own
Redis. The "$N × realms = $N·realms total leakage" failure mode of
having only `max_total_usd` disappears.

**Simple deployments don't pay for any of this.** A single-swarm
operator never sets a `realm_id`, never types "realm" in a
macaroon, and gets exactly the phase-4 behavior they have today.

## The simple case: no realms

A single-swarm deployment (one Bifrost, one plugin, one Redis,
one workspace) doesn't configure or mention realms anywhere:

- The swarm's trust registry has no `realm_id`.
- The issuer mints macaroons with no `realm_budgets`.
- The plugin's realm-membership check is a no-op.

The minimum useful macaroon for a single-swarm deployment:

```jsonc
"user_authorization": {
  "user_id": "u_alice",
  "user_pubkey": { ... },
  "agents": ["coder", "browser"],
  "budget": {
    "max_per_invocation_usd": 25,
    "max_total_usd": 1000
  },
  "iat": "...", "exp": "...", "nonce": "...",
  "org_sig": { ... }
}

"invocation": {
  "agents": ["coder"],
  "run_id": "...",
  "max_cost_usd": 5,
  "iat": "...", "exp": "...", "nonce": "...",
  "user_sig": { ... }
}
```

Org caps alice at $1000 UA-cumulative and $25 per call. User caps
this run at $5. The plugin enforces all three caps against its
local Redis. The word "realm" appears nowhere.

## Multi-realm: opt-in for multi-swarm operators

Operators with two or more swarms who want **exact** per-swarm
budget enforcement (vs. accepting that `max_total_usd` is a
per-swarm cap that leaks across swarms) opt into `realm_budgets`:

```jsonc
"user_authorization": {
  "user_id": "u_alice",
  "user_pubkey": { ... },
  "agents": ["coder", "browser"],
  "budget": {
    "max_per_invocation_usd": 25,
    "max_total_usd": 1000,
    "realm_budgets": {
      "w1": { "max_total_usd": 500 },
      "w2": { "max_total_usd": 200 }
    }
  },
  "iat": "...", "exp": "...", "nonce": "...",
  "org_sig": { ... }
}
```

When `realm_budgets` is present, the keys of the map are the
realms the macaroon authorizes. Each swarm verifies that its own
`realm_id` is one of those keys, then enforces the entry's
`max_total_usd` against its local cumulative-spend Redis bucket.
The realms a swarm is NOT named in are simply not its concern.

A swarm can be in the multi-realm world (it has a `realm_id`) and
still accept macaroons without `realm_budgets` — the org chose not
to scope this UA per-realm, and that's fine, the swarm enforces
the non-realm caps as before. The two opt-ins are independent.

## Wire format

Every signed layer carries:

| Field | Type | Required | Notes |
|---|---|---|---|
| `agents` | string[] | yes | Permitted agents; non-empty. Narrows at each layer. |
| `budget` | object | no | Optional cap block; see below. |
| `max_cost_usd` | number | no | Per-run cap (invocation/attenuation only, not UA). |
| `max_steps` | number | no | Per-run step cap (invocation/attenuation only). |
| `run_id` | string | yes on inv/att | One run per invocation; new run_id per attenuation. |
| `iat`, `exp`, `nonce` | string | yes | Time bounds + replay protection. |

The optional `budget` block:

| Field | Type | Required | Notes |
|---|---|---|---|
| `max_per_invocation_usd` | number | no | Per-call cap. Signature-time check. |
| `max_total_usd` | number | no | UA-cumulative cap, enforced per-swarm against `bifrost:cost:ua:<nonce>`. |
| `realm_budgets` | object | no | Map of realm-id → `{ max_total_usd: number }`. Opt-in for multi-swarm. |

The `realm_budgets` shape:

```jsonc
"realm_budgets": {
  "w1": { "max_total_usd": 500 },
  "w2": { "max_total_usd": 200 }
}
```

Keys are realm-ids (the same strings the trust registry stores).
The map itself is the set of permitted realms; there's no separate
`realms[]` field.

## Verifier rule (one uniform check at every layer)

For every parent → child boundary (org → UA, UA → invocation,
invocation → attenuation, attenuation → attenuation):

1. **`agents` — direction depends on the layer.**
   - **UA → invocation:** `child.agents ⊆ parent.agents` (the human's
     invocation picks a subset of the org-granted set; non-empty).
   - **invocation → attenuation, attenuation → attenuation:**
     `child.agents ⊇ parent.agents` (lineage extension; the child
     must preserve every ancestor entry and may append its own
     identity at the tail). The last entry is the most-specific
     agent (used as the billing dim).

   This is the one axis whose direction flips. Below the invocation,
   `agents[]` is a provenance chain, not a permission set — see
   phase 4 §"Caveat narrowing rules" for the rationale.

2. `child.budget.max_per_invocation_usd ≤ parent.budget.max_per_invocation_usd`
   (if parent set it).
3. `child.budget.max_total_usd ≤ parent.budget.max_total_usd`
   (if parent set it).
4. For each `r` in `child.budget.realm_budgets`:
   - `r` must be a key in `parent.budget.realm_budgets`
     (or `parent.budget.realm_budgets` must be absent — see "Mixed
     mode" below).
   - `child.budget.realm_budgets[r].max_total_usd ≤
      parent.budget.realm_budgets[r].max_total_usd` when both set.
5. `child.max_cost_usd ≤ parent.max_cost_usd` (when both set).
6. `child.exp ≤ parent.exp`.
7. Signature (or HMAC) verifies.

One function: `narrow(parent, child) -> effective | err`. Called
in a loop down the chain from UA to leaf. The function takes the
layer kind as an argument so rule 1 can dispatch on direction; every
other rule is uniform.

### Mixed mode

`realm_budgets` is opt-in per layer. The honest interpretations:

- Parent has `realm_budgets`, child has `realm_budgets`: normal
  narrowing per rule 4.
- Parent has `realm_budgets`, child omits it: child inherits the
  parent's set unchanged (no narrowing on this axis).
- Parent omits `realm_budgets`, child adds it: child is introducing
  a constraint that didn't exist; that's narrowing, allowed.
- Both omit: no per-realm enforcement at any layer.

## The plugin's realm-membership check

A swarm runs PreLLMHook after the macaroon verifies. If the
swarm has a configured `realm_id` AND the verified claims carry
`realm_budgets`:

- Assert `swarm.realm_id ∈ keys(claims.realm_budgets)`. Reject
  with `realm_not_permitted` (401) if not.
- The cap to compare against `bifrost:cost:ua:<nonce>` is
  `claims.realm_budgets[swarm.realm_id].max_total_usd`.
  (If `max_total_usd` is also set at the budget top level, both
  must hold — they're independent caps.)

If the swarm has no `realm_id` and the macaroon has no
`realm_budgets`: simple-deployment mode, no membership check.

If the swarm has a `realm_id` but the macaroon has no
`realm_budgets`: the org didn't scope this UA per-realm. The
swarm accepts the call and enforces only the non-realm caps.

If the swarm has no `realm_id` but the macaroon has
`realm_budgets`: configuration error — a multi-realm macaroon
landed on a swarm that doesn't claim an identity to match
against. Reject with `realm_not_configured` (401).

## Redis bucket keys: unchanged

Phase 6's bucket keys (`bifrost:cost:ua:<ua_nonce>`,
`bifrost:cost:run:<run_id>`, the `agent_budgets` keys, kill
switches, revocations) stay exactly as defined. Every key on a
swarm's Redis is implicitly for that swarm's realm — adding a
`:realm:<r>` suffix would be redundant with the database's own
identity. What changes is which *cap* the plugin compares the
bucket against, not the *key*.

## The `realm-id` dim goes away from logs

Today `realm-id` is one of the signature-bound dims that
PreLLMHook stamps onto `BifrostContextKeyDimensions`, which the
Bifrost built-in logging plugin snapshots into `logs.metadata`.
After phase 11 the macaroon no longer carries a single realm, so
there's no scalar to stamp. We could synthesize one from the
swarm's `realm_id`, but: every row in a swarm's `logs.db` is by
definition for that swarm's realm. The column is redundant with
the database's own identity. We drop it.

Cross-swarm analytics (the central-aggregator path) adds the
realm column at import time, keyed off which swarm the slice
came from.

This means:

- `realm-id` is removed from `signatureBoundDims` in
  `gateway/internal/pluginctx/dims.go`.
- `CanonicalizeFromClaims` drops its `realmID` parameter.
- `parseDimensionParam` in `adminapi/observability.go` drops
  `realm-id` from its allow-list.
- `parseMetadataFilters` drops the `realm_id` query param.
- Phase-7's `/_plugin/spend/by-realm` and phase-8's realm filter
  widget go away.

The `DimRealmID` constant can stay defined for caller-supplied
`x-bf-dim-realm-id` headers (ad-hoc observability with no
semantic guarantee), but no plugin code writes it.

## Where the swarm's `realm_id` lives: the trust registry

Multi-swarm deployments need a per-swarm self-identity. Rather
than a new env var, it lives in the trust registry's persisted
state alongside the existing org list. The registry already owns
"what this swarm knows about itself and the world," persists
across restarts, has admin auth, and exposes a status endpoint.
Self-identity fits the same shape.

```jsonc
{
  "realm_id": "w1",                        // ← new, optional
  "orgs": [ { "org_id": "...", "pubkey": "..." } ]
}
```

`realm_id` is optional. A single-swarm deployment omits it; the
trust registry accepts org entries without any swarm-identity.
Hive's reconciler sets it at workspace provisioning for
multi-swarm deployments.

## Implementation hints (for the next agent)

### Wire format (`gateway/auth/`)

- **`types.go` / `types.ts`:**
  - On `UserAuthorization`: move `Agents` to top-level (was under
    `Permissions.Agents`). Delete the `Permissions` wrapper and
    the `UserPermissions` type.
  - On `Invocation` and `AttenuationCaveats`: delete `Realm string`
    (was singular phase-4 field).
  - On `UserBudget` (rename to just `Budget`, used at every
    layer's optional `budget` field): keep `MaxPerInvocationUSD`
    and `MaxTotalUSD`. Add `RealmBudgets map[string]RealmBudget`
    where `RealmBudget struct { MaxTotalUSD float64 }`.
  - On `Invocation` and `AttenuationCaveats`: add `Budget *Budget`
    (optional, same shape as on the UA).

- **`Claims`:** delete `Realm string`. Add
  `PermittedRealms []string` (the keys of `EffectiveCaveats.Budget.RealmBudgets`,
  or nil if no `realm_budgets` anywhere in the chain). Keep
  `AgentName` (the leaf agent for billing).

- **Narrowing in `verify.go` / `verify.ts`:** factor narrowing into
  one `narrow(parent, child) -> effective | err` and call it at
  every layer boundary. Follow the seven-rule check in "Verifier
  rule" above.

- **Fixtures:** existing fixtures (01-simple, 02-one-attenuation,
  03-two-attenuations, 04-multisig-2of3, 05-budget-envelope) all
  need regeneration because the wire shape changes. Add:
  - `06-multi-realm.json`: UA + invocation both carry
    `realm_budgets`; no attenuations.
  - `07-cross-realm-attenuation.json`: parent allows w1+w2, child
    attenuates to w2 only with a smaller cap.

  Regenerate via `regenerate-fixtures.ts`; both Go and TS tests
  must pass byte-identically.

### Trust registry (`gateway/internal/trust/`)

- **`types.go`:** add `RealmID string` to `trust.File` and
  `trust.Seed` (both optional, omitempty). Validate non-empty +
  no whitespace + no slashes when set.
- **`persistence.go`:** include `realm_id` in the atomic write
  payload. Loading is permissive — `realm_id` may be empty (the
  simple-deployment case).
- **`registry.go`:** add `RealmID() string` accessor (empty string
  if unset).
- **`StatusResponse`:** add `RealmID string` so
  `GET /_plugin/trust/status` shows it.
- **Admin API:** new `PUT /_plugin/trust/realm_id` endpoint, same
  bearer auth as existing trust endpoints.

### Hot-path enforcement (`gateway/internal/auth/`)

The adapter already holds a `*trust.Registry`. Add one more
accessor call to read `registry.RealmID()` — same in-memory state,
no new wiring.

PreLLMHook after macaroon verify:

```
realm_id      = registry.RealmID()                    // may be ""
realm_budgets = claims.EffectiveCaveats.Budget.RealmBudgets  // may be nil

if realm_budgets != nil:
    if realm_id == "":
        return 401 realm_not_configured
    if realm_id not in realm_budgets:
        return 401 realm_not_permitted
    realm_cap = realm_budgets[realm_id].MaxTotalUSD
else:
    realm_cap = 0  // no realm-scoped cap

# Existing phase-6 cap walk, with one extra comparison:
ua_spend  = HGET bifrost:cost:ua:<nonce> total
if realm_cap > 0 && ua_spend + this_call > realm_cap:
    return 402 realm_budget_exceeded
if claims.Budget.MaxTotalUSD > 0 && ua_spend + this_call > claims.Budget.MaxTotalUSD:
    return 402 ua_budget_exceeded
# ... rest unchanged
```

Redis bucket keys and the chain-walk pipeline shape are
**unchanged** from phase 6.

### Dim canonicalization (`gateway/internal/pluginctx/dims.go`)

- Remove `DimRealmID` from `signatureBoundDims`.
- `CanonicalizeFromClaims`: drop the `realmID` parameter.
- `DimRealmID` constant can stay defined for caller-supplied
  `x-bf-dim-realm-id` headers (no plugin code writes it).
- `hooks/llm_prehook.go`: drop the `claims.Realm` argument from
  the `CanonicalizeFromClaims` call.

### Admin / observability (`gateway/internal/adminapi/`)

- `observability.go` `parseDimensionParam`: remove `"realm-id"`
  from the allow-list.
- `observability.go` `parseMetadataFilters`: remove the
  `realm_id` query param mapping.
- Any phase-7 `/_plugin/spend/by-realm` handler and phase-8
  realm filter widget are deleted.

### Issuer (`/macaroons/issue`)

- Request body: `agents` (single string or array; the issuer
  builds the UA's `agents` set), optional `realm_budgets` map,
  `run_id`, optional `max_cost_usd`, optional `override`.
- The `realm` (singular) request field is deleted.
- If the request includes `realm_budgets`, the issuer enforces:
  every key must be in the UA's `realm_budgets`; every per-realm
  cap must be ≤ the UA's; agents must be ⊆ UA's.
- Cross-realm sub-agent spawning does NOT go through the issuer.
  Parents attenuate locally with the desired `realm_budgets`
  subset.

### Agent SDK / attenuator

- `attenuate(parentSigBytes, caveats)` is unchanged in primitive
  shape — just JCS + HMAC-SHA256 + hex. The caveats may now
  include an optional `budget` block with `realm_budgets`.
- Recommended convention for parents spawning cross-realm
  sub-agents: leave at least 25% headroom in your own per-realm
  bucket. Convention only; the verifier enforces just the upper
  bound.

### Docs to update

- `phase-4-macaroon-shape.md`: rewrite `Invocation` and
  `AttenuationCaveats` sections; rename `UserBudget` to `Budget`;
  drop the `Permissions` wrapper from the UA; add `realm_budgets`
  to `Budget`; describe the symmetric narrowing rule.
- `phase-5-trust-registry.md`: add optional `realm_id` to the
  on-disk schema and the new `PUT /_plugin/trust/realm_id`
  endpoint. The "deliberately NOT in the trust registry"
  subsection stays — `realm_id` is the swarm's own identity,
  not a per-org policy.
- `phase-6-plugin-enforcement.md`: Redis bucket keys unchanged;
  PreLLMHook gains the realm-membership check; drop `realm-id`
  from the dim canonicalization rule entirely.
- `cryptographic-identity.md`: rephrase "permissions.realms" /
  "realm" mentions to match the new shape.
- `llm-governance-v2.md`: update the request-flow walk; add a
  brief cross-swarm sub-agent example for the multi-realm case.

## Cutover (no backward compatibility)

Phase 11 is a breaking wire-format change, and that's fine. The
macaroon system is barely deployed (one internal workspace at
time of writing). We don't ship `v=2` alongside `v=1`; we change
the shape in place and regenerate fixtures.

1. Disable macaroon enforcement on the one active workspace.
2. Land the wire-format change in `gateway/auth/` + fixtures +
   plugin + issuer in one PR series.
3. Deploy plugin and Hive issuer together.
4. Re-enable enforcement.

If a deployment ever does need to migrate without a maintenance
window, the macaroon already carries a `v` field — we can add
version dispatch in `verify.go` at that point. We just don't
need to pre-build it.

## Open questions

1. **Issuer-side default allocation policy.** When a spawner
   calls `/macaroons/issue` for a multi-swarm scenario but
   doesn't specify per-realm caps, what does the issuer fill in?
   Options: even split of the agent registry's default;
   proportional to the UA's `realm_budgets` ratios; require
   explicit caps from the caller.

2. **Per-realm rate / step limits.** `realm_budgets[r]`
   currently carries only `max_total_usd`. Should it also carry
   `max_steps` for completeness? Probably yes when needed; defer
   until the first real use case demands it.

3. **Cross-swarm analytics aggregator.** Once `realm-id` is gone
   from per-row metadata, "spend across all swarms by realm" is
   strictly an aggregator-side concern — the central store
   imports each swarm's slice and stamps the realm at import
   time. Phase-7 / phase-8 update accordingly. No protocol
   change.

4. **PR ordering within the cutover.** Suggested sequence inside
   the maintenance window: (a) `gateway/auth/` types + verifier
   + regenerated fixtures, byte-equivalence Go↔TS; (b)
   `gateway/internal/trust/` extended with optional `RealmID`
   (file shape + seed shape + status + `PUT /_plugin/trust/realm_id`);
   (c) `gateway/internal/auth/` adapter + new
   `claims.PermittedRealms` consumption + the membership check;
   (d) admin/observability dim allow-list cleanup; (e) Hive
   issuer + reconciler cut over to the new shape and start
   writing `realm_id` on multi-swarm provisioning; (f) re-enable
   enforcement. Each PR is mergeable independently as long as
   the whole sequence lands before enforcement comes back on.
