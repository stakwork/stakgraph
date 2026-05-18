# Phase 6 — Plugin Enforcement: Redis Schema, Hot-Path Ops, Failure Modes

> Concrete operational spec for the Bifrost-plugin adapter at
> `gateway/internal/auth/`. Companion to `phase-4-macaroon-shape.md`
> (which defines the cryptographic verifier the plugin wraps) and
> `phase-5-trust-registry.md` (which defines where the plugin gets
> the org pubkeys it verifies against). Builds on `llm-governance-v2.md`
> §475-575, which sketched the Redis schema and hook flow in
> pseudocode; this doc pins down the bytes, the bucket-key formats,
> the TTL policy, the atomicity model, and the failure modes that
> v2 left implicit.
>
> This phase produces no new wire format and no new cryptographic
> primitives. It produces the operational contract the plugin adapter
> implements against — enough that a v2-style "per-run cost cap +
> kill switch + per-agent budget" enforcement layer can land in
> `gateway/internal/auth/` against the verifier in `gateway/auth/go/`
> without re-litigating any of the questions about bucket boundaries,
> TTLs, or race semantics.

## What this phase decides

v2 §561-575 specified seven Redis key prefixes and the per-hook ops
that touch them. What v2 did **not** specify:

- The day-bucket key format for `cost:agent:<name>:<day>` (UTC?
  workspace-local? what about weekly / monthly windows?)
- TTLs that work for long-lived invocations (v2's hardcoded 1h breaks
  the 8-hour sessions and 24-hour workflows that the cryptographic-
  identity doc explicitly allows)
- Whether multi-write PostHook ops are atomic, pipelined, or
  independent — and what overshoot is acceptable
- How the ancestor cap walk pairs `run_id` with its `max_cost_usd`
  from the macaroon chain
- The `kill:agent:<name>` switch (proposed in design discussion but
  never specified)
- PostHook retry / replay semantics
- The `revoke_user_before:<user_id>` time comparison axis (compared
  against ua.iat? invocation.iat? both?)

This doc resolves each of those. After this:

- The plugin adapter in `gateway/internal/auth/` can be written
  against a definitive schema and ops list.
- Operators can reason about Redis memory usage, throughput, and
  failure modes.
- Per-agent budgets can be configured with any of Bifrost's standard
  duration windows (`1d`, `1w`, `1M`, `1Y`, sub-day rolling) without
  re-asking what "the right bucket" is.

## Relation to v2

This is the third leg of the spec triangle:

| Phase | Decides |
|---|---|
| Phase 4 | What bytes a macaroon contains |
| Phase 5 | Which orgs the swarm trusts |
| **Phase 6** | **How the plugin enforces caveats against Redis state** |

Where v2 and this phase disagree, **phase 6 wins.** v2 §561-575 is
superseded by the schema in "Redis schema" below. v2 §475-556 is
superseded by the hot-path algorithm in "Hot path" below.

## Duration vocabulary

Per-agent budgets and any future windowed limit use Bifrost's
standard duration vocabulary, identical to what Hive's reconciler
passes to `Customer.budget.reset_duration` (`phases/phase-1-reconciler.md`
§472-485):

| Suffix | Meaning | Bucket boundary |
|---|---|---|
| `s` | seconds | rolling window (no calendar boundary) |
| `m` | minutes | rolling window |
| `h` | hours | rolling window |
| `d` | days | midnight UTC of the current day |
| `w` | weeks | midnight UTC of the most recent Monday |
| `M` | months | midnight UTC of the 1st of the current month |
| `Y` | years | midnight UTC of Jan 1 of the current year |

Numeric prefix can vary (`"10m"`, `"2h"`, `"3d"`). Case matters:
`"1M"` is months, `"1m"` is minutes.

**Mirroring, not importing.** Bifrost's parser lives in
`bifrost/framework/configstore/tables/utils.go` (`ParseDuration` and
`GetCalendarPeriodStart`). The plugin re-implements both in a small
internal package — `gateway/internal/duration/` — rather than taking
a dep on `bifrost/framework` (the plugin already depends on
`bifrost/core` and adding framework grows the dep tree
disproportionately). The re-implementation is ~30 LOC; its tests
cross-check the output against Bifrost's reference implementation
in CI to catch any drift if Bifrost's vocabulary ever extends.

The plugin's duration handling is wire-compatible with anything Hive
configures on the Bifrost side, today and after future Bifrost
extensions, with the caveat that the plugin must explicitly add
support for any new suffix Bifrost adds.

## Redis schema

Canonical key shapes, value types, and TTLs. Supersedes v2 §561-575.

```
bifrost:cost:run:<run_id>                       HASH    { total: float }
bifrost:cost:ua:<ua_nonce>                      HASH    { total: float }
bifrost:steps:run:<run_id>                      HASH    { total: int   }
bifrost:tools:run:<run_id>                      LIST    [<tool>, ...] (capped at 10)
bifrost:kill:<run_id>                           STRING  "1"
bifrost:kill:agent:<agent_name>                 STRING  "1"
bifrost:revoke:<nonce>                          STRING  "1"      TTL = nonce-bearing layer.exp
bifrost:revoke_user_before:<user_id>            STRING  <RFC 3339 UTC>
bifrost:cost:agent:<agent_name>:<bucket_key>    HASH    { total: float }
```

### Namespace

**Every key the gateway plugin reads or writes is prefixed with
`bifrost:`.** This is load-bearing in the Sphinx-swarm deployment, where
the gateway shares the `redis.sphinx` instance with Jarvis (see
`gateway/images/bifrost.rs` and `gateway/docker-compose.yml`). The
prefix keeps the plugin's keyspace cleanly partitioned from every other
consumer's so:

- `KEYS bifrost:*` / `SCAN MATCH bifrost:*` enumerates exactly the
  plugin's state for ops debugging.
- `FLUSHDB`-equivalent operations can be scoped via `SCAN + DEL` on the
  prefix without touching Jarvis state.
- Redis Cluster (future) routes via hash tags; cluster mode does not
  support multi-DB but does support prefixes, so this scheme stays
  valid if the deployment grows past a single redis instance.

The prefix is fixed at `bifrost:` and is **not** configurable. A
configurable prefix would split fixture vocabulary and admin-tooling
docs in two for no operational gain — there is one plugin, it owns one
prefix. Throughout the rest of this doc, every key shape shown is
implicitly preceded by `bifrost:`; the body of each section omits the
prefix for readability after this point, but every Go-side
`redis.Client` call MUST emit it.

The TS issuer side (Hive's `/macaroons/revoke`, etc.) MUST write the
same prefixed keys when fanning out to a workspace's Redis. The
operator-facing CLI examples in `llm-governance-v2.md` and
`cryptographic-identity.md` use the prefixed form.

### `cost:run:<run_id>`

Per-run cost accumulator. Float in dollars (4 decimal places of
precision; Redis `HINCRBYFLOAT` handles this natively).

- **Updated by:** PostLLMHook on each call, walking every `run_id`
  in the macaroon chain (leaf + every ancestor).
- **Read by:** PreLLMHook on each call, compared against
  `effective_caveats.max_cost_usd` from the verified macaroon.
- **TTL:** `max(macaroon.exp - now + 1h, 1h)`, capped at 7d. See
  "TTL policy" below.

### `cost:ua:<ua_nonce>`

Per-`user_authorization` cumulative cost accumulator. Tracks the
**org-signed spending envelope** described in
`phase-4-macaroon-shape.md` ("Budget envelope") — i.e. the org leader
signs a UA from cold storage with `budget.max_total_usd = $X`, the
employee's hot key signs many invocations under it, and the plugin
enforces the cumulative cap across all of them.

The accumulator is per-UA (keyed on `ua.nonce`), not per-user or
per-org, because each UA represents one delegation envelope with its
own ceiling and its own expiry. When the org leader signs a new UA
the following week, that's a fresh nonce and a fresh bucket. The old
UA's bucket TTLs out naturally.

- **Updated by:** PostLLMHook on each call, only when
  `claims.UABudget != nil && claims.UABudget.MaxTotalUSD > 0`.
  Single `HINCRBYFLOAT cost:ua:<ua.nonce> total <cost>` added to the
  existing accumulator pipeline — no extra round-trip.
- **Read by:** PreLLMHook on each call, only when budget is set.
  Single `HGET cost:ua:<ua.nonce> total` added to the existing
  cap-walk pipeline. Compared against `claims.UABudget.MaxTotalUSD`.
- **TTL:** `clamp(ua.exp - now + 1h, 1h, 7d)`. Same formula as
  `cost:run`, but the lifetime axis is the UA's expiry (the
  delegation envelope) rather than the invocation's. This keeps the
  bucket alive across many short invocations under a long-lived UA.

**No bucket ⇒ no enforcement.** Per-invocation budget caps
(`max_per_invocation_usd`) are checked at signature-time in the
verifier and need no Redis state.

### `steps:run:<run_id>`

Per-run step counter. Integer.

- **Updated by:** PostLLMHook, walking ancestors as with cost.
- **Read by:** PreLLMHook, compared against
  `effective_caveats.max_steps`.
- **TTL:** same policy as `cost:run`.

### `tools:run:<run_id>`

Last-10 tool calls for tool-loop detection. Redis LIST capped via
LTRIM after each push.

- **Updated by:** PostLLMHook when the response contains a tool call.
- **Read by:** PreLLMHook tool-loop check (last 10 entries, count
  duplicates against threshold from plugin config).
- **TTL:** same policy as `cost:run`.

### `kill:<run_id>`

Per-run kill switch. Set by Hive UI or admin API when an operator
clicks "stop this run."

- **Updated by:** plugin's `POST /_plugin/runs/:id/kill` admin
  endpoint, or external Redis client.
- **Read by:** PreLLMHook, checks leaf run_id AND every ancestor —
  killing a parent kills every descendant.
- **TTL:** 1h (the kill is a hot operation; if the run isn't dead
  in an hour, something else has gone wrong).

### `kill:agent:<agent_name>`

Per-agent kill switch. Set when an operator wants to halt all runs
under a given agent name across the entire swarm. Matches on
`agents[last]` from the verified macaroon (the most-specific agent).

- **Updated by:** plugin's `POST /_plugin/agents/:name/kill` admin
  endpoint, or external Redis client.
- **Read by:** PreLLMHook, after the per-run kill check.
- **TTL:** 24h. Re-set if the kill needs to persist longer. (A kill
  of "browser-agent" that lasts forever would be a configuration
  question, not a hot-state question — express it by setting the
  agent_budget cap to 0 instead.)

### `revoke:<nonce>`

Macaroon-chain revocation. Presence of any nonce from the chain in
this set rejects the macaroon.

- **Updated by:** Hive's `/macaroons/revoke` (fans out to every
  workspace's Redis) or plugin's admin endpoint.
- **Read by:** PreLLMHook, checks every nonce in
  `Claims.Nonces` (which is `[ua.nonce, inv.nonce, atts[*].caveats.nonce]`).
- **TTL:** the exp of the layer whose nonce is revoked. A revoke of
  an invocation nonce TTLs at `invocation.exp`; a revoke of a
  user_authorization nonce TTLs at `user_authorization.exp`. After
  that point the macaroon is naturally dead and the revocation entry
  serves no purpose.

### `revoke_user_before:<user_id>`

User-level revocation. Stores an RFC 3339 UTC timestamp; any
macaroon whose `user_authorization.iat` is **before** that timestamp
is rejected.

- **Comparison axis:** the plugin checks
  `user_authorization.iat < revoke_user_before:<user_id>`. **Not**
  the invocation's iat — the user-level revocation is about
  invalidating org-issued user authorizations, not individual
  invocations. An invocation signed after the revocation timestamp
  (against a re-issued user_authorization) is fine.
- **Updated by:** Hive's offboarding flow (`PUT /customers/:id
  {is_active: false}` is paired with this Redis write).
- **Read by:** PreLLMHook after macaroon verification.
- **TTL:** none. User revocation is a permanent state until
  explicitly cleared.

### `cost:agent:<agent_name>:<bucket_key>`

Per-agent cost accumulator. `<bucket_key>` is computed from the
configured window and the current time, using `GetCalendarPeriodStart`
semantics from "Duration vocabulary" above.

**Bucket-key format by window suffix:**

| Window | Bucket key format | Example |
|---|---|---|
| `Nd` | `YYYY-MM-DD` | `bifrost:cost:agent:browser:2026-05-16` |
| `Nw` | `YYYY-Www` (ISO week starting Monday) | `bifrost:cost:agent:browser:2026-W20` |
| `NM` | `YYYY-MM` | `bifrost:cost:agent:browser:2026-05` |
| `NY` | `YYYY` | `bifrost:cost:agent:browser:2026` |
| Sub-day (`Nh`, `Nm`, `Ns`) | unix epoch of window start | `bifrost:cost:agent:browser:1747353600` |

Sub-day windows use rolling semantics: the bucket key is the unix
epoch of `now - N * unit`, rounded to that unit. This avoids the
"there is no natural boundary" problem `GetCalendarPeriodStart`
flags for sub-day durations.

For `N > 1` calendar windows (e.g. `2w`, `3M`), the bucket aligns to
the standard calendar boundary of the **smaller** unit (e.g. `2w`
buckets on the most recent Monday; the plugin tracks bucket
ownership for the window length internally). In practice phase-6
ships only `N=1` calendar windows; larger N is a phase-7 extension.

- **Updated by:** PostLLMHook, keyed on the verified
  `agents[last]` from the macaroon.
- **Read by:** PreLLMHook if the agent has a configured budget in
  plugin config.
- **TTL:** 2 × window duration. So a `1d` bucket lives 48h, a `1w`
  bucket lives 14d, a `1M` bucket lives ~60d. Long enough to handle
  late PostHooks; not so long that buckets pile up indefinitely.
  Bucket rollover is implicit — a new day produces a new key with
  no value (Redis treats this as 0).

## TTL policy

The hardcoded 1h TTL in v2 §573 is wrong for any invocation whose
`exp` exceeds 1h, which under the cryptographic-identity model is
common (sessions can be 8h, workflows 24h). Phase 6 replaces it
with a macaroon-aware policy.

**For `cost:run`, `steps:run`, `tools:run` (per-run keys):**

```
TTL = clamp(macaroon.exp - now + 1h, 1h, 7d)
```

Where:
- The +1h grace ensures a final PostHook firing right at exp doesn't
  race with TTL eviction.
- The 1h floor handles edge cases where the macaroon is already
  near or past expiry when the call comes in (the verifier rejects
  the call anyway, but we want the bookkeeping window to be sensible
  if the call somehow lands).
- The 7d ceiling bounds Redis memory growth for any macaroon issued
  with an absurdly long exp. A 30-day invocation would have a 7d
  cost-tracking window, after which a fresh write resets the
  accumulator. The macaroon itself remains valid until its real
  exp; the plugin will lose visibility into spend that happened
  more than 7d before the current call, but practically: an
  invocation spending continuously for 7d has either hit its cap or
  is being abused, and a 7d ceiling forces a refresh of the macaroon
  in either case.

The TTL is **refreshed on every write** via `EXPIRE` (or by using
`HSET … EX`). So an actively-spending run keeps its keys alive past
the nominal TTL for its full lifetime, capped at the 7d ceiling
from the most recent write.

**For `cost:agent` (per-agent windowed keys):** TTL is 2× the window
duration, as specified in the schema. Doesn't depend on macaroon
lifetime — the bucket exists for the configured window regardless
of which runs touched it.

**For `revoke:<nonce>`:** TTL = the exp of the layer whose nonce is
revoked. Hardcoded in Hive's revoke endpoint when it writes.

**For `kill:<run_id>` and `kill:agent:<name>`:** Fixed TTLs (1h and
24h respectively). These are hot operational state; if you need a
longer kill, re-set the key.

**For `revoke_user_before:<user_id>`:** No TTL. Permanent until
cleared.

## Hot path

Concrete algorithm for `PreLLMHook` and `PostLLMHook`, replacing v2
§475-556. All Redis ops within a single hook are issued as one
pipeline (see "Pipelining and atomicity" below).

### PreLLMHook

```
PreLLMHook(ctx, req) → response | bifrost.Error:

  1. CRYPTOGRAPHIC VERIFICATION (in-process, no I/O)
     - Extract x-macaroon header.
     - If missing → 401 macaroon_required.
     - Parse just enough to extract org_id; look up policy from
       in-memory trust registry (phase 5).
     - If untrusted org → 401 untrusted_org.
     - Call auth.Verify(macaroon_b64, policy, time.Now()).
     - If err is *auth.VerifyError → 401 with err.Code.
     - On success, hold the verified Claims for the rest of the hook.

  2. PIPELINE 1 — REVOCATION + KILL CHECKS (one Redis round-trip)
      Build a pipeline issuing:
       - EXISTS bifrost:revoke:<ua.nonce>
       - EXISTS bifrost:revoke:<inv.nonce>
       - EXISTS bifrost:revoke:<att[i].nonce> for each attenuation
       - GET    bifrost:revoke_user_before:<user_id>
       - EXISTS bifrost:kill:<inv.run_id>
       - EXISTS bifrost:kill:<att[i].caveats.run_id> for each attenuation
       - EXISTS bifrost:kill:agent:<agents[last]>
     Submit pipeline; await response.

     Evaluate:
       - Any bifrost:revoke:<nonce> EXISTS → 401 macaroon_revoked.
       - bifrost:revoke_user_before present AND ua.iat < that timestamp
         → 401 user_authorization_revoked.
       - Any bifrost:kill:<run_id> EXISTS → 402 run_killed.
       - bifrost:kill:agent:<agents[last]> EXISTS → 402 agent_killed.

  3. PIPELINE 2 — COST + STEP READS (one Redis round-trip)
     Build a pipeline issuing:
       - HGET bifrost:cost:run:<inv.run_id>           total
       - HGET bifrost:cost:run:<att[i].run_id>        total      (per attenuation)
       - HGET bifrost:cost:ua:<ua.nonce>              total      (if claims.UABudget.MaxTotalUSD > 0)
       - HGET bifrost:steps:run:<leaf_run_id>         total
       - HGET bifrost:cost:agent:<agents[last]>:<bucket>  total  (if configured)
     Submit pipeline; await response.

     Evaluate (cap walk, leaf first, then each ancestor, then envelope):
       - bifrost:cost:run:<leaf> >= claims.effective_caveats.max_cost_usd
         → 402 run_cost_exceeded.
       - For each ancestor in claims.macaroon_chain:
           bifrost:cost:run:<ancestor.run_id> >= ancestor.max_cost_usd
           → 402 run_cost_exceeded.
       - If claims.UABudget.MaxTotalUSD > 0 AND
         bifrost:cost:ua:<ua.nonce> >= claims.UABudget.MaxTotalUSD
         → 402 ua_budget_exceeded.
       - bifrost:steps:run:<leaf> >= claims.effective_caveats.max_steps
         → 402 run_step_exceeded.
       - If agent budget configured AND
         bifrost:cost:agent:<name>:<bucket> >= configured cap
         → 402 agent_cost_exceeded.

     Note: `max_per_invocation_usd` is enforced at signature-time in
     the verifier (`phase-4-macaroon-shape.md` step 8), not here.
     The verifier already rejected the macaroon if
     `invocation.max_cost_usd > ua.budget.max_per_invocation_usd`,
     so by this point we know the per-call cap is honoured.

  4. TOOL-LOOP CHECK (optional pipeline 3, only if request has tools)
     - LRANGE tools:run:<leaf_run_id> 0 9
     - If last N tool names are >= threshold-of-N matches → 402
       tool_loop_detected.
     (Skip the pipeline entirely if the request has no tool history
     yet — first N calls of any run.)

  5. DEFENSE IN DEPTH (in-process)
     - claims.UserID != ctx.CustomerID → 401 macaroon_user_mismatch.
     - claims.effective_caveats.max_cost_usd > plugin.hard_ceiling
       → 402 budget_unreasonable.

  6. STAMP CONTEXT for downstream hooks:
     - ctx.VerifiedClaims = claims
     - ctx.LeafRunID      = claims.RunID
     - ctx.AncestorRunIDs = [inv.run_id] + [att[i].run_id for i in chain]
     - ctx.LeafAgent      = agents[last]
     - ctx.AgentBucketKey = bucketKeyFor(agentBudget.window, now)
                            (cached for PostHook to reuse)
     - ctx.UANonce        = claims.UANonce
     - ctx.UABudget       = claims.UABudget   // nil if absent
```

Total: at most three Redis round-trips. Two for the common case
(no tools). ~1 ms wall-clock on co-located Redis. Plus ~70-200 µs
for the cryptographic verification (~70 µs with the
`(org_id, ua.nonce)` cache warm).

### PostLLMHook

```
PostLLMHook(ctx, resp) → none:

  1. EXTRACT COST AND OUTCOME from resp.Usage / resp.Cost.
     - For streams, this is the cost from the final chunk's
       accumulator.
     - If the call errored before billable usage, cost = 0 but step
       still increments.

  2. COMPUTE TTL for per-run keys:
     ttl = clamp(claims.effective_caveats.exp - now + 1h, 1h, 7d)

   3. PIPELINE 1 — ACCUMULATOR WRITES (one Redis round-trip)
      Build a pipeline issuing, for each run_id in
      [ctx.LeafRunID] + ctx.AncestorRunIDs:
        - HINCRBYFLOAT bifrost:cost:run:<r>   total <cost>
        - HINCRBY      bifrost:steps:run:<r>  total 1
        - EXPIRE       bifrost:cost:run:<r>   <ttl>
        - EXPIRE       bifrost:steps:run:<r>  <ttl>
      If ctx.UABudget.MaxTotalUSD > 0:
        - HINCRBYFLOAT bifrost:cost:ua:<ctx.UANonce>  total <cost>
        - EXPIRE       bifrost:cost:ua:<ctx.UANonce>  <ua_ttl>
        // ua_ttl = clamp(ua.exp - now + 1h, 1h, 7d)
      If agent budget configured:
        - HINCRBYFLOAT bifrost:cost:agent:<ctx.LeafAgent>:<ctx.AgentBucketKey>  total <cost>
        - EXPIRE       bifrost:cost:agent:<ctx.LeafAgent>:<ctx.AgentBucketKey>  <agent_ttl>
      If response includes a tool call:
        - LPUSH        bifrost:tools:run:<ctx.LeafRunID>  <tool_name>
        - LTRIM        bifrost:tools:run:<ctx.LeafRunID>  0 9
        - EXPIRE       bifrost:tools:run:<ctx.LeafRunID>  <ttl>
     Submit pipeline; do NOT await response (fire-and-forget — see
     "Failure modes" below).

  4. RETURN (PostHook is non-blocking from the caller's perspective).
```

Total: one Redis round-trip. ~0.5 ms wall-clock.

## Pipelining and atomicity

**All multi-op sequences within a hook use Redis pipelining**
(`MULTI`-free batching of commands in one round-trip). The plugin
issues N commands and awaits N replies.

**Atomicity is NOT used in phase 6.** This is a conscious choice:

- **For revocation/kill checks (PreHook pipeline 1):** the checks are
  read-only and idempotent; no atomicity issue.
- **For cost cap checks (PreHook pipeline 2):** two concurrent calls
  from the same run can both read `cost:run:<r>` before either has
  written via PostHook, and both pass the cap check, and both
  proceed — briefly exceeding the cap by `(concurrency - 1) ×
  per_call_cost`. In practice this overshoot is small: at $0.50/call
  with 10 concurrent calls against a $200 cap, the worst-case
  overshoot is $4.50 — 2.25% over the nominal cap. **Accepted.**
- **For PostHook writes:** each `HINCRBYFLOAT` is individually atomic
  on the Redis server. The pipeline doesn't need to be transactional
  because every command is idempotent **on its own value**
  (HINCRBYFLOAT by `+X` is commutative; LPUSH+LTRIM is bounded; EXPIRE
  is set-not-add). Out-of-order delivery of the pipeline writes
  produces the same final state regardless of order.

**When atomicity would matter** (deferred to phase 7 if real
operational data demands it):

- **Strict cap enforcement** for keys with high write contention.
  A Lua script doing `GET + compare + INCRBYFLOAT` atomically would
  eliminate overshoot at the cost of one round-trip becoming one
  Lua eval. Worth considering for per-agent budgets if many agents
  share one key under heavy concurrency.
- **Cross-run consistency** during multi-write PostHooks. Currently
  if the plugin crashes mid-pipeline, some ancestor counters may
  have been updated and others not. Acceptable because the loss
  is bounded and re-tries don't compound it; the next call's
  accumulator reads will reflect whatever subset of writes
  succeeded.

The overshoot bound and the consciously-accepted inconsistencies
are documented here so an operator reading dashboards never
mistakes "spend $4.50 over the $200 cap" for a bug.

## Failure modes

### Redis unreachable

**Fail closed for auth checks. Fail open for accounting.**

- PreLLMHook pipeline 1 fails (revocation/kill reads): **401
  revocation_check_unavailable**. The plugin cannot determine
  whether the macaroon is revoked; rejecting is the safe path.
- PreLLMHook pipeline 2 fails (cost reads): **402
  budget_check_unavailable**. The plugin cannot determine current
  spend; rejecting is the safe path.
- PostLLMHook pipeline fails: **log loudly, do not retry, do not
  block the response.** The call's spend goes uncounted for this
  workspace. Alerting fires from the plugin's own metrics. The
  resulting accumulator drift is bounded by Redis outage duration ×
  call rate; it is acceptable for accounting to lag during outages
  because the **next** PreHook (once Redis is back) will see the
  partially-updated state, and the auth correctness invariant (no
  unverified call passes) is preserved throughout.

This matches v2 §602 ("Auth-correctness wins over availability")
and is the only sane posture for a security-sensitive enforcement
layer.

### PostLLMHook retry / replay

Bifrost may invoke PostLLMHook more than once for the same response
under failure-recovery scenarios (network blip during streaming,
plugin restart between PreHook and PostHook, etc.).

**Phase 6 PostHook is NOT idempotent on duplicate invocation.** A
duplicate fire produces a duplicate `HINCRBYFLOAT`. This is a known
quasi-bug whose remediation is in phase 7:

- The simplest fix is a per-call dedupe key: `SETNX
  posthook:done:<response_id> 1 EX 600` at the top of PostHook;
  bail if it returns 0.
- Phase 6 doesn't ship this because Bifrost's exact PostHook
  retry semantics aren't pinned down yet, and the duplicate-fire
  risk in practice is low (one duplicate per provider failure ≈ a
  ~0.1% accounting error, well below the cap overshoot tolerance
  already accepted).

When Bifrost's semantics solidify, the dedupe key is a one-line
addition. The schema field `posthook:done:<response_id>` is
reserved here so it doesn't need a separate spec.

### Crash between PreHook and PostHook

The call has hit the upstream provider and burned money; the
plugin crashes before PostHook can record the spend. On restart,
the next call's PreHook reads `bifrost:cost:run:<r>` and gets an
under-count.

**Accepted.** Same posture as Redis-down accounting: the bound on
inconsistency is one call's cost per crash, which is acceptable
given the cap and the kill-switch availability. No protocol
mechanism exists to recover the lost write — the upstream provider
has no callback that would let us know "you successfully spent $X
for run Y but never recorded it."

The mitigations are operational: monitor the plugin's process
health, alert on crashes, reconcile against Bifrost's logs.db
spend (which is written by Bifrost-core's logging plugin, *after*
our PostHook, and so survives our crash) on a daily basis.

### Macaroon expires mid-PostHook

The macaroon is valid at PreHook time but its `exp` has passed by
the time PostHook fires. Two questions:

- **Should the cost still be recorded?** Yes — the call happened
  and spent money. The accumulator must reflect reality regardless
  of whether the macaroon is still alive.
- **What TTL should the write get?** The same formula:
  `clamp(macaroon.exp - now + 1h, 1h, 7d)`. Since `now > exp`, the
  computed value is `< 1h`, so the clamp floor of 1h applies. The
  key TTL is 1h after the write, which is enough for any straggler
  state inspection to read it.

### Bucket boundary mid-call

The call arrives at 11:59:58 UTC and the PostHook fires at 12:00:01
UTC, crossing a day boundary. PreHook checks `bifrost:cost:agent:browser:
2026-05-16`; PostHook writes to `bifrost:cost:agent:browser:2026-05-17`.

**Accepted.** The boundary is unambiguous from the time the op is
issued (PostHook computes its own bucket key from its own clock,
not from PreHook's). The effect: the call's cost is charged to the
new day's bucket; the old day's bucket reflects only calls whose
PostHook fired before midnight. This matches what any time-bucketed
metering system does.

The alternative — passing the bucket key from PreHook to PostHook
via context — produces stranger behavior (a call billed against
yesterday despite its accumulator updating today). The current
choice is the principle of "bucket by the time of the write."

## Configuration

The plugin's enforcement layer takes the following configuration
inputs (passed via `plugin.yaml` or environment, format defined in
the plugin's deployment doc):

```yaml
# Enforcement hard ceilings (defense in depth — the verifier already
# narrows on macaroon caveats, but a misissued macaroon can't exceed
# these).
hard_ceiling:
  per_invocation_cost_usd: 1000.00
  per_invocation_steps:    10000

# Per-agent budgets. Empty by default; turn on per agent.
agent_budgets:
  browser:
    cap_usd: 500.00
    window:  "1d"        # Bifrost duration vocabulary
  pr-monitor:
    cap_usd: 1000.00
    window:  "1d"
  monthly-research-agent:
    cap_usd: 5000.00
    window:  "1M"

# Tool-loop detection.
tool_loop:
  window:    10           # last N tool calls
  threshold: 8            # this many duplicates triggers 402

# Redis connection.
redis:
  url:           ${REDIS_URL}
  pool_size:     50
  pipeline_size: 64       # max ops per pipeline before forced flush
```

Per-agent `window` accepts any Bifrost duration suffix. The bucket
key format follows the table in "Duration vocabulary" above.

## Plugin HTTP surface

All phase-6 endpoints live under the plugin's `/_plugin/*` namespace,
served by `gateway/internal/adminapi/` on a loopback HTTP server
(`127.0.0.1:8189`), reverse-proxied to the container's public port
by the wrapper binary. See
[`phase-3-swarm-handoff.md`](./phase-3-swarm-handoff.md) for the full
wrapper architecture, and v2 §"Observability" for the broader
endpoint inventory. All endpoints require the per-swarm shared
bearer secret (same auth scheme as phase 5's `/_plugin/trust/*`).

### Hot-state admin endpoints (Redis only)

```
POST /_plugin/runs/:run_id/kill
  effect:  SET bifrost:kill:<run_id> "1" EX 3600
  returns: { run_id, killed_at }

POST /_plugin/agents/:name/kill
  effect:  SET bifrost:kill:agent:<name> "1" EX 86400
  returns: { agent_name, killed_at }

DELETE /_plugin/runs/:run_id/kill
  effect:  DEL bifrost:kill:<run_id>

DELETE /_plugin/agents/:name/kill
  effect:  DEL bifrost:kill:agent:<name>

GET /_plugin/runs/:run_id/state
  returns: {
    cost:  <float>,        // from HGET bifrost:cost:run total
    steps: <int>,          // from HGET bifrost:steps:run total
    tools: [<str>...],     // from LRANGE bifrost:tools:run 0 9
    killed: <bool>,        // EXISTS bifrost:kill:run
    ttl_seconds: <int>,    // TTL bifrost:cost:run
  }

GET /_plugin/agents/:name/state?window=1d
  returns: {
    agent_name:      <str>,
    window:          <duration>,
    bucket_key:      <str>,
    current_spend:   <float>,  // HGET bifrost:cost:agent:<name>:<bucket> total
    configured_cap:  <float | null>,  // from agent_budgets
    killed:          <bool>,
  }
```

### Observability endpoints (logs.db, optionally mixed with Redis)

Read endpoints that aggregate over Bifrost's `logs.db`. These do
NOT touch SQLite/Postgres directly — instead they call Bifrost's
own `/api/logs` API over loopback (`http://127.0.0.1:8080/api/logs`)
and compose `SearchFilters.MetadataFilters` against the dim keys
the plugin stamped during `PreLLMHook` canonicalization. This keeps
the plugin decoupled from Bifrost's storage schema and works
identically against SQLite and Postgres backends.

```
GET /_plugin/spend/by-user?window=24h
GET /_plugin/spend/by-agent?window=24h         ← MetadataFilters: agent-name
GET /_plugin/spend/by-realm?window=24h         ← MetadataFilters: realm-id
GET /_plugin/spend/by-session?window=24h       ← MetadataFilters: session-id
GET /_plugin/spend/by-model?window=24h
GET /_plugin/histogram/cost?window=24h&bucket=1h&dimension=agent-name
GET /_plugin/histogram/tokens?window=24h&bucket=1h&dimension=user-id

GET /_plugin/runs/:run_id                      ← logs.db: all calls in run
GET /_plugin/sessions/:session_id              ← logs.db: GetSessionLogs (native to LogStore)
GET /_plugin/users/:user_id/spend?window=24h
GET /_plugin/users/:user_id/quota              ← Bifrost customer cap + live Redis spend
GET /_plugin/agents/:name/spend?window=24h
```

Observability shape evolves with Hive UI demand — the table above
is the v1 surface. The full inventory lives in v2 §"Observability".

## Performance characteristics

Total per-call overhead in steady state:

| Phase | Wall-clock | Notes |
|---|---|---|
| Bifrost VK auth | ~1 µs | In-process, no I/O |
| Crypto verify (warm cache) | ~70 µs | secp verify cached by `(org_id, ua.nonce)` |
| PreHook pipeline 1 (revoke/kill) | ~0.5 ms | 5-8 Redis ops, one round-trip |
| PreHook pipeline 2 (cost/steps) | ~0.5 ms | 3-6 Redis ops, one round-trip |
| PreHook pipeline 3 (tools, optional) | ~0.5 ms | Skipped on first N calls |
| Model API call | 500 ms - 30 s | The actual work |
| PostHook pipeline | ~0.5 ms | 4-8 Redis ops, one round-trip |
| **Auth/enforcement overhead** | **~1.5 ms** | **≪ 1% of model latency** |

**Cost grows O(1) per call** in the count of accumulated calls.
HINCRBYFLOAT does not slow down as the counter grows; HGET does
not slow down as the key has been incremented more. An 8-hour
session with 100,000 calls has the same per-call enforcement cost
as a 10-minute session with 50 calls.

**Cost grows O(depth) per call** in the macaroon chain depth (one
ancestor cost lookup per level). Typical chains are 2-3 deep;
10-deep chains add ~5 ms total. Not a meaningful overhead.

**Redis throughput on a single co-located instance:** ~16k LLM
calls/sec at ~6 ops per call. Throughput per-workspace; multiple
workspaces have independent Redis instances. Per workspace this is
roughly an order of magnitude above expected steady-state load.

## What is NOT in this phase

- **PostHook idempotency / deduplication.** Reserved schema key
  `posthook:done:<response_id>` but no implementation; depends on
  Bifrost's retry semantics being pinned down. Phase 7.
- **Lua atomic check-and-increment for cap enforcement.** Deferred
  unless operational data shows real overshoot pain. Phase 7.
- **Multi-Redis sharding.** Single Redis per workspace, as v2
  assumes. If a single workspace's throughput ever outgrows this,
  the plugin grows a hash-by-run_id sharder. Not phase 6.
- **Per-user, per-workspace, or per-org budgets** at the plugin
  level. These are Bifrost-Customer-side concerns (v2 §115-122)
  and don't live in plugin Redis.
- **Cross-workspace aggregation.** Each workspace's plugin has its
  own Redis and its own per-agent counters. "What did browser-agent
  spend across all workspaces this week" is a Hive-side question
  answered against logs.db, not against plugin Redis.
- **Revocation list distribution mechanism.** Phase 6 reads
  `bifrost:revoke:<nonce>` from Redis; how that key got there (Hive
  push, swarm pull, macaroon-carried) is decided by phase 5 and the
  identity doc.

## Wire-up checklist

**Plugin adapter (`gateway/internal/auth/`):**

- [ ] `verifier.go`: wraps `gateway/auth/go` with bifrost.Request
      header extraction, trust-registry lookup, and bifrost.Error
      shaping for crypto failures
- [ ] `redis.go`: pipeline helpers — `revokeAndKillPipeline`,
      `costAndStepsPipeline`, `toolsPipeline`, `accumulatorPipeline`
- [ ] `ttl.go`: `runKeyTTL(macaroonExp time.Time) time.Duration`
      implementing clamp(exp-now+1h, 1h, 7d)
- [ ] `enforcement.go`: PreLLMHook + PostLLMHook implementations
      against the verified Claims
- [ ] `admin.go`: state-query and mutation primitives (`RunState`,
      `AgentState`, `KillRun`, `UnkillRun`, `KillAgent`,
      `UnkillAgent`) — pure Redis logic, no HTTP. Consumed by the
      `adminapi` handlers below.
- [ ] `config.go`: structured `agent_budgets`, `hard_ceiling`,
      `tool_loop`, `redis` config types
- [ ] Test fixtures covering: per-run cap walk, per-agent budget
      hit, kill:run, kill:agent, revocation, tool-loop, Redis-down

**Plugin admin HTTP (`gateway/internal/adminapi/`):**

- [ ] `runs.go`: `/_plugin/runs/:run_id/{state,kill}` handlers,
      thin wrappers over `auth.RunState` / `auth.KillRun` and
      friends
- [ ] `agents.go`: `/_plugin/agents/:name/{state,kill,spend}`
      handlers (state+kill → `auth/`; spend → logstore client)
- [ ] `spend.go`: `/_plugin/spend/by-*` handlers — compose
      `MetadataFilters` and call the logstore client
- [ ] `histogram.go`: `/_plugin/histogram/*` handlers — call
      `GetDimensionCostHistogram` etc. via the logstore client
- [ ] `sessions.go`, `users.go`: drill-down handlers
- [ ] `logstore_client.go`: HTTP client to
      `http://127.0.0.1:8080/api/logs` with `SearchFilters`
      serialization; reused by all observability handlers
- [ ] Route registration in `server.go` (mirror the
      `SetTrustRegistry` pattern: a `SetRedisClient` setter that
      `main.Init` calls before `Start()`)

**Internal duration util (`gateway/internal/duration/`):**

- [ ] `duration.go`: `ParseDuration` mirroring Bifrost's
      `framework/configstore/tables/utils.go`
- [ ] `bucket.go`: `BucketKey(window string, now time.Time) string`
      returning the strings from "Duration vocabulary" table
- [ ] `bucket_test.go`: cross-check tests asserting our parser
      agrees with Bifrost's reference implementation byte-for-byte
      on a fixed input set (run as part of CI; alerts if either
      side drifts)

**Plugin context (`gateway/internal/pluginctx/`):**

- [ ] Add `VerifiedClaims`, `LeafRunID`, `AncestorRunIDs`,
      `LeafAgent`, `AgentBucketKey` to the context shape so
      downstream hooks can read what PreHook stamped

**Documentation:**

- [ ] Update `gateway/internal/auth/doc.go` to drop "stub" and
      reference this phase doc
- [ ] Update `gateway/internal/adminapi/server.go` doc to note the
      new route families (runs, agents, spend, histogram, sessions,
      users) alongside the existing trust/credentials routes
- [ ] Update `llm-governance-v2.md` §"Gateway plugin" and
      §"Observability" with forward-pointers to this phase

**Gate:** plugin enforcement doesn't ship until the test fixtures
demonstrate all six enforcement modes (per-run cap, per-agent
budget, run kill, agent kill, nonce revocation, user revocation)
work end-to-end against a real Redis.

## What this design buys

- **One Redis schema, one set of ops, one TTL policy** — operators
  can reason about Redis memory and throughput from this doc alone.
- **Bifrost-compatible duration vocabulary** — `1d`, `1w`, `1M`,
  `1Y` mean exactly what they mean in Hive's reconciler config.
  No new vocabulary for operators to learn.
- **TTLs that actually work** for the cryptographic-identity model's
  long-lived invocations — the 8-hour session and 24-hour workflow
  cases described in the identity doc now have accumulators that
  outlive the session.
- **Per-agent kill switch first-class** — fills the gap identified
  in design discussion. Operators can stop a misbehaving agent
  across the whole swarm with one admin call.
- **Explicit failure-mode contract** — operators know what happens
  on Redis outages (auth fails closed; accounting fails open with
  bounded drift), on plugin crashes (one call's worth of unrecorded
  spend at most), on bucket boundaries (charge by PostHook time),
  and on macaroon-expired-mid-call (still record).
- **Sub-millisecond hot path** — three Redis round-trips per call
  in the common case, ~1.5 ms total, ≪ 1% of model API latency.
  Performance is not a deployment blocker at any realistic load.
- **No new wire format and no new crypto** — this phase is pure
  operational plumbing. The cryptographic correctness invariants
  from phase 4 are not touched.
