# Cryptographic Identity for LLM Governance

> Companion to `llm-governance-v2.md`. v2 describes the full governance
> stack (Bifrost, macaroons, plugin, Redis). This doc replaces v2's
> identity model — its symmetric HMAC root key and Hive-issued daily
> root macaroons — with a layered, asymmetric trust chain rooted in
> the **organization** itself.
>
> Read v2 first for context. Then read this for what changes at the
> identity layer.

## Why this exists

v2's macaroons are signed with a single symmetric HMAC root key shared
between Hive's macaroon issuer and every workspace (realm) plugin. That key
is the crown jewel: anyone holding it can mint a macaroon claiming to
be any user, in any organization, for any amount. From a user's point
of view, the **system can spoof them.** From an org's point of view,
the **system is the org.**

That's wrong for what this platform actually is. The platform is
infrastructure that _carries_ org-authorized agent work — it isn't the
authority issuing that work. The cryptography should match the
principal hierarchy in the real world:

```
ORGANIZATION  ── delegates authority to ──▶  USERS
USERS         ── delegate per-invocation ──▶  AGENTS
AGENTS        ── attenuate to sub-agents ──▶  SUB-AGENTS
```

Every LLM call should carry a verifiable signature chain back to an
**org's** root key, not to a system-held secret. Once that property
holds, a second thing becomes possible: **swarms can trust each
other's work without coordinating through a central authority.** A
macaroon presented to swarm B by an agent acting on swarm A's behalf
is verifiable from the signatures alone, no callback to Hive, no
shared database. That's the cross-swarm story this platform is built
for: agents that pull from multiple swarms' databases and knowledge
bases to complete a task, all of it bounded by one verifiable chain
of authorization from one org.

## The three-principal model

There are three principals in this system, and exactly three keys:

```
┌────────────────────────────────────────────────────────────────┐
│  ORG ROOT KEY        secp256k1, multisig-capable               │
│    Represents the organization itself.                         │
│    Held by the org (any multisig configuration the org wants). │
│    Signs: user-key authorizations and revocations.             │
│    Frequency: rare — user onboarding, key rotation.            │
├────────────────────────────────────────────────────────────────┤
│  USER KEY            Ed25519, hardware-friendly                │
│    Represents one human (or service identity).                 │
│    Held by the user (Yubikey, Passkey, mobile secure enclave,  │
│    or by Hive in custodial mode).                              │
│    Signs: invocation macaroon roots.                           │
│    Frequency: per-invocation, per-session, or whatever cadence │
│    the org's policy allows — see "Policy is local."            │
├────────────────────────────────────────────────────────────────┤
│  INVOCATION MACAROON   HMAC chain, attenuates to sub-agents    │
│    Represents one unit of agent work in flight.                │
│    Held in the agent process for the run lifetime.             │
│    Carries: caveats (realm, agent, max_cost_usd, exp, …).  │
│    Extends via HMAC attenuation when sub-agents are spawned.   │
└────────────────────────────────────────────────────────────────┘
```

**There is no fourth key.** No shared HMAC root, no platform-held
signing key, no system identity. Every signature in the chain comes
from one of these three principals.

## Curve choices

**Org root: secp256k1.** Chosen for multisig flexibility. Orgs need to
be able to compose root authority from multiple parties — a swarm
operator's automated key plus a CEO's mobile-app key, or any other
combination the org defines. secp256k1 has mature multisig tooling,
threshold signature schemes, and hardware-wallet support across the
ecosystems orgs are likely to already operate in. It also gives
unhardened child derivation for orgs that want to issue
sub-organization keys without re-registering each one.

**User key: Ed25519.** Chosen for hardware availability at the
end-user level. Yubikey (PIV / FIDO2), WebAuthn passkeys, mobile
secure enclaves, and modern OS keychains all support Ed25519 natively.
The user-level key is what real humans interact with, so it has to
work with whatever device they already have.

**Macaroon attenuation: HMAC-SHA256.** Unchanged from v2 in form, but
no longer rooted in a shared system secret. The HMAC chain now hangs
off the user's Ed25519 signature over the invocation macaroon's root
caveats. Children extending the chain HMAC over the previous
signature, exactly as the macaroon protocol specifies — the chain
math is unchanged, only its root is.

The three curves don't compose with each other beyond the
"previous-signature → next-signature" handoff at each layer
boundary. Each layer's verifier only needs to understand its own
curve.

## The signature chain on the wire

A complete invocation macaroon carries everything a verifier needs to
walk the chain back to the org root, without callbacks:

```
{
  org_id:                "org_acme",
  org_pubkey_hint:       <kid / multisig-policy-id>,

  user_authorization: {                       ← signed by ORG ROOT
    user_id:             "u_alice",
    user_pubkey:         <Ed25519 pubkey>,
    permissions:         { realms: […], agents: […], … },
    exp:                 <iso8601>,
    nonce:               <random>,
    org_sig:             <secp256k1 multisig over the above>,
  },

  invocation: {                               ← signed by USER KEY
    realms:           "w1",
    agents:              ["coder"],
    run_id:              <uuid>,
    max_cost_usd:        5.00,
    max_steps:           100,
    max_wallclock_s:     600,
    iat:                 <iso8601>,
    exp:                 <iso8601>,
    nonce:               <random>,
    user_sig:            <Ed25519 over the above>,
  },

  attenuations: [                             ← HMAC chain
    { caveats: { agents: ["coder","web-search"],
                 max_cost_usd: 2.00,
                 max_wallclock_s: 120,
                 run_id: <child uuid>,
                 exp: <iso8601>,
                 nonce: <random> },
      hmac:    <HMAC over (prev_sig, caveats)> },
    …
  ],
}
```

Three signatures (or sigs+chain), three layers. The **org signs the
user**, the **user signs the invocation**, the **invocation attenuates
to sub-agents via HMAC**.

Wire format details (CBOR vs. JSON, exact field encoding, canonicalization
rules) are an implementation choice, not a design decision. The structure
above is the abstract shape; the implementation should follow whatever
the chosen macaroon library produces, with the additional `user_authorization`
envelope wrapped around it.

## Per invocation, not per day

v2 had a daily root macaroon held server-side and re-minted as
sessions refreshed, with shorter invocation macaroons attenuated off
it. That layer goes away.

Every invocation gets its own freshly-signed macaroon. The user
signature is over the invocation's caveats directly — including
`exp`, `max_cost_usd`, `run_id`. There is no intermediate
"session-level" macaroon as a separate cryptographic object.

That said, **what counts as an invocation is local policy.** An
invocation can be:

- A single agent run (10-minute scope, $5 budget) — the default for
  ad-hoc spawns.
- A long-lived session (8-hour scope, $200 budget, single workspace)
  — for a user actively pairing with an agent.
- An overnight workflow (24-hour scope, $50 budget, narrow agent set)
  — for a scheduled cron task.

The protocol doesn't distinguish. Only the caveats differ. The
user-key signature commits to _some_ set of caveats; whatever those
caveats permit, the resulting macaroon can do. Sub-agents always
attenuate within whatever the user signed.

## Plugin verification

Each workspace's plugin holds:

- A **trust registry** — which org root pubkeys (or multisig policies)
  the plugin's swarm accepts macaroons from. Configured at swarm
  setup; updatable.
- A **cache** of recently-seen user pubkeys (extracted from
  `user_authorization` envelopes). Pure verification optimization;
  the macaroon carries the pubkey itself, so the cache is just for
  skipping redundant org-sig verifications.

The plugin holds **no signing keys.** It's a pure verifier. (This is
the structural win over v2: in v2 every workspace's plugin held the
HMAC root key and was therefore part of the trusted-issuer set; here
it's just a checker.)

On each LLM call:

```
1. Parse the macaroon (org_id, user_authorization, invocation,
   attenuations).

2. Look up the trusted org pubkey/policy for org_id.
   If not in trust registry → 401 untrusted_org.

3. Verify user_authorization.org_sig against the org's pubkey/policy.
   - For multisig orgs: verify the signature satisfies the org's
     declared policy (m-of-n, weighted, etc.).
   If invalid → 401 invalid_user_authorization.

4. Check user_authorization caveats:
   - exp not passed
   - user not in revocation list for org_id
   - workspace/agent permissions allow what the invocation claims
   If any fail → 401 user_authorization_violated.

5. Verify invocation.user_sig against user_authorization.user_pubkey.
   If invalid → 401 invalid_invocation_signature.

6. Check invocation caveats:
   - exp not passed
   - nonce not revoked
   - max_cost_usd, max_steps within plugin's hard ceiling
   - user_id matches Bifrost ctx customer_id (defense-in-depth)
   If any fail → 401 invocation_violated.

7. Walk the HMAC attenuation chain (as v2 already specifies).
   - Each attenuation extends prev_sig via HMAC over its caveats.
   - Verify each link.
   - At each level, check that caveats strictly narrow the parent.
   If any fail → 401 attenuation_invalid.

8. Check all enforcement conditions (run cost, kill switch, tool loop)
   exactly as v2 plugin already does.
```

Steps 1–7 are the new identity verification. Step 8 is the v2 plugin's
existing enforcement, unchanged.

Total cost on the hot path: one secp256k1 verify (or one multisig
verify, which is a small constant multiple), one Ed25519 verify, plus
the HMAC chain walk. Sub-millisecond on the workloads this platform
sees, and **the secp verify can be cached** by `(org_id,
user_authorization.nonce)` because user_authorization changes far less
often than per-invocation.

## Issuer endpoints

The Hive macaroon issuer replaces v2's auth service. The endpoint
surface changes to match the new model: spawners no longer attenuate
a server-side daily root, they request a freshly-signed invocation.

```
POST /macaroons/issue
  body:    {
    org_id:        "org_acme",
    user_id:       "u_alice",
    realm:     "w1",
    agent:         "coder",                   ← name from the agent registry
    run_id:        "r_01H...",
    override?:     {                          ← optional caller narrowing
      max_cost_usd?:    number,
      max_steps?:       number,
      max_wallclock_s?: number,
      exp?:             iso8601,
    },
  }
  returns: complete invocation macaroon (the wire shape in "The
           signature chain on the wire" above)
  behavior:
    1. Resolve agent defaults from the agent registry (see
       agent-registry.md). Reject if the agent is unknown or
       disabled for org_id.
    2. Apply override fields as min() narrowings on the defaults;
       the issuer never widens.
    3. In phase 1 (custodial): sign user_authorization with the
       org's custodial key, sign invocation with the user's
       custodial key, return the assembled macaroon.
    4. In phase 2+: hold a pre-signed user_authorization for the
       (org, user) pair; request an invocation signature from the
       user's device-held key; assemble and return.

POST /macaroons/revoke
  body:    { org_id, nonce } | { org_id, user_id }
  effects: writes bifrost:revoke:<nonce> or
           bifrost:revoke_user_before:<user_id> to every relevant
           workspace's Redis (fan-out; see phase-6 "Namespace").
  auth:    requires org admin authorization in phase 1;
           requires an org-root signature in phase 3.
```

v2's `POST /macaroons/login` and `POST /macaroons/attenuate`
endpoints are **deleted.** Login produces a Hive session as it always
did; the session no longer holds a daily root macaroon. Spawners that
v2 documented as calling `/macaroons/attenuate` call `/macaroons/issue`
instead — the difference is that the issuer is producing a new
user-signed invocation rather than attenuating a server-held root.

Parent-to-child attenuation **inside a running agent** is unchanged
from v2: the parent attenuates its own macaroon locally with the
HMAC chain, no network call. Only the issuance of a brand-new
invocation root touches the issuer.

## Trust registration

A swarm operator decides which orgs that swarm trusts. There is no
implicit global trust.

```
swarm.trust.orgs:
  - org_id: "org_acme"
    pubkey: "0x..."                  # or multisig policy object
    policy:
      max_invocation_cost_usd: 100   # local backstop
      allowed_realms: [w1, w2]   # if subset desired
  - org_id: "org_partner"
    pubkey: "0x..."
    policy:
      max_invocation_cost_usd: 25    # smaller trust budget
```

A swarm can trust:

- **Its own org** — the common case. The swarm runs inside an org's
  infrastructure and trusts that org's root.
- **Other orgs** — for cross-org agent work (partner integrations,
  service offerings). Each trusted org gets its own local policy
  layered on top of whatever the org's user_authorization claims.

This is what makes cross-swarm agent work clean. Swarm B doesn't need
to call Hive or Org A's servers to verify an agent acting under Org
A's authority — Swarm B verifies the signatures locally against Org
A's registered pubkey.

A swarm's local policy is a **further attenuation** at verification
time: even if Org A's user_authorization permits $1000 invocations,
if Swarm B's local trust policy caps Org A at $100, the invocation is
bounded by $100 on Swarm B. The min of all caveats and policies along
the chain wins.

### Two separate auth concerns

It's worth being explicit about a distinction that's easy to conflate:

- **Transport auth on the trust-registry admin surface** — who is
  allowed to mutate the swarm's trust registry over HTTP. This is a
  swarm-operator concern, not an org concern. In the sphinx-swarm
  integration this reuses the existing per-swarm shared secret
  (`stakwork_secret` in boltwall); in non-swarm deployments it's any
  bearer the operator configures. The plugin doesn't care which.
- **Cryptographic identity of the org being registered** — the
  secp256k1 pubkey (or multisig policy) that signs `user_authorization`
  envelopes. This is the org's concern and is what every macaroon
  ultimately verifies against.

A swarm operator with the transport-auth secret can register an org's
pubkey; they cannot _forge_ that org's signatures. Conversely, an org
that holds its root key cannot mutate a swarm's trust registry over
HTTP without also holding the swarm's transport-auth secret. The two
authorities are independent, which is correct: the swarm decides
"do I trust this org," the org decides "am I the org I claim to be."

How a swarm sources its trust registry (env-var seed, admin-API call,
persisted state across restarts) is operator policy; see
`phases/phase-2-trust-registry.md`.

## Revocation

Revocation happens at each layer independently:

| Layer                             | Mechanism                                                                               | Effect                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Org revokes a user                | Org publishes a revocation entry signed by org root                                     | Plugin rejects any user_authorization for that user_id |
| Org rotates root key              | Org publishes a new root pubkey; old continues to verify until configured grace expires | All future user_authorizations signed by new root      |
| User revokes a session/invocation | User key signs a revocation entry for a nonce                                           | Plugin rejects macaroons with that nonce               |
| Sub-agent kill                    | Run-id-keyed kill switch in plugin's Redis (as v2)                                      | Plugin returns 402 run_killed                          |
| Macaroon natural expiry           | `exp` caveat                                                                            | Plugin rejects expired                                 |

How revocation lists are **distributed** is a separate concern:

- **Pull** — plugin periodically fetches `revocations.json` from each
  trusted org's well-known URL. Simple, eventually consistent, no
  inbound connections needed.
- **Push** — org pushes revocations to subscribing swarms. Faster but
  requires connectivity.
- **Macaroon-carried** — recent revocations included in the macaroon
  itself. Smallest deployment surface, but caveat-bloat.

Pull is the default. The doc doesn't prescribe a frequency — that's
local policy. Critical revocations (compromised key) propagate at
whatever the polling interval is, which sets the worst-case window.

The plugin **fails closed** on revocation-list staleness past a
configured threshold: if it hasn't been able to fetch updated
revocations for an org in N minutes, it rejects that org's macaroons
rather than risk accepting a revoked one. Sets a hard floor on
revocation latency at the cost of availability if the source is down.

## Policy is local

Everything above is the **protocol.** How an org or user actually
operates the keys is **local policy** and is intentionally out of
scope of this doc.

Examples of policy that vary by deployment:

**Org root key custody.**

- One swarm operator's key (effectively 1-of-1; convenient for solo
  developers).
- Multisig with the swarm key + the CEO's Sphinx-app key (2-of-2;
  routine ops require both, but the swarm key alone can sign nothing).
- Threshold multisig with weighted parties (board approval for large
  budget changes, swarm-only for routine user provisioning).

The plugin only needs the **verification rule** for the org's policy;
it doesn't care how signatures get produced.

**User key custody.**

- Custodial — Hive holds the privkey. Phase-1 default.
- Yubikey — user touches device per invocation. Appropriate for
  high-value or long-running invocations.
- Yubikey + session delegation — user touches device once per login,
  software key signs invocations after. Appropriate for high-frequency
  use.
- Sphinx app / mobile secure enclave — biometric per invocation.
- Hardware wallet — for users who already use one.

The plugin only needs to verify the Ed25519 signature against the
pubkey in the user_authorization; it doesn't care which device
produced it.

**Signing cadence.**

- Per-invocation — fresh signature every spawn. Strongest binding,
  highest signing volume.
- Per-session — one signature covers a session's worth of invocations,
  attenuated thereafter via HMAC. Lower signing volume, longer-lived
  authority.
- Hybrid — short-running tasks signed per-invocation; long-running
  sessions signed once at start with a long `exp`.

Different orgs will land in different places. A finance team running
a five-hour analytical agent might want a Yubikey touch per
invocation. A developer team running dozens of small ad-hoc spawns
per day might want a per-session signature with software-key
sub-signing. The protocol supports both with the same verifier.

## Phase staging

The protocol stays the same across phases. What changes is who holds
which keys.

**Phase 1: custodial mode.**

- Org root key: held by Hive (or by the swarm). Effectively a renamed
  HMAC root, with the structural advantage that it's the _org's_ key
  in name and in revocation semantics — and can be migrated to a
  user-held / multisig key without protocol changes.
- User keys: held by Hive on behalf of each user.
- Signing: Hive's macaroon issuer signs on behalf of org and user.
- Cryptographic spoofing resistance: same as today (Hive can mint
  anything). The **structural** spoofing resistance is real (every
  call traces to an org-key signature, revocation is per-org, trust
  registries make swarms first-class).

**Phase 2: user-key migration.**

- Users opt in to holding their own keys (Yubikey, Passkey, Sphinx
  app). The user_authorization envelope is signed once by the org
  (at onboarding, with the user's pubkey embedded). The invocation
  signature happens on the user's device per the cadence the org
  allows.
- Hive stops being able to mint as that user.

**Phase 3: org-key migration.**

- Org adopts multisig for root. Swarm key + CEO Sphinx-app key (or
  whatever shape the org chooses). New user onboardings require the
  configured multisig threshold. Routine ops can still be 1-of-n
  swarm-only if the policy allows.
- Hive stops being able to mint as the org.

Each phase is opt-in per org and per user. A single workspace can
have phase-1 users alongside phase-2 users; the plugin verifies each
according to the keys present in their macaroons. No flag day.

## Relationship to v2

v2 stands as written for everything **except**:

- v2's `hmac_root_key` plugin config field — **delete.** The plugin
  no longer holds any signing key. Its verification key material is
  the trust registry (org pubkeys) plus the user pubkeys carried in
  each macaroon.

- v2's "auth service" — **rescope and rename to Hive macaroon
  issuer.** What v2 calls the auth service is the Hive subsystem that
  issued daily root macaroons and attenuations. That subsystem is
  renamed **Hive macaroon issuer** and its job changes: in phase 1
  (custodial) it produces signatures on behalf of the org and the
  user; in phase 2 it produces signatures only on behalf of the org
  (and only for the user_authorization envelope, not invocations);
  in phase 3 it produces no signatures on its own — it orchestrates
  signing requests to org-key holders and user-key holders. The
  endpoint surface also changes — see "Issuer endpoints" below.

- v2's "daily root macaroon" — **delete.** Each invocation gets its
  own user-signed root. Sessions are an invocation with longer
  caveats; the protocol doesn't distinguish.

- v2's threat-model "HMAC root signing key" crown-jewel row —
  **replace** with three rows: org root key (high; custodial in
  phase 1, multisig in phase 3), user key (medium; custodial in
  phase 1, user-held in phase 2+), session/invocation signature
  (low; bounded by caveats).

- v2's rollout — **insert** a step between current step 6 ("Build the
  Hive macaroon issuer") and step 7 ("Plugin v2: macaroon
  enforcement") for the cryptographic-identity work: stand up the org
  and user keys (custodial first), wire the plugin verifier to the
  chain, register swarm trust. Note that step 6 itself is reshaped by
  this doc — its endpoint surface is `/macaroons/issue` and
  `/macaroons/revoke`, not v2's original three endpoints.

Everything else in v2 — Bifrost-per-workspace, customer/VK
provisioning, per-run cost tracking, kill switches, tool-loop
detection, dimension stamping, the entire enforcement plugin
architecture — is unchanged.

## What this design buys

- **The system cannot spoof users (eventually).** In phase 2, user
  invocation signatures come from user-held keys. No platform
  component holds material that could produce them.

- **The system cannot spoof orgs (eventually).** In phase 3, org
  authorizations require the org's configured multisig. No platform
  component holds enough material to authorize a new user.

- **Cross-swarm agent work is verifiable without callbacks.** Any
  swarm that trusts an org's root pubkey can verify any of that org's
  agents' work from the signatures alone. Agents can pull from
  multiple swarms' databases and knowledge bases, all bounded by one
  verifiable authorization chain.

- **Revocation is per-principal.** Org revokes a user. User revokes
  a session. Plugin revokes a run. Each layer has its own mechanism;
  none requires the others.

- **Plugins are pure verifiers.** No swarm's plugin holds signing
  material. A plugin compromise affects accounting state for in-flight
  runs in that swarm, not the ability to forge identity.

- **Policy is local, protocol is universal.** Every org picks its own
  multisig shape, every user picks their own key custody, every
  invocation picks its own cadence. The plugin verifies the same way
  regardless.

- **Phased migration without flag days.** Custodial → user-held →
  multisig is per-org and per-user. Old and new coexist in the same
  swarm. The plugin handles both.
