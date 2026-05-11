# xpub Key Trees + Signed Auth (Hybrid with Macaroons)

**Status:** Exploratory. Alternative / extension to
[`macaroon-auth-and-attribution.md`](./macaroon-auth-and-attribution.md).
Per-run plugin + Bifrost gateway design carries over unchanged.

## The pitch in one paragraph

Instead of (or alongside) macaroons as bearer tokens, every user has a
BIP32 secp256k1 xpub registered with the issuer. Each agent invocation
gets a *derived child private key*. The agent **signs each LLM request**
with that child key. The plugin verifies the signature and checks that
the child pubkey derives from the user's registered root xpub. This
gives cryptographic provenance + replay protection + non-repudiation +
keeps minting power out of the plugin — at the cost of a more complex
wire protocol and per-request signing in every caller.

## Why this is worth considering

Three properties macaroons can't give you:

1. **Signing instead of bearing.** Macaroons are bearer tokens. Whoever
   has the bytes can use them. With xpub trees, the agent holds a
   *private key* and signs each request. The signature crosses the wire;
   the key never does. Intercepting traffic yields nothing reusable.
2. **Plugin never holds minting power.** With macaroons, the plugin
   needs the HMAC root key to verify — which is the same key used to
   mint. Compromise the plugin, you compromise minting. With xpubs, the
   plugin only needs *public* keys to verify. The root xprv lives only
   in the user's wallet / the issuer. Plugin compromise yields
   verification material only.
3. **Non-repudiation.** An HMAC macaroon can be forged by anyone with
   the root key (the issuer). An ECDSA signature can only be produced
   by whoever holds the private key. "Alice signed this" becomes a
   statement the issuer cannot fake.

And one stack-specific reason:

4. **Sphinx already uses secp256k1 keys.** Users may already have
   wallets, xpubs, signing infrastructure. Reusing that for agent auth
   is incremental work, not a new credential class.

## How it works

### Key tree shape

```
                          alice's xprv (in her wallet)
                                    │
                                    │ BIP32 derivation
                                    ▼
            ┌───────────────────────┴───────────────────────┐
            │                                               │
   m/agent'/browser/...                          m/agent'/org-chat/...
            │                                               │
   m/agent'/browser/ws/w1/run/<uuid>            m/agent'/org-chat/run/<uuid>
   (child xprv → agent process)                 (child xprv → agent process)
```

The path itself encodes scope. Conventions to lock down:

```
m / agent' / <agent-name> / ws / <ws-id-hash> / run / <run-id-hash>
m / agent' / <agent-name> / run / <run-id-hash>             (cross-workspace)
```

Path components map BIP32 31-bit integers via a stable hash. Each
component appears in a registry the plugin knows about, so it can
reconstruct the expected path from request headers.

### Registration

When alice signs up / connects her wallet:

```
POST /keys/register
{ "user_id": "u_01H...", "xpub": "xpub6F..." }
```

Issuer stores `(user_id → xpub)`. That's the only long-lived state
needed for verification. The plugin reads this mapping (cached, refreshed
periodically) and uses it to verify every request.

### Invocation flow

```
1. Alice triggers `browser` on w1 from the app.

2. App backend asks her wallet to derive child xprv at:
       m/agent'/browser/ws/<hash(w1)>/run/<hash(run_id)>
   (or has the wallet do this and return just the child xprv)

3. App backend spawns the browser agent process, passing:
   - child xprv (in env, short-lived in memory)
   - derivation path
   - run metadata (user_id, workspace_id, agent_name, run_id, exp)

4. Agent process makes LLM call. Before sending, it constructs:
       canonical_request = method + path + body_hash + timestamp + nonce
       signature = sign(child_xprv, canonical_request)
   and adds headers:
       x-user-id: alice
       x-agent-path: m/agent'/browser/ws/<hash>/run/<hash>
       x-agent-pubkey: <child pubkey>
       x-agent-scope: <signed json: workspaces, exp, run_id, agent>
       x-agent-timestamp: 2026-05-11T17:00:00Z
       x-agent-nonce: <random-128-bit>
       x-agent-sig: <signature over canonical_request>

5. Plugin PreHook:
   a. Look up xpub for x-user-id
   b. Derive expected pubkey from xpub + x-agent-path
   c. Assert derived pubkey == x-agent-pubkey
   d. Verify x-agent-sig over canonical_request
   e. Verify x-agent-scope signature with same pubkey
   f. Check scope: x-workspace-id ∈ scope.workspaces, etc.
   g. Check x-agent-timestamp within freshness window (e.g. 60s)
   h. Check x-agent-nonce not seen in last freshness window (Redis)
   i. Run-state checks (cost, steps, kill — unchanged)

6. PostHook: same attribution stamps as macaroon design.
```

### What the agent process actually holds

The child xprv for its invocation, in memory, for the lifetime of the
run (≤10 minutes typically). Never written to disk. Never sent over the
wire. When the run ends, the process exits and the key is gone.

Sub-agent spawn: parent derives a *grandchild* (further down the path,
narrower scope), passes the grandchild xprv to the sub-agent. Parent
cannot grant more than it has — derivation paths are append-only.

## Compared to macaroons

| Property | Macaroons | xpub trees |
|---|---|---|
| Tie request to specific human | Signed caveat | Derivation proof + signature |
| Local verification | Yes | Yes |
| Holder can attenuate without issuer | Add caveats | Derive narrower child |
| Scope encoding | First-class typed caveats | Path + signed scope blob |
| Replay protection | Needs separate nonce mechanism | **Built in** — sig covers nonce+ts |
| Bearer-token leak risk | Stolen token = usable until exp | Stolen key = usable for that subtree until exp; **stolen wire data = nothing** |
| Plugin compromise yields minting? | **Yes** | No |
| Non-repudiation | No (issuer can mint) | Yes (issuer can't sign) |
| Wire size | ~1 KB token | Pubkey + sig + scope = ~200–400 B |
| Per-request CPU cost | One HMAC verify | One ECDSA verify (~30 µs) |
| Caller complexity | Send a header | Sign every request |
| Library maturity | Decent | Excellent (Bitcoin/Lightning) |
| Mental model | "JWT with caveats" | "Bitcoin key tree" |
| User key custody | Not user's problem | **User holds an xprv somewhere** |

## The hybrid model (recommended if going this route)

Keep macaroon-shaped *content* (typed scope fields), but make it a
signed payload instead of a bearer token. Drop the macaroon HMAC root
key entirely.

```
┌─────────────────────────────────────────────────────────────┐
│  Wire:                                                      │
│   x-user-id: alice                                          │
│   x-agent-pubkey: <child pubkey>                            │
│   x-agent-path: m/agent'/browser/ws/<h>/run/<h>             │
│   x-agent-scope: { workspaces, agents, env, exp, run_id }   │
│   x-agent-timestamp: <iso8601>                              │
│   x-agent-nonce: <random>                                   │
│   x-agent-sig: <sig over canonical request bytes>           │
│   Authorization: Bearer sk-bf-<agent-vk>                    │
│   x-run-id: <uuid>                                          │
│   x-session-id: <persistent>                                │
│   x-workspace-id: w1                                        │
│   x-agent-name: browser                                     │
└─────────────────────────────────────────────────────────────┘
```

You get:

- **Provenance** (sig from alice's tree)
- **Replay protection** (sig covers ts + nonce)
- **Plugin holds only public material**
- **Scope is still typed, structured, easy to extend**
- **No HMAC minting key to protect anywhere**

The cost is signing code in every caller and ~3x the protocol surface
of "send one macaroon header."

## Plugin responsibilities (hybrid model)

### PreHook (auth gate)

```
1. RESOLVE USER XPUB
   - look up x-user-id → xpub (cached, refresh every N minutes)
   - if unknown user                       → 401 unknown_user

2. VERIFY DERIVATION
   - derive pubkey from (xpub, x-agent-path)
   - if derived != x-agent-pubkey          → 401 key_mismatch

3. VERIFY REQUEST SIGNATURE
   - build canonical_request from method, path, body hash, timestamp, nonce
   - verify x-agent-sig with x-agent-pubkey over canonical_request
   - if bad sig                             → 401 sig_invalid

4. FRESHNESS / REPLAY
   - if |now - x-agent-timestamp| > 60s     → 401 ts_skew
   - if SETNX nonce:<x-agent-nonce> 1 EX 120 fails → 401 replay

5. VERIFY SCOPE BLOB
   - verify x-agent-scope sig with same pubkey
   - if exp passed                          → 401 scope_expired
   - if revoke:<x-agent-pubkey> set         → 401 key_revoked

6. CHECK VK ↔ SCOPE BINDING
   - vk.agent == scope.agent ?              → else 403 agent_mismatch

7. CHECK HEADERS MATCH SCOPE
   - x-workspace-id ∈ scope.workspaces ?    → else 403 scope_violation
   - x-agent-name == scope.agent ?          → else 403 scope_violation
   - x-user-id == scope.user_id ?           → else 403 scope_violation

8. PER-RUN ENFORCEMENT (unchanged)
   - kill / cost / steps / tool-loop checks

9. PER-WORKSPACE DAILY CAP (unchanged)
```

### PostHook (accounting)

Identical to the macaroon design. Once auth passes, accounting is just
counter increments on trusted headers.

## What this costs you (honest list)

1. **Per-request signing in every caller.** TS, Rust, Python, Goose,
   Stakwork — all need secp256k1 signing code on the hot path. Not hard
   (one library call), but it's new code in every place.

2. **Canonical request serialization.** Signing arbitrary HTTP requests
   is a known footgun. We'd lift HTTP Message Signatures (RFC 9421) or
   adapt AWS SigV4's canonical form rather than invent. Adds a real
   spec-compliance dimension.

3. **Per-agent key lifecycle.** The child xprv has to get into the
   agent process and stay there only as long as the run. New plumbing
   in every spawn path.

4. **User key custody.** Alice needs an xprv somewhere. Options:
   - Sphinx wallet (if she already has one — natural)
   - Server-held with a passphrase (defeats half the benefit — server
     can sign as her)
   - Browser-held (WebCrypto / passkey-derived) — possible but each
     browser session is a new key custody story
   - Hardware-backed (secure enclave / YubiKey) — most secure, worst UX

5. **Plugin needs xpub lookup.** New service dependency for resolving
   `user_id → xpub`. Cache it, but cache freshness is now a thing.

6. **Revocation list semantics differ.** With macaroons you revoke by
   nonce (one token). With xpubs you revoke by pubkey (one derived
   subtree). Bulk revoke: "revoke alice's root xpub" — invalidates her
   entire active tree. Plugin needs a `revoked_xpubs` set and to walk
   up the derivation path checking each ancestor. More complex than the
   macaroon case.

7. **Auditor / team mental model.** Explaining BIP32 + secp256k1 +
   derivation paths is harder than explaining JWT-with-caveats. Real
   cost for an internal tool.

## When this is the right call

Strong case if any of these are true:

- **Sphinx wallets are already in the picture.** Users have keys; we'd
  be reusing infrastructure, not introducing a new credential class.
  This is the biggest single reason to consider it.
- **Non-repudiation matters.** Audit/compliance pressure where "the
  user authorized this" needs to be cryptographically provable, not
  just logged.
- **Plugin compromise is a serious concern.** If the plugin runs in
  shared infra, near attacker-controlled data, with broad network
  exposure — keeping minting power out of it is meaningful.
- **Federation across orgs is on the roadmap.** Public-key trees
  federate trivially (publish xpubs). HMAC-based macaroons require
  shared secrets between every pair of organizations.

Weak case if:

- The threat model is "employees abusing access + leaky bearer tokens"
  and we trust our own infrastructure. Macaroons handle that, simpler.
- Users don't already have keys. Adding key custody UX is a real
  product cost.
- We want to ship fast. Macaroons are ~half the plumbing.

## Compatibility with the macaroon plan

Three viable rollout shapes:

**A. Pure xpub (replace macaroons).** Drop the macaroon design entirely;
all auth is signature-based with derivation proofs. Cleanest if we
commit, but biggest upfront cost.

**B. Hybrid (xpub envelope + macaroon-shaped scope).** Recommended if
we go this direction. Reuse macaroon scope semantics in a signed JSON
blob; replace macaroon HMAC with secp256k1 signature. Best of both:
typed scope from macaroons, cryptographic properties from xpubs.

**C. Macaroon now, xpub later.** Ship the macaroon design. Define the
plugin's auth interface as pluggable. Add an xpub verifier later if/when
sphinx wallet integration or compliance pressure pushes us. This is the
"keep options open" path.

(C) is genuinely the lowest-risk path if we're not sure yet. The plugin
PreHook is one function; today it calls `verifyMacaroon(...)`, tomorrow
it dispatches on header presence to `verifyXpubSig(...)` instead. We
just don't paint ourselves into a corner by hardcoding macaroon
semantics into headers used elsewhere.

## Open questions for the team chat

1. **Do users already have keys via sphinx?** If yes, half the
   objections to xpubs disappear. If no, key custody is a real product
   decision that probably blocks adoption.

2. **Where does the user's xprv live?** Sphinx wallet, browser, server-
   held with passphrase, hardware. Each has different threat-model and
   UX implications.

3. **Signing spec.** Roll our own (fast, custom-shaped, footguns) or
   adopt RFC 9421 HTTP Message Signatures (slower start, interop
   benefits, well-reviewed). Lean toward the RFC.

4. **Curve / sig scheme.** secp256k1 ECDSA (matches Bitcoin/sphinx) or
   secp256k1 Schnorr (cleaner, smaller, but newer in this ecosystem) or
   Ed25519 (best crypto, breaks the BIP32 + xpub story — would need a
   different derivation scheme like SLIP-0010).

5. **xpub registration & rotation.** Treat xpub like a long-lived
   identity. Rotation = register new xpub, old one still valid until
   sunset window. Standard PKI patterns apply.

6. **Per-agent derivation path convention.** Lock down a registry early
   so the plugin can always reconstruct paths from headers. Path
   collisions would be a subtle bug source.

7. **Sub-agent key passing.** Parent agent derives a grandchild xprv
   and passes it to the sub-agent. How does that handoff happen
   securely between processes? (env var? unix socket? IPC?)

8. **Plugin xpub cache.** How fresh does the user-xpub mapping need to
   be? On revocation, how fast must the plugin pick it up? Likely
   acceptable to cache for minutes if we have a revocation list as the
   fast path.

9. **Server-side signing for non-interactive agents.** Some agents run
   from CI / cron / webhooks where there's no human in the loop at
   trigger time. Who holds their signing key? A "system identity" xpub
   tree? That weakens the "tied to a human" property for those runs —
   may be fine, but worth flagging.

## What I'd actually recommend

If sphinx wallet infrastructure is already part of the stack and users
have keys: **Option B (hybrid)** is genuinely compelling. You get
cryptographic provenance, replay protection, and plugin-can't-mint for
roughly 1.5× the implementation cost of macaroons alone.

If sphinx wallets aren't in the picture for this user base, or we want
to ship in weeks not months: **Option C** — ship the macaroon design,
define the plugin auth interface to be pluggable, leave the xpub door
open. The hybrid can land as v2 of the plugin without rearchitecting.

The wrong answer is to spend a quarter debating crypto schemes while
runaway agents burn through the budget. The macaroon plan is good
enough to start with and the xpub upgrade is additive.

## Summary

| Property | Macaroons (current plan) | Hybrid xpub + signed scope |
|---|---|---|
| Auth primary layer | HMAC bearer token | secp256k1 signature |
| Plugin needs to mint? | Yes (HMAC root key) | **No (verify only)** |
| Replay protection | Separate nonce mechanism | **Built into sig** |
| Non-repudiation | No | **Yes** |
| Wire protocol complexity | One header | ~7 headers + canonical signing |
| Caller implementation cost | Header set | Sign every request |
| Mental model burden | Low | Medium-high |
| Natural fit if sphinx wallets exist | Neutral | **Strong** |
| Time to ship | Weeks | Weeks + signing infra |
