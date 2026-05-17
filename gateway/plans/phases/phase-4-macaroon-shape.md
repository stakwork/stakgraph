# Phase 4 — Macaroon Shape: Wire Format, Encoding, Verification

> Concrete cryptographic spec for the three-layer macaroon described
> abstractly in `cryptographic-identity.md`. Pins down byte-level
> encoding, signing inputs, the HMAC chain definition, and the
> verifier algorithm — enough that a Go verifier (`gateway/auth/go/`,
> with a Bifrost-plugin adapter in `gateway/internal/auth/`) and a
> TypeScript signer + attenuator + verifier (`gateway/auth/ts/`, an
> open-source npm package consumed by Hive's macaroon issuer and
> `/mcp`) can be implemented against the same fixtures
> (`gateway/auth/fixtures/`) and known to agree.
>
> This phase produces no code on its own. It produces the contract
> that phase 5 (trust registry) stores pubkeys against and that the
> eventual plugin enforcement and issuer implementations both target.

## What this phase decides

The identity doc gives the abstract shape:

```
ORG-signed user_authorization → USER-signed invocation → HMAC attenuations
```

and explicitly punts on wire format:

> Wire format details (CBOR vs. JSON, exact field encoding,
> canonicalization rules) are an implementation choice, not a design
> decision.

Phase 5 (trust registry) needs pubkeys it can store and verify against.
Agent processes need to extend the HMAC chain in whatever language they
happen to run in. The plugin needs to verify deterministically. None of
that can move until the bytes are pinned down.

This doc pins them down. After this:

- A pure Go verifier in `gateway/auth/go/` can be written.
- A TS signer + attenuator + verifier in `gateway/auth/ts/` can be
  written. Hive's `/macaroons/issue` handler and `/mcp` both depend on
  this package; agent processes can also depend on it for local
  attenuation.
- An HMAC attenuator in any other agent language (Python, Rust, …) can
  be written from `crypto.createHmac`-equivalent primitives alone.
- Test vectors in `gateway/auth/fixtures/` are the single source of
  truth that every implementation builds against. CI in this repo
  runs both `go test ./gateway/auth/go/...` and the TS package's
  fixture tests; drift between signer and verifier is impossible to
  miss.

## Implementation split

| Component                | Language   | Location                       | Responsibilities                                         |
| ------------------------ | ---------- | ------------------------------ | -------------------------------------------------------- |
| Pure verifier            | Go         | `gateway/auth/go/`             | Parse, JCS-canonicalize, verify org-sig + user-sig + HMAC chain, enforce caveat narrowing. No I/O, no Bifrost types. |
| Bifrost-plugin adapter   | Go         | `gateway/internal/auth/`       | Pull `x-macaroon` from `bifrost.Request`, look up the org's policy from the in-memory trust registry, call `gateway/auth/go`, layer Redis-backed revocation checks on top, translate failures into `bifrost.Error` codes. Holds no cryptographic knowledge. |
| Signer + attenuator + verifier | TypeScript | `gateway/auth/ts/` (published as a public npm package) | JCS-canonicalize, sign org layer (custodial phase 1), sign user layer (custodial phase 1), attenuate (HMAC chain), optionally verify (for tests + Hive sanity-checks). Hive's issuer and `/mcp` import this package. |
| Polyglot attenuators     | Other      | Agent processes (Python, Rust, …) | JCS-canonicalize own caveats, HMAC-SHA256 with prev_sig as key, append to chain. Not provided as a Stakwork lib; spec + fixtures are the contract. |

The Go pure verifier and the TS signer are the two sides that must
produce bit-identical canonicalized JSON for the same logical object.
The attenuator only needs JCS + HMAC-SHA256, both of which are
one-file implementations in every target language.

The split between `gateway/auth/go/` (pure) and `gateway/internal/auth/`
(plugin adapter) keeps the cryptographic spec testable without Bifrost
plumbing, and keeps the package importable by any future Go consumer
(CLIs, debug tools, non-plugin validators) that the Go `internal/`
visibility rule would otherwise exclude.

## Encoding choices

**Outer wire format.** `base64url(JCS-JSON(macaroon))`, no padding.
Carried in the `x-macaroon` HTTP header on every LLM call, exactly as
v2 already specifies. Base64url is chosen over raw JSON in the header
because the JSON contains characters that don't survive header
transport reliably (whitespace handling, embedded `:` in URLs, etc.).

**Canonicalization.** JSON Canonicalization Scheme (RFC 8785, JCS).
Chosen because:

- Mature implementations in Go (`github.com/cyberphone/json-canonicalization`),
  TypeScript (`canonicalize`), Python (`jcs`), Rust, and others.
- Lets agents debug-print macaroons during development.
- The size win of CBOR (~30% on objects this shape) is irrelevant when
  the whole macaroon is ~1 KB and travels in one header per call.
- Deterministic on numeric edge cases (integer vs. float, leading
  zeros, exponent normalization) which is necessary for HMAC
  reproducibility.

JCS handles object key ordering (alphabetical), numeric normalization
(I-JSON rules per RFC 7493), and string escaping. The verifier and
signer both run JCS on the same logical object and get the same bytes.

**Hex strings.** All binary fields (pubkeys, signatures, HMAC outputs,
nonces) are lowercase hex, **no `0x` prefix**. (Phase 5's prior
examples used `0x…`; that's an inconsistency to clean up when phase 5
is revisited. Phase 4 wire format is canonical.)

## Top-level macaroon object

The JSON object that gets JCS-canonicalized and base64url-encoded into
the `x-macaroon` header:

```json
{
  "v":                  1,
  "org_id":             "org_acme",
  "user_authorization": { … },
  "invocation":         { … },
  "attenuations":       [ … ]
}
```

| Field                | Type   | Notes                                                    |
| -------------------- | ------ | -------------------------------------------------------- |
| `v`                  | int    | Wire-format version. `1` for everything in this doc.     |
| `org_id`             | string | Lookup key for the trust registry.                       |
| `user_authorization` | object | Org-signed envelope; shape below.                        |
| `invocation`         | object | User-signed root caveats; shape below.                   |
| `attenuations`       | array  | Zero or more HMAC links; shape below. Empty for top-level invocations (no sub-agents yet). |

The verifier rejects any unknown top-level field at `v=1` to keep the
upgrade surface clean. New fields land at `v=2` and the verifier
either rejects or branches on version.

## The `user_authorization` layer (org-signed)

```json
{
  "user_id":     "u_alice",
  "user_pubkey": {
    "alg":   "ed25519",
    "key":   "8b2…32-byte-hex…"
  },
  "permissions": {
    "workspaces": ["w1", "w2"],
    "agents":     ["coder", "browser", "repair-agent"]
  },
  "budget": {
    "max_total_usd":          1000.00,
    "max_per_invocation_usd": 25.00
  },
  "iat":         "2026-05-14T09:00:00Z",
  "exp":         "2026-06-14T09:00:00Z",
  "nonce":       "9f4e…32-hex-chars…",
  "org_sig":     {
    "alg":       "ecdsa-secp256k1-sha256",
    "sig":       "3045…compact-or-policy-shaped…"
  }
}
```

| Field         | Type   | Notes                                                              |
| ------------- | ------ | ------------------------------------------------------------------ |
| `user_id`     | string | Hive's stable user identifier. Same string used as Bifrost Customer name (phase 1). |
| `user_pubkey` | object | `{alg, key}`. `alg="ed25519"`, `key` = 32-byte hex (Ed25519 raw pubkey). |
| `permissions` | object | What the user is allowed to do. `workspaces` and `agents` are arrays of identifiers; further fields can be added at `v=1` only if backward-compatible. |
| `budget`      | object | **Optional.** Org-signed spending envelope. Omit (or set fields to 0) for "no org-side cap; user's `Invocation.max_cost_usd` is the only limit." See "Budget envelope" below. |
| `iat`         | string | RFC 3339 / ISO 8601 UTC.                                           |
| `exp`         | string | RFC 3339 / ISO 8601 UTC. Verifier rejects when `now > exp`.        |
| `nonce`       | string | 128-bit random, lowercase hex, 32 chars. Used for revocation **and** as the Redis key for `cost:ua:<nonce>` cumulative-spend tracking when `budget` is set. |
| `org_sig`     | object | See "Signature objects" below. Signs all other fields in this object. |

### Budget envelope

The optional `budget` substruct lets the org bound what the user-key
can spend under this `user_authorization` without the org key needing
to be online for every invocation. This is the canonical pattern for
"org leader signs once a week from cold storage, employee's hot key
signs many invocations under that ceiling."

```json
"budget": {
  "max_total_usd":          1000.00,
  "max_per_invocation_usd": 25.00
}
```

| Field                     | Type   | Notes                                                              |
| ------------------------- | ------ | ------------------------------------------------------------------ |
| `max_total_usd`           | number | **Cumulative** cap across all invocations under this `user_authorization`. Plugin tracks via Redis key `cost:ua:<ua.nonce>` (same HASH/TTL pattern as `cost:run:*`). `0` ⇒ no cumulative cap. |
| `max_per_invocation_usd`  | number | **Per-invocation** cap. Verifier rejects when `invocation.max_cost_usd > budget.max_per_invocation_usd`. Pure signature-time check — no Redis. `0` ⇒ no per-invocation cap. |

**Both fields are independently optional.** Setting only
`max_total_usd` lets the user pick per-invocation amounts freely up to
the envelope total. Setting only `max_per_invocation_usd` caps each
call but doesn't bound the total. Setting both gives the org full
envelope control.

**Backwards compatibility.** Absent `budget` (or both fields = 0) is
indistinguishable on the wire from a pre-budget macaroon and verifies
under the same rules — no UA-level budget checks. The verifier never
fails open on a budget; an absent budget is "no cap by design," not
"missing data."

**Cold-storage flow** (the motivating scenario):

1. Org leader, holding their secp256k1 key on a hardware wallet,
   signs a `user_authorization` with
   `budget = { max_total_usd: 5000, max_per_invocation_usd: 200 }`
   and `exp = next-sunday`. Done once per week.
2. The signed UA is handed to the employee (delivered via Hive's
   user store). The employee's Ed25519 hot key signs many
   `invocation`s under it through the week.
3. Plugin tracks `cost:ua:<ua.nonce>` cumulatively; when it crosses
   `max_total_usd`, every subsequent invocation under this UA is
   rejected with `402 ua_budget_exceeded`.
4. To extend or top up, the org leader signs a new UA next Sunday.
   Old UA naturally expires at its `exp`; its Redis bucket TTLs out
   shortly after.

**Verifier rules.** See `phase-6-plugin-enforcement.md` for the full
cap-walk integration. In summary:

- Signature-time: reject if `budget.max_per_invocation_usd > 0` and
  `invocation.max_cost_usd > budget.max_per_invocation_usd`.
- Runtime (PreLLMHook): if `budget.max_total_usd > 0`, add one
  `HGET cost:ua:<ua.nonce> total` to the existing cap-walk pipeline;
  reject when accumulated total would meet or exceed the cap.
- Runtime (PostLLMHook): if `budget.max_total_usd > 0`, add one
  `HINCRBYFLOAT cost:ua:<ua.nonce> total <cost>` to the existing
  accumulator pipeline. No new round-trips.

**Why not just one `max_total_usd` field?** `max_per_invocation_usd`
costs nothing at runtime (no Redis) and prevents an employee's
compromised hot key from burning the full envelope in a single
`Invocation.max_cost_usd: 5000` call. The two caps are
complementary, not redundant.

**Cadence.** `user_authorization` is long-lived (issued once per user
onboarding or key rotation; per the identity doc, "rare — user
onboarding, key rotation"). The same `user_authorization` is reused
across many `invocation`s.

**Signing input.** Take the `user_authorization` object, remove the
`org_sig` field, run JCS on what remains, UTF-8 encode → those are the
bytes signed. See "Signing inputs" for the exact algorithm.

## The `invocation` layer (user-signed)

```json
{
  "workspace":    "w1",
  "agents":       ["coder"],
  "run_id":       "r_01H8…",
  "max_cost_usd": 5.00,
  "max_steps":    100,
  "iat":          "2026-05-14T10:00:00Z",
  "exp":          "2026-05-14T10:10:00Z",
  "nonce":        "7c2a…32-hex-chars…",
  "user_sig":     {
    "alg":        "ed25519",
    "sig":        "a3b1…128-hex-chars…"
  }
}
```

| Field          | Type           | Notes                                                              |
| -------------- | -------------- | ------------------------------------------------------------------ |
| `workspace`    | string         | Workspace identifier. Verifier checks it's in `user_authorization.permissions.workspaces`. |
| `agents`       | array<string>  | Always single-element on first issuance. Sub-agents append on attenuation. The last element is the "most specific" agent (matches v2's plugin canonicalization). Verifier checks every entry is in `user_authorization.permissions.agents`. |
| `run_id`       | string         | Stable id for this run. Used as the cost-accumulation key in plugin Redis (`cost:run:<run_id>`). |
| `max_cost_usd` | number         | USD budget for this run. Plugin enforces via Redis accumulator.    |
| `max_steps`    | int            | Step budget. Plugin enforces.                                      |
| `iat`          | string         | RFC 3339 UTC.                                                      |
| `exp`          | string         | RFC 3339 UTC. Replaces v2's `max_wallclock_s` — duration caps are computed by the issuer at signing time and stamped as a concrete `exp`. The macaroon only carries the timestamp. |
| `nonce`        | string         | 128-bit random, lowercase hex, 32 chars. Per-invocation.           |
| `user_sig`     | object         | See "Signature objects". Signs all other fields in this object.    |

**On `exp` vs `max_wallclock_s`.** v2 and the identity doc carried
both an absolute `exp` and a duration `max_wallclock_s`. They're
redundant — at issuance time `exp = iat + max_wallclock_s` and the
verifier only ever consults `exp`. Phase 4 carries `exp` only. The
issuer reads `defaultMaxWallclockS` from the agent registry, computes
`exp`, and stamps it; the duration never appears on the wire.

**Signing input.** Take the `invocation` object, remove the `user_sig`
field, run JCS, UTF-8 encode → bytes to sign with the Ed25519 key
whose pubkey is in `user_authorization.user_pubkey.key`.

## The `attenuations` chain (HMAC)

Each entry is one sub-agent's narrowing of its parent. Top-level
invocations carry `"attenuations": []`. A parent spawning a sub-agent
appends one entry; that sub-agent spawning a grandchild appends a
second; and so on.

```json
{
  "caveats": {
    "agents":       ["coder", "web-search"],
    "max_cost_usd": 2.00,
    "max_steps":    40,
    "run_id":       "r_01H8…child…",
    "exp":          "2026-05-14T10:02:00Z",
    "nonce":        "1e8d…32-hex-chars…"
  },
  "hmac": "5f3c…64-hex-chars…"
}
```

**HMAC definition.** For attenuation `i` (0-indexed):

```
prev_sig_bytes_0    = hex_decode(invocation.user_sig.sig)
prev_sig_bytes_i>0  = hex_decode(attenuations[i-1].hmac)

hmac_i              = HMAC_SHA256(
  key = prev_sig_bytes_i,
  msg = utf8(JCS(attenuations[i].caveats))
)

attenuations[i].hmac = hex_encode(hmac_i)
```

**Why this exact definition.**

- **Key = previous signature bytes.** Standard macaroon chain
  construction. A child cannot produce a valid HMAC without holding
  the parent's signature bytes; the parent gives them to the child by
  handing it the (parent + child caveats) macaroon at spawn.
- **Message = JCS of `caveats` only.** Not the whole attenuation
  object (which contains the HMAC itself), not anything from layers
  above. Keeps the input self-contained and language-portable.
- **Hex output.** So the assembled JSON-with-attenuations is itself
  pure JSON, JCS-able, base64url-able.

An attenuator in any language needs only:

1. JCS (one library, or even hand-rolled — RFC 8785 is ~200 LOC).
2. HMAC-SHA256 (standard library everywhere).
3. Hex encode/decode (standard library everywhere).

No curve operations, no key handling, no PKI. Sub-agent containment
is the cheapest layer to implement.

**Caveat narrowing rules.** At each link, the verifier checks the
attenuation's `caveats` strictly narrow the parent's effective caveats
(parent = invocation for the first link, or the previous attenuation's
caveats for later links, walked transitively up to the invocation):

| Field          | Rule                                                                 |
| -------------- | -------------------------------------------------------------------- |
| `agents`       | `child.agents ⊇ parent.agents` (must include all parent entries; may add). The last entry is the most-specific agent. |
| `max_cost_usd` | `child.max_cost_usd ≤ parent.max_cost_usd`                           |
| `max_steps`    | `child.max_steps ≤ parent.max_steps`                                 |
| `exp`          | `child.exp ≤ parent.exp` (lexicographic on RFC 3339 UTC works)       |
| `run_id`       | Free choice — child gets its own run_id, lets the plugin accumulate cost both at the child level and (transitively, via chain walk) at every ancestor. |
| `nonce`        | Free choice — fresh per attenuation, used for per-link revocation. |

`agents` is the one field that legitimately *grows*: a sub-agent's
identity is appended to the list, not replacing the parent's. The
last element wins as the "agent-name" dimension for billing. This
matches v2's existing plugin canonicalization (line 507:
`dims["agent-name"] = macaroon.agents[last]`).

Any other field present in a child's `caveats` that wasn't present in
the parent is permitted only if it strictly narrows (in the
field-specific sense above). Fields the parent had that the child
omits are treated as "inherited unchanged" by the verifier.

## Signature objects

Both `org_sig` and `user_sig` follow the same envelope shape:

```json
{
  "alg": "ecdsa-secp256k1-sha256" | "ed25519" | "multisig-v1",
  "sig": "<hex-encoded signature bytes>",
  …    // alg-specific fields
}
```

### `alg: "ed25519"`

Used for `user_sig`. Ed25519 signs raw bytes (no prehash). Signature
is 64 bytes, hex-encoded → 128 chars.

```json
{
  "alg": "ed25519",
  "sig": "a3b1…128-hex-chars…"
}
```

Pubkey (in `user_authorization.user_pubkey`) is 32 raw bytes,
hex-encoded → 64 chars.

### `alg: "ecdsa-secp256k1-sha256"`

Used for single-key `org_sig` in phase 1 custodial mode. ECDSA over
secp256k1 with SHA-256 prehash, signature in **compact `r||s`** form
(64 bytes, hex-encoded → 128 chars), low-`s` normalized (BIP 62 rule)
to make signatures deterministically comparable.

```json
{
  "alg": "ecdsa-secp256k1-sha256",
  "sig": "3f2a…128-hex-chars…"
}
```

Pubkey (in the trust registry) is the 33-byte **compressed** secp256k1
pubkey, hex-encoded → 66 chars.

DER-encoded signatures are not used on the wire — compact form is
shorter and avoids DER parsing ambiguities. The signer normalizes to
low-`s` before encoding; the verifier rejects high-`s` signatures.

### `alg: "multisig-v1"` (org-side, phase 3+)

Used for `org_sig` when the org's trust-registry entry has a multisig
policy. Carries one signature per participating key.

```json
{
  "alg":       "multisig-v1",
  "sigs": [
    { "key_index": 0, "alg": "ecdsa-secp256k1-sha256", "sig": "…hex…" },
    { "key_index": 2, "alg": "ecdsa-secp256k1-sha256", "sig": "…hex…" }
  ]
}
```

Each entry references a key by `key_index` into the multisig policy
object stored in the trust registry. The verifier checks the policy's
threshold is met by valid signatures over the same signing-input bytes.

Schnorr (BIP-340) and MuSig2 are future-compatible: the `sigs[].alg`
field can become `"schnorr-secp256k1-bip340"` or `"musig2-secp256k1"`
without changing the wire format. Phase 4 ships ECDSA only; phase 3
hardening can add Schnorr/MuSig2 by extending the verifier's alg
dispatch.

## Multisig policy object (trust registry)

Stored in the trust registry per org (phase 5). Two shapes:

**Single-key:**

```json
{
  "type": "single",
  "key": {
    "alg": "ecdsa-secp256k1-sha256",
    "key": "02a1…66-hex-chars…"
  }
}
```

**Multisig:**

```json
{
  "type":      "multisig",
  "threshold": 2,
  "keys": [
    { "alg": "ecdsa-secp256k1-sha256", "key": "02a1…66-hex-chars…" },
    { "alg": "ecdsa-secp256k1-sha256", "key": "03b2…66-hex-chars…" },
    { "alg": "ecdsa-secp256k1-sha256", "key": "02c3…66-hex-chars…" }
  ]
}
```

`threshold` is the minimum count of valid signatures required. The
identity doc mentions weighted multisig as a possibility; phase 4
defines simple m-of-n only. Weighted policies can land as
`"type": "multisig-weighted"` later without invalidating this format.

The verifier dispatches on `policy.type`:

- `"single"` → expect `org_sig.alg ∈ { "ecdsa-secp256k1-sha256" }`,
  verify against `policy.key`.
- `"multisig"` → expect `org_sig.alg == "multisig-v1"`, verify each
  inner signature against `policy.keys[key_index]`, count successes,
  require `count ≥ policy.threshold`.

Phase 1 ships only `"type": "single"`. The multisig path is specified
here so phase 5 can store the right shape from day one.

## Signing inputs

Both signature layers use the same algorithm for "what bytes get signed":

```
sign_bytes(obj, sig_field_name) =
  let obj' = obj with sig_field_name removed
  let canonical = JCS(obj')          // returns a UTF-8-string
  return utf8_bytes(canonical)
```

For `user_authorization.org_sig`:

```
msg = sign_bytes(user_authorization, "org_sig")
sig = sign(org_priv_key, sha256(msg))  // ECDSA needs the SHA-256 prehash
```

For `invocation.user_sig`:

```
msg = sign_bytes(invocation, "user_sig")
sig = ed25519_sign(user_priv_key, msg)  // Ed25519 takes raw bytes
```

The verifier mirror:

```
verify_org(user_authorization, org_pubkey):
  msg = sign_bytes(user_authorization, "org_sig")
  return ecdsa_verify(org_pubkey, sha256(msg), user_authorization.org_sig.sig)

verify_user(invocation, user_pubkey):
  msg = sign_bytes(invocation, "user_sig")
  return ed25519_verify(user_pubkey, msg, invocation.user_sig.sig)
```

The signature value is the only field excluded from its own input.
Every other field in the layer is bound by the signature.

## Nonces, run_ids, and other identifiers

| Identifier              | Format                              | Scope                                |
| ----------------------- | ----------------------------------- | ------------------------------------ |
| `user_authorization.nonce` | 128-bit random, hex lowercase, 32 chars | Per `user_authorization` issuance |
| `invocation.nonce`      | 128-bit random, hex lowercase, 32 chars | Per invocation                       |
| `attenuations[i].caveats.nonce` | 128-bit random, hex lowercase, 32 chars | Per attenuation                |
| `user_id`               | Hive's stable string                | Globally meaningful within Hive      |
| `org_id`                | string                              | Globally meaningful (trust registry key) |
| `run_id`                | string, recommended `r_<26-char-ulid>` or `r_<uuid-v4>` | Per agent run (each attenuation gets a fresh one) |
| `workspace`             | string                              | Workspace-id, matches Bifrost Customer scoping |

Nonces are uniformly 16 random bytes hex-encoded to 32 chars, generated
from a cryptographically secure RNG. The verifier doesn't impose any
structure beyond format — they're opaque revocation handles.

## Verifier algorithm

Concrete byte-level version of the 8-step pseudocode from the identity
doc. Steps 1–10 are the pure verifier in `gateway/auth/go/`; step 11
is the Bifrost-plugin adapter in `gateway/internal/auth/` plus the v2
plugin enforcement that already exists. Revocation lookups (Redis) sit
in the adapter, not in the pure verifier:

### Pure verifier (`gateway/auth/go/`, `gateway/auth/ts/`)

Pure function: takes the macaroon, the policy, and the current time;
returns verified claims or an error. No Redis, no Bifrost types, no
ambient state. The same pseudocode applies to both Go and TS
implementations.

```
Verify(macaroon_b64url, policy, now) → (Claims, error):
  1. macaroon_bytes  = base64url_decode(macaroon_b64url)
  2. macaroon        = JSON_parse(macaroon_bytes)
  3. if macaroon.v != 1                                  → macaroon_unsupported_version
  4. if any unknown top-level fields                     → macaroon_malformed

  5. ua = macaroon.user_authorization
     ua_msg = utf8(JCS(ua without org_sig))
     dispatch on policy.type:
       "single":
         if ua.org_sig.alg != "ecdsa-secp256k1-sha256"   → invalid_user_authorization
         ok = ecdsa_verify(policy.key.key, sha256(ua_msg), ua.org_sig.sig)
       "multisig":
         if ua.org_sig.alg != "multisig-v1"              → invalid_user_authorization
         valid_count = 0
         for each entry in ua.org_sig.sigs:
           pk = policy.keys[entry.key_index]
           if entry.alg != pk.alg                        → skip
           if verify_alg(entry.alg, pk.key, ua_msg, entry.sig)
             valid_count++
         ok = valid_count >= policy.threshold
     if !ok                                              → invalid_user_authorization

  6. check ua caveats (time only — revocation is the adapter's job):
       now > ua.exp                                      → user_authorization_expired

  7. inv = macaroon.invocation
     inv_msg = utf8(JCS(inv without user_sig))
     if inv.user_sig.alg != ua.user_pubkey.alg           → invalid_invocation_signature
     if !ed25519_verify(ua.user_pubkey.key, inv_msg, inv.user_sig.sig)
                                                         → invalid_invocation_signature

   8. check inv caveats (intrinsic only):
        inv.workspace not in ua.permissions.workspaces    → invocation_violated
        any inv.agents not in ua.permissions.agents       → invocation_violated
        now > inv.exp                                     → macaroon_expired
        // Per-invocation budget cap (pure signature-time check).
        if ua.budget != nil && ua.budget.max_per_invocation_usd > 0
           && inv.max_cost_usd > ua.budget.max_per_invocation_usd
                                                          → ua_per_invocation_exceeded

  9. walk attenuations:
       prev_sig_bytes = hex_decode(inv.user_sig.sig)
       prev_caveats   = effective_caveats(inv)
       for each att in macaroon.attenuations:
         expected_hmac = HMAC_SHA256(prev_sig_bytes, utf8(JCS(att.caveats)))
         if hex_encode(expected_hmac) != att.hmac        → attenuation_invalid
         enforce_narrowing(prev_caveats, att.caveats)    → attenuation_widened (on fail)
         now > att.caveats.exp                           → macaroon_expired
         prev_sig_bytes = hex_decode(att.hmac)
         prev_caveats   = merge(prev_caveats, att.caveats)

  10. return Claims {
        org_id, user_id, workspace,
        effective_caveats: prev_caveats,    // narrowed by entire chain
        ua_nonce:          ua.nonce,        // for cost:ua:<nonce> tracking
        ua_budget:         ua.budget,       // nil if absent
        nonces: [ua.nonce, inv.nonce, att[*].caveats.nonce],
        iat:   inv.iat,
      }
```

### Bifrost-plugin adapter (`gateway/internal/auth/`)

Wraps the pure verifier with the I/O the plugin needs. Returns
`bifrost.Error`-shaped responses.

```
VerifyRequest(ctx, req, registry, redis) → bifrost.Error:
  1. macaroon_b64 = req.header("x-macaroon")
     if missing                                         → 401 macaroon_required

  2. macaroon_org_id = peek_org_id(macaroon_b64)        // parse just enough to look up policy
     policy = registry[macaroon_org_id]
     if policy == nil                                   → 401 untrusted_org

  3. claims, err = auth.Verify(macaroon_b64, policy, time.Now())
     if err != nil                                      → 401 with err.code

  4. revocation checks (Redis):
     redis.exists("revoke:" + claims.nonces[0])         → 401 user_authorization_revoked
     redis.get("revoke_user_before:" + claims.user_id)
       > claims.iat                                     → 401 user_authorization_revoked
     for each nonce in claims.nonces[1:]:
       redis.exists("revoke:" + nonce)                  → 401 macaroon_revoked

  5. defense-in-depth + plugin ceilings:
     claims.user_id != ctx.customer_id                  → 401 macaroon_user_mismatch
     claims.effective_caveats.max_cost_usd
       > plugin.hard_ceiling                            → 402 budget_unreasonable

  6. stamp ctx.VerifiedClaims = claims; return nil
     (downstream hooks: v2 plugin enforcement — cost accumulation,
     kill switch, tool-loop detection — unchanged from v2 except the
     caveats now come from claims.effective_caveats rather than
     plugin config)
```

The pure-verifier steps (1–10 in the first block) are the new
cryptographic identity check; they're identical across the Go and TS
implementations and are what the fixture tests assert. The adapter
steps (1–6 in the second block) layer I/O — header extraction, trust
registry lookup, Redis-backed revocation, defense-in-depth — and only
live in Go because Bifrost is Go. Beyond step 6 of the adapter, v2's
existing plugin enforcement (cost accumulation, kill switch,
tool-loop detection) runs untouched.

**Performance notes.** A single verify involves: one base64 decode,
one JSON parse, two JCS canonicalizations (one per signed layer), one
ECDSA-secp256k1 verify (or multisig-count of them), one Ed25519 verify,
one HMAC-SHA256 per attenuation. On x86_64 with a modern Go crypto
library this is dominated by the secp256k1 verify (~100 µs single-key).
Sub-millisecond total. The secp verify result can be cached by
`(org_id, user_authorization.nonce)` since the `user_authorization`
changes far less often than invocations — same caching note the
identity doc already makes.

## Test vectors

Phase 4 publishes a minimal set of fixtures in
`gateway/auth/fixtures/` — sibling to `gateway/auth/go/` and
`gateway/auth/ts/`, so both implementations load them by relative
path:

```
gateway/auth/fixtures/
├── keys.json              # the keys used in all vectors (priv+pub, hex)
├── 01-simple.json         # one ua, one invocation, zero attenuations
├── 02-one-attenuation.json
├── 03-two-attenuations.json
├── 04-multisig-2of3.json
└── README.md              # how to run the vectors against either side
```

Each `NN-*.json` is shaped as:

```json
{
  "description": "single attenuation under a custodial single-key org",
  "inputs": {
    "org_priv":  "…hex…",
    "user_priv": "…hex…",
    "ua_unsigned":  { … },
    "inv_unsigned": { … },
    "atts_unsigned": [ { … } ]
  },
  "expected": {
    "ua_signing_bytes_hex":      "…",
    "inv_signing_bytes_hex":     "…",
    "att_0_hmac_input_hex":      "…",
    "att_0_hmac_output_hex":     "…",
    "macaroon_canonical_json":   "…",
    "macaroon_b64url":           "…",
    "verify_result":             "ok"
  }
}
```

Both `gateway/auth/go/fixtures_test.go` and `gateway/auth/ts/test/fixtures.test.ts`
load every `NN-*.json`, reproduce each `expected.*` value from the
`inputs` (re-sign with the given private keys, re-canonicalize, re-HMAC),
and fail the build if any byte differs. This is the gate that keeps
signer and verifier from drifting.

The fixtures land in this directory **with** phase 4 — the doc and the
fixtures are one deliverable. The pure-verifier package in Go, the
public npm package in TS, phase 5's trust registry, and any future
agent-side attenuator in another language all use them as the
cross-language source of truth.

## What is NOT in this phase

- **Issuer endpoint implementation** — covered by
  `cryptographic-identity.md` ("Issuer endpoints"). Phase 4 only
  defines what bytes the issuer must produce.
- **Trust-registry storage and admin API** — phase 5. Phase 4 defines
  what shape phase 5 stores (the policy object).
- **Revocation distribution** — identity doc covers the shape; the
  concrete pull/push implementation lands later. Phase 4 says
  "revocation handles are nonces, hex-encoded, 32 chars."
- **Plugin enforcement plumbing** — v2 covers cost accumulators, kill
  switches, dim stamping. Phase 4 only specifies up to "verify
  succeeded; here are the caveats."
- **Phase 2 user-key migration** — the wire format works unchanged
  when the user key moves from Hive's custody to a hardware device;
  only the signer location changes. Phase 4 doesn't need to anticipate.
- **Phase 3 org-key multisig migration** — wire format already
  accommodates `multisig-v1`. Phase 4 specifies the verifier path so
  phase-5 storage can land with the right shape; phase 1 ships only
  the `"single"` path.

## Wire-up checklist for phase 4

**Cross-language contract:**

- [ ] `gateway/auth/` sibling directory created with `README.md`,
      `fixtures/`, `go/`, `ts/` subdirs
- [ ] `gateway/auth/fixtures/keys.json` populated with reproducible
      org + user keypairs (priv+pub, hex)
- [ ] `gateway/auth/fixtures/01-simple.json` through `04-multisig-2of3.json`
      populated with concrete bytes at every intermediate step
- [ ] `gateway/auth/fixtures/README.md` documents the fixture schema
      and how each side loads them

**Go pure verifier (`gateway/auth/go/`):**

- [ ] Package skeleton: `verify.go`, `jcs.go`, `sigs.go`, `hmac.go`,
      `narrow.go`, `encoding.go`, `types.go`
- [ ] Public API: `func Verify(macaroonB64 string, policy Policy, now time.Time) (*Claims, error)`
- [ ] JCS dependency: `github.com/gowebpki/jcs` (or
      `github.com/cyberphone/json-canonicalization`) — pick during impl
- [ ] secp256k1 dependency: `github.com/decred/dcrd/dcrec/secp256k1/v4`
      (used by go-ethereum, well-maintained, supports compact sig + low-s
      normalization)
- [ ] Ed25519 from Go stdlib (`crypto/ed25519`)
- [ ] HMAC-SHA256 from Go stdlib (`crypto/hmac`)
- [ ] `fixtures_test.go` loads `../fixtures/*.json`, asserts every
      `expected.*` value reproduces byte-for-byte

**Go Bifrost-plugin adapter (`gateway/internal/auth/`):**

- [ ] Update `doc.go` to reflect the split (pure verifier elsewhere,
      this package is the plugin adapter only)
- [ ] `plugin.go`: `func VerifyRequest(ctx, req, registry, redis) bifrost.Error`
- [ ] `revocation.go`: Redis lookups for `revoke:<nonce>` and
      `revoke_user_before:<user_id>`
- [ ] Imports `gateway/auth/go` for all cryptographic work
- [ ] Adapter has no JCS / curve / HMAC code of its own

**TypeScript package (`gateway/auth/ts/`):**

- [ ] `package.json` with public name (e.g. `@stakwork/macaroon`),
      MIT license, types, exports map
- [ ] `tsconfig.json` targeting ES2022, ESM + CJS dual output
- [ ] Source: `index.ts`, `types.ts`, `encoding.ts` (base64url + hex),
      `jcs.ts`, `sigs.ts`, `sign.ts`, `attenuate.ts`, `verify.ts`
- [ ] JCS dependency: `canonicalize` (small, MIT, RFC 8785-compliant)
- [ ] secp256k1 dependency: `@noble/secp256k1` (pure-JS, audited,
      supports low-s normalization)
- [ ] Ed25519 dependency: `@noble/ed25519` (pure-JS, audited)
- [ ] HMAC-SHA256 dependency: `@noble/hashes` (sibling to noble curves)
- [ ] `test/fixtures.test.ts` loads `../../fixtures/*.json`, asserts
      every `expected.*` value reproduces byte-for-byte
- [ ] Build emits both ESM and CJS so Hive (Node ESM) and `/mcp` (also
      ESM but may include CJS consumers) can both consume

**Consumers:**

- [ ] Hive's `/macaroons/issue` handler imports `@stakwork/macaroon`
      and calls `signUserAuthorization` + `signInvocation`
- [ ] `/mcp` imports `@stakwork/macaroon` via workspace path (no
      publish needed for in-repo consumers)
- [ ] Any agent-side TS code that attenuates imports the same package
      and calls `attenuate(parentSigBytes, caveats)`

**Gate:** neither Go nor TS ships until both pass the same fixtures.

## What this design buys

- **Signer and verifier are independently implementable.** JCS plus
  standard crypto primitives means the Go verifier and the TS issuer
  share no code, only the spec and the fixtures.
- **Attenuators are trivial to port.** HMAC-SHA256 over JCS is one
  page of code in any language. Agents can be written in whatever the
  agent author prefers without trusting a Hive-provided SDK.
- **Phase 5 has a concrete storage target.** The multisig policy
  object shape is specified here; phase 5 stores exactly that JSON.
- **Hardware crypto fits.** ECDSA-secp256k1 is universally supported
  by hardware wallets and HSMs. Ed25519 is supported by Yubikeys,
  Passkeys, and mobile secure enclaves. The wire format doesn't
  presuppose where the keys live.
- **Schnorr / MuSig2 land without breaking changes.** Adding a new
  `alg` value extends the `multisig-v1` envelope. Phase 4's signers
  and verifiers reject unknown `alg`s; phase 4's wire shape doesn't
  need to change.
- **Test vectors prevent drift.** The single source-of-truth fixture
  set is the contract. If signer and verifier disagree, exactly one
  of them is wrong, and the fixtures say which.
