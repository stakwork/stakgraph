# gatekey

Three-layer macaroon signer, attenuator, and verifier for cryptographic
LLM governance. Issued by an organization, signed by a user, attenuated
by sub-agents — every LLM call carries a chain of authority you can
verify offline.

```
gatekey
  org   ──ECDSA──▶  user_authorization   (who, for how long, in which realms)
  user  ──Ed25519─▶ invocation           (this specific run, this budget)
  agent ──HMAC───▶  attenuation[]        (narrower scope for each sub-agent)
```

The wire format, signing inputs, and HMAC chain are pinned to a
language-neutral spec with fixtures, so a TS signer and a Go verifier
(or any other implementation) agree byte-for-byte.

## Install

```sh
npm install gatekey
```

Requires Node >= 20.

## Quick start

### Mint an invocation macaroon

The custodial / phase-1 case: the issuer holds both the org private key
and the user's private key, and produces a complete macaroon ready to
put in an `x-macaroon` header.

```ts
import {
  signUserAuthorizationSingle,
  signInvocation,
  encodeMacaroon,
  type UserAuthorizationUnsigned,
  type InvocationUnsigned,
} from "gatekey";

const uaUnsigned: UserAuthorizationUnsigned = {
  user_id: "user_123",
  user_pubkey: { alg: "ed25519", key: userPubkeyHex },
  agents: ["browser-agent", "pr-monitor"],
  // Optional. Single-swarm deployments can omit `budget` entirely.
  // Multi-swarm deployments can opt into per-realm caps via
  // `budget.realm_budgets`.
  budget: {
    max_per_invocation_usd: 25,
    max_total_usd: 1000,
  },
  iat: new Date().toISOString(),
  exp: new Date(Date.now() + 24 * 3600 * 1000).toISOString(),
  nonce: randomNonceHex(),
};
const ua = signUserAuthorizationSingle(uaUnsigned, orgPrivKey);

const invUnsigned: InvocationUnsigned = {
  agents: ["browser-agent"],
  run_id: "run_abc",
  max_cost_usd: 10.0,
  max_steps: 200,
  iat: new Date().toISOString(),
  exp: new Date(Date.now() + 3600 * 1000).toISOString(),
  nonce: randomNonceHex(),
};
const invocation = signInvocation(invUnsigned, userPrivKey);

const macaroon = encodeMacaroon({
  v: 1,
  org_id: "org_acme",
  user_authorization: ua,
  invocation,
  attenuations: [],
});

// → base64url string. Send as `x-macaroon: <macaroon>`.
```

### Attenuate before spawning a sub-agent

A parent agent narrows the scope (smaller budget, narrower agent list)
before handing control to a child. No network call, no extra signature —
one HMAC.

```ts
import { attenuate, invocationSigBytes } from "gatekey";

const childCaveats = {
  agents: ["browser-agent", "browser-agent.search"],
  max_cost_usd: 2.0,        // ≤ parent
  max_steps: 50,             // ≤ parent
  run_id: "run_child_xyz",
  exp: new Date(Date.now() + 600 * 1000).toISOString(),  // ≤ parent
  nonce: randomNonceHex(),
};

const link = attenuate(invocationSigBytes(invocation), childCaveats);
// Append `link` to macaroon.attenuations and re-encode.
```

### Verify

```ts
import { verify, VerifyError } from "gatekey";

try {
  const claims = verify(macaroon, policy, new Date());
  // claims.agent_name, claims.run_id, claims.effective_caveats, ...
} catch (e) {
  if (e instanceof VerifyError) {
    // e.code is one of: macaroon_malformed, invalid_user_authorization,
    // user_authorization_expired, invalid_invocation_signature,
    // invocation_violated, macaroon_expired, attenuation_invalid,
    // attenuation_widened, macaroon_unsupported_version.
  }
  throw e;
}
```

The verifier is pure: no I/O, no clock except the one you hand it, no
revocation list. Wrap it with whatever transport + freshness checks
your environment needs.

## Concepts

- **`user_authorization`** — signed once by the organization. Names a
  user, embeds their public key, and lists the agents they may act
  as. Optionally carries a `budget` block with per-invocation and
  cumulative caps, plus opt-in per-realm caps (`realm_budgets`) for
  multi-swarm deployments. ECDSA-secp256k1 single-key or multisig
  (`multisig-v1`).
- **`invocation`** — signed by the user (with the key embedded in the
  UA above). Names one specific run with its own budget, step ceiling,
  and expiry. Ed25519.
- **`attenuation`** — appended by a parent agent for each sub-agent.
  HMAC-SHA256 chained from the user's invocation signature, then from
  each previous attenuation's HMAC. Each link narrows the effective
  caveats; widening is rejected at verify time.

Every signed layer (UA, invocation, attenuation) shares the same
shape — a flat `agents` set plus an optional `budget` — and phase
11's symmetric `narrow(parent, child)` rule applies at every
boundary. See
[`phase-11-symmetric-recursive-authorization.md`](https://github.com/stakwork/stakgraph/blob/main/gateway/plans/phases/phase-11-symmetric-recursive-authorization.md)
for the protocol; the `agents[]` axis flips direction at the
UA→invocation boundary (subset) versus the attenuation boundaries
(lineage extension).

The verifier returns `Claims` with the **effective caveats** — the
intersection of the invocation and every attenuation in the chain.
This is the authoritative answer to "what is this call allowed to do."

## API surface

| Export | Purpose |
|---|---|
| `signUserAuthorizationSingle`, `signUserAuthorizationMultisig` | Sign the org→user layer. |
| `signInvocation`, `invocationSigBytes` | Sign the user→run layer. |
| `attenuate`, `computeAttenuationHmac`, `attenuationSigBytes` | Append a sub-agent link. |
| `encodeMacaroon`, `decodeMacaroon` | Base64url transport encoding. |
| `verify`, `VerifyError` | Full end-to-end verifier. |
| `jcs`, `signingBytes` | RFC 8785 canonicalization (advanced). |
| `ed25519Sign/Verify/PublicKey`, `ecdsaSign/Verify/PublicKey` | Raw primitives (advanced / keygen). |
| `bytesToHex`, `hexToBytes`, `bytesToBase64url`, `base64urlToBytes`, `utf8Bytes` | Encoding helpers. |
| `rootFromLeaves`, `rootFromLeafHashes`, `hashLeaf`, `leafHash`, `nodeHash`, `emptyRoot` | Transparency log: RFC 9162 Merkle tree hashing. |
| `verifyInclusion`, `verifyConsistency` | Transparency log: proof verifiers (RFC 9162 §2.1.3.2 / §2.1.4.2). |
| `verifySth`, `signSth`, `sthSigningBytes` | Transparency log: signed tree heads. |

Types: `Macaroon`, `UserAuthorization`, `Invocation`, `Attenuation`,
`Claims`, `EffectiveCaveats`, `Policy`, `PubKey`, plus their `*Unsigned`
variants. See `src/types.ts` for the wire shape.

## Transparency log

Every accounted LLM call that passes through the gateway becomes a
leaf in an append-only Merkle log (RFC 9162 §2.1 hashing, the scheme
RFC 6962 introduced). The gateway serves a signed tree head plus the
leaves since the last one at `GET /_plugin/tlog/sth?since=N`; a
witness outside the swarm (Hive) verifies and countersigns it. Once a
head is witnessed, nothing below it can be edited or dropped without
the recomputed root disagreeing. Spec:
`gateway/plans/phases/phase-12-transparency-log.md`.

The witness check, end to end:

```ts
import { rootFromLeaves, verifySth, bytesToHex, type TlogSthResponse } from "gatekey";

const page: TlogSthResponse = await fetchSth(since); // your transport
if (!verifySth(page.sth, page.log_pubkey)) throw new Error("bad sth signature");

// A full replica recomputes the root over every leaf it holds plus
// the new page. Store leaves as their canonical JSON string so the
// recompute never depends on a JSON round-trip.
const allLeaves = [...storedCanonicalLeaves, ...page.leaves];
if (bytesToHex(rootFromLeaves(allLeaves)) !== page.sth.root_hash) {
  throw new Error("root does not reproduce from leaves");
}
// Only now: countersign `page.sth` with the org key and store head + leaves.
```

Light witnesses that hold only heads use `verifyConsistency` with the
served `consistency_proof`; per-call receipts (Part 2, later) verify
with `verifyInclusion`. Leaf and head shapes are `TlogLeaf` and
`SignedTreeHead`; the head signature is the macaroon construction,
`ECDSA-secp256k1-SHA256` over `JCS(sth \ sig)`.

## Wire format

```
base64url( JCS({
  v: 1,
  org_id: "...",
  user_authorization: {
    user_id, user_pubkey, agents, budget?, iat, exp, nonce,
    org_sig: { alg, sig } | { alg: "multisig-v1", sigs: [...] }
  },
  invocation: {
    agents, run_id, max_cost_usd, max_steps, budget?,
    iat, exp, nonce,
    user_sig: { alg: "ed25519", sig }
  },
  attenuations: [
    { caveats: { agents, max_cost_usd, max_steps, run_id, budget?,
                 exp, nonce },
      hmac: "..." },
    ...
  ]
}) )
```

`budget` is the same shape at every layer:

```
budget: {
  max_per_invocation_usd?: number,
  max_total_usd?:          number,
  realm_budgets?:          { [realm_id]: { max_total_usd: number } }
}
```

All binary fields are lowercase hex (no `0x` prefix). All timestamps
are RFC 3339 UTC strings. Canonical JSON is RFC 8785 (JCS).

## Cross-language

A Go verifier exists in the same source repository
([`stakgraph/gateway/auth/go/`](https://github.com/stakwork/stakgraph/tree/main/gateway/auth/go))
and shares the test fixtures, so byte-equivalence is enforced in CI.
For the transparency log the roles flip: the Go gateway is the
producer and generates `gateway/auth/fixtures/tlog-*.json`
(`go test ./internal/tlog -update`), and this package's test checks
them byte-for-byte.
Sub-agent attenuation only requires HMAC-SHA256 + JCS + hex encoding —
the spec and fixtures together let any language implement an attenuator
in a few dozen lines.

## License

MIT
