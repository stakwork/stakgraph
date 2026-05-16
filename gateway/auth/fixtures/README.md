# Macaroon test vectors

Language-neutral fixtures that pin down every byte of the phase-4
wire format. Every implementation (Go, TS, polyglot) loads these,
reproduces every `expected.*` value from the `inputs.*` private keys
and unsigned objects, and fails if any byte differs.

## Files

```
keys.json                  the deterministic seed keys used by all fixtures
01-simple.json             single-key org, one ua + one invocation, zero attenuations
02-one-attenuation.json    single-key org, one sub-agent attenuation
03-two-attenuations.json   single-key org, two-deep attenuation chain
04-multisig-2of3.json      2-of-3 multisig org, signers 0 and 2 participate
```

## Reproducibility

These files are **generated**, not hand-written. The source of truth
is `gateway/auth/ts/scripts/regenerate-fixtures.ts`. To regenerate:

```sh
cd gateway/auth/ts
npm install
npm run regenerate-fixtures
```

The seed private keys are fixed in the script (well-known low-numbered
secp256k1 keys, fixed 32-byte Ed25519 seeds). The script produces
deterministic output because:

- `@noble/curves` ECDSA signing uses RFC 6979 deterministic nonces.
- `@noble/curves` Ed25519 signing is deterministic by RFC 8032.
- Everything else is just JCS canonicalization + HMAC-SHA256, both
  deterministic by definition.

Anyone running the script gets byte-identical output. The fixtures are
checked in alongside the spec so anyone can read them without running
anything.

## Fixture shape

Each `NN-*.json` has two top-level keys:

```jsonc
{
  "description": "human-readable summary",
  "inputs":   { /* private keys + unsigned objects */ },
  "expected": { /* every intermediate value, byte-for-byte */ }
}
```

### `inputs`

What an implementation needs in order to produce the macaroon:

| Field                            | Notes                                                 |
| -------------------------------- | ----------------------------------------------------- |
| `org_id`                         | The `org_id` that goes in the macaroon's top-level field. |
| `org_priv_hex`                   | Single-sig only. 32-byte secp256k1 private key.       |
| `org_privs_hex`                  | Multisig only. Map of `key_index → priv_hex`.         |
| `participating_signers`          | Multisig only. Array of key_indexes that actually sign. |
| `user_priv_hex`                  | 32-byte Ed25519 seed.                                 |
| `policy`                         | The trust-registry policy a verifier would store for this org. |
| `ua_unsigned`                    | `user_authorization` without `org_sig`.               |
| `inv_unsigned`                   | `invocation` without `user_sig`.                      |
| `atts_unsigned`                  | Array of `caveats` objects, one per attenuation in the chain. |

### `expected`

Every byte at every intermediate step:

| Field                            | What it pins down                                     |
| -------------------------------- | ----------------------------------------------------- |
| `ua_signing_bytes_hex`           | UTF-8 bytes of `JCS(ua_unsigned)` (the ECDSA input).  |
| `inv_signing_bytes_hex`          | UTF-8 bytes of `JCS(inv_unsigned)` (the Ed25519 input). |
| `ua_canonical_json`              | The exact canonical JSON string of the signed `user_authorization` minus `org_sig`. |
| `inv_canonical_json`             | Same for `invocation` minus `user_sig`.               |
| `macaroon_canonical_json`        | JCS of the assembled macaroon (signatures included). |
| `macaroon_b64url`                | base64url of the canonical macaroon — the value that goes in `x-macaroon`. |
| `attenuation_hmac_inputs[i]`     | For each attenuation: the prev_sig bytes, the canonical caveats, the HMAC input, the HMAC output. |
| `claims`                         | What the verifier returns on success.                 |
| `verify_result`                  | `"ok"` for the success-path fixtures here.            |

## How implementations use this

Both the Go pure verifier (`gateway/auth/go/`) and the TS package
(`gateway/auth/ts/`) ship a fixture test that:

1. Reads each `NN-*.json`.
2. Re-signs `ua_unsigned` with `org_priv_hex` (or multisig signers).
3. Re-signs `inv_unsigned` with `user_priv_hex`.
4. Re-HMACs each entry in `atts_unsigned`.
5. Asserts every intermediate value matches `expected.*` byte-for-byte.
6. Verifies the assembled macaroon end-to-end against `inputs.policy`.

If both implementations pass, they agree on the wire format. If a spec
change shifts the bytes, regenerate the fixtures (the script is the
source of truth) and commit the new fixtures alongside the spec change.
