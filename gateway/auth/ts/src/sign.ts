/**
 * Layer-level signers. Phase 1 custodial mode: Hive holds both org
 * and user private keys and signs both layers itself. Phase 2+ moves
 * the user sign step to the user's device; the API stays the same
 * (the caller swaps which key it uses where).
 */

import { bytesToHex, hexToBytes } from "./encoding.js";
import { signingBytes } from "./jcs.js";
import { ecdsaSign, ed25519Sign } from "./sigs.js";
import type {
  EcdsaSecp256k1Sig,
  Ed25519Sig,
  Invocation,
  InvocationUnsigned,
  MultisigV1Sig,
  UserAuthorization,
  UserAuthorizationUnsigned,
} from "./types.js";

/**
 * Sign a `user_authorization` envelope with a single-key org private
 * key. Custodial phase 1 uses this; phase 3+ uses
 * `signUserAuthorizationMultisig` when the org adopts an m-of-n
 * policy.
 *
 * @param unsigned the layer without `org_sig`
 * @param orgPrivKey 32-byte secp256k1 private key
 */
export function signUserAuthorizationSingle(
  unsigned: UserAuthorizationUnsigned,
  orgPrivKey: Uint8Array,
): UserAuthorization {
  const msg = signingBytes(
    unsigned as unknown as Record<string, unknown>,
    "org_sig",
  );
  const sigBytes = ecdsaSign(orgPrivKey, msg);
  const orgSig: EcdsaSecp256k1Sig = {
    alg: "ecdsa-secp256k1-sha256",
    sig: bytesToHex(sigBytes),
  };
  return { ...unsigned, org_sig: orgSig };
}

/**
 * Sign a `user_authorization` envelope under a multisig policy. Each
 * signer is identified by its `key_index` into the policy's `keys`
 * array. Returns the assembled `multisig-v1` envelope. Threshold
 * enforcement happens at verify time.
 *
 * @param unsigned the layer without `org_sig`
 * @param signers  array of { key_index, privKey } — one entry per
 *                 participating signer. Must be sorted by key_index
 *                 ascending so the canonical output is deterministic.
 */
export function signUserAuthorizationMultisig(
  unsigned: UserAuthorizationUnsigned,
  signers: Array<{ key_index: number; privKey: Uint8Array }>,
): UserAuthorization {
  const msg = signingBytes(
    unsigned as unknown as Record<string, unknown>,
    "org_sig",
  );
  const sorted = [...signers].sort((a, b) => a.key_index - b.key_index);
  const sigs = sorted.map(({ key_index, privKey }) => {
    const sigBytes = ecdsaSign(privKey, msg);
    return {
      key_index,
      alg: "ecdsa-secp256k1-sha256" as const,
      sig: bytesToHex(sigBytes),
    };
  });
  const orgSig: MultisigV1Sig = { alg: "multisig-v1", sigs };
  return { ...unsigned, org_sig: orgSig };
}

/**
 * Sign an `invocation` envelope with the user's Ed25519 private key.
 *
 * @param unsigned the layer without `user_sig`
 * @param userPrivKey 32-byte Ed25519 private key (seed)
 */
export function signInvocation(
  unsigned: InvocationUnsigned,
  userPrivKey: Uint8Array,
): Invocation {
  const msg = signingBytes(
    unsigned as unknown as Record<string, unknown>,
    "user_sig",
  );
  const sigBytes = ed25519Sign(userPrivKey, msg);
  const userSig: Ed25519Sig = {
    alg: "ed25519",
    sig: bytesToHex(sigBytes),
  };
  return { ...unsigned, user_sig: userSig };
}

/**
 * Pull the raw signature bytes out of an invocation's `user_sig`,
 * for use as the HMAC key of the first attenuation.
 */
export function invocationSigBytes(inv: Invocation): Uint8Array {
  return hexToBytes(inv.user_sig.sig);
}
