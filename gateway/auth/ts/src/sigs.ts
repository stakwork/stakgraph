/**
 * Signing + verifying primitives for the two curves phase 4 uses:
 * Ed25519 (user keys) and ECDSA-secp256k1-SHA256 (org keys, single-key
 * and multisig members).
 *
 * All inputs and outputs are raw `Uint8Array`s. Hex conversion happens
 * at the boundary in `sign.ts` / `verify.ts`.
 */

import { secp256k1 } from "@noble/curves/secp256k1";
import { ed25519 } from "@noble/curves/ed25519";
import { sha256 } from "@noble/hashes/sha2";

// ─── Ed25519 ──────────────────────────────────────────────────────────

/**
 * Sign raw bytes with Ed25519. No prehash — Ed25519 signs the message
 * directly (per RFC 8032).
 *
 * @param privKey 32-byte private key
 * @param msg     bytes to sign
 * @returns       64-byte signature
 */
export function ed25519Sign(privKey: Uint8Array, msg: Uint8Array): Uint8Array {
  return ed25519.sign(msg, privKey);
}

/**
 * @param pubKey 32-byte raw public key
 * @param msg    bytes that were signed
 * @param sig    64-byte signature
 */
export function ed25519Verify(
  pubKey: Uint8Array,
  msg: Uint8Array,
  sig: Uint8Array,
): boolean {
  try {
    return ed25519.verify(sig, msg, pubKey);
  } catch {
    return false;
  }
}

export function ed25519PublicKey(privKey: Uint8Array): Uint8Array {
  return ed25519.getPublicKey(privKey);
}

// ─── ECDSA-secp256k1-SHA256 ───────────────────────────────────────────

/**
 * Sign with ECDSA over secp256k1. We SHA-256 the message first
 * (phase 4 requires the prehash), then produce a compact `r||s`
 * signature with low-s normalization (BIP 62) so two signers
 * producing "the same" signature produce byte-identical output.
 *
 * @param privKey 32-byte private key
 * @param msg     bytes to sign (will be SHA-256'd internally)
 * @returns       64-byte compact signature (r||s), low-s normalized
 */
export function ecdsaSign(privKey: Uint8Array, msg: Uint8Array): Uint8Array {
  const digest = sha256(msg);
  // @noble/curves' sign() is low-s by default and returns a Signature
  // object; toCompactRawBytes() emits the 64-byte r||s form.
  const sig = secp256k1.sign(digest, privKey, { lowS: true });
  return sig.toCompactRawBytes();
}

/**
 * @param pubKey 33-byte compressed public key
 * @param msg    bytes that were signed (will be SHA-256'd internally)
 * @param sig    64-byte compact signature (r||s)
 * @returns      true iff signature is valid AND low-s normalized
 */
export function ecdsaVerify(
  pubKey: Uint8Array,
  msg: Uint8Array,
  sig: Uint8Array,
): boolean {
  try {
    const digest = sha256(msg);
    return secp256k1.verify(sig, digest, pubKey, { lowS: true });
  } catch {
    return false;
  }
}

/**
 * Compressed (33-byte) pubkey from a 32-byte private key.
 */
export function ecdsaPublicKey(privKey: Uint8Array): Uint8Array {
  return secp256k1.getPublicKey(privKey, true);
}
