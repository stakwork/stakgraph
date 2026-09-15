/**
 * Signed tree head: signing input, signing (for tests and any TS log
 * producer), and verification against the gateway's `log_pubkey`.
 */

import { bytesToHex, hexToBytes } from "../encoding.js";
import { signingBytes } from "../jcs.js";
import { ecdsaSign, ecdsaVerify } from "../sigs.js";
import type { SignedTreeHead, SignedTreeHeadUnsigned } from "./types.js";

/**
 * The exact bytes the log key signs: `JCS({tree_size, root_hash,
 * signed_at})`. Built from the three named fields only, so an object
 * carrying extra keys still yields the producer's bytes.
 */
export function sthSigningBytes(sth: SignedTreeHeadUnsigned): Uint8Array {
  return signingBytes(
    { tree_size: sth.tree_size, root_hash: sth.root_hash, signed_at: sth.signed_at },
    "sig",
  );
}

/** Sign a tree head with a 32-byte secp256k1 private key. */
export function signSth(unsigned: SignedTreeHeadUnsigned, logPrivKey: Uint8Array): SignedTreeHead {
  return {
    tree_size: unsigned.tree_size,
    root_hash: unsigned.root_hash,
    signed_at: unsigned.signed_at,
    sig: bytesToHex(ecdsaSign(logPrivKey, sthSigningBytes(unsigned))),
  };
}

/**
 * Verify `sth.sig` with the compressed-hex `log_pubkey` served next
 * to it. The log key is per gateway boot and attests only "this
 * gateway served this head"; a witness must still recompute the root
 * from the leaves before countersigning.
 */
export function verifySth(sth: SignedTreeHead, logPubkeyHex: string): boolean {
  try {
    return ecdsaVerify(hexToBytes(logPubkeyHex), sthSigningBytes(sth), hexToBytes(sth.sig));
  } catch {
    return false;
  }
}
