/**
 * Sub-agent attenuation. Appends one HMAC link to the chain.
 *
 * HMAC definition from phase 4:
 *
 *     prev_sig_bytes  = invocation.user_sig.sig (first link)
 *                     | previous attenuation's hmac (later links)
 *     hmac_i          = HMAC-SHA256(prev_sig_bytes, utf8(JCS(caveats)))
 *
 * This is the only crypto primitive a sub-agent needs, in any
 * language. The polyglot port (Python, Rust, …) reproduces this
 * function exactly.
 */

import { hmac } from "@noble/hashes/hmac";
import { sha256 } from "@noble/hashes/sha2";

import { bytesToHex, hexToBytes, utf8Bytes } from "./encoding.js";
import { jcs } from "./jcs.js";
import type { Attenuation, AttenuationCaveats } from "./types.js";

/**
 * Compute the HMAC for one attenuation link.
 *
 * @param prevSigBytes raw bytes of the previous link's signature:
 *                     - For the first attenuation, this is the raw
 *                       bytes of `invocation.user_sig.sig` (hex-decoded).
 *                     - For later attenuations, this is the previous
 *                       attenuation's `hmac` (hex-decoded).
 * @param caveats      this attenuation's caveats
 * @returns            32-byte HMAC-SHA256 output
 */
export function computeAttenuationHmac(
  prevSigBytes: Uint8Array,
  caveats: AttenuationCaveats,
): Uint8Array {
  const msg = utf8Bytes(jcs(caveats));
  return hmac(sha256, prevSigBytes, msg);
}

/**
 * Build a complete `Attenuation` object: compute the HMAC and pair
 * it with the caveats.
 */
export function attenuate(
  prevSigBytes: Uint8Array,
  caveats: AttenuationCaveats,
): Attenuation {
  const hmacBytes = computeAttenuationHmac(prevSigBytes, caveats);
  return { caveats, hmac: bytesToHex(hmacBytes) };
}

/**
 * Convenience: pull the HMAC bytes out of an existing attenuation
 * for use as the `prevSigBytes` of the next link.
 */
export function attenuationSigBytes(att: Attenuation): Uint8Array {
  return hexToBytes(att.hmac);
}
