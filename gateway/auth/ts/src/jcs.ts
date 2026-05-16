/**
 * JSON Canonicalization Scheme (RFC 8785) wrapper.
 *
 * Single source of truth for the canonical-JSON step. All signing
 * inputs and HMAC inputs run through this exactly once.
 *
 * Returns the canonical JSON STRING (not bytes). Use `utf8Bytes` to
 * convert when feeding to crypto primitives.
 */

// canonicalize is shipped as CJS with `module.exports = function`; under
// NodeNext + esModuleInterop the default import resolves to that function.
// The package's own `.d.ts` declares `export default function serialize`.
import canonicalizeNs from "canonicalize";

import { utf8Bytes } from "./encoding.js";

type Canonicalize = (input: unknown) => string | undefined;

// Some TS module-resolution settings see the namespace rather than the
// default; tolerate both.
const canonicalize: Canonicalize =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  typeof (canonicalizeNs as unknown) === "function"
    ? (canonicalizeNs as unknown as Canonicalize)
    : ((canonicalizeNs as unknown as { default: Canonicalize }).default);

export function jcs(value: unknown): string {
  const out = canonicalize(value);
  if (typeof out !== "string") {
    throw new Error(
      "canonicalize() returned non-string (input may contain undefined or non-JSON types)",
    );
  }
  return out;
}

/**
 * Strip one named field from a layer object, then canonicalize the
 * rest, then return UTF-8 bytes. This is the "signing input" recipe
 * from phase 4.
 */
export function signingBytes(
  layer: Record<string, unknown>,
  sigFieldName: string,
): Uint8Array {
  const stripped: Record<string, unknown> = {};
  for (const key of Object.keys(layer)) {
    if (key !== sigFieldName) {
      stripped[key] = layer[key];
    }
  }
  return utf8Bytes(jcs(stripped));
}
