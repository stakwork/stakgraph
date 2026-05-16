/**
 * Hex and base64url helpers. Phase 4 wire format uses lowercase hex
 * WITHOUT `0x` prefix, and base64url WITHOUT padding (RFC 4648 §5).
 */

const HEX_CHARS = "0123456789abcdef";

export function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i]!;
    out += HEX_CHARS[b >>> 4]! + HEX_CHARS[b & 0x0f]!;
  }
  return out;
}

export function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new Error(`hex string has odd length: ${hex.length}`);
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    const hi = parseHexDigit(hex.charCodeAt(i * 2));
    const lo = parseHexDigit(hex.charCodeAt(i * 2 + 1));
    out[i] = (hi << 4) | lo;
  }
  return out;
}

function parseHexDigit(code: number): number {
  if (code >= 48 && code <= 57) return code - 48;         // 0-9
  if (code >= 97 && code <= 102) return code - 97 + 10;   // a-f
  if (code >= 65 && code <= 70) return code - 65 + 10;    // A-F (tolerated on input)
  throw new Error(`invalid hex digit: ${String.fromCharCode(code)}`);
}

/**
 * Base64url encode without padding. RFC 4648 §5.
 */
export function bytesToBase64url(bytes: Uint8Array): string {
  // Node 20+ has Buffer; use it for speed and correctness.
  return Buffer.from(bytes).toString("base64url");
}

export function base64urlToBytes(s: string): Uint8Array {
  return new Uint8Array(Buffer.from(s, "base64url"));
}

const UTF8 = new TextEncoder();

export function utf8Bytes(s: string): Uint8Array {
  return UTF8.encode(s);
}
