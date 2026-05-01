import { getPublicKey, etc } from "@noble/ed25519";
import { sha512 } from "@noble/hashes/sha512";
import { randomBytes } from "node:crypto";

// @noble/ed25519 v2 ships without a built-in sha512 backend; the caller has
// to plug one in. We use @noble/hashes' sha512. This must run before any
// call into getPublicKey/sign/verify.
etc.sha512Sync = (...msgs: Uint8Array[]) => {
  const h = sha512.create();
  for (const m of msgs) h.update(m);
  return h.digest();
};

/**
 * Convert a 32-byte hex-encoded ed25519 private key into the 32-byte hex
 * pubkey. Throws if the input isn't exactly 64 hex chars.
 */
export function privHexToPubHex(hexPriv: string): string {
  const priv = hexToBytes(hexPriv);
  if (priv.length !== 32) {
    throw new Error(
      `private key must be 32 bytes (64 hex chars); got ${priv.length} bytes`,
    );
  }
  const pub = getPublicKey(priv);
  return bytesToHex(pub);
}

/**
 * Format the agent's git identity. `full` is the standard "Name <email>"
 * rendering used by `git log --format='%an <%ae>'`.
 */
export function authorString(
  child: number,
  pubkeyHex: string,
): { name: string; email: string; full: string } {
  const name = `agent-${child}`;
  const email = `${pubkeyHex}@agents.sphinx.chat`;
  return { name, email, full: `${name} <${email}>` };
}

/**
 * Build an OpenSSH unencrypted ed25519 private key file (PEM-armored).
 *
 * Layout (all lengths big-endian uint32; "string" = uint32 length + bytes):
 *
 *   "openssh-key-v1\0"             (15 bytes magic, NUL-terminated)
 *   string  ciphername             "none"
 *   string  kdfname                "none"
 *   string  kdfoptions             ""
 *   uint32  num_keys               1
 *   string  public_key_blob
 *   string  encrypted_section      (cipher=none; padded to 8-byte alignment)
 *
 * public_key_blob:
 *   string  "ssh-ed25519"
 *   string  pubkey_32
 *
 * encrypted_section:
 *   uint32  checkint1              (random)
 *   uint32  checkint2              (== checkint1)
 *   string  "ssh-ed25519"
 *   string  pubkey_32
 *   string  privkey_64             (raw_priv_32 || pubkey_32)
 *   string  comment
 *   bytes   padding                (1,2,3,...n)
 */
export function buildOpenSSHPrivateKey(
  hexPriv: string,
  comment: string,
): string {
  const priv = hexToBytes(hexPriv);
  if (priv.length !== 32) {
    throw new Error(
      `private key must be 32 bytes (64 hex chars); got ${priv.length} bytes`,
    );
  }
  const pub = getPublicKey(priv);
  const expandedPriv = concat(priv, pub); // 64 bytes — OpenSSH's "secret + public"

  const MAGIC = Buffer.from("openssh-key-v1\0", "utf8");

  const publicKeyBlob = concat(
    sshString(Buffer.from("ssh-ed25519", "utf8")),
    sshString(Buffer.from(pub)),
  );

  // checkint pair must match; OpenSSH uses these to detect bad-passphrase
  // decryption. With cipher=none they're not load-bearing but we still
  // randomize them.
  const checkint = randomBytes(4);

  let encryptedSection = concat(
    checkint,
    checkint,
    sshString(Buffer.from("ssh-ed25519", "utf8")),
    sshString(Buffer.from(pub)),
    sshString(Buffer.from(expandedPriv)),
    sshString(Buffer.from(comment, "utf8")),
  );

  // Pad to 8-byte alignment with bytes 1, 2, 3, ...
  const padLen = (8 - (encryptedSection.length % 8)) % 8;
  if (padLen > 0) {
    const pad = Buffer.alloc(padLen);
    for (let i = 0; i < padLen; i++) pad[i] = i + 1;
    encryptedSection = concat(encryptedSection, pad);
  }

  const blob = concat(
    MAGIC,
    sshString(Buffer.from("none", "utf8")),
    sshString(Buffer.from("none", "utf8")),
    sshString(Buffer.alloc(0)),
    u32(1),
    sshString(publicKeyBlob),
    sshString(encryptedSection),
  );

  return pemArmor(blob, "OPENSSH PRIVATE KEY");
}

// ---------- helpers ----------

function hexToBytes(hex: string): Uint8Array {
  if (typeof hex !== "string" || !/^[0-9a-fA-F]*$/.test(hex) || hex.length % 2) {
    throw new Error("invalid hex string");
  }
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return out;
}

function bytesToHex(b: Uint8Array): string {
  let s = "";
  for (let i = 0; i < b.length; i++) s += b[i].toString(16).padStart(2, "0");
  return s;
}

function u32(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32BE(n >>> 0, 0);
  return b;
}

function sshString(payload: Buffer | Uint8Array): Buffer {
  const buf = Buffer.isBuffer(payload) ? payload : Buffer.from(payload);
  return Buffer.concat([u32(buf.length), buf]);
}

function concat(...parts: (Buffer | Uint8Array)[]): Buffer {
  return Buffer.concat(parts.map((p) => (Buffer.isBuffer(p) ? p : Buffer.from(p))));
}

function pemArmor(blob: Buffer, label: string): string {
  const b64 = blob.toString("base64");
  const lines: string[] = [];
  for (let i = 0; i < b64.length; i += 70) {
    lines.push(b64.slice(i, i + 70));
  }
  return `-----BEGIN ${label}-----\n${lines.join("\n")}\n-----END ${label}-----\n`;
}
