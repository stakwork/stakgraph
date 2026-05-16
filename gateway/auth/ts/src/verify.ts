/**
 * Pure verifier — TS mirror of `gateway/auth/go/verify.go`. Takes a
 * macaroon (either base64url-encoded or already parsed), the trusting
 * org's policy, and a clock; returns `Claims` or throws a
 * `VerifyError` with a machine-readable `code`.
 *
 * No I/O. No revocation list. Adapters wrap this with their own
 * revocation + transport plumbing.
 */

import { base64urlToBytes, hexToBytes, utf8Bytes } from "./encoding.js";
import { jcs, signingBytes } from "./jcs.js";
import { computeAttenuationHmac } from "./attenuate.js";
import { ecdsaVerify, ed25519Verify } from "./sigs.js";
import type {
  Attenuation,
  AttenuationCaveats,
  Claims,
  EffectiveCaveats,
  Invocation,
  Macaroon,
  MultisigV1Sig,
  Policy,
  UserAuthorization,
} from "./types.js";

export type VerifyErrorCode =
  | "macaroon_malformed"
  | "macaroon_unsupported_version"
  | "invalid_user_authorization"
  | "user_authorization_expired"
  | "invalid_invocation_signature"
  | "invocation_violated"
  | "macaroon_expired"
  | "attenuation_invalid"
  | "attenuation_widened";

export class VerifyError extends Error {
  readonly code: VerifyErrorCode;
  constructor(code: VerifyErrorCode, message?: string) {
    super(message ?? code);
    this.code = code;
    this.name = "VerifyError";
  }
}

const KNOWN_TOP_LEVEL_FIELDS = new Set([
  "v",
  "org_id",
  "user_authorization",
  "invocation",
  "attenuations",
]);

/**
 * Verify a macaroon end-to-end.
 *
 * @param macaroon either the base64url-encoded wire form or a parsed object
 * @param policy   the trusted org's pubkey or multisig policy
 * @param now      the verifier's current time
 */
export function verify(
  macaroon: string | Macaroon,
  policy: Policy,
  now: Date,
): Claims {
  const m = typeof macaroon === "string" ? decodeMacaroon(macaroon) : macaroon;

  validateTopLevel(m);

  verifyUserAuthorization(m.user_authorization, policy);
  if (asDate(m.user_authorization.exp) < now) {
    throw new VerifyError("user_authorization_expired");
  }

  verifyInvocation(m.invocation, m.user_authorization);
  enforceInvocationCaveats(m.invocation, m.user_authorization, now);

  const { effective, runId, agentName, nonces } = walkAttenuations(
    m.invocation,
    m.attenuations,
    now,
  );

  return {
    org_id: m.org_id,
    user_id: m.user_authorization.user_id,
    workspace: m.invocation.workspace,
    agent_name: agentName,
    run_id: runId,
    effective_caveats: effective,
    nonces: [m.user_authorization.nonce, m.invocation.nonce, ...nonces],
    iat: m.invocation.iat,
  };
}

// ─── decode + structural validation ───────────────────────────────────

export function decodeMacaroon(b64url: string): Macaroon {
  let parsed: unknown;
  try {
    const bytes = base64urlToBytes(b64url);
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new VerifyError("macaroon_malformed", "base64url or JSON decode failed");
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new VerifyError("macaroon_malformed", "top-level is not an object");
  }
  return parsed as Macaroon;
}

export function encodeMacaroon(m: Macaroon): string {
  // base64url(JCS(m)) — same canonicalization the signing inputs use,
  // applied at the top level for transport.
  const canonical = jcs(m as unknown as Record<string, unknown>);
  const bytes = utf8Bytes(canonical);
  return Buffer.from(bytes).toString("base64url");
}

function validateTopLevel(m: Macaroon): void {
  if (m.v !== 1) {
    throw new VerifyError("macaroon_unsupported_version", `v=${String(m.v)}`);
  }
  for (const key of Object.keys(m)) {
    if (!KNOWN_TOP_LEVEL_FIELDS.has(key)) {
      throw new VerifyError("macaroon_malformed", `unknown top-level field: ${key}`);
    }
  }
  if (typeof m.org_id !== "string" || !m.user_authorization || !m.invocation
      || !Array.isArray(m.attenuations)) {
    throw new VerifyError("macaroon_malformed", "missing required fields");
  }
}

// ─── org-signed user_authorization ────────────────────────────────────

function verifyUserAuthorization(ua: UserAuthorization, policy: Policy): void {
  const msg = signingBytes(ua as unknown as Record<string, unknown>, "org_sig");

  if (policy.type === "single") {
    if (ua.org_sig.alg !== "ecdsa-secp256k1-sha256") {
      throw new VerifyError("invalid_user_authorization",
        `expected single-key ecdsa, got alg=${ua.org_sig.alg}`);
    }
    const pub = hexToBytes(policy.key.key);
    const sig = hexToBytes(ua.org_sig.sig);
    if (!ecdsaVerify(pub, msg, sig)) {
      throw new VerifyError("invalid_user_authorization", "ecdsa verify failed");
    }
    return;
  }

  // multisig
  if (ua.org_sig.alg !== "multisig-v1") {
    throw new VerifyError("invalid_user_authorization",
      `expected multisig-v1, got alg=${ua.org_sig.alg}`);
  }
  const multisig: MultisigV1Sig = ua.org_sig;
  let validCount = 0;
  const seen = new Set<number>();
  for (const entry of multisig.sigs) {
    if (seen.has(entry.key_index)) continue;  // ignore duplicates
    seen.add(entry.key_index);
    const pk = policy.keys[entry.key_index];
    if (!pk) continue;
    if (entry.alg !== pk.alg) continue;
    if (entry.alg !== "ecdsa-secp256k1-sha256") continue;
    const ok = ecdsaVerify(hexToBytes(pk.key), msg, hexToBytes(entry.sig));
    if (ok) validCount++;
  }
  if (validCount < policy.threshold) {
    throw new VerifyError("invalid_user_authorization",
      `multisig threshold not met: ${validCount}/${policy.threshold}`);
  }
}

// ─── user-signed invocation ───────────────────────────────────────────

function verifyInvocation(inv: Invocation, ua: UserAuthorization): void {
  if (inv.user_sig.alg !== "ed25519" || ua.user_pubkey.alg !== "ed25519") {
    throw new VerifyError("invalid_invocation_signature",
      "non-ed25519 user signature/pubkey not supported in v=1");
  }
  const msg = signingBytes(inv as unknown as Record<string, unknown>, "user_sig");
  const pub = hexToBytes(ua.user_pubkey.key);
  const sig = hexToBytes(inv.user_sig.sig);
  if (!ed25519Verify(pub, msg, sig)) {
    throw new VerifyError("invalid_invocation_signature", "ed25519 verify failed");
  }
}

function enforceInvocationCaveats(
  inv: Invocation,
  ua: UserAuthorization,
  now: Date,
): void {
  if (!ua.permissions.workspaces.includes(inv.workspace)) {
    throw new VerifyError("invocation_violated",
      `workspace ${inv.workspace} not in user permissions`);
  }
  for (const a of inv.agents) {
    if (!ua.permissions.agents.includes(a)) {
      throw new VerifyError("invocation_violated", `agent ${a} not in user permissions`);
    }
  }
  if (asDate(inv.exp) < now) {
    throw new VerifyError("macaroon_expired", `invocation expired at ${inv.exp}`);
  }
}

// ─── attenuation chain walk ───────────────────────────────────────────

interface WalkResult {
  effective: EffectiveCaveats;
  runId: string;
  agentName: string;
  nonces: string[];
}

function walkAttenuations(
  inv: Invocation,
  atts: Attenuation[],
  now: Date,
): WalkResult {
  let prevSigBytes = hexToBytes(inv.user_sig.sig);
  let effective: EffectiveCaveats = {
    agents: [...inv.agents],
    max_cost_usd: inv.max_cost_usd,
    max_steps: inv.max_steps,
    exp: inv.exp,
  };
  let runId = inv.run_id;
  const nonces: string[] = [];

  for (const att of atts) {
    const expected = computeAttenuationHmac(prevSigBytes, att.caveats);
    const expectedHex = bytesToHexLocal(expected);
    if (expectedHex !== att.hmac) {
      throw new VerifyError("attenuation_invalid", "hmac mismatch");
    }
    enforceNarrowing(effective, att.caveats);
    if (asDate(att.caveats.exp) < now) {
      throw new VerifyError("macaroon_expired", `attenuation expired at ${att.caveats.exp}`);
    }
    effective = {
      agents: att.caveats.agents,
      max_cost_usd: att.caveats.max_cost_usd,
      max_steps: att.caveats.max_steps,
      exp: att.caveats.exp,
    };
    runId = att.caveats.run_id;
    nonces.push(att.caveats.nonce);
    prevSigBytes = hexToBytes(att.hmac);
  }

  const agentName = effective.agents[effective.agents.length - 1] ?? "";
  return { effective, runId, agentName, nonces };
}

function enforceNarrowing(parent: EffectiveCaveats, child: AttenuationCaveats): void {
  // agents: child must include every parent entry (child ⊇ parent)
  for (const a of parent.agents) {
    if (!child.agents.includes(a)) {
      throw new VerifyError("attenuation_widened",
        `child agents dropped parent entry: ${a}`);
    }
  }
  if (child.max_cost_usd > parent.max_cost_usd) {
    throw new VerifyError("attenuation_widened",
      `child max_cost_usd ${child.max_cost_usd} > parent ${parent.max_cost_usd}`);
  }
  if (child.max_steps > parent.max_steps) {
    throw new VerifyError("attenuation_widened",
      `child max_steps ${child.max_steps} > parent ${parent.max_steps}`);
  }
  // exp: child.exp <= parent.exp (RFC 3339 UTC lexicographic compare works)
  if (child.exp > parent.exp) {
    throw new VerifyError("attenuation_widened",
      `child exp ${child.exp} > parent ${parent.exp}`);
  }
}

// ─── helpers ──────────────────────────────────────────────────────────

function asDate(iso: string): Date {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    throw new VerifyError("macaroon_malformed", `invalid timestamp: ${iso}`);
  }
  return d;
}

// Local copy to avoid a circular import.
const HEX = "0123456789abcdef";
function bytesToHexLocal(b: Uint8Array): string {
  let out = "";
  for (let i = 0; i < b.length; i++) {
    const v = b[i]!;
    out += HEX[v >>> 4]! + HEX[v & 0x0f]!;
  }
  return out;
}
