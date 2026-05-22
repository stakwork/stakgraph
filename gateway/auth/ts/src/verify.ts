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
  Budget,
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
  | "ua_per_invocation_exceeded"
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
    m.user_authorization.budget ?? null,
    m.attenuations,
    now,
  );

  return {
    org_id: m.org_id,
    user_id: m.user_authorization.user_id,
    agent_name: agentName,
    run_id: runId,
    effective_caveats: effective,
    ua_nonce: m.user_authorization.nonce,
    ua_budget: m.user_authorization.budget ?? null,
    permitted_realms: permittedRealms(effective),
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

/**
 * UA→invocation boundary check. This is the one boundary where
 * `agents` narrows (`child ⊆ parent`) — the user grants a set of
 * agents on the UA, and the invocation picks a subset to actually
 * use this run. Subsequent attenuation boundaries extend lineage
 * (`child ⊇ parent`); see `narrowAttenuation`.
 */
function enforceInvocationCaveats(
  inv: Invocation,
  ua: UserAuthorization,
  now: Date,
): void {
  if (inv.agents.length === 0) {
    throw new VerifyError("invocation_violated",
      "invocation.agents must be non-empty");
  }
  for (const a of inv.agents) {
    if (!ua.agents.includes(a)) {
      throw new VerifyError("invocation_violated", `agent ${a} not in user permissions`);
    }
  }
  if (asDate(inv.exp) < now) {
    throw new VerifyError("macaroon_expired", `invocation expired at ${inv.exp}`);
  }
  // Exp narrowing: invocation must not outlive the UA. Signature-
  // time field comparison, no clock dependency.
  if (inv.exp > ua.exp) {
    throw new VerifyError("invocation_violated",
      `invocation exp ${inv.exp} > ua exp ${ua.exp}`);
  }
  // Per-invocation budget cap. Pure signature-time check — the
  // cumulative cap (max_total_usd) is enforced by the adapter via
  // Redis in PreLLMHook, not here.
  if (
    ua.budget &&
    ua.budget.max_per_invocation_usd !== undefined &&
    ua.budget.max_per_invocation_usd > 0 &&
    inv.max_cost_usd > ua.budget.max_per_invocation_usd
  ) {
    throw new VerifyError(
      "ua_per_invocation_exceeded",
      `invocation max_cost_usd ${inv.max_cost_usd} > ua.budget.max_per_invocation_usd ${ua.budget.max_per_invocation_usd}`,
    );
  }
  // Budget narrowing: phase 11's symmetric rule applies between the
  // UA's Budget and the invocation's Budget block. The invocation
  // block, when present, must not widen any axis the UA constrained.
  narrowBudget(ua.budget ?? null, inv.budget ?? null);
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
  uaBudget: Budget | null,
  atts: Attenuation[],
  now: Date,
): WalkResult {
  let prevSigBytes = hexToBytes(inv.user_sig.sig);
  // Effective Budget at the invocation layer = the invocation's own
  // block when set, else inherit the UA's. This mirrors the "Mixed
  // mode" rule in phase-11: parent has budget, child omits → child
  // inherits unchanged. Without this, realm_budgets set only at the
  // UA wouldn't propagate to the membership check.
  let effective: EffectiveCaveats = {
    agents: [...inv.agents],
    max_cost_usd: inv.max_cost_usd,
    max_steps: inv.max_steps,
    budget: mergeAttenuationBudget(uaBudget, inv.budget ?? null),
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
    narrowAttenuation(effective, att.caveats);
    if (asDate(att.caveats.exp) < now) {
      throw new VerifyError("macaroon_expired", `attenuation expired at ${att.caveats.exp}`);
    }
    effective = {
      agents: att.caveats.agents,
      max_cost_usd: att.caveats.max_cost_usd,
      max_steps: att.caveats.max_steps,
      budget: mergeAttenuationBudget(effective.budget, att.caveats.budget ?? null),
      exp: att.caveats.exp,
    };
    runId = att.caveats.run_id;
    nonces.push(att.caveats.nonce);
    prevSigBytes = hexToBytes(att.hmac);
  }

  const agentName = effective.agents[effective.agents.length - 1] ?? "";
  return { effective, runId, agentName, nonces };
}

/**
 * Parent→child check at every attenuation boundary. Agents is the
 * lineage-extension axis (child ⊇ parent); every other axis is
 * shrink-only (child ≤ parent). This is the half of phase 11's
 * symmetric rule that differs from the UA→invocation boundary
 * handled in `enforceInvocationCaveats`.
 */
function narrowAttenuation(parent: EffectiveCaveats, child: AttenuationCaveats): void {
  // agents: child must include every parent entry (child ⊇ parent).
  // The last entry remains "the most-specific agent" for billing.
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
  narrowBudget(parent.budget, child.budget ?? null);
}

/**
 * Symmetric budget-narrowing rule between any parent→child layer
 * boundary (UA→invocation, invocation→attenuation, attenuation→
 * attenuation). Throws `attenuation_widened` when the child widens
 * any axis. null child means "inherits parent unchanged"; null
 * parent + non-null child means "child introduces a constraint that
 * didn't exist" — that's narrowing, allowed.
 *
 * Realm-budgets narrowing (rule 4 in phase-11): for each realm-id
 * key in `child.realm_budgets`, the same key must exist in
 * `parent.realm_budgets` (if parent set realm_budgets at all), and
 * the child's per-realm cap must be ≤ parent's. Child may also OMIT
 * realms the parent permitted — that's narrowing, allowed.
 */
function narrowBudget(parent: Budget | null, child: Budget | null): void {
  if (!child) return;
  if (parent) {
    const ppi = parent.max_per_invocation_usd ?? 0;
    const cpi = child.max_per_invocation_usd ?? 0;
    if (ppi > 0 && cpi > 0 && cpi > ppi) {
      throw new VerifyError("attenuation_widened",
        `child budget.max_per_invocation_usd ${cpi} > parent ${ppi}`);
    }
    const pt = parent.max_total_usd ?? 0;
    const ct = child.max_total_usd ?? 0;
    if (pt > 0 && ct > 0 && ct > pt) {
      throw new VerifyError("attenuation_widened",
        `child budget.max_total_usd ${ct} > parent ${pt}`);
    }
  }
  const childRealms = child.realm_budgets;
  if (!childRealms || Object.keys(childRealms).length === 0) return;
  // If parent set realm_budgets, every child key must appear in it
  // and not widen its cap. If parent did NOT set realm_budgets, the
  // child is introducing per-realm scoping that didn't exist upstream
  // — that's narrowing, allowed.
  const parentRealms = parent?.realm_budgets;
  if (!parentRealms || Object.keys(parentRealms).length === 0) return;
  for (const [r, cb] of Object.entries(childRealms)) {
    const pb = parentRealms[r];
    if (!pb) {
      throw new VerifyError("attenuation_widened",
        `child budget.realm_budgets[${r}] not in parent`);
    }
    if (pb.max_total_usd > 0 && cb.max_total_usd > 0 &&
        cb.max_total_usd > pb.max_total_usd) {
      throw new VerifyError("attenuation_widened",
        `child budget.realm_budgets[${r}].max_total_usd ${cb.max_total_usd} > parent ${pb.max_total_usd}`);
    }
  }
}

/**
 * Propagate the effective Budget down the chain. The rule mirrors
 * `narrowBudget`'s intent: a child that omits budget inherits the
 * parent's; a child that sets budget replaces the parent's (already
 * validated as narrowing by `narrowBudget`).
 */
function mergeAttenuationBudget(parent: Budget | null, child: Budget | null): Budget | null {
  if (!child) return parent;
  return child;
}

/**
 * Sorted list of realm-ids the effective caveats authorize spend on.
 * Returns null when no `realm_budgets` appears anywhere in the chain
 * — single-swarm deployments rely on this null-ness to skip the
 * membership check entirely.
 */
function permittedRealms(eff: EffectiveCaveats): string[] | null {
  if (!eff.budget || !eff.budget.realm_budgets) return null;
  const keys = Object.keys(eff.budget.realm_budgets);
  if (keys.length === 0) return null;
  return keys.sort();
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
