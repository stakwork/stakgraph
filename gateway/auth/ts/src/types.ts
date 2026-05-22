/**
 * Wire types for the three-layer macaroon. Names and shapes match
 * `gateway/plans/phases/phase-4-macaroon-shape.md` exactly.
 *
 * All binary fields (pubkeys, signatures, HMACs, nonces) are lowercase
 * hex strings WITHOUT `0x` prefix. All timestamps are RFC 3339 / ISO
 * 8601 UTC strings.
 */

// ─── signature envelopes ──────────────────────────────────────────────

export type SigAlg =
  | "ed25519"
  | "ecdsa-secp256k1-sha256"
  | "multisig-v1";

export interface Ed25519Sig {
  alg: "ed25519";
  /** 64-byte signature, hex-encoded (128 chars). */
  sig: string;
}

export interface EcdsaSecp256k1Sig {
  alg: "ecdsa-secp256k1-sha256";
  /** Compact r||s, low-s normalized (BIP 62). 64 bytes, hex-encoded (128 chars). */
  sig: string;
}

export interface MultisigV1Sig {
  alg: "multisig-v1";
  sigs: Array<{
    key_index: number;
    alg: Exclude<SigAlg, "multisig-v1">;
    sig: string;
  }>;
}

export type OrgSig = EcdsaSecp256k1Sig | MultisigV1Sig;
export type UserSig = Ed25519Sig;

// ─── pubkeys + policies (trust-registry shape) ────────────────────────

export interface Ed25519PubKey {
  alg: "ed25519";
  /** 32-byte raw pubkey, hex-encoded (64 chars). */
  key: string;
}

export interface EcdsaSecp256k1PubKey {
  alg: "ecdsa-secp256k1-sha256";
  /** 33-byte compressed pubkey, hex-encoded (66 chars). */
  key: string;
}

export type PubKey = Ed25519PubKey | EcdsaSecp256k1PubKey;

export interface SinglePolicy {
  type: "single";
  key: EcdsaSecp256k1PubKey;
}

export interface MultisigPolicy {
  type: "multisig";
  threshold: number;
  keys: EcdsaSecp256k1PubKey[];
}

export type Policy = SinglePolicy | MultisigPolicy;

// ─── macaroon layers ──────────────────────────────────────────────────

export interface UserPermissions {
  realms: string[];
  agents: string[];
}

/**
 * Org-signed spending envelope for a `user_authorization`. Optional;
 * if omitted, no UA-level budget enforcement happens.
 *
 * Both fields are independently optional. Zero means "no cap on this
 * axis" — consistent with the empty-by-default convention used for
 * agent_budgets in phase 6.
 *
 * See `gateway/plans/phases/phase-4-macaroon-shape.md` ("Budget
 * envelope") for the motivating cold-storage flow: org leader signs
 * a UA from cold storage with `max_total_usd: $X`, the employee's
 * hot key signs many invocations under it through the week.
 *
 * The verifier:
 *  - Rejects at signature time if `max_per_invocation_usd > 0` and
 *    the invocation's `max_cost_usd` exceeds it (pure field
 *    comparison; no Redis).
 *  - In phase 6's hot path, the plugin tracks cumulative spend in
 *    Redis key `cost:ua:<ua.nonce>` and rejects when the total
 *    would meet or exceed `max_total_usd`. The pure verifier (this
 *    package) is I/O-free and surfaces the budget on Claims for
 *    the adapter to enforce.
 */
export interface UserBudget {
  max_total_usd: number;
  max_per_invocation_usd: number;
}

export interface UserAuthorizationUnsigned {
  user_id: string;
  user_pubkey: Ed25519PubKey;
  permissions: UserPermissions;
  /**
   * Optional. Absent on the wire ⇒ no UA-level budget enforcement.
   * Byte-identical to a pre-budget macaroon when omitted.
   */
  budget?: UserBudget;
  iat: string;
  exp: string;
  nonce: string;
}

export interface UserAuthorization extends UserAuthorizationUnsigned {
  org_sig: OrgSig;
}

export interface InvocationUnsigned {
  realm: string;
  agents: string[];
  run_id: string;
  max_cost_usd: number;
  max_steps: number;
  iat: string;
  exp: string;
  nonce: string;
}

export interface Invocation extends InvocationUnsigned {
  user_sig: UserSig;
}

export interface AttenuationCaveats {
  agents: string[];
  max_cost_usd: number;
  max_steps: number;
  run_id: string;
  exp: string;
  nonce: string;
}

export interface Attenuation {
  caveats: AttenuationCaveats;
  /** HMAC-SHA256 output, hex-encoded (64 chars). */
  hmac: string;
}

export interface Macaroon {
  v: 1;
  org_id: string;
  user_authorization: UserAuthorization;
  invocation: Invocation;
  attenuations: Attenuation[];
}

// ─── verifier output ──────────────────────────────────────────────────

/**
 * Effective caveats: invocation caveats narrowed by every attenuation in
 * the chain. What the plugin actually enforces.
 */
export interface EffectiveCaveats {
  agents: string[];
  max_cost_usd: number;
  max_steps: number;
  exp: string;
}

export interface Claims {
  org_id: string;
  user_id: string;
  realm: string;
  /** Most-specific agent name (last element of the final `agents` list). */
  agent_name: string;
  /** Run id of the innermost attenuation, or the invocation if no attenuations. */
  run_id: string;
  effective_caveats: EffectiveCaveats;
  /**
   * `user_authorization.nonce` — surfaced separately from the general
   * `nonces` list because phase-6 cumulative-spend tracking
   * (`cost:ua:<ua_nonce>`) needs it by itself, before any attenuation
   * processing.
   */
  ua_nonce: string;
  /**
   * `user_authorization.budget` passed through unchanged. `null` when
   * the UA carried no budget — adapters MUST treat null as "no
   * UA-level cap" rather than substituting defaults; absent budget
   * is a design choice, not missing data.
   */
  ua_budget: UserBudget | null;
  /** Nonces in order: user_authorization, invocation, attenuations[0..]. */
  nonces: string[];
  /** Invocation iat — used by adapters for revoke_user_before checks. */
  iat: string;
}
