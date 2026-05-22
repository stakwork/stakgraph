/**
 * Wire types for the three-layer macaroon. Names and shapes match
 * `gateway/plans/phases/phase-4-macaroon-shape.md` and the
 * symmetric-recursive refinement in
 * `gateway/plans/phases/phase-11-symmetric-recursive-authorization.md`.
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

/**
 * Per-realm spending cap inside a `Budget`. Phase 11 adds the
 * `realm_budgets` map so multi-swarm deployments can pin exact
 * per-swarm cumulative caps without relying on the implicit
 * "max_total_usd leaks across swarms" behavior.
 *
 * Only `max_total_usd` is defined today. Per-realm step / rate caps
 * are an open question in phase 11 (see "Open questions" §2) and
 * can be added without breaking compatibility — `RealmBudget` is a
 * struct, not a bare number, precisely to leave that door open.
 */
export interface RealmBudget {
  max_total_usd: number;
}

/**
 * Spending envelope carried by any signed layer. Phase 11 renamed
 * `UserBudget` → `Budget` so the same shape appears on
 * `UserAuthorization`, `Invocation`, and `AttenuationCaveats`.
 * Children narrow against parents using the rules in `verify.ts`'s
 * narrowing functions.
 *
 * All fields are independently optional. Zero / undefined means "no
 * cap on this axis" — consistent with the empty-by-default
 * convention used for `agent_budgets` in phase 6. The verifier
 * checks structural narrowing (child ≤ parent at each axis); the
 * adapter enforces the cumulative caps (`max_total_usd`,
 * `realm_budgets[r].max_total_usd`) against Redis at request time.
 *
 * Layers that omit `budget` produce byte-identical wire bytes to a
 * pre-phase-11 macaroon when the field was present only on the UA —
 * important for the "single-swarm operators don't pay for any of
 * this" promise.
 */
export interface Budget {
  max_total_usd?: number;
  max_per_invocation_usd?: number;
  realm_budgets?: Record<string, RealmBudget>;
}

/**
 * Deprecated alias preserved so external callers can continue to
 * read the budget field by its phase-4 name during the phase-11
 * cutover. New code should use `Budget` directly.
 */
export type UserBudget = Budget;

export interface UserAuthorizationUnsigned {
  user_id: string;
  user_pubkey: Ed25519PubKey;
  /**
   * Permitted agents (phase 11 lifted this from the `permissions`
   * wrapper to top-level). Invocations narrow by picking a subset.
   */
  agents: string[];
  /**
   * Optional. Absent on the wire ⇒ no UA-level budget enforcement.
   * Byte-identical to a pre-budget macaroon when omitted.
   */
  budget?: Budget;
  iat: string;
  exp: string;
  nonce: string;
}

export interface UserAuthorization extends UserAuthorizationUnsigned {
  org_sig: OrgSig;
}

export interface InvocationUnsigned {
  agents: string[];
  run_id: string;
  max_cost_usd: number;
  max_steps: number;
  /**
   * Optional invocation-level budget block. When present, must
   * narrow against the UA's budget (phase 11 symmetric rule).
   */
  budget?: Budget;
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
  /**
   * Optional attenuation-level budget block. Used by parents
   * spawning cross-realm sub-agents to authorize spend on specific
   * swarms with locally-attenuated caps (no Hive round-trip).
   */
  budget?: Budget;
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
 *
 * `budget` carries the layer's narrowed Budget block (or null if no
 * budget appears anywhere in the chain). The plugin reads
 * `budget.realm_budgets` for the per-realm membership + cap check.
 */
export interface EffectiveCaveats {
  agents: string[];
  max_cost_usd: number;
  max_steps: number;
  budget: Budget | null;
  exp: string;
}

export interface Claims {
  org_id: string;
  user_id: string;
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
  ua_budget: Budget | null;
  /**
   * Sorted list of realm-ids the verified chain authorizes spend
   * on. Derived from `effective_caveats.budget.realm_budgets` —
   * sorted keys, or `null` when no `realm_budgets` appears anywhere
   * in the chain (single-swarm deployments). The plugin's
   * realm-membership check uses `effective_caveats.budget.realm_budgets`
   * directly; `permitted_realms` is for logging and observability.
   */
  permitted_realms: string[] | null;
  /** Nonces in order: user_authorization, invocation, attenuations[0..]. */
  nonces: string[];
  /** Invocation iat — used by adapters for revoke_user_before checks. */
  iat: string;
}
