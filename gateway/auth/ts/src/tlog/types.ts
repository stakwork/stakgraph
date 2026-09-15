/**
 * Transparency-log wire shapes (phase 12, Part 1). Produced by the
 * gateway plugin (`gateway/internal/tlog`, Go); verified here.
 * Spec: `gateway/plans/phases/phase-12-transparency-log.md`.
 */

/**
 * One accounted LLM call. Every field is always present and nulls are
 * explicit, so `jcs(leaf)` reproduces the exact bytes the gateway
 * hashed. Identity fields come from the verified macaroon, not from
 * caller headers. `request_sha256` is over the raw request body and
 * is `null` only when the gateway never saw the body (bifrost's
 * large-payload mode). `response_sha256` and `agent_request_sig` are
 * reserved `null` in v1.
 */
export interface TlogLeaf {
  v: 1;
  leaf_id: string;
  ts: string;
  org_id: string;
  user_id: string;
  run_id: string;
  agent: string;
  macaroon_sha256: string;
  request_sha256: string | null;
  response_sha256: string | null;
  model: string;
  provider: string;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  status: "ok" | "error";
  agent_request_sig: string | null;
}

/**
 * Signed tree head. `sig` is the gateway's per-boot log key's
 * ECDSA-secp256k1-SHA256 signature over `JCS(sth \ sig)` — the same
 * construction as a macaroon layer signature.
 */
export interface SignedTreeHead {
  tree_size: number;
  root_hash: string;
  signed_at: string;
  sig: string;
}

/** `SignedTreeHead` before signing. */
export type SignedTreeHeadUnsigned = Omit<SignedTreeHead, "sig">;

/**
 * Response body of `GET /_plugin/tlog/sth?since=N`. `leaves` are
 * indices `[since, next)`, capped per page — page with `since = next`
 * until `next === sth.tree_size`. `consistency_proof` runs from
 * `since` to `sth.tree_size` and is empty when `since` is 0 or equals
 * the tree size.
 */
export interface TlogSthResponse {
  sth: SignedTreeHead;
  log_pubkey: string;
  consistency_proof: string[];
  leaves: TlogLeaf[];
  next: number;
}

/** A hash as bytes or lowercase hex. Every verifier accepts either. */
export type HashLike = Uint8Array | string;
