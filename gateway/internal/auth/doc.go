// Package auth is the gateway plugin's Bifrost-side adapter for
// macaroon verification. It glues the pure cryptographic verifier in
// gateway/auth/go (no I/O, no Bifrost types) onto the request
// lifecycle: extract the x-macaroon header, look up the issuing org's
// policy in the in-memory trust registry, run macaroon.Verify, layer
// Redis-backed revocation on top, and stamp ctx.VerifiedClaims for
// downstream hooks.
//
// What's in scope here
// --------------------
//   - config.go       enforce_macaroons flag (shadow → enforce rollout)
//   - verifier.go     Verify() — header extraction + trust lookup + pure verify
//   - revocation.go   CheckRevocations() — bifrost:revoke:* / revoke_user_before:*
//   - ttl.go          clamp(exp-now+1h, 1h, 7d) shared with phase-6 accumulators
//   - enforcement.go  Evaluate() + ApplyToLLMPre() — hook glue
//   - admin.go        admin endpoints (revoke management — minimal scope)
//
// What's out of scope (phase 6)
// -----------------------------
//   - Per-run cost cap walk (cost:run:<run_id> with ancestor walking)
//   - Per-UA cumulative spend (cost:ua:<ua_nonce>)
//   - Per-agent windowed budgets (cost:agent:<name>:<bucket_key>)
//   - Step counters / tool-loop detection
//   - Kill switches (kill:<run_id>, kill:agent:<name>)
//   - PostLLMHook accumulator pipelines
//
// All of the above land in a follow-up PR against this same package.
// The Claims surfaced here already carry everything that work needs
// (RunID, UANonce, UABudget, EffectiveCaveats); the missing piece is
// just the Redis pipeline plumbing.
//
// Operational posture
// -------------------
// Shadow vs enforce is the load-bearing rollout knob. With
// enforce_macaroons=false (default) the adapter:
//
//   - Verifies every macaroon end-to-end.
//   - Stamps claims on the context for downstream visibility.
//   - LOGS LOUDLY when a macaroon would have been rejected.
//   - Does NOT reject — the request continues to the provider.
//
// With enforce_macaroons=true the failure path becomes 401/402 with
// a stable AdapterError.Code. Operators flip the flag per-swarm once
// the shadow-mode logs show no false positives. See
// gateway/plans/phases/phase-4-macaroon-shape.md ("Verifier
// algorithm → Bifrost-plugin adapter").
//
// Observability mode
// ------------------
// When redisclient.Client() returns nil (BIFROST_PLUGIN_REDIS_URL
// unset or unreachable at startup), the revocation pipeline is
// skipped. Signature verification still runs. Phase-6 "Failure modes"
// names this state explicitly — auth correctness is preserved without
// Redis; revocation enforcement is the piece that requires it.
package auth
