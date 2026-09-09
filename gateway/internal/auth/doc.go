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
//   - config.go       enforce_macaroons flag (shadow → enforce rollout),
//     agent_budgets, model_pricing
//   - verifier.go     Verify() — header extraction + trust lookup + pure verify
//   - revocation.go   CheckRevocations() — bifrost:revoke:* / revoke_user_before:*
//   - ttl.go          clamp(exp-now+1h, 1h, 7d) shared by revocation + accumulators
//   - enforcement.go  Evaluate() + ApplyToLLMPre() — hook glue
//   - accumulator.go  ApplyToLLMPost() — phase-6 PostLLMHook pipeline:
//     cost:run / steps:run per chain layer, cost:ua envelope,
//     cost:agent windowed buckets, tools:run history
//   - pricing.go      PriceCall() — model_pricing table → dollars
//   - admin.go        admin endpoints (revoke management — minimal scope)
//
// What's still out of scope (phase 6 read side)
// ---------------------------------------------
//   - Per-run cost/step cap walk in PreLLMHook (reads the
//     accumulators this package now writes)
//   - ua_budget / realm_budget / agent budget 402 rejections
//   - Tool-loop detection (reads tools:run)
//   - Kill switches (kill:<run_id>, kill:agent:<name>) + admin routes
//
// The write side landing first is deliberate: accumulators are
// shadow-safe (they reject nothing), they light up the phase-8
// budget endpoint that currently falls back to logs.db, and the cap
// walk needs weeks of real accumulated state to validate against
// before it starts rejecting.
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
