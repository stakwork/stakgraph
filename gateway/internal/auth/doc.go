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
//   - config.go       enforce_macaroons + enforce_budgets flags (shadow →
//     enforce rollout, each with an env override), agent_budgets,
//     model_pricing
//   - verifier.go     Verify() — header extraction + trust lookup + pure verify
//   - revocation.go   CheckRevocations() — phase-6 PIPELINE 1:
//     bifrost:revoke:* / revoke_user_before:* (401) and the
//     kill:<run_id> (every chain layer) / kill:agent:<name> switches (402)
//   - kill.go         KillRun/KillAgent + Unkill*, GetRunState/GetAgentState —
//     the admin primitives behind /_plugin/{runs,agents}/:id/{kill,state}
//   - capwalk.go      CheckCaps() — phase-6 PIPELINE 2: per-run cost/steps
//     for every chain layer, UA envelope, realm cap, agent bucket (402s;
//     gated by enforce_budgets)
//   - ttl.go          clamp(exp-now+1h, 1h, 7d) shared by revocation + accumulators
//   - enforcement.go  Evaluate() + ApplyToLLMPre() — hook glue
//   - accumulator.go  ApplyToLLMPost() — phase-6 PostLLMHook pipeline:
//     cost:run / steps:run per chain layer, cost:ua envelope,
//     cost:agent windowed buckets, tools:run history
//   - pricing.go      PriceCall() — model_pricing table → dollars
//   - admin.go        revoke primitives behind /_plugin/revoke/*
//
// What's still out of scope (phase 6)
// -----------------------------------
//   - Tool-loop detection (reads tools:run)
//   - hard_ceiling defense-in-depth + the user_id == customer_id cross-check
//   - tool_loop config + the /_plugin/config/* overrides
//
// Rollout order was deliberate: accumulators first (shadow-safe,
// they reject nothing), then kill switches (no dependency on
// accumulated state), then the cap walk behind its own
// enforce_budgets flag — it needs real accumulated spend to validate
// against before it starts rejecting, and a swarm that already
// enforces macaroons must be able to watch "budget shadow: would
// reject" lines before flipping it.
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
// With enforce_macaroons=true the failure path becomes 401 (bad,
// missing or revoked macaroon) or 402 (valid macaroon, but the run
// or agent was killed) with a stable AdapterError.Code. Spend caps
// are a second knob: enforce_budgets=true (only effective alongside
// enforce_macaroons) turns the cap walk's "budget shadow" log lines
// into 402s. Operators flip the flags per-swarm once
// the shadow-mode logs show no false positives — either in the
// config.json plugin block or, without rebuilding the image, via the
// BIFROST_PLUGIN_ENFORCE_MACAROONS env var (which wins when set; an
// unparseable value is logged at ERROR and ignored, because a plugin
// that fails Init leaves bifrost-http serving with no verification at
// all — see Init). See
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
