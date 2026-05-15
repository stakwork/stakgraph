// Package auth will hold the gateway's cryptographic auth path:
// macaroon parsing, caveat enforcement, trust-registry lookup.
//
// Status
// ------
// Stub. Reserved import path so the first auth PR lands here.
//
// Planned scope
// -------------
//   - Macaroon parsing (root + chained delegations).
//   - HMAC verification against the trust registry's root pubkey for
//     the issuer org.
//   - Caveat enforcement: max_invocation_cost_usd, allowed_workspaces,
//     allowed_agents, expiry, etc. Each caveat type gets its own file
//     here.
//   - Trust-registry sync: bootstrapped from config, refreshed
//     periodically via /_plugin/trust/refresh. See
//     gateway/plans/cryptographic-identity.md.
//   - VerifiedClaims struct stashed on pluginctx for downstream hooks
//     and ratelimit lookups.
//
// Until the first feature lands this package exports nothing.
package auth
