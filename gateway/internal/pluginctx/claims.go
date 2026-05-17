package pluginctx

import (
	"github.com/maximhq/bifrost/core/schemas"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// Macaroon-related context keys. Like keyDimensions / keyStartTime
// these are private; callers go through the typed Set/Get helpers
// below so the storage shape stays under our control.
const (
	keyRawMacaroon    schemas.BifrostContextKey = "stakgraph-gateway/raw-macaroon"
	keyVerifiedClaims schemas.BifrostContextKey = "stakgraph-gateway/verified-claims"
	keyLeafRunID      schemas.BifrostContextKey = "stakgraph-gateway/leaf-run-id"
	keyLeafAgent      schemas.BifrostContextKey = "stakgraph-gateway/leaf-agent"
)

// SetRawMacaroon stashes the raw x-macaroon header value extracted
// in HTTPTransportPreHook so PreLLMHook can pick it up without
// re-parsing headers. Bifrost's per-request HTTPRequest is not
// available at PreLLMHook time, so this is the canonical handoff.
//
// Empty string is allowed and means "no macaroon header present" —
// downstream hooks treat it as "missing" rather than nil-deref.
func SetRawMacaroon(ctx *schemas.BifrostContext, raw string) {
	ctx.SetValue(keyRawMacaroon, raw)
}

// RawMacaroon returns the value stashed by SetRawMacaroon, or "" if
// the transport hook didn't see one (or didn't fire — SDK mode).
func RawMacaroon(ctx *schemas.BifrostContext) string {
	if v, ok := ctx.Value(keyRawMacaroon).(string); ok {
		return v
	}
	return ""
}

// SetVerifiedClaims stamps the result of a successful macaroon
// verification on the context. PostLLMHook reads these to drive
// cost-accumulation writes (phase 6) and structured logging.
func SetVerifiedClaims(ctx *schemas.BifrostContext, claims *macaroon.Claims) {
	if claims == nil {
		return
	}
	ctx.SetValue(keyVerifiedClaims, claims)
	ctx.SetValue(keyLeafRunID, claims.RunID)
	ctx.SetValue(keyLeafAgent, claims.AgentName)
}

// VerifiedClaims returns the stashed *macaroon.Claims, or nil when
// no macaroon was verified (observability mode, shadow-mode reject,
// missing header, etc.). Callers MUST nil-check.
func VerifiedClaims(ctx *schemas.BifrostContext) *macaroon.Claims {
	v := ctx.Value(keyVerifiedClaims)
	if v == nil {
		return nil
	}
	claims, _ := v.(*macaroon.Claims)
	return claims
}

// LeafRunID returns the innermost run_id from the verified macaroon
// chain (equivalent to claims.RunID), or "" if no claims are
// available. Useful as a shorter spelling in hot paths.
func LeafRunID(ctx *schemas.BifrostContext) string {
	if v, ok := ctx.Value(keyLeafRunID).(string); ok {
		return v
	}
	return ""
}

// LeafAgent returns the most-specific agent name from the verified
// macaroon chain (the last entry of effective_caveats.agents), or ""
// if no claims are available.
func LeafAgent(ctx *schemas.BifrostContext) string {
	if v, ok := ctx.Value(keyLeafAgent).(string); ok {
		return v
	}
	return ""
}
