package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// LLMPre is the body of PreLLMHook. It fires after Bifrost has parsed
// the request into its internal BifrostRequest type, so this is the
// first place we can see provider + model resolved (vs the raw URL
// path in TransportPre).
//
// Per gateway/plans/phases/phase-6-plugin-enforcement.md, PreLLMHook
// is the canonical site for verified-claims-aware logic — not
// TransportPre, which fires before the request has been parsed. The
// raw x-macaroon header was stashed on context by TransportPre; we
// pick it up here and run the full verify pipeline.
//
// Shadow-mode failures log loudly and pass through; enforce-mode
// failures short-circuit with a *BifrostError carrying the
// machine-readable AdapterError.Code.
func LLMPre(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	provider, model, _ := req.GetRequestFields()
	dims := pluginctx.Dims(ctx)

	pluginlog.Logf(
		"PreLLMHook provider=%s model=%s request_type=%s run_id=%s agent=%s session_id=%s",
		provider,
		model,
		req.RequestType,
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		dims[pluginctx.DimSessionID],
	)

	// Macaroon adapter — verifies + stamps claims + decides shadow
	// vs enforce. Returns nil short-circuit on pass-through (success
	// or shadow-mode failure); non-nil short-circuit when
	// enforce_macaroons is on and verification failed.
	if shortCircuit := auth.ApplyToLLMPre(ctx); shortCircuit != nil {
		return req, shortCircuit, nil
	}

	// If a macaroon verified successfully, the verified claims are
	// now stamped on context. Make them authoritative over the
	// caller-supplied x-bf-dim-* headers for the four signature-
	// bound dims (run-id, user-id, agent-name, realm-id) — anything
	// the caller stamped that disagreed is overwritten so the
	// metadata that lands in logs.db tells the cryptographic truth.
	//
	// REQUIRES the plugin to run at placement="pre_builtin" in
	// config.json. The built-in logging plugin reads
	// BifrostContextKeyDimensions in its PreLLMHook and snapshots
	// the map into the pending log entry's metadata. If we run
	// post_builtin (the default), the snapshot is already taken by
	// the time we get here and the overwrite is too late — see the
	// rationale in pluginctx/dims.go CanonicalizeFromClaims.
	//
	// In shadow mode with no macaroon (or a verify failure),
	// VerifiedClaims is nil and the caller-stamped headers pass
	// through unchanged — that preserves observability during the
	// rollout phases where some callers haven't onboarded yet.
	if claims := pluginctx.VerifiedClaims(ctx); claims != nil {
		pluginctx.CanonicalizeFromClaims(
			ctx,
			claims.RunID,
			claims.UserID,
			claims.AgentName,
			claims.Realm,
			claims.OrgID,
		)
	}

	return req, nil, nil
}
