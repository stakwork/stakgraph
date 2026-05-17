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

	return req, nil, nil
}
