package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// LLMPre is the body of PreLLMHook. It fires after Bifrost has parsed
// the request into its internal BifrostRequest type, so this is the
// first place we can see provider + model resolved (vs the raw URL
// path in TransportPre).
//
// Dimensions stashed in TransportPre are read back via
// pluginctx.Dims — at this stage Bifrost has also populated its own
// BifrostContextKeyDimensions, but using our copy keeps the plugin's
// auth path independent of Bifrost's internal context lifecycle (so
// the same logic works for SDK-mode users too, when we get there).
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
	return req, nil, nil
}
