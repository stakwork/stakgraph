package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// TransportPost is the body of HTTPTransportPostHook. It fires after
// the upstream provider call has returned, for NON-STREAMING responses.
// (Streaming responses go through StreamChunk instead — PostHook does
// not fire for them.)
//
// This is where cost accounting + budget decrement will eventually
// live, once macaroon-verified identity is available in ctx.
func TransportPost(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	elapsed := pluginctx.Elapsed(ctx)
	dims := pluginctx.Dims(ctx)

	pluginlog.Logf(
		"HTTPTransportPostHook path=%s status=%d body_bytes=%d run_id=%s agent=%s elapsed_ms=%d",
		req.Path,
		resp.StatusCode,
		len(resp.Body),
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		elapsed.Milliseconds(),
	)
	return nil
}
