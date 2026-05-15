package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// StreamChunk is the body of HTTPTransportStreamChunkHook. It fires
// once per streamed chunk. PostHook does NOT fire for streaming
// responses, so cost accounting on streams will eventually hook in
// here (specifically on the final chunk that carries usage).
//
// Logging policy: one log line per chunk would flood output, so we
// only emit on error chunks. The "interesting boundary" chunks
// (first/last) will eventually deserve their own targeted logging,
// but for now stay quiet to keep `docker logs` readable.
func StreamChunk(
	ctx *schemas.BifrostContext,
	req *schemas.HTTPRequest,
	chunk *schemas.BifrostStreamChunk,
) (*schemas.BifrostStreamChunk, error) {
	if chunk == nil {
		return chunk, nil
	}
	if chunk.BifrostError != nil {
		dims := pluginctx.Dims(ctx)
		pluginlog.Logf(
			"StreamChunk error path=%s run_id=%s agent=%s err=%v",
			req.Path,
			dims[pluginctx.DimRunID],
			dims[pluginctx.DimAgentName],
			chunk.BifrostError.Error,
		)
	}
	return chunk, nil
}
