package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// StreamChunk is the body of HTTPTransportStreamChunkHook. It fires
// once per streamed chunk. PostLLMHook does NOT carry usage for
// streaming responses, so phase-6 cost accounting for streams lives
// here: Bifrost normalizes providers to the OpenAI stream shape,
// where usage arrives once on the final chunk (include_usage is on
// by default). MarkAccounted guards the "provider emits usage on
// more than one chunk" case — first usage-bearing chunk wins, and a
// second one is ignored rather than double-counted.
//
// Stream tool-call names arrive as per-chunk deltas that would need
// cross-chunk assembly to reconstruct; the tools:run history
// currently comes from non-streaming responses only. Revisit when
// the phase-6 PreLLMHook tool-loop check lands.
//
// Logging policy: one log line per chunk would flood output, so we
// only emit on error chunks and on the accounting chunk.
func StreamChunk(
	ctx *schemas.BifrostContext,
	req *schemas.HTTPRequest,
	chunk *schemas.BifrostStreamChunk,
) (*schemas.BifrostStreamChunk, error) {
	if chunk == nil {
		return chunk, nil
	}
	claims := pluginctx.VerifiedClaims(ctx)

	if chunk.BifrostError != nil {
		dims := pluginctx.Dims(ctx)
		pluginlog.Logf(
			"StreamChunk error path=%s run_id=%s agent=%s err=%v",
			req.Path,
			dims[pluginctx.DimRunID],
			dims[pluginctx.DimAgentName],
			chunk.BifrostError.Error,
		)
		// A stream that died still burned a call slot: count the
		// step with zero cost (any partial usage was never
		// delivered on a final chunk).
		if claims != nil && pluginctx.MarkAccounted(ctx) {
			auth.ApplyToLLMPost(claims, 0, nil)
		}
		return chunk, nil
	}

	chat := chunk.BifrostChatResponse
	if claims == nil || chat == nil || chat.Usage == nil || chat.Usage.TotalTokens == 0 {
		return chunk, nil
	}
	if !pluginctx.MarkAccounted(ctx) {
		return chunk, nil
	}

	dims := pluginctx.Dims(ctx)
	costUSD := resolveCost(chat, chat.Usage, dims)
	auth.ApplyToLLMPost(claims, costUSD, nil)
	pluginlog.Logf(
		"StreamChunk accounted run_id=%s agent=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%.6f",
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		chat.Usage.PromptTokens,
		chat.Usage.CompletionTokens,
		chat.Usage.TotalTokens,
		costUSD,
	)
	return chunk, nil
}
