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
// here. Which chunk carries the usage depends on the request type —
// see streamCallUsage. MarkAccounted guards the "provider emits
// usage on more than one chunk" case — first accountable chunk
// wins, and a second one is ignored rather than double-counted.
//
// Chat-stream tool-call names arrive as per-chunk deltas that would
// need cross-chunk assembly to reconstruct, so chat streams feed no
// tools:run history. Responses streams do: the terminal event
// carries the fully assembled output items, function_call included.
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

	call := streamCallUsage(chunk)
	if claims == nil || call.usage == nil || call.usage.TotalTokens == 0 {
		return chunk, nil
	}
	if !pluginctx.MarkAccounted(ctx) {
		return chunk, nil
	}

	dims := pluginctx.Dims(ctx)
	costUSD := resolveCost(call, dims)
	auth.ApplyToLLMPost(claims, costUSD, call.tools)
	pluginlog.Logf(
		"StreamChunk accounted run_id=%s agent=%s prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%.6f",
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		call.usage.PromptTokens,
		call.usage.CompletionTokens,
		call.usage.TotalTokens,
		costUSD,
	)
	return chunk, nil
}

// streamCallUsage returns the accountable usage a chunk carries, or
// the zero value (nil usage) for every chunk that must not be
// billed. The rule differs per stream shape:
//
//   - chat_completion_stream: Bifrost normalizes providers to the
//     OpenAI stream shape, where usage arrives once on the final
//     chunk (include_usage is on by default). Any usage-bearing
//     chunk is therefore the final one.
//   - responses_stream: usage rides on the nested Response object of
//     the terminal event (response.completed / .incomplete /
//     .failed). It is NOT safe to take the first usage-bearing
//     event: the Anthropic provider forwards message_start's
//     input-only usage on response.created, so a first-wins rule
//     would bill the prompt and drop the completion tokens.
func streamCallUsage(chunk *schemas.BifrostStreamChunk) callUsage {
	switch {
	case chunk.BifrostChatResponse != nil:
		chat := chunk.BifrostChatResponse
		return callUsage{
			usage:    chat.Usage,
			model:    resolvedModel(chat.Model, &chat.ExtraFields),
			provider: extraFieldsProvider(&chat.ExtraFields),
		}
	case chunk.BifrostResponsesStreamResponse != nil:
		sr := chunk.BifrostResponsesStreamResponse
		if sr.Response == nil || !isTerminalResponsesEvent(sr.Type) {
			return callUsage{}
		}
		// Routing lives on the chunk; fall back to the nested
		// response's block only when the chunk carries none.
		ef := &sr.ExtraFields
		if extraFieldsProvider(ef) == "" && ef.ResolvedModelUsed == "" {
			ef = &sr.Response.ExtraFields
		}
		return responsesCallUsage(sr.Response, ef)
	}
	return callUsage{}
}

// isTerminalResponsesEvent reports whether a Responses stream event
// closes the response and so carries its final usage.
func isTerminalResponsesEvent(t schemas.ResponsesStreamResponseType) bool {
	switch t {
	case schemas.ResponsesStreamResponseTypeCompleted,
		schemas.ResponsesStreamResponseTypeIncomplete,
		schemas.ResponsesStreamResponseTypeFailed:
		return true
	}
	return false
}
