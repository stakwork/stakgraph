package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// LLMPost is the body of PostLLMHook. It fires after the upstream
// provider call (or after a short-circuit). For STREAMING requests
// this fires once when the response is set up; the actual chunks go
// through StreamChunk.
//
// We pull usage tokens here when present. Once macaroon-verified
// identity lands, this is the function that increments the per-(run,
// agent, user) cost counters.
func LLMPost(
	ctx *schemas.BifrostContext,
	resp *schemas.BifrostResponse,
	bifrostErr *schemas.BifrostError,
) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	elapsed := pluginctx.Elapsed(ctx)
	dims := pluginctx.Dims(ctx)

	var (
		hadResp = resp != nil
		hadErr  = bifrostErr != nil
	)

	// Try to pull usage/cost if the provider populated it (Chat responses).
	var promptTokens, completionTokens, totalTokens int
	if hadResp && resp.ChatResponse != nil && resp.ChatResponse.Usage != nil {
		u := resp.ChatResponse.Usage
		promptTokens = u.PromptTokens
		completionTokens = u.CompletionTokens
		totalTokens = u.TotalTokens
	}

	pluginlog.Logf(
		"PostLLMHook run_id=%s agent=%s had_resp=%t had_err=%t prompt_tokens=%d completion_tokens=%d total_tokens=%d elapsed_ms=%d",
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		hadResp,
		hadErr,
		promptTokens,
		completionTokens,
		totalTokens,
		elapsed.Milliseconds(),
	)
	return resp, bifrostErr, nil
}
