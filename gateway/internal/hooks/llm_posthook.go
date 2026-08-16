package hooks

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// LLMPost is the body of PostLLMHook. It fires after the upstream
// provider call (or after a short-circuit). For STREAMING requests
// this fires once when the response is set up — usage arrives on the
// final chunk instead, so streaming accounting lives in StreamChunk
// and this function skips it (see the request-type gate below).
//
// Phase-6 accounting: when a verified macaroon stamped claims on the
// context, every completed call feeds the Redis accumulators
// (cost:run / steps:run / cost:ua / cost:agent / tools:run) through
// auth.ApplyToLLMPost. An errored call accounts cost=0 but still
// counts the step — the run burned a call slot even if the provider
// returned nothing billable.
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
	var usage *schemas.BifrostLLMUsage
	if hadResp && resp.ChatResponse != nil {
		usage = resp.ChatResponse.Usage
	}
	var promptTokens, completionTokens, totalTokens int
	if usage != nil {
		promptTokens = usage.PromptTokens
		completionTokens = usage.CompletionTokens
		totalTokens = usage.TotalTokens
	}

	// Phase-6 accumulator writes. Streaming requests are accounted
	// on their final usage-bearing chunk (StreamChunk), not here —
	// this firing has no usage yet. MarkAccounted keeps the two
	// sites from ever both counting the same call.
	costUSD := 0.0
	if claims := pluginctx.VerifiedClaims(ctx); claims != nil {
		switch {
		case hadErr:
			if pluginctx.MarkAccounted(ctx) {
				auth.ApplyToLLMPost(claims, 0, nil)
			}
		case hadResp && !isStreamRequest(resp):
			if pluginctx.MarkAccounted(ctx) {
				costUSD = resolveCost(resp.ChatResponse, usage, dims)
				auth.ApplyToLLMPost(claims, costUSD, toolCallNames(resp.ChatResponse))
			}
		}
	}

	pluginlog.Logf(
		"PostLLMHook run_id=%s agent=%s had_resp=%t had_err=%t prompt_tokens=%d completion_tokens=%d total_tokens=%d cost_usd=%.6f elapsed_ms=%d",
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		hadResp,
		hadErr,
		promptTokens,
		completionTokens,
		totalTokens,
		costUSD,
		elapsed.Milliseconds(),
	)
	return resp, bifrostErr, nil
}

// isStreamRequest reports whether the response belongs to a
// streaming request type ("chat_completion_stream", etc.), whose
// usage is delivered on the final chunk rather than here.
func isStreamRequest(resp *schemas.BifrostResponse) bool {
	ef := resp.GetExtraFields()
	if ef == nil {
		return false
	}
	return strings.HasSuffix(string(ef.RequestType), "_stream")
}

// resolveCost turns a chat response's usage into dollars. Precedence
// per the phase-6 accumulator design:
//
//  1. Provider-computed Usage.Cost.TotalCost (only some providers).
//  2. auth.PriceCall on the resolved model name — operator
//     model_pricing config first, then the internal/pricing catalog
//     (bifrost's own datasheet, refreshed daily).
//  3. $0, with a loud log — an unpriced model must be visible in
//     `docker logs`, not silently guessed at.
func resolveCost(chat *schemas.BifrostChatResponse, usage *schemas.BifrostLLMUsage, dims map[string]string) float64 {
	if usage == nil {
		return 0
	}
	if usage.Cost != nil && usage.Cost.TotalCost > 0 {
		return usage.Cost.TotalCost
	}
	model := ""
	if chat != nil {
		if model = chat.ExtraFields.ResolvedModelUsed; model == "" {
			model = chat.Model
		}
	}
	if cost, ok := auth.PriceCall(model, usage.PromptTokens, usage.CompletionTokens); ok {
		return cost
	}
	if usage.TotalTokens > 0 {
		pluginlog.Warnf(
			"accounting: no price for model=%q (run_id=%s) — %d tokens accumulated as $0; add a model_pricing entry",
			model, dims[pluginctx.DimRunID], usage.TotalTokens,
		)
	}
	return 0
}

// toolCallNames collects the tool names invoked in a non-streaming
// chat response, feeding the tools:run history that phase-6's
// tool-loop heuristic reads. Nil when the response called no tools.
func toolCallNames(chat *schemas.BifrostChatResponse) []string {
	if chat == nil {
		return nil
	}
	var names []string
	for _, choice := range chat.Choices {
		if choice.ChatNonStreamResponseChoice == nil || choice.Message == nil {
			continue
		}
		if choice.Message.ChatAssistantMessage == nil {
			continue
		}
		for _, tc := range choice.Message.ToolCalls {
			if tc.Function.Name != nil && *tc.Function.Name != "" {
				names = append(names, *tc.Function.Name)
			}
		}
	}
	return names
}
