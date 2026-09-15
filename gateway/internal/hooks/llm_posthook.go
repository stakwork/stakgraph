package hooks

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
	"github.com/stakwork/stakgraph/gateway/internal/tlog"
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
//
// Phase-12 transparency log: the same two sites, under the same
// gate, append one leaf per call (appendTlogLeaf). Errored calls get
// a leaf with status "error".
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

	// Pull usage/cost from whichever response shape the provider
	// populated (Chat or Responses — see callUsageOf).
	call := callUsageOf(resp)
	var (
		tokens      pricing.Usage
		totalTokens int
	)
	if call.usage != nil {
		tokens = tokensOf(call.usage)
		totalTokens = call.usage.TotalTokens
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
				appendTlogLeaf(ctx, claims, call, 0, tlog.StatusError)
			}
		case hadResp && !isStreamRequest(resp):
			if pluginctx.MarkAccounted(ctx) {
				costUSD = resolveCost(call, dims)
				auth.ApplyToLLMPost(claims, costUSD, call.tools)
				appendTlogLeaf(ctx, claims, call, costUSD, tlog.StatusOK)
			}
		}
	}

	pluginlog.Logf(
		"PostLLMHook run_id=%s agent=%s had_resp=%t had_err=%t prompt_tokens=%d completion_tokens=%d total_tokens=%d cache_read=%d cache_write=%d cost_usd=%.6f elapsed_ms=%d",
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimAgentName],
		hadResp,
		hadErr,
		tokens.Prompt,
		tokens.Completion,
		totalTokens,
		tokens.CacheRead,
		tokens.CacheWrite,
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

// callUsage is the accounting view of one completed call, normalized
// across the Bifrost response shapes that can carry billable usage:
//
//   - Chat (RequestType "chat_completion[_stream]"): the OpenAI
//     /chat/completions shape, usage as BifrostLLMUsage.
//   - Responses (RequestType "responses[_stream]"): the OpenAI
//     /responses shape, usage as ResponsesResponseUsage. Bifrost core
//     v1.6 routes the Anthropic-native /anthropic/v1/messages
//     integration through this type, so every Claude call made with
//     the Anthropic SDK lands here — with ChatResponse nil.
//
// Everything downstream (resolveCost, the accumulator, the log line)
// reads this struct and never touches the raw shapes again.
type callUsage struct {
	usage    *schemas.BifrostLLMUsage
	model    string // resolved model, falling back to the wire model
	provider string // Bifrost provider id that served the call
	tools    []string
}

// callUsageOf picks whichever shape a non-streaming BifrostResponse
// carries. Zero value (nil usage) for nil, for a shape without
// usage, or for the streaming set-up firing.
func callUsageOf(resp *schemas.BifrostResponse) callUsage {
	switch {
	case resp == nil:
		return callUsage{}
	case resp.ChatResponse != nil:
		return chatCallUsage(resp.ChatResponse)
	case resp.ResponsesResponse != nil:
		return responsesCallUsage(resp.ResponsesResponse, &resp.ResponsesResponse.ExtraFields)
	}
	return callUsage{}
}

func chatCallUsage(chat *schemas.BifrostChatResponse) callUsage {
	return callUsage{
		usage:    chat.Usage,
		model:    resolvedModel(chat.Model, &chat.ExtraFields),
		provider: extraFieldsProvider(&chat.ExtraFields),
		tools:    toolCallNames(chat),
	}
}

// responsesCallUsage builds the view of a Responses-shaped result. ef
// is the extra-fields block that carries routing for this call: the
// response's own for a non-streaming result, the chunk's for a
// stream terminal event (core stamps routing on the chunk, and the
// nested Response.ExtraFields is typically blank there).
func responsesCallUsage(r *schemas.BifrostResponsesResponse, ef *schemas.BifrostResponseExtraFields) callUsage {
	return callUsage{
		usage:    responsesUsage(r.Usage),
		model:    resolvedModel(r.Model, ef),
		provider: extraFieldsProvider(ef),
		tools:    responsesToolCallNames(r.Output),
	}
}

// responsesUsage maps ResponsesResponseUsage onto the chat-shaped
// BifrostLLMUsage the pricing path consumes, through core's own
// converter: input→prompt, output→completion, the cached-token
// breakdown (input_tokens_details→prompt_tokens_details), and the
// provider-computed Cost verbatim. Bifrost folds cached tokens into
// InputTokens (same convention as PromptTokens on the chat shape);
// the details block is what lets resolveCost price them at their own
// rates instead of the full input rate — a hand-rolled mapping that
// dropped it is exactly how cache reads came to bill at 10×.
// TotalTokens is recomputed only when the provider left it zero.
func responsesUsage(u *schemas.ResponsesResponseUsage) *schemas.BifrostLLMUsage {
	if u == nil {
		return nil
	}
	usage := u.ToBifrostLLMUsage()
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return usage
}

// tokensOf flattens a BifrostLLMUsage into the breakdown the pricing
// formula consumes. Core folds cache reads and writes into
// PromptTokens and reports the split on PromptTokensDetails — the
// Anthropic provider fills it from cache_read / cache_creation (with
// the 5m/1h TTL split), OpenAI-compatible ones from cached_tokens. A
// usage without the details block is all fresh prompt.
func tokensOf(u *schemas.BifrostLLMUsage) pricing.Usage {
	t := pricing.Usage{Prompt: u.PromptTokens, Completion: u.CompletionTokens}
	if d := u.PromptTokensDetails; d != nil {
		t.CacheRead = d.CachedReadTokens
		t.CacheWrite = d.CachedWriteTokens
		if d.CachedWriteTokenDetails != nil {
			t.CacheWrite1h = d.CachedWriteTokenDetails.CachedWriteTokens1h
		}
	}
	return t
}

// resolveCost turns a call's usage into dollars. Precedence per the
// phase-6 accumulator design:
//
//  1. Provider-computed Usage.Cost.TotalCost (only some providers).
//  2. auth.PriceCall on the (provider, resolved model) pair —
//     operator model_pricing config first, then the internal/pricing
//     catalog (bifrost's own datasheet, refreshed daily). The
//     provider matters: the datasheet keys some providers' rows as
//     "<provider>/<model>" ("xai/grok-4") while Bifrost reports the
//     wire model bare ("grok-4"). The call's cached-token breakdown
//     goes along (tokensOf) so cache reads and writes price at their
//     own rates.
//  3. $0, with a loud log — an unpriced model must be visible in
//     `docker logs`, not silently guessed at.
func resolveCost(call callUsage, dims map[string]string) float64 {
	usage := call.usage
	if usage == nil {
		return 0
	}
	if usage.Cost != nil && usage.Cost.TotalCost > 0 {
		return usage.Cost.TotalCost
	}
	if cost, ok := auth.PriceCall(call.provider, call.model, tokensOf(usage)); ok {
		return cost
	}
	if usage.TotalTokens > 0 {
		pluginlog.Warnf(
			"accounting: no price for provider=%q model=%q (run_id=%s) — %d tokens accumulated as $0; add a model_pricing entry",
			call.provider, call.model, dims[pluginctx.DimRunID], usage.TotalTokens,
		)
	}
	return 0
}

// resolvedModel prefers the model Bifrost actually resolved the
// request to over the wire model the response echoes, so aliases
// price against the real row.
func resolvedModel(wire string, ef *schemas.BifrostResponseExtraFields) string {
	if ef != nil && ef.ResolvedModelUsed != "" {
		return ef.ResolvedModelUsed
	}
	return wire
}

// extraFieldsProvider returns the Bifrost provider id that actually
// served a response. RoutingInfo.Provider is the current field;
// ExtraFields.Provider is its deprecated twin, still populated by
// core v1.6 and kept as the fallback for chunks that only carry the
// old shape.
func extraFieldsProvider(ef *schemas.BifrostResponseExtraFields) string {
	if ef == nil {
		return ""
	}
	if p := ef.RoutingInfo.Provider; p != "" {
		return string(p)
	}
	return string(ef.Provider)
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

// responsesToolCallNames is toolCallNames for the Responses shape,
// where a tool call is an output item of type "function_call" whose
// name rides on the embedded ResponsesToolMessage. Other item kinds
// (message, reasoning, server-side tool calls like web_search_call)
// are not client tool invocations and are skipped. Nil when the
// response called no tools.
func responsesToolCallNames(output []schemas.ResponsesMessage) []string {
	var names []string
	for _, item := range output {
		if item.Type == nil || *item.Type != schemas.ResponsesMessageTypeFunctionCall {
			continue
		}
		if item.ResponsesToolMessage == nil || item.Name == nil || *item.Name == "" {
			continue
		}
		names = append(names, *item.Name)
	}
	return names
}
