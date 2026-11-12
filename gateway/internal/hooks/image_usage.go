package hooks

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// imageCallUsage is the accounting view of an image generation, edit
// or variation (Bifrost answers all three with the image generation
// shape). resp is the response as Bifrost's catalog prices it: the
// whole BifrostResponse for a non-streaming call, the completed
// event wrapped in one for a stream. u is the provider's usage, nil
// when it reported none (a model priced per image may not).
//
// The usage is never nil for an image call: the gates downstream
// read a nil usage as "nothing to account", and an image priced per
// image costs money with no tokens at all.
func imageCallUsage(
	resp *schemas.BifrostResponse,
	u *schemas.ImageUsage,
	wireModel string,
	ef *schemas.BifrostResponseExtraFields,
) callUsage {
	return callUsage{
		usage:    imageLLMUsage(u),
		model:    resolvedModel(wireModel, ef),
		provider: extraFieldsProvider(ef),
		image:    resp,
	}
}

// imageLLMUsage maps an ImageUsage onto the chat-shaped usage the log
// line and the transparency log read: input tokens (text, and input
// images where the provider counts them) as prompt, output tokens
// (the generated images) as completion. The provider's cost rides
// along: xAI reports cost_in_usd_ticks, which core normalizes into
// Cost; a copy is normalized here too so a path that skipped it
// still bills what the provider said, without mutating the response.
func imageLLMUsage(u *schemas.ImageUsage) *schemas.BifrostLLMUsage {
	if u == nil {
		return &schemas.BifrostLLMUsage{}
	}
	c := u.DeepCopy()
	c.NormalizeProviderCost()
	usage := &schemas.BifrostLLMUsage{
		PromptTokens:     c.InputTokens,
		CompletionTokens: c.OutputTokens,
		TotalTokens:      c.TotalTokens,
		Cost:             c.Cost,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	return usage
}

// imageCost prices an image call. The per-token tables resolveCost
// reads (model_pricing, the runtime table, the plugin's copy of the
// datasheet) know text rates only, so an image call never goes
// through them. Precedence:
//
//  1. The provider's own cost (xAI's grok-imagine-image: $0.02 an
//     image, reported on every call).
//  2. Bifrost's model catalog, through ctx.CalculateCost: the same
//     datasheet the text path prices from, read by the code that
//     fills the cost column of Bifrost's own logs — text and image
//     tokens at their own rates (gpt-image-1-mini), else per pixel,
//     else per image by size and quality (dall-e-3), plus any
//     pricing override set in Bifrost.
//  3. $0, with a loud log: an image is never free, and an unpriced
//     one must show in `docker logs`, not slip past the run's cap.
func imageCost(ctx *schemas.BifrostContext, call callUsage, dims map[string]string) float64 {
	if u := call.usage; u != nil && u.Cost != nil && u.Cost.TotalCost > 0 {
		return u.Cost.TotalCost
	}
	if ctx != nil {
		if cost := ctx.CalculateCost(call.image); cost > 0 {
			return cost
		}
	}
	tokens := 0
	if call.usage != nil {
		tokens = call.usage.TotalTokens
	}
	pluginlog.Warnf(
		"accounting: no price for image call provider=%q model=%q (run_id=%s, %d tokens) — accumulated as $0; the provider reported no cost and Bifrost's model catalog has no image rate for it",
		call.provider, call.model, dims[pluginctx.DimRunID], tokens,
	)
	return 0
}

// isImageStreamCompleted reports whether an image stream event closes
// an image: the event that carries the call's usage (OpenAI sends it
// once, after the last image) and the one an image stream is
// accounted on.
func isImageStreamCompleted(t schemas.ImageEventType) bool {
	return t == schemas.ImageGenerationEventTypeCompleted || t == schemas.ImageEditEventTypeCompleted
}
