package auth

import (
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// PriceCall converts token usage into dollars. Returns (cost, true)
// when a price source knows the model, (0, false) when none does —
// the caller decides how loudly to complain about an unpriced model.
//
// Source precedence (the hook layer sits one level above this: a
// provider-computed Usage.Cost.TotalCost wins over everything here):
//
//  1. The operator's model_pricing config block — the explicit
//     override, and the only source in air-gapped deployments.
//  2. The pricing catalog (internal/pricing) — bifrost's own
//     datasheet, fetched at boot and refreshed daily. Same file
//     bifrost prices logs.db rows from, so enforcement dollars and
//     reported dollars agree.
//
// Model matching follows pricing.Keys: the bare name, then
// "<provider>/<model>", then the name with any "provider/" prefix
// stripped — so "claude-sonnet-5", "anthropic/claude-sonnet-5", and
// ("xai", "grok-4") → "xai/grok-4" all resolve against either source
// whichever form the caller holds. provider is the Bifrost provider
// id that served the call; "" is allowed and skips the second form.
func PriceCall(provider, model string, promptTokens, completionTokens int) (float64, bool) {
	keys := pricing.Keys(provider, model)
	if len(keys) == 0 {
		return 0, false
	}
	const mtok = 1_000_000
	if entry, ok := configPrice(keys); ok {
		return float64(promptTokens)*entry.InputPerMTok/mtok +
			float64(completionTokens)*entry.OutputPerMTok/mtok, true
	}
	if p, ok := pricing.Lookup(provider, model); ok {
		return float64(promptTokens)*p.InputPerMTok/mtok +
			float64(completionTokens)*p.OutputPerMTok/mtok, true
	}
	return 0, false
}

func configPrice(keys []string) (ModelPrice, bool) {
	table := GetConfig().ModelPricing
	if len(table) == 0 {
		return ModelPrice{}, false
	}
	for _, k := range keys {
		if entry, ok := table[k]; ok {
			return entry, true
		}
	}
	return ModelPrice{}, false
}
