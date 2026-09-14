package auth

import (
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// PriceCall converts one call's token usage into dollars. Returns
// (cost, true) when a price source knows the model, (0, false) when
// none does — the caller decides how loudly to complain about an
// unpriced model.
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
// Whichever source wins, the dollars come from pricing.Price.Cost,
// which prices cached prompt tokens at their own rates. Bifrost
// folds cache reads and writes into the prompt count, so pricing
// that count flat at the input rate — what this did before — billed
// cache-heavy Claude runs several times their real spend.
//
// Model matching follows pricing.Keys: the bare name, then
// "<provider>/<model>", then the name with any "provider/" prefix
// stripped — so "claude-sonnet-5", "anthropic/claude-sonnet-5", and
// ("xai", "grok-4") → "xai/grok-4" all resolve against either source
// whichever form the caller holds. provider is the Bifrost provider
// id that served the call; "" is allowed and skips the second form.
func PriceCall(provider, model string, usage pricing.Usage) (float64, bool) {
	keys := pricing.Keys(provider, model)
	if len(keys) == 0 {
		return 0, false
	}
	if entry, ok := configPrice(keys); ok {
		return entry.price().Cost(usage), true
	}
	if p, ok := pricing.Lookup(provider, model); ok {
		return p.Cost(usage), true
	}
	return 0, false
}

// price converts an operator row into the catalog's Price so both
// sources go through the one cost formula.
func (m ModelPrice) price() pricing.Price {
	return pricing.Price{
		InputPerMTok:        m.InputPerMTok,
		OutputPerMTok:       m.OutputPerMTok,
		CacheReadPerMTok:    m.CacheReadPerMTok,
		CacheWritePerMTok:   m.CacheWritePerMTok,
		CacheWrite1hPerMTok: m.CacheWrite1hPerMTok,
	}
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
