package auth

import (
	"strings"

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
// Model matching: exact key first, then the name with any
// "provider/" prefix stripped, so "anthropic/claude-sonnet-5" and
// "claude-sonnet-5" resolve to the same entry whichever form the
// caller holds.
func PriceCall(model string, promptTokens, completionTokens int) (float64, bool) {
	if model == "" {
		return 0, false
	}
	const mtok = 1_000_000
	if entry, ok := configPrice(model); ok {
		return float64(promptTokens)*entry.InputPerMTok/mtok +
			float64(completionTokens)*entry.OutputPerMTok/mtok, true
	}
	if p, ok := pricing.Lookup(model); ok {
		return float64(promptTokens)*p.InputPerMTok/mtok +
			float64(completionTokens)*p.OutputPerMTok/mtok, true
	}
	return 0, false
}

func configPrice(model string) (ModelPrice, bool) {
	table := GetConfig().ModelPricing
	if len(table) == 0 {
		return ModelPrice{}, false
	}
	if entry, ok := table[model]; ok {
		return entry, true
	}
	if i := strings.LastIndexByte(model, '/'); i >= 0 {
		if entry, ok := table[model[i+1:]]; ok {
			return entry, true
		}
	}
	return ModelPrice{}, false
}
