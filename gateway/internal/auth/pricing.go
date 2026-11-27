package auth

import (
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// PriceCall converts one call's token usage into dollars. Returns
// (cost, true) when a price source knows the model, (0, false) when
// none does — the caller decides how loudly to complain about an
// unpriced model.
//
// Text calls only: an image call is priced in the hook layer
// (hooks.imageCost), since these sources hold per-token text rates.
//
// Source precedence (the hook layer sits one level above this: a
// provider-computed Usage.Cost.TotalCost wins over everything here):
//
//  1. The operator's model_pricing config block — the explicit
//     override baked into the image, and the only source in
//     air-gapped deployments.
//  2. The runtime price table (runtime_pricing.go) — rows pushed
//     through PUT /_plugin/pricing on a running gateway, for models
//     the datasheet lacks: custom providers such as Maple or Numes.
//     It also lets an operator correct a datasheet row without a
//     new image.
//  3. The pricing catalog (internal/pricing) — bifrost's own
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
// whichever form the caller holds; "maple/gemma4-31b" matches the
// ("maple", "gemma4-31b") a custom provider's call reports. provider
// is the Bifrost provider id that served the call; "" is allowed and
// skips the second form.
func PriceCall(provider, model string, usage pricing.Usage) (float64, bool) {
	r, ok := ResolvePrice(provider, model)
	if !ok {
		return 0, false
	}
	return r.Price.Cost(usage), true
}

// Price sources ResolvePrice reports.
const (
	SourceConfig  = "config"
	SourceRuntime = "runtime"
	SourceCatalog = "catalog"
)

// ResolvedPrice is the outcome of ResolvePrice: which source won, the
// key that matched in it, and the rates (as stored, before the
// zero-rate fallbacks — see pricing.Price.Effective).
type ResolvedPrice struct {
	Source string
	Key    string
	Price  pricing.Price
}

// ResolvePrice walks the sources in PriceCall's order (config,
// runtime, catalog) with the same pricing.Keys candidates. PriceCall
// is built on it, so what a read-out reports is what billing uses.
func ResolvePrice(provider, model string) (ResolvedPrice, bool) {
	keys := pricing.Keys(provider, model)
	if len(keys) == 0 {
		return ResolvedPrice{}, false
	}
	if key, entry, ok := configPrice(keys); ok {
		return ResolvedPrice{SourceConfig, key, entry.price()}, true
	}
	if key, entry, ok := runtimePrice(keys); ok {
		return ResolvedPrice{SourceRuntime, key, entry.price()}, true
	}
	if key, p, ok := pricing.LookupKey(provider, model); ok {
		return ResolvedPrice{SourceCatalog, key, p}, true
	}
	return ResolvedPrice{}, false
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

func configPrice(keys []string) (string, ModelPrice, bool) {
	table := GetConfig().ModelPricing
	if len(table) == 0 {
		return "", ModelPrice{}, false
	}
	for _, k := range keys {
		if entry, ok := table[k]; ok {
			return k, entry, true
		}
	}
	return "", ModelPrice{}, false
}

// ConfigPricingRows is the number of rows in the config's model_pricing.
func ConfigPricingRows() int { return len(GetConfig().ModelPricing) }

// RuntimePricingRows is the number of rows in the runtime table.
func RuntimePricingRows() int {
	if m := runtimeTable.Load(); m != nil {
		return len(*m)
	}
	return 0
}
