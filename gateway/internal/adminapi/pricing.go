package adminapi

import (
	"errors"
	"net/http"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// pricingPath is the runtime price table: GET reads it, PUT replaces
// it. The body of both is the table itself,
//
//	{ "maple/gemma4-31b": { "input_per_mtok": 0.5, "output_per_mtok": 1.5 }, … }
//
// each row the shape of a config model_pricing row (input, output,
// and optional cache_read, cache_write and cache_write_1h rates, in
// dollars per million tokens). PriceCall consults it after the
// config's model_pricing and before bifrost's datasheet; the table is
// kept on the data volume, so it survives restarts. See
// internal/auth/runtime_pricing.go.
const pricingPath = "/_plugin/pricing"

// maxPricingBody bounds a PUT. A row is ~120 bytes, so this is room
// for thousands of models.
const maxPricingBody = 1 << 20

// pricingHandler serves GET and PUT /_plugin/pricing. Bearer-only,
// like the other machine-to-plugin writes: the host that owns the
// model catalog (stadeum) pushes the whole table on each reconcile.
func pricingHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, auth.RuntimePricing())
	case http.MethodPut:
		r.Body = http.MaxBytesReader(w, r.Body, maxPricingBody)
		var rows map[string]auth.ModelPrice
		if err := decodeJSON(r, &rows); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
			return
		}
		if err := auth.SetRuntimePricing(rows); err != nil {
			if errors.Is(err, auth.ErrInvalidPrice) {
				writeError(w, http.StatusBadRequest, "invalid_price", err.Error())
				return
			}
			pluginlog.Errf("adminapi: runtime pricing: %v", err)
			writeError(w, http.StatusInternalServerError, "pricing_store_failed", "could not save the price table")
			return
		}
		pluginlog.Logf("adminapi: runtime price table replaced rows=%d", len(rows))
		writeJSON(w, http.StatusOK, auth.RuntimePricing())
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodPut)
	}
}

const (
	pricingModelPath  = "/_plugin/pricing/model"
	pricingStatusPath = "/_plugin/pricing/status"
)

// pricingModelHandler serves GET /_plugin/pricing/model?provider=&model=:
// the price the gateway would bill the model at, resolved through
// auth.ResolvePrice (the same walk PriceCall bills with). Rates are the
// EFFECTIVE ones, zero-rate cache fallbacks applied, so a client
// multiplies token counts by them with no rules of its own. Anonymous
// and read-only: prices and datasheet limits only.
func pricingModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	provider := r.URL.Query().Get("provider")
	model := r.URL.Query().Get("model")
	if model == "" {
		writeError(w, http.StatusBadRequest, "missing_model", "the model query parameter is required")
		return
	}
	res, ok := auth.ResolvePrice(provider, model)
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]any{
			"error":    "unpriced",
			"provider": provider,
			"model":    model,
		})
		return
	}
	eff := res.Price.Effective()
	body := map[string]any{
		"provider":                provider,
		"model":                   model,
		"key":                     res.Key,
		"source":                  res.Source,
		"input_per_mtok":          eff.InputPerMTok,
		"output_per_mtok":         eff.OutputPerMTok,
		"cache_read_per_mtok":     eff.CacheReadPerMTok,
		"cache_write_per_mtok":    eff.CacheWritePerMTok,
		"cache_write_1h_per_mtok": eff.CacheWrite1hPerMTok,
	}
	if l, ok := pricing.Limits(provider, model); ok {
		if l.MaxInputTokens > 0 {
			body["max_input_tokens"] = l.MaxInputTokens
		}
		if l.MaxOutputTokens > 0 {
			body["max_output_tokens"] = l.MaxOutputTokens
		}
	}
	w.Header().Set("Cache-Control", "max-age=300")
	writeJSON(w, http.StatusOK, body)
}

// pricingStatusHandler serves GET /_plugin/pricing/status: the catalog's
// state and the row counts of the other two price sources.
func pricingStatusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	st := pricing.Status()
	catalog := map[string]any{
		"loaded": st.Loaded,
		"models": st.Models,
		"url":    st.URL,
	}
	if !st.FetchedAt.IsZero() {
		catalog["fetched_at"] = st.FetchedAt.UTC().Format(time.RFC3339)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"catalog":      catalog,
		"runtime_rows": auth.RuntimePricingRows(),
		"config_rows":  auth.ConfigPricingRows(),
	})
}
