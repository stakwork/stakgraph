package adminapi

import (
	"errors"
	"net/http"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
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
