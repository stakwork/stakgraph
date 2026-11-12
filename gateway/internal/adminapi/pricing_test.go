package adminapi

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// newPricingServer serves the real routing table with the runtime
// price table on a temp file.
func newPricingServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pricing.json")
	auth.LoadRuntimePricing(path)
	t.Cleanup(func() { auth.ResetRuntimePricingForTest("") })

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{adminUser: "admin", adminPass: "secret", provisioningToken: testToken})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, path
}

func putRaw(t *testing.T, srv *httptest.Server, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPut, srv.URL+pricingPath, bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+testToken)
	req.Header.Set("Content-Type", "application/json")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestPricing_RequiresBearer(t *testing.T) {
	srv, _ := newPricingServer(t)
	for _, method := range []string{http.MethodGet, http.MethodPut} {
		resp := do(t, srv, method, pricingPath, map[string]any{}, false)
		resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("%s without bearer = %d, want 401", method, resp.StatusCode)
		}
	}
}

func TestPricing_PutThenGet(t *testing.T) {
	srv, path := newPricingServer(t)

	// Empty table reads as {}.
	resp := do(t, srv, http.MethodGet, pricingPath, nil, true)
	var got map[string]auth.ModelPrice
	decode(t, resp, &got)
	if resp.StatusCode != http.StatusOK || got == nil || len(got) != 0 {
		t.Fatalf("GET empty = %d %#v, want 200 {}", resp.StatusCode, got)
	}

	body := map[string]any{
		"maple/gemma4-31b": map[string]any{"input_per_mtok": 0.5, "output_per_mtok": 1.5},
		"numes/qwen3-free": map[string]any{"input_per_mtok": 0, "output_per_mtok": 0},
		"acme/llama-4": map[string]any{
			"input_per_mtok": 1, "output_per_mtok": 2, "cache_read_per_mtok": 0.1, "cache_write_per_mtok": 1.25,
		},
	}
	resp = do(t, srv, http.MethodPut, pricingPath, body, true)
	got = nil // json merges into a non-nil map
	decode(t, resp, &got)
	if resp.StatusCode != http.StatusOK || len(got) != 3 {
		t.Fatalf("PUT = %d %v, want 200 and the 3 rows back", resp.StatusCode, got)
	}

	resp = do(t, srv, http.MethodGet, pricingPath, nil, true)
	got = nil // json merges into a non-nil map
	decode(t, resp, &got)
	if got["acme/llama-4"].CacheWritePerMTok != 1.25 || got["maple/gemma4-31b"].OutputPerMTok != 1.5 {
		t.Fatalf("GET after PUT = %v", got)
	}
	if _, ok := got["numes/qwen3-free"]; !ok {
		t.Fatal("a free (all-zero) row must be kept")
	}

	// Survives a restart: the next boot loads the same file.
	auth.ResetRuntimePricingForTest("")
	auth.LoadRuntimePricing(path)
	resp = do(t, srv, http.MethodGet, pricingPath, nil, true)
	got = nil // json merges into a non-nil map
	decode(t, resp, &got)
	if len(got) != 3 {
		t.Fatalf("GET after reload = %v, want the 3 rows", got)
	}

	// {} clears it.
	resp = do(t, srv, http.MethodPut, pricingPath, map[string]any{}, true)
	got = nil // json merges into a non-nil map
	decode(t, resp, &got)
	if resp.StatusCode != http.StatusOK || len(got) != 0 {
		t.Fatalf("PUT {} = %d %v, want 200 {}", resp.StatusCode, got)
	}
}

func TestPricing_RejectsBadBodies(t *testing.T) {
	srv, _ := newPricingServer(t)
	if resp := putRaw(t, srv, `{"maple/a": {"input_per_mtok": 1, "output_per_mtok": 2}}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("seed PUT = %d", resp.StatusCode)
	}

	for name, tc := range map[string]struct{ body, code string }{
		"null":          {`null`, "invalid_price"},
		"array":         {`[]`, "invalid_body"},
		"typo in field": {`{"maple/a": {"input_per_mtoken": 1}}`, "invalid_body"},
		"negative rate": {`{"maple/a": {"input_per_mtok": -1}}`, "invalid_price"},
		"blank key":     {`{"": {"input_per_mtok": 1}}`, "invalid_price"},
		"key w/ space":  {`{"maple/gemma 4": {"input_per_mtok": 1}}`, "invalid_price"},
		"not json":      {`{"maple/a": `, "invalid_body"},
		"too large":     {`{"k": "` + strings.Repeat("x", maxPricingBody) + `"}`, "invalid_body"},
	} {
		resp := putRaw(t, srv, tc.body)
		var env struct {
			Error struct{ Code string } `json:"error"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&env)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest || env.Error.Code != tc.code {
			t.Fatalf("%s: %d %q, want 400 %q", name, resp.StatusCode, env.Error.Code, tc.code)
		}
	}

	// None of them touched the table.
	got := auth.RuntimePricing()
	if len(got) != 1 || got["maple/a"].OutputPerMTok != 2 {
		t.Fatalf("table after refused PUTs = %v", got)
	}
}

func TestPricing_MethodNotAllowed(t *testing.T) {
	srv, _ := newPricingServer(t)
	resp := do(t, srv, http.MethodPost, pricingPath, map[string]any{}, true)
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed || resp.Header.Get("Allow") != "GET, PUT" {
		t.Fatalf("POST = %d Allow=%q, want 405 GET, PUT", resp.StatusCode, resp.Header.Get("Allow"))
	}
}

func getAnon(t *testing.T, srv *httptest.Server, path string) (*http.Response, map[string]any) {
	t.Helper()
	resp := do(t, srv, http.MethodGet, path, nil, false)
	defer resp.Body.Close()
	var body map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&body)
	return resp, body
}

func TestPricingModel_Anonymous(t *testing.T) {
	srv, _ := newPricingServer(t)
	auth.SetConfigForTest(auth.Config{})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}); pricing.SetTableForTest(nil) })
	pricing.SetTableForTest(map[string]pricing.Price{
		"xai/grok-4": {InputPerMTok: 3, OutputPerMTok: 15},
	})
	pricing.SetLimitsForTest(map[string]pricing.ModelLimits{"xai/grok-4": {MaxInputTokens: 256000}})

	resp, body := getAnon(t, srv, pricingModelPath+"?provider=xai&model=grok-4")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if resp.Header.Get("Cache-Control") != "max-age=300" {
		t.Fatalf("Cache-Control = %q", resp.Header.Get("Cache-Control"))
	}
	if body["key"] != "xai/grok-4" || body["source"] != "catalog" || body["provider"] != "xai" || body["model"] != "grok-4" {
		t.Fatalf("body = %v", body)
	}
	// Zero cache rates fall back to the effective ones.
	if body["input_per_mtok"] != 3.0 || body["output_per_mtok"] != 15.0 ||
		body["cache_read_per_mtok"] != 3.0 || body["cache_write_per_mtok"] != 3.0 ||
		body["cache_write_1h_per_mtok"] != 3.0 {
		t.Fatalf("rates = %v", body)
	}
	if body["max_input_tokens"] != 256000.0 {
		t.Fatalf("max_input_tokens = %v", body["max_input_tokens"])
	}
	if _, has := body["max_output_tokens"]; has {
		t.Fatal("max_output_tokens must be omitted when unknown")
	}

	resp, body = getAnon(t, srv, pricingModelPath+"?model=nope")
	if resp.StatusCode != http.StatusNotFound || body["error"] != "unpriced" || body["model"] != "nope" {
		t.Fatalf("unpriced = %d %v", resp.StatusCode, body)
	}
	resp, _ = getAnon(t, srv, pricingModelPath+"?provider=xai")
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("no model = %d, want 400", resp.StatusCode)
	}
}

func TestPricingStatus_Anonymous(t *testing.T) {
	srv, _ := newPricingServer(t)
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{"a": {InputPerMTok: 1}}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}); pricing.SetTableForTest(nil) })
	pricing.SetTableForTest(map[string]pricing.Price{"x": {InputPerMTok: 1}, "y": {InputPerMTok: 1}})

	resp, body := getAnon(t, srv, pricingStatusPath)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	cat, _ := body["catalog"].(map[string]any)
	if cat["loaded"] != true || cat["models"] != 2.0 {
		t.Fatalf("catalog = %v", cat)
	}
	if body["runtime_rows"] != 0.0 || body["config_rows"] != 1.0 {
		t.Fatalf("rows = %v", body)
	}
	// The exact bearer-only path still routes to the old handler.
	r := do(t, srv, http.MethodGet, pricingPath, nil, false)
	r.Body.Close()
	if r.StatusCode != http.StatusUnauthorized {
		t.Fatalf("/_plugin/pricing anon = %d, want 401", r.StatusCode)
	}
}
