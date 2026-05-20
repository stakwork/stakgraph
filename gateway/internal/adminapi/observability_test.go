package adminapi

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

// fakeLog is the wire shape our logstoreClient decodes from Bifrost.
// We construct a small fixture by hand here so the tests don't take
// a dependency on bifrost's framework/logstore package (which would
// drag a lot of GORM weight into the test binary).
//
// Body fields (input_history etc.) are present here so the same
// fixture drives both the list endpoint (which trims them via
// `omitempty`) and the by-id endpoint (which returns them).
type fakeLog struct {
	ID         string            `json:"id"`
	Timestamp  string            `json:"timestamp"`
	Provider   string            `json:"provider"`
	Model      string            `json:"model"`
	Status     string            `json:"status"`
	Cost       float64           `json:"cost"`
	Latency    float64           `json:"latency"`
	CustomerID string            `json:"customer_id"`
	Metadata   map[string]string `json:"metadata"`

	InputHistory   json.RawMessage `json:"input_history,omitempty"`
	OutputMessage  json.RawMessage `json:"output_message,omitempty"`
	Params         json.RawMessage `json:"params,omitempty"`
	Tools          json.RawMessage `json:"tools,omitempty"`
	ErrorDetails   json.RawMessage `json:"error_details,omitempty"`
	RawRequest     string          `json:"raw_request,omitempty"`
	RawResponse    string          `json:"raw_response,omitempty"`
	ContentSummary string          `json:"content_summary,omitempty"`

	TokenUsage json.RawMessage `json:"token_usage,omitempty"`
	CacheDebug json.RawMessage `json:"cache_debug,omitempty"`

	StopReason      string `json:"stop_reason,omitempty"`
	Stream          bool   `json:"stream"`
	NumberOfRetries int    `json:"number_of_retries"`
	FallbackIndex   int    `json:"fallback_index"`
}

// fakeBifrost is a small httptest.Server that mimics the slice of
// /api/logs phase-8 actually calls. It validates Basic auth (the
// real Bifrost gates this endpoint behind auth_config) and supports
// metadata_<key>=<value> filtering + pagination so we can drive the
// run-detail tests realistically.
type fakeBifrost struct {
	srv          *httptest.Server
	logs         []fakeLog
	authUser     string
	authPass     string
	requireAuth  bool
	failNextWith int // if non-zero, the next call returns this status
}

func newFakeBifrost(t *testing.T, logs []fakeLog) *fakeBifrost {
	t.Helper()
	f := &fakeBifrost{
		logs:        logs,
		authUser:    "admin",
		authPass:    "hunter2",
		requireAuth: true,
	}
	f.srv = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeBifrost) handle(w http.ResponseWriter, r *http.Request) {
	if f.failNextWith != 0 {
		w.WriteHeader(f.failNextWith)
		fmt.Fprintf(w, `{"error":"forced"}`)
		f.failNextWith = 0
		return
	}
	if f.requireAuth {
		user, pass, ok := r.BasicAuth()
		if !ok || user != f.authUser || pass != f.authPass {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
	}
	switch {
	case r.URL.Path == "/api/logs":
		f.serveLogs(w, r)
	case strings.HasPrefix(r.URL.Path, "/api/logs/"):
		f.serveLogByID(w, r, strings.TrimPrefix(r.URL.Path, "/api/logs/"))
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

// serveLogByID mimics Bifrost's GET /api/logs/{id} — returns the
// full log row (body fields included) or 404.
func (f *fakeBifrost) serveLogByID(w http.ResponseWriter, _ *http.Request, id string) {
	for _, l := range f.logs {
		if l.ID == id {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(l)
			return
		}
	}
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, `{"error":"not found"}`)
}

func (f *fakeBifrost) serveLogs(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	limit := atoiOr(q.Get("limit"), 50)
	offset := atoiOr(q.Get("offset"), 0)

	// Metadata filters: every metadata_<key>=<value> must match.
	mdf := map[string]string{}
	for k, v := range q {
		if strings.HasPrefix(k, "metadata_") {
			mdf[strings.TrimPrefix(k, "metadata_")] = v[0]
		}
	}

	// Filter by metadata.
	var rows []fakeLog
	for _, l := range f.logs {
		ok := true
		for k, want := range mdf {
			if l.Metadata[k] != want {
				ok = false
				break
			}
		}
		if ok {
			rows = append(rows, l)
		}
	}
	total := int64(len(rows))

	// Paginate.
	start := offset
	if start > len(rows) {
		start = len(rows)
	}
	end := start + limit
	if end > len(rows) {
		end = len(rows)
	}
	page := rows[start:end]

	// Compute stats over the *filtered* set (matches Bifrost).
	var totalCost float64
	for _, l := range rows {
		totalCost += l.Cost
	}

	resp := map[string]any{
		"logs": page,
		"pagination": map[string]any{
			"limit":       limit,
			"offset":      offset,
			"total_count": total,
		},
		"stats": map[string]any{
			"total_requests": total,
			"total_cost":     totalCost,
			"total_tokens":   int64(0),
		},
		"has_logs": total > 0,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func atoiOr(s string, fallback int) int {
	if s == "" {
		return fallback
	}
	var n int
	_, _ = fmt.Sscanf(s, "%d", &n)
	if n == 0 {
		return fallback
	}
	return n
}

// newObservabilityTestServer wires the four phase-8 observability
// handlers against a fakeBifrost. Auth is bearer-only here — the
// session/cookie path is exercised in login_test.go, this file is
// about the read paths working end-to-end.
func newObservabilityTestServer(t *testing.T, bifrost *fakeBifrost) *httptest.Server {
	t.Helper()
	c := &logstoreClient{
		base:       bifrost.srv.URL,
		httpClient: &http.Client{Timeout: 2 * time.Second},
		authHeader: basicAuth(bifrost.authUser, bifrost.authPass),
	}
	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         "admin",
		adminPass:         "hunter2",
		provisioningToken: testToken,
		trust:             nil,
		sessions:          nil, // no cookie auth in these tests; bearer is fine
		logstore:          c,
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func bearerGet(t *testing.T, srv *httptest.Server, path string) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, srv.URL+path, nil)
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// ─── fixtures ────────────────────────────────────────────────────────

func sampleLogs(now time.Time) []fakeLog {
	return []fakeLog{
		{
			ID: "1", Timestamp: now.Add(-30 * time.Minute).Format(time.RFC3339Nano),
			Provider: "anthropic", Model: "claude-3-5-haiku",
			Status: "success", Cost: 0.05, Latency: 800, CustomerID: "u_alice",
			Metadata: map[string]string{"agent-name": "coder", "run-id": "r1", "user-id": "u_alice"},
			// Body fields exist on the row but the list endpoint
			// strips them (Bifrost's listSelectColumns). The
			// fakeBifrost above echoes them only on /api/logs/{id}.
			InputHistory:  json.RawMessage(`[{"role":"user","content":"hello"}]`),
			OutputMessage: json.RawMessage(`{"role":"assistant","content":"hi"}`),
			Params:        json.RawMessage(`{"temperature":0.7}`),
			RawResponse:   `{"id":"resp_1"}`,
			TokenUsage:    json.RawMessage(`{"prompt_tokens":20,"completion_tokens":22,"total_tokens":42,"prompt_tokens_details":{"cached_read_tokens":8}}`),
			CacheDebug:    json.RawMessage(`{"cache_hit":false,"similarity":0.42,"threshold":0.85}`),
			StopReason:    "stop",
		},
		{
			ID: "2", Timestamp: now.Add(-25 * time.Minute).Format(time.RFC3339Nano),
			Provider: "anthropic", Model: "claude-3-5-haiku",
			Status: "success", Cost: 0.10, Latency: 1200, CustomerID: "u_alice",
			Metadata: map[string]string{"agent-name": "coder", "run-id": "r1", "user-id": "u_alice"},
		},
		{
			ID: "3", Timestamp: now.Add(-20 * time.Minute).Format(time.RFC3339Nano),
			Provider: "openai", Model: "gpt-4o-mini",
			Status: "success", Cost: 0.02, Latency: 600, CustomerID: "u_bob",
			Metadata: map[string]string{"agent-name": "web-search", "run-id": "r2", "user-id": "u_bob"},
		},
	}
}

// ─── spend.by-agent ──────────────────────────────────────────────────

func TestSpendByAgent(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/spend/by-agent?window=1h")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out SpendByAgentResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Window != "1h" || len(out.Results) != 2 {
		t.Fatalf("unexpected: %+v", out)
	}
	// Sort order: coder (0.15) before web-search (0.02).
	if out.Results[0].AgentName != "coder" || out.Results[0].TotalCost < 0.149 {
		t.Errorf("coder: %+v", out.Results[0])
	}
	if out.Results[1].AgentName != "web-search" {
		t.Errorf("web-search: %+v", out.Results[1])
	}
}

func TestSpendByAgent_BadWindow(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	resp := bearerGet(t, srv, "/_plugin/spend/by-agent?window=99y")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

// ─── spend.by-user ───────────────────────────────────────────────────

func TestSpendByUser(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/spend/by-user?window=1h")
	defer resp.Body.Close()
	var out SpendByUserResponse
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Results) != 2 {
		t.Fatalf("results: %+v", out)
	}
	if out.Results[0].UserID != "u_alice" {
		t.Errorf("expected alice first, got %+v", out.Results[0])
	}
}

// ─── spend.by-agent-user ─────────────────────────────────────────────

func TestSpendByAgentUser(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/spend/by-agent-user?window=1h")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out SpendByAgentUserResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	// Sample data: (coder, u_alice, 0.15), (web-search, u_bob, 0.02).
	if out.Window != "1h" || len(out.Results) != 2 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.Results[0].AgentName != "coder" || out.Results[0].UserID != "u_alice" {
		t.Errorf("first row: %+v", out.Results[0])
	}
	if out.Results[0].TotalCost < 0.149 || out.Results[0].RequestCount != 2 {
		t.Errorf("coder/alice agg: %+v", out.Results[0])
	}
	if out.Results[1].AgentName != "web-search" || out.Results[1].UserID != "u_bob" {
		t.Errorf("second row: %+v", out.Results[1])
	}
}

func TestSpendByAgentUser_FiltersUnattributed(t *testing.T) {
	now := time.Now().UTC()
	logs := []fakeLog{
		// no agent-name → excluded
		{
			ID: "a", Timestamp: now.Add(-10 * time.Minute).Format(time.RFC3339Nano),
			Status: "success", Cost: 1.0,
			Metadata: map[string]string{"user-id": "u_alice"},
		},
		// no user-id → excluded
		{
			ID: "b", Timestamp: now.Add(-10 * time.Minute).Format(time.RFC3339Nano),
			Status: "success", Cost: 1.0,
			Metadata: map[string]string{"agent-name": "coder"},
		},
		// both → included
		{
			ID: "c", Timestamp: now.Add(-10 * time.Minute).Format(time.RFC3339Nano),
			Status: "success", Cost: 0.5,
			Metadata: map[string]string{"agent-name": "coder", "user-id": "u_alice"},
		},
	}
	bf := newFakeBifrost(t, logs)
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/spend/by-agent-user?window=1h")
	defer resp.Body.Close()
	var out SpendByAgentUserResponse
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if len(out.Results) != 1 {
		t.Fatalf("expected 1 row, got %d: %+v", len(out.Results), out.Results)
	}
	if out.Results[0].AgentName != "coder" || out.Results[0].UserID != "u_alice" {
		t.Errorf("row: %+v", out.Results[0])
	}
}

// ─── histogram.cost ──────────────────────────────────────────────────

func TestHistogramCost_ByAgent(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv,
		"/_plugin/histogram/cost?window=1h&bucket=10m&dimension=agent-name")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out HistogramCostResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.BucketSizeSeconds != 600 || out.Dimension != "agent-name" {
		t.Errorf("envelope: %+v", out)
	}
	// Heaviest series first.
	if len(out.Series) == 0 || out.Series[0].DimensionValue != "coder" {
		t.Errorf("series order: %+v", out.Series)
	}
}

func TestHistogramCost_BucketExceedsWindow(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	resp := bearerGet(t, srv,
		"/_plugin/histogram/cost?window=1h&bucket=1d&dimension=agent-name")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestHistogramCost_BadDimension(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	resp := bearerGet(t, srv,
		"/_plugin/histogram/cost?window=1h&bucket=10m&dimension=bogus")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

// ─── runs.detail ─────────────────────────────────────────────────────

func TestRunDetail_FiltersByRunID(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/runs/r1")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out RunDetailResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.RunID != "r1" || len(out.Logs) != 2 {
		t.Fatalf("logs: %+v", out)
	}
	if out.Stats.TotalRequests != 2 {
		t.Errorf("stats: %+v", out.Stats)
	}
}

func TestRunDetail_404OnSubpath(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	resp := bearerGet(t, srv, "/_plugin/runs/r1/state")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404 on /:id/state (phase-6 territory), got %d", resp.StatusCode)
	}
}

// ─── runs.call_detail ────────────────────────────────────────────────

func TestRunCallDetail_ReturnsBodyFields(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/runs/r1/calls/1")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out CallDetailResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.ID != "1" || out.RunID != "r1" {
		t.Fatalf("id/run: %+v", out)
	}
	if string(out.InputHistory) == "" {
		t.Error("input_history missing — handler should pass through Bifrost's parsed body")
	}
	if string(out.OutputMessage) == "" {
		t.Error("output_message missing")
	}
	if string(out.TokenUsage) == "" {
		t.Error("token_usage missing — should pass through provider usage breakdown")
	}
	if string(out.CacheDebug) == "" {
		t.Error("cache_debug missing — should pass through semantic cache record")
	}
	if out.StopReason != "stop" {
		t.Errorf("stop_reason: %s", out.StopReason)
	}
	if out.Provider != "anthropic" {
		t.Errorf("provider: %s", out.Provider)
	}
}

// Cross-run access is rejected: a valid call-id belonging to one
// run must 404 when fetched under a different run's URL. Prevents
// the SPA from being tricked into rendering a call from a run the
// operator didn't ask for.
func TestRunCallDetail_404OnRunMismatch(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	// Call id "3" belongs to run r2.
	resp := bearerGet(t, srv, "/_plugin/runs/r1/calls/3")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404 on cross-run mismatch, got %d", resp.StatusCode)
	}
}

func TestRunCallDetail_404OnUnknownID(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, sampleLogs(now))
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/runs/r1/calls/does-not-exist")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
}

func TestRunCallDetail_404OnEmptyCallID(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	resp := bearerGet(t, srv, "/_plugin/runs/r1/calls/")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404 on empty call id, got %d", resp.StatusCode)
	}
}

// ─── upstream failure mapping ───────────────────────────────────────

func TestUpstreamUnavailable_Maps502(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	bf.failNextWith = http.StatusInternalServerError
	srv := newObservabilityTestServer(t, bf)

	resp := bearerGet(t, srv, "/_plugin/spend/by-agent?window=1h")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("want 502, got %d", resp.StatusCode)
	}
	body := map[string]any{}
	_ = json.NewDecoder(resp.Body).Decode(&body)
	gotErr, ok := body["error"].(map[string]any)
	if !ok || gotErr["code"] != "upstream_unavailable" {
		t.Fatalf("error envelope: %+v", body)
	}
}

// ─── auth ────────────────────────────────────────────────────────────

func TestObservability_RequiresAuth(t *testing.T) {
	bf := newFakeBifrost(t, nil)
	srv := newObservabilityTestServer(t, bf)
	paths := []string{
		"/_plugin/spend/by-agent",
		"/_plugin/spend/by-user",
		"/_plugin/histogram/cost?bucket=1h&dimension=agent-name",
		"/_plugin/runs/r1",
		"/_plugin/runs/r1/calls/1",
	}
	for _, p := range paths {
		resp, _ := http.Get(srv.URL + p)
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s: want 401, got %d", p, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

// silence unused import staticcheck noise for url in some configs
var _ = url.QueryEscape
