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
	switch r.URL.Path {
	case "/api/logs":
		f.serveLogs(w, r)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
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
		t.Errorf("stats requests: %+v", out.Stats)
	}
	// total_cost is derived from row sums in Go, not Bifrost's
	// stats block. r1 = 0.05 + 0.10 in sampleLogs.
	if out.Stats.TotalCost < 0.149 || out.Stats.TotalCost > 0.151 {
		t.Errorf("stats cost: got %v, want ~0.15", out.Stats.TotalCost)
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
