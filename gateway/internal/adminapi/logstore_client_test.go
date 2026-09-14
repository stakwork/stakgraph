package adminapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"
)

// Unit tests for the loopback client itself (phase-7 wire-up
// checklist: "logstore_client_test.go … hits a fake Bifrost"). The
// handler tests cover the shapes end-to-end; these pin the query
// composition, paging, auth header and error mapping the handlers
// rely on without going through HTTP twice.

// recordingBifrost captures every request the client makes and
// answers with a canned handler.
type recordingBifrost struct {
	srv  *httptest.Server
	reqs []*http.Request
	h    http.HandlerFunc
}

func newRecordingBifrost(t *testing.T, h http.HandlerFunc) *recordingBifrost {
	t.Helper()
	rb := &recordingBifrost{h: h}
	rb.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rb.reqs = append(rb.reqs, r.Clone(context.Background()))
		rb.h(w, r)
	}))
	t.Cleanup(rb.srv.Close)
	return rb
}

func (rb *recordingBifrost) client() *logstoreClient {
	return &logstoreClient{
		base:       rb.srv.URL,
		httpClient: &http.Client{Timeout: 2 * time.Second},
		authHeader: basicAuth("admin", "hunter2"),
	}
}

func emptyPage(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprint(w, `{"logs":[],"pagination":{"limit":50,"offset":0,"total_count":0},"stats":{"total_requests":0,"total_cost":0,"total_tokens":0},"has_logs":false}`)
}

func TestLogstoreClient_SearchComposesQuery(t *testing.T) {
	rb := newRecordingBifrost(t, emptyPage)
	start := time.Date(2026, 5, 14, 9, 0, 0, 0, time.UTC)
	end := start.Add(time.Hour)

	_, err := rb.client().search(context.Background(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"run-id": "r1", "agent-name": "coder"},
		Limit:     25,
		Offset:    50,
		SortBy:    "cost",
		Order:     "asc",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rb.reqs) != 1 {
		t.Fatalf("requests: %d", len(rb.reqs))
	}
	r := rb.reqs[0]
	if r.URL.Path != "/api/logs" {
		t.Errorf("path: %s", r.URL.Path)
	}
	if got := r.Header.Get("Authorization"); got != basicAuth("admin", "hunter2") {
		t.Errorf("auth header: %q", got)
	}
	q := r.URL.Query()
	want := url.Values{
		"start_time":          {start.Format(time.RFC3339Nano)},
		"end_time":            {end.Format(time.RFC3339Nano)},
		"metadata_run-id":     {"r1"},
		"metadata_agent-name": {"coder"},
		"limit":               {"25"},
		"offset":              {"50"},
		"sort_by":             {"cost"},
		"order":               {"asc"},
	}
	for k, v := range want {
		if q.Get(k) != v[0] {
			t.Errorf("query %s = %q, want %q", k, q.Get(k), v[0])
		}
	}
	if len(q) != len(want) {
		t.Errorf("unexpected extra params: %v", q)
	}
}

func TestLogstoreClient_SearchAll_PagesAndCaps(t *testing.T) {
	// 2500 rows served in pages of whatever `limit` asks for.
	rb := newRecordingBifrost(t, func(w http.ResponseWriter, r *http.Request) {
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
		const total = 2500
		var rows []map[string]any
		for i := offset; i < offset+limit && i < total; i++ {
			rows = append(rows, map[string]any{"id": strconv.Itoa(i), "cost": 0.01})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"logs":       rows,
			"pagination": map[string]any{"limit": limit, "offset": offset, "total_count": total},
			"stats":      map[string]any{},
		})
	})
	c := rb.client()

	all, err := c.searchAll(context.Background(), searchOpts{}, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2500 {
		t.Fatalf("rows = %d, want 2500", len(all))
	}
	// 1000 + 1000 + 500: the short page ends the walk.
	if len(rb.reqs) != 3 {
		t.Fatalf("requests = %d, want 3", len(rb.reqs))
	}
	if all[2499].ID != "2499" {
		t.Errorf("last row: %+v", all[2499])
	}

	// maxRows stops the walk even though more pages exist.
	rb.reqs = nil
	capped, err := c.searchAll(context.Background(), searchOpts{}, 1000, 1500)
	if err != nil {
		t.Fatal(err)
	}
	if len(capped) != 2000 || len(rb.reqs) != 2 {
		t.Errorf("capped walk: rows=%d requests=%d (cap applies after the page that crosses it)", len(capped), len(rb.reqs))
	}
	// Oversized page size clamps to Bifrost's 1000 ceiling.
	rb.reqs = nil
	_, _ = c.searchAll(context.Background(), searchOpts{}, 5000, 100)
	if got := rb.reqs[0].URL.Query().Get("limit"); got != "1000" {
		t.Errorf("page size not clamped: limit=%s", got)
	}
}

func TestLogstoreClient_FindByID_404IsNil(t *testing.T) {
	rb := newRecordingBifrost(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/logs/known" {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"id":"known","provider":"anthropic","raw_response":"{}","stream":true}`)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	})
	c := rb.client()

	got, err := c.findByID(context.Background(), "known")
	if err != nil || got == nil || got.ID != "known" || !got.Stream {
		t.Fatalf("known: %+v (%v)", got, err)
	}
	missing, err := c.findByID(context.Background(), "nope")
	if err != nil || missing != nil {
		t.Fatalf("404 must be (nil, nil): %+v (%v)", missing, err)
	}
	// Path-escaping: an id with a slash can't walk the URL.
	rb.reqs = nil
	_, _ = c.findByID(context.Background(), "a/b")
	if rb.reqs[0].URL.EscapedPath() != "/api/logs/a%2Fb" {
		t.Errorf("id not escaped: %s", rb.reqs[0].URL.EscapedPath())
	}
}

func TestLogstoreClient_UpstreamErrors(t *testing.T) {
	// Non-2xx maps to upstreamError carrying the status + a body excerpt.
	rb := newRecordingBifrost(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		fmt.Fprint(w, `{"error":"db locked"}`)
	})
	_, err := rb.client().search(context.Background(), searchOpts{})
	var ue *upstreamError
	if !errors.As(err, &ue) || ue.status != http.StatusBadGateway || ue.body != `{"error":"db locked"}` {
		t.Fatalf("non-2xx: %v", err)
	}

	// Unreachable maps to upstreamError with a cause.
	dead := &logstoreClient{
		base:       "http://127.0.0.1:1", // nothing listens on port 1
		httpClient: &http.Client{Timeout: time.Second},
		authHeader: basicAuth("a", "b"),
	}
	_, err = dead.search(context.Background(), searchOpts{})
	if !errors.As(err, &ue) || ue.cause == nil {
		t.Fatalf("unreachable: %v", err)
	}

	// A 2xx with a non-JSON body is a decode error, not upstreamError —
	// the handler maps that to 500 rather than 502.
	rb2 := newRecordingBifrost(t, func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "<html>oops</html>")
	})
	_, err = rb2.client().search(context.Background(), searchOpts{})
	if err == nil || errors.As(err, &ue) {
		t.Fatalf("decode failure must not be upstreamError: %v", err)
	}
}

func TestLogstoreClient_Customer(t *testing.T) {
	rb := newRecordingBifrost(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/governance/customers/u_alice":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"customer":{"id":"u_alice","name":"alice","budgets":[{"id":"b1","max_limit":1000,"reset_duration":"1d","last_reset":"2026-05-14T00:00:00Z","current_usage":42.5}]}}`)
		default:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"Customer not found"}`)
		}
	})
	c := rb.client()

	got, err := c.customer(context.Background(), "u_alice")
	if err != nil || got == nil {
		t.Fatalf("customer: %+v (%v)", got, err)
	}
	if got.ID != "u_alice" || len(got.Budgets) != 1 || got.Budgets[0].MaxLimit != 1000 ||
		got.Budgets[0].ResetDuration != "1d" || got.Budgets[0].CurrentUsage != 42.5 {
		t.Errorf("decoded: %+v", got)
	}
	missing, err := c.customer(context.Background(), "u_ghost")
	if err != nil || missing != nil {
		t.Fatalf("404 must be (nil, nil): %+v (%v)", missing, err)
	}
	if got := rb.reqs[0].Header.Get("Authorization"); got != basicAuth("admin", "hunter2") {
		t.Errorf("governance call must carry the admin basic auth: %q", got)
	}
}

func TestLogstoreLog_Tokens(t *testing.T) {
	var l logstoreLog
	if l.tokens() != 0 {
		t.Error("nil usage must be 0")
	}
	if err := json.Unmarshal([]byte(`{"id":"1","token_usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`), &l); err != nil {
		t.Fatal(err)
	}
	if l.tokens() != 15 {
		t.Errorf("tokens = %d", l.tokens())
	}
	// total_tokens left at 0 by the provider: fall back to the sum.
	l.TokenUsage.TotalTokens = 0
	if l.tokens() != 15 {
		t.Errorf("fallback tokens = %d", l.tokens())
	}
}

func TestNewLogstoreClient_RequiresCreds(t *testing.T) {
	if newLogstoreClient("", "x") != nil || newLogstoreClient("x", "") != nil {
		t.Error("missing creds must yield nil (routes skipped), not a client that 401s")
	}
	if c := newLogstoreClient("u", "p"); c == nil || c.base != logstoreBaseURL {
		t.Errorf("client: %+v", c)
	}
}
