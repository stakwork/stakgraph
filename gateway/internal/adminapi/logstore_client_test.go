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

// keysetBifrost serves `rows` (newest first, stable) the way Bifrost's
// /api/logs does for the walker: `end_time` is inclusive, then
// offset/limit. Pass the row set newest-first.
func keysetBifrost(t *testing.T, rows []map[string]any) *recordingBifrost {
	t.Helper()
	return newRecordingBifrost(t, func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		limit, _ := strconv.Atoi(q.Get("limit"))
		offset, _ := strconv.Atoi(q.Get("offset"))
		var end *time.Time
		if v := q.Get("end_time"); v != "" {
			ts, err := time.Parse(time.RFC3339Nano, v)
			if err != nil {
				t.Errorf("bad end_time %q: %v", v, err)
			}
			end = &ts
		}
		var inRange []map[string]any
		for _, row := range rows {
			if end != nil {
				ts, _ := time.Parse(time.RFC3339Nano, row["timestamp"].(string))
				if ts.After(*end) {
					continue
				}
			}
			inRange = append(inRange, row)
		}
		if offset > len(inRange) {
			offset = len(inRange)
		}
		page := inRange[offset:]
		if len(page) > limit {
			page = page[:limit]
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"logs":       page,
			"pagination": map[string]any{"limit": limit, "offset": offset, "total_count": len(inRange)},
			"stats":      map[string]any{},
		})
	})
}

func TestLogstoreClient_SearchAll_PagesAndCaps(t *testing.T) {
	// 2500 rows with distinct timestamps, newest first.
	base := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)
	var rows []map[string]any
	for i := 0; i < 2500; i++ {
		rows = append(rows, map[string]any{
			"id": strconv.Itoa(i), "cost": 0.01,
			"timestamp": base.Add(-time.Duration(i) * time.Second).Format(time.RFC3339Nano),
		})
	}
	rb := keysetBifrost(t, rows)
	c := rb.client()

	all, err := c.searchAll(context.Background(), searchOpts{}, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2500 {
		t.Fatalf("rows = %d, want 2500", len(all))
	}
	// 1000 + 1000 + 500: the short page ends the walk. Page 2 and 3
	// are keyed by end_time (the boundary row comes back and is
	// dropped), never by offset.
	if len(rb.reqs) != 3 {
		t.Fatalf("requests = %d, want 3", len(rb.reqs))
	}
	for i, req := range rb.reqs {
		q := req.URL.Query()
		if q.Get("sort_by") != "timestamp" || q.Get("order") != "desc" {
			t.Errorf("request %d: sort=%s/%s, want timestamp/desc", i, q.Get("sort_by"), q.Get("order"))
		}
		if q.Get("offset") != "" {
			t.Errorf("request %d: offset=%s, want keyset paging (no offset)", i, q.Get("offset"))
		}
	}
	if got := rb.reqs[1].URL.Query().Get("end_time"); got != rows[999]["timestamp"] {
		t.Errorf("page 2 end_time = %s, want the oldest ts of page 1 (%s)", got, rows[999]["timestamp"])
	}
	seen := map[string]bool{}
	for _, l := range all {
		if seen[l.ID] {
			t.Fatalf("row %s returned twice", l.ID)
		}
		seen[l.ID] = true
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
	// Two pages: 1000 + (1000 − 1 boundary duplicate). The cap applies
	// after the page that crosses it.
	if len(capped) != 1999 || len(rb.reqs) != 2 {
		t.Errorf("capped walk: rows=%d requests=%d, want 1999/2", len(capped), len(rb.reqs))
	}
	// Oversized page size clamps to Bifrost's 1000 ceiling.
	rb.reqs = nil
	_, _ = c.searchAll(context.Background(), searchOpts{}, 5000, 100)
	if got := rb.reqs[0].URL.Query().Get("limit"); got != "1000" {
		t.Errorf("page size not clamped: limit=%s", got)
	}
}

func TestLogstoreClient_SearchAll_BoundaryTies(t *testing.T) {
	base := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)

	// Triplets share a timestamp, so page boundaries fall inside a
	// tie: the boundary rows re-served by the inclusive end_time must
	// be dropped exactly once.
	var rows []map[string]any
	for i := 0; i < 1500; i++ {
		rows = append(rows, map[string]any{
			"id":        strconv.Itoa(i),
			"timestamp": base.Add(-time.Duration(i/3) * time.Second).Format(time.RFC3339Nano),
		})
	}
	rb := keysetBifrost(t, rows)
	all, err := rb.client().searchAll(context.Background(), searchOpts{}, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1500 || len(rb.reqs) != 2 {
		t.Fatalf("tied pages: rows=%d requests=%d, want 1500/2", len(all), len(rb.reqs))
	}
	seen := map[string]bool{}
	for _, l := range all {
		if seen[l.ID] {
			t.Fatalf("row %s returned twice", l.ID)
		}
		seen[l.ID] = true
	}

	// A page that sits entirely on one timestamp cannot move the
	// boundary; the walk steps past it with offset and still finds
	// every row.
	rows = nil
	for i := 0; i < 1200; i++ {
		rows = append(rows, map[string]any{
			"id": strconv.Itoa(i), "timestamp": base.Format(time.RFC3339Nano),
		})
	}
	rb = keysetBifrost(t, rows)
	all, err = rb.client().searchAll(context.Background(), searchOpts{}, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1200 {
		t.Fatalf("single-ts rows = %d, want 1200", len(all))
	}
	seen = map[string]bool{}
	for _, l := range all {
		if seen[l.ID] {
			t.Fatalf("row %s returned twice", l.ID)
		}
		seen[l.ID] = true
	}
	if last := rb.reqs[len(rb.reqs)-1].URL.Query(); last.Get("offset") != "1000" {
		t.Errorf("stuck boundary must step by offset; last request offset=%s", last.Get("offset"))
	}
}

func TestLogstoreClient_SearchAll_ExplicitSortUsesOffset(t *testing.T) {
	base := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)
	var rows []map[string]any
	for i := 0; i < 1200; i++ {
		rows = append(rows, map[string]any{
			"id": strconv.Itoa(i), "timestamp": base.Add(-time.Duration(i) * time.Second).Format(time.RFC3339Nano),
		})
	}
	rb := keysetBifrost(t, rows)
	all, err := rb.client().searchAll(context.Background(), searchOpts{SortBy: "cost", Order: "asc"}, 1000, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 1200 || len(rb.reqs) != 2 {
		t.Fatalf("rows=%d requests=%d", len(all), len(rb.reqs))
	}
	q := rb.reqs[1].URL.Query()
	if q.Get("offset") != "1000" || q.Get("end_time") != "" || q.Get("sort_by") != "cost" {
		t.Errorf("non-timestamp sort must page by offset: %v", q)
	}
}

func TestLogstoreClient_WindowLogs_BypassesCacheWhenUnbounded(t *testing.T) {
	rb := newRecordingBifrost(t, emptyPage)
	c := rb.client()
	walks := 0
	c.window = newLogWindowCache(func(ctx context.Context, o searchOpts) ([]logstoreLog, error) {
		walks++
		return c.searchAll(ctx, o, windowPageSize, windowMaxRows)
	})
	// No time bounds (the session summary) ⇒ direct walk, not cached.
	for i := 0; i < 2; i++ {
		if _, err := c.windowLogs(context.Background(), searchOpts{Metadata: map[string]string{"session-id": "s1"}}); err != nil {
			t.Fatal(err)
		}
	}
	if walks != 0 || len(rb.reqs) != 2 {
		t.Errorf("unbounded opts: cache walks=%d bifrost requests=%d, want 0/2", walks, len(rb.reqs))
	}
	// Bounded ⇒ cached: the second call is served from the entry.
	rb.reqs = nil
	start, end := time.Now().Add(-time.Hour), time.Now()
	for i := 0; i < 2; i++ {
		if _, err := c.windowLogs(context.Background(), searchOpts{StartTime: &start, EndTime: &end}); err != nil {
			t.Fatal(err)
		}
	}
	if walks != 1 || len(rb.reqs) != 1 {
		t.Errorf("bounded opts: cache walks=%d bifrost requests=%d, want 1/1", walks, len(rb.reqs))
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

// Bifrost's logging plugin stamps non-string values into metadata
// (`realtime: true` on realtime turns, `isAsyncRequest: true` on
// x-bf-async jobs). One such row on a page used to fail the whole
// decode and 502 every rollup for the window. Strings must survive
// verbatim, scalars as their JSON text, and nested/null values must
// simply be dropped — never an error.
func TestMetadataMap_ToleratesNonStringValues(t *testing.T) {
	var l logstoreLog
	err := json.Unmarshal([]byte(`{
		"id": "rt1",
		"metadata": {
			"run-id": "r1",
			"agent-name": "canvas-agent",
			"realtime": true,
			"isAsyncRequest": true,
			"retries": 3,
			"ratio": 0.5,
			"nested": {"a": "b"},
			"list": [1, 2],
			"gone": null
		}
	}`), &l)
	if err != nil {
		t.Fatalf("bool/number metadata must decode, got: %v", err)
	}
	want := map[string]string{
		"run-id":         "r1",
		"agent-name":     "canvas-agent",
		"realtime":       "true",
		"isAsyncRequest": "true",
		"retries":        "3",
		"ratio":          "0.5",
	}
	if len(l.Metadata) != len(want) {
		t.Errorf("metadata = %v, want %v", l.Metadata, want)
	}
	for k, v := range want {
		if l.Metadata[k] != v {
			t.Errorf("metadata[%q] = %q, want %q", k, l.Metadata[k], v)
		}
	}
	if dimensionValue(l, "run-id") != "r1" {
		t.Errorf("dimensionValue must still index the map: %q", dimensionValue(l, "run-id"))
	}

	// Detail rows decode through the same type.
	var d logstoreLogDetail
	if err := json.Unmarshal([]byte(`{"id":"rt1","metadata":{"realtime":true,"user-id":"u1"}}`), &d); err != nil {
		t.Fatalf("detail row: %v", err)
	}
	if d.Metadata["realtime"] != "true" || d.Metadata["user-id"] != "u1" {
		t.Errorf("detail metadata = %v", d.Metadata)
	}

	// null / absent metadata stay nil so callers can index freely.
	var n logstoreLog
	if err := json.Unmarshal([]byte(`{"id":"x","metadata":null}`), &n); err != nil || n.Metadata != nil {
		t.Errorf("null metadata: err=%v map=%v", err, n.Metadata)
	}
	var a logstoreLog
	if err := json.Unmarshal([]byte(`{"id":"x"}`), &a); err != nil || a.Metadata["run-id"] != "" {
		t.Errorf("absent metadata: err=%v map=%v", err, a.Metadata)
	}
	// A non-object metadata value is still a decode error, not silent data loss.
	var bad logstoreLog
	if err := json.Unmarshal([]byte(`{"id":"x","metadata":"oops"}`), &bad); err == nil {
		t.Error("string-typed metadata must be rejected")
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
