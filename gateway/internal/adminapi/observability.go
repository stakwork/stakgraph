package adminapi

import (
	"errors"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// ─── public response shapes (consumed by the SPA & by tygo) ──────────
//
// Every struct here is what the dashboard receives over the wire. The
// frontend imports the tygo-generated TS bindings of these types; the
// shapes are part of phase 8's stable contract and shouldn't change
// without bumping the SPA in lockstep.

// AgentSpend is one row of /_plugin/spend/by-agent.
type AgentSpend struct {
	AgentName    string  `json:"agent_name"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// SpendByAgentResponse is the envelope for /_plugin/spend/by-agent.
type SpendByAgentResponse struct {
	Window  string       `json:"window"`
	Results []AgentSpend `json:"results"`
}

// UserSpend is one row of /_plugin/spend/by-user.
type UserSpend struct {
	UserID       string  `json:"user_id"`
	UserName     string  `json:"user_name"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// SpendByUserResponse is the envelope for /_plugin/spend/by-user.
type SpendByUserResponse struct {
	Window  string      `json:"window"`
	Results []UserSpend `json:"results"`
}

// HistogramPoint is one (timestamp, cost) datum in a per-dimension
// series. Timestamp is the bucket's start time in RFC3339.
type HistogramPoint struct {
	Timestamp string  `json:"ts"`
	Cost      float64 `json:"cost"`
}

// HistogramSeries is one line on a stacked-area chart.
type HistogramSeries struct {
	DimensionValue string           `json:"dimension_value"`
	Points         []HistogramPoint `json:"points"`
}

// HistogramCostResponse is the envelope for
// /_plugin/histogram/cost.
type HistogramCostResponse struct {
	BucketSizeSeconds int64             `json:"bucket_size_seconds"`
	Dimension         string            `json:"dimension"`
	Series            []HistogramSeries `json:"series"`
}

// RunLogEntry is one row inside a RunDetailResponse's `logs` field.
// A trimmed view of Bifrost's Log — phase 8 only renders the columns
// the call-log table actually shows.
type RunLogEntry struct {
	ID        string            `json:"id"`
	Timestamp string            `json:"timestamp"`
	Provider  string            `json:"provider"`
	Model     string            `json:"model"`
	Status    string            `json:"status"`
	Cost      float64           `json:"cost"`
	Latency   float64           `json:"latency"`
	Metadata  map[string]string `json:"metadata"`
}

// RunStats is the aggregate-card summary at the top of /runs/:id.
type RunStats struct {
	TotalRequests int64   `json:"total_requests"`
	TotalCost     float64 `json:"total_cost"`
	TotalTokens   int64   `json:"total_tokens"`
}

// RunDetailResponse is the envelope for /_plugin/runs/:run_id.
type RunDetailResponse struct {
	RunID string        `json:"run_id"`
	Logs  []RunLogEntry `json:"logs"`
	Stats RunStats      `json:"stats"`
}

// ─── handler scaffold ────────────────────────────────────────────────

// observabilityHandlers carries the logstore client into the handler
// methods. Construction is trivial; the heavy work is in the
// per-handler funcs below.
type observabilityHandlers struct {
	logs *logstoreClient
}

func newObservabilityHandlers(c *logstoreClient) *observabilityHandlers {
	return &observabilityHandlers{logs: c}
}

// ─── /_plugin/spend/by-agent ─────────────────────────────────────────

// Aggregation strategy
// --------------------
// Bifrost's native dimension support is column-bound (provider /
// team_id / customer_id / user_id / business_unit_id). agent-name
// lives in metadata, so we can't ask Bifrost to group by it.
//
// Instead we page through /api/logs with no metadata filter (just
// the time window), then bucket+sum in Go by `metadata.agent-name`.
// This is the approach the user confirmed during phase-8 planning —
// see plans/phases/phase-8-observability-dashboard.md §"Architecture".
//
// Practical bounds: phase 8 caps the read at 200k rows (see
// logstoreClient.searchAll). A workspace doing >200k LLM calls in a
// 30-day window will see results that under-count the oldest slice
// — at which point per-agent aggregations want to be moved to a
// materialised view, which is a phase-9+ concern.
func (h *observabilityHandlers) spendByAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}

	// Optional ?user_id=<id> scopes the aggregation to one user's
	// calls — used by the People > UserDetail page to render
	// "which agents did this person use?". Without the filter the
	// handler returns the same swarm-wide rollup it always did.
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "spend.by_agent")
		return
	}

	type agg struct {
		cost   float64
		tokens int64 // not in our trimmed Log; left 0 in phase 8
		count  int64
	}
	by := map[string]*agg{}
	for _, l := range logs {
		name := l.Metadata["agent-name"]
		if name == "" {
			continue // unattributed call; surface separately in phase 9 if needed
		}
		if _, ok := by[name]; !ok {
			by[name] = &agg{}
		}
		by[name].cost += l.Cost
		by[name].count++
	}

	out := SpendByAgentResponse{
		Window:  window,
		Results: make([]AgentSpend, 0, len(by)),
	}
	for name, a := range by {
		out.Results = append(out.Results, AgentSpend{
			AgentName:    name,
			TotalCost:    a.cost,
			TotalTokens:  a.tokens,
			RequestCount: a.count,
		})
	}
	sort.Slice(out.Results, func(i, j int) bool {
		return out.Results[i].TotalCost > out.Results[j].TotalCost
	})
	writeJSON(w, http.StatusOK, out)
}

// ─── /_plugin/spend/by-user ──────────────────────────────────────────
//
// `customer_id` is a first-class indexed column on Bifrost's logs
// table (v2 invariant: customer_id = user_id). We could call
// Bifrost's rankings endpoint, but it adds a "trends" calculation
// against the previous window that complicates the shape and is
// phase-9's job. Phase 8 just sums per-user from the same paged
// row scan we already do for by-agent.
func (h *observabilityHandlers) spendByUser(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}

	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "spend.by_user")
		return
	}

	type agg struct {
		cost   float64
		tokens int64
		count  int64
	}
	by := map[string]*agg{}
	for _, l := range logs {
		uid := l.CustomerID
		if uid == "" {
			// Fall back to the verified macaroon dim so logs from
			// before phase-6 canonicalisation still attribute.
			uid = l.Metadata["user-id"]
		}
		if uid == "" {
			continue
		}
		if _, ok := by[uid]; !ok {
			by[uid] = &agg{}
		}
		by[uid].cost += l.Cost
		by[uid].count++
	}

	out := SpendByUserResponse{
		Window:  window,
		Results: make([]UserSpend, 0, len(by)),
	}
	for uid, a := range by {
		out.Results = append(out.Results, UserSpend{
			UserID:       uid,
			UserName:     uid, // see plans note: user_id == user_name in v2
			TotalCost:    a.cost,
			TotalTokens:  a.tokens,
			RequestCount: a.count,
		})
	}
	sort.Slice(out.Results, func(i, j int) bool {
		return out.Results[i].TotalCost > out.Results[j].TotalCost
	})
	writeJSON(w, http.StatusOK, out)
}

// ─── /_plugin/histogram/cost ─────────────────────────────────────────
//
// Same agent-name-in-metadata story as by-agent: bucket+sum in Go.
//
// Bucket policy: the SPA passes `bucket=1h`; we honour any of {1m,
// 5m, 10m, 1h, 6h, 1d}. Buckets are aligned to UTC epoch so the same
// window/bucket pair produces identical bucket edges across calls
// — important for incremental refresh and for visual stability when
// the page polls.
func (h *observabilityHandlers) histogramCost(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}
	bucket, ok := parseBucket(w, r, end.Sub(start))
	if !ok {
		return
	}
	dimension, ok := parseDimensionParam(w, r)
	if !ok {
		return
	}

	// Optional ?user_id= / ?agent_name= scope the histogram to one
	// dim value. Used by UserDetail / AgentDetail pages so the
	// chart shows only this person/agent's contribution rather
	// than all activity sliced by dimension.
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "histogram.cost")
		return
	}

	// bucket aligned to epoch seconds so two calls with identical
	// args produce identical bucket boundaries.
	bucketSec := int64(bucket.Seconds())
	// series[dim][bucketStart] = cost
	series := map[string]map[int64]float64{}
	for _, l := range logs {
		dim := dimensionValue(l, dimension)
		if dim == "" {
			continue
		}
		ts, err := time.Parse(time.RFC3339Nano, l.Timestamp)
		if err != nil {
			continue
		}
		b := (ts.Unix() / bucketSec) * bucketSec
		if _, ok := series[dim]; !ok {
			series[dim] = map[int64]float64{}
		}
		series[dim][b] += l.Cost
	}

	// Materialise. Empty buckets are omitted (the chart renders
	// gaps naturally); a phase-9 enhancement could emit zero points
	// for visual continuity. Sort series by total descending so
	// the dashboard renders the heaviest contributor first.
	out := HistogramCostResponse{
		BucketSizeSeconds: bucketSec,
		Dimension:         dimension,
		Series:            make([]HistogramSeries, 0, len(series)),
	}
	for dim, buckets := range series {
		pts := make([]HistogramPoint, 0, len(buckets))
		var total float64
		for ts, cost := range buckets {
			pts = append(pts, HistogramPoint{
				Timestamp: time.Unix(ts, 0).UTC().Format(time.RFC3339),
				Cost:      cost,
			})
			total += cost
		}
		sort.Slice(pts, func(i, j int) bool { return pts[i].Timestamp < pts[j].Timestamp })
		out.Series = append(out.Series, HistogramSeries{
			DimensionValue: dim,
			Points:         pts,
		})
	}
	sort.SliceStable(out.Series, func(i, j int) bool {
		return seriesTotal(out.Series[i]) > seriesTotal(out.Series[j])
	})

	// Silence unused-vars: `window` is part of the request input
	// and shouldn't show up in the response when the bucket-aligned
	// times already convey the range, but stash it as a header so
	// curl users can confirm what they queried.
	w.Header().Set("X-Window", window)
	writeJSON(w, http.StatusOK, out)
}

func seriesTotal(s HistogramSeries) float64 {
	var t float64
	for _, p := range s.Points {
		t += p.Cost
	}
	return t
}

// dimensionValue returns the value of `dim` from a logstoreLog.
// Metadata-backed dims (agent-name, run-id, session-id, realm-id)
// are pulled from l.Metadata; the native customer_id column is
// pulled from l.CustomerID. Unknown dimensions return "" — the
// caller already rejected those in parseDimensionParam, so this
// branch only fires for genuinely-empty rows.
func dimensionValue(l logstoreLog, dim string) string {
	switch dim {
	case "user-id":
		// user-id and customer_id are aliased per v2.
		if l.CustomerID != "" {
			return l.CustomerID
		}
		return l.Metadata["user-id"]
	default:
		return l.Metadata[dim]
	}
}

// ─── /_plugin/runs/:run_id ───────────────────────────────────────────
//
// Routed under `/_plugin/runs/` (subtree); the trailing segment is
// the run_id. We don't pull in a router library for this since
// phase 8 only has one such route — extracting the segment with
// string ops is shorter than gluing in a dependency.
func (h *observabilityHandlers) runDetail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	const prefix = "/_plugin/runs/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	if rest == "" || strings.ContainsRune(rest, '/') {
		// Phase 6 will introduce /_plugin/runs/:id/state and
		// /_plugin/runs/:id/kill — both contain a slash after the
		// id. Phase 8 doesn't serve those, so reject early with a
		// crisp 404 rather than confusing the caller with a
		// stripped-id query.
		http.NotFound(w, r)
		return
	}
	runID := rest

	limit, offset, ok := parsePagination(w, r)
	if !ok {
		return
	}

	res, err := h.logs.search(r.Context(), searchOpts{
		Metadata: map[string]string{"run-id": runID},
		Limit:    limit,
		Offset:   offset,
		SortBy:   "timestamp",
		Order:    "desc",
	})
	if err != nil {
		writeUpstreamError(w, err, "runs.detail")
		return
	}

	out := RunDetailResponse{
		RunID: runID,
		Logs:  make([]RunLogEntry, 0, len(res.Logs)),
		Stats: RunStats{
			TotalRequests: res.Stats.TotalRequests,
			TotalCost:     res.Stats.TotalCost,
			TotalTokens:   res.Stats.TotalTokens,
		},
	}
	for _, l := range res.Logs {
		out.Logs = append(out.Logs, RunLogEntry{
			ID:        l.ID,
			Timestamp: l.Timestamp,
			Provider:  l.Provider,
			Model:     l.Model,
			Status:    l.Status,
			Cost:      l.Cost,
			Latency:   l.Latency,
			Metadata:  l.Metadata,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// ─── parameter parsing ───────────────────────────────────────────────

// parseWindow reads ?window=1h|24h|7d|30d and returns the canonical
// label plus the resolved start/end pair (UTC). 24h is the default
// when unset.
//
// Returns (window, start, end, ok). When ok is false the handler
// has already written a 400 and must return.
func parseWindow(w http.ResponseWriter, r *http.Request) (string, time.Time, time.Time, bool) {
	q := r.URL.Query().Get("window")
	if q == "" {
		q = "24h"
	}
	now := time.Now().UTC()
	var from time.Time
	switch q {
	case "1h":
		from = now.Add(-time.Hour)
	case "6h":
		from = now.Add(-6 * time.Hour)
	case "24h":
		from = now.Add(-24 * time.Hour)
	case "7d":
		from = now.AddDate(0, 0, -7)
	case "30d":
		from = now.AddDate(0, 0, -30)
	default:
		writeError(w, http.StatusBadRequest, "bad_request",
			"window must be one of: 1h, 6h, 24h, 7d, 30d")
		return "", time.Time{}, time.Time{}, false
	}
	return q, from, now, true
}

// parseBucket reads ?bucket=… and enforces a small whitelist. Must
// not exceed the window (otherwise a 7d window with a 30d bucket
// would produce one point — useless, and the kind of silent
// degradation that's confusing to debug).
func parseBucket(w http.ResponseWriter, r *http.Request, window time.Duration) (time.Duration, bool) {
	q := r.URL.Query().Get("bucket")
	if q == "" {
		// Default: 1h matches the canonical "hour buckets across
		// today" use case the dashboard does on first paint.
		q = "1h"
	}
	d, ok := map[string]time.Duration{
		"1m":  time.Minute,
		"5m":  5 * time.Minute,
		"10m": 10 * time.Minute,
		"1h":  time.Hour,
		"6h":  6 * time.Hour,
		"1d":  24 * time.Hour,
	}[q]
	if !ok {
		writeError(w, http.StatusBadRequest, "bad_request",
			"bucket must be one of: 1m, 5m, 10m, 1h, 6h, 1d")
		return 0, false
	}
	if d > window {
		writeError(w, http.StatusBadRequest, "bad_request",
			"bucket must be ≤ window")
		return 0, false
	}
	return d, true
}

// parseDimensionParam reads ?dimension=… and enforces the set of
// dims phase 8 supports. agent-name, run-id, session-id, realm-id
// are metadata-backed; user-id is aliased to customer_id (see
// dimensionValue).
func parseDimensionParam(w http.ResponseWriter, r *http.Request) (string, bool) {
	q := r.URL.Query().Get("dimension")
	if q == "" {
		// Default to the dashboard's headline view.
		q = "agent-name"
	}
	switch q {
	case "agent-name", "run-id", "session-id", "realm-id", "user-id":
		return q, true
	default:
		writeError(w, http.StatusBadRequest, "bad_request",
			"dimension must be one of: agent-name, run-id, session-id, realm-id, user-id")
		return "", false
	}
}

// parsePagination reads ?limit=&offset=. Defaults match phase 8's
// "RunDetail call log: default 50 most recent" — large enough to be
// useful, small enough to never timeout the upstream call.
func parsePagination(w http.ResponseWriter, r *http.Request) (int, int, bool) {
	limit := 50
	offset := 0
	if s := r.URL.Query().Get("limit"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v <= 0 || v > 500 {
			writeError(w, http.StatusBadRequest, "bad_request",
				"limit must be 1..500")
			return 0, 0, false
		}
		limit = v
	}
	if s := r.URL.Query().Get("offset"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil || v < 0 {
			writeError(w, http.StatusBadRequest, "bad_request",
				"offset must be ≥ 0")
			return 0, 0, false
		}
		offset = v
	}
	return limit, offset, true
}

// metadataFilterFromQuery is the common parser for the optional
// `?user_id=` / `?agent_name=` / `?run_id=` / `?session_id=` /
// `?realm_id=` / `?org_id=` query params. Each maps to the matching
// `metadata.<dim>` filter that Bifrost's /api/logs accepts.
//
// Why this exists in one place: every handler that rolls up
// metadata accepts the same scoping params, and we want adding a
// new dim (e.g. business-unit later) to be a one-line edit here
// rather than scattered across spend / histogram / future endpoints.
//
// Empty values are skipped — `?user_id=` with no value doesn't add
// a filter (so the SPA can pass `userID` directly without
// short-circuiting on the empty string).
func metadataFilterFromQuery(r *http.Request) map[string]string {
	q := r.URL.Query()
	out := map[string]string{}
	pairs := []struct{ qparam, dim string }{
		{"user_id", "user-id"},
		{"agent_name", "agent-name"},
		{"run_id", "run-id"},
		{"session_id", "session-id"},
		{"realm_id", "realm-id"},
		{"org_id", "org-id"},
	}
	for _, p := range pairs {
		if v := q.Get(p.qparam); v != "" {
			out[p.dim] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// writeUpstreamError maps an upstream-failure error to phase-7's
// 502 / `upstream_unavailable`. Logs the underlying detail so an
// operator can see what Bifrost actually said, but never bleeds the
// upstream body to the browser.
func writeUpstreamError(w http.ResponseWriter, err error, where string) {
	var ue *upstreamError
	if errors.As(err, &ue) {
		pluginlog.Warnf("adminapi: %s: %v", where, ue)
		writeError(w, http.StatusBadGateway, "upstream_unavailable",
			"bifrost log store is unavailable")
		return
	}
	pluginlog.Errf("adminapi: %s: %v", where, err)
	writeError(w, http.StatusInternalServerError, "internal", "internal error")
}
