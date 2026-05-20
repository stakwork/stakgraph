package adminapi

import (
	"encoding/json"
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

// AgentUserSpend is one row of /_plugin/spend/by-agent-user — the
// fan-out crossing of (agent-name × user-id) so the flowchart UI can
// render one box per pairing in a single round-trip. Rows missing
// either dim are excluded (same policy as by-agent / by-user).
type AgentUserSpend struct {
	AgentName    string  `json:"agent_name"`
	UserID       string  `json:"user_id"`
	UserName     string  `json:"user_name"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// SpendByAgentUserResponse is the envelope for
// /_plugin/spend/by-agent-user.
type SpendByAgentUserResponse struct {
	Window  string           `json:"window"`
	Results []AgentUserSpend `json:"results"`
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

// CallDetailResponse is the envelope for
// /_plugin/runs/:run_id/calls/:call_id — the full request/response
// content for a single LLM call, fetched on-demand when the
// operator clicks a row in the RunDetail call log.
//
// Run-scoping rationale: we verify that the fetched log's
// metadata.run-id matches the URL's run_id before returning, so a
// caller can't enumerate other workspaces' logs by guessing IDs.
// Bifrost's /api/logs/{id} doesn't do this check itself (it just
// looks up by primary key), so the plugin enforces it.
//
// Body fields are pass-through json.RawMessage from Bifrost — the
// SPA pretty-prints them; the plugin doesn't introspect them. This
// keeps the schema coupling minimal: new fields upstream surface in
// the UI without code changes here.
type CallDetailResponse struct {
	ID         string            `json:"id"`
	RunID      string            `json:"run_id"`
	Timestamp  string            `json:"timestamp"`
	Provider   string            `json:"provider"`
	Model      string            `json:"model"`
	Status     string            `json:"status"`
	Cost       float64           `json:"cost"`
	Latency    float64           `json:"latency"`
	CustomerID string            `json:"customer_id"`
	Metadata   map[string]string `json:"metadata"`

	// Per-request descriptors stamped by Bifrost. `stop_reason`
	// tells the operator why the model stopped (stop, length,
	// content_filter, tool_calls, refusal). `stream` flags
	// streaming responses (which lack a single output_message and
	// surface their content via Bifrost's stream chunk replay).
	// Retries / fallback_index are zero on the happy path; non-zero
	// means Bifrost had to retry the call or fall back to a
	// different provider, which is useful provenance.
	StopReason      string `json:"stop_reason,omitempty"`
	Stream          bool   `json:"stream"`
	NumberOfRetries int    `json:"number_of_retries"`
	FallbackIndex   int    `json:"fallback_index"`

	// TokenUsage is the provider-reported usage breakdown
	// (BifrostLLMUsage). Includes prompt/completion/total totals
	// plus cached read/write splits (Anthropic prompt-cache,
	// OpenAI cached_tokens), audio/image token counts, and a
	// per-call cost record. Pass-through JSON — the SPA introspects
	// it. Bifrost's row-level prompt_tokens / completion_tokens /
	// total_tokens columns are denormalized helpers tagged
	// `json:"-"`, so this is the only place token data is on the
	// wire.
	TokenUsage json.RawMessage `json:"token_usage,omitempty"`

	// CacheDebug carries Bifrost's *semantic* cache verdict for
	// this call (hit/miss + similarity score). Distinct from
	// prompt-cache tokens, which live in TokenUsage above. Absent
	// when no semantic cache is configured for this swarm.
	CacheDebug json.RawMessage `json:"cache_debug,omitempty"`

	// All optional; missing on failures, realtime turns, or rows
	// recorded before a given column existed.
	InputHistory   json.RawMessage `json:"input_history,omitempty"`
	OutputMessage  json.RawMessage `json:"output_message,omitempty"`
	Params         json.RawMessage `json:"params,omitempty"`
	Tools          json.RawMessage `json:"tools,omitempty"`
	ErrorDetails   json.RawMessage `json:"error_details,omitempty"`
	RawRequest     string          `json:"raw_request,omitempty"`
	RawResponse    string          `json:"raw_response,omitempty"`
	ContentSummary string          `json:"content_summary,omitempty"`
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
// Identity source-of-truth
// ------------------------
// We aggregate by `metadata["user-id"]` only, NOT by the indexed
// `customer_id` column. Two reasons:
//
//  1. The v2 invariant `customer_id = user_id` holds in theory, but
//     `customer_id` is populated from the virtual key (whatever Hive
//     issued), which for production traffic is a Hive UUID — not the
//     human-readable identifier callers stamp on `x-bf-dim-user-id`.
//     Mixing the two confuses the dashboard (UUID rows in the list,
//     username rows on RunDetail) and breaks click-through (clicking
//     a UUID queries by username and gets empty).
//
//  2. Post-phase-6 `metadata.user-id` is overwritten from the
//     verified macaroon claim, so it becomes the cryptographically
//     attested identity. Treating it as authoritative now means
//     phase-6 lights up without any UI change.
//
// Rows with no `metadata.user-id` (e.g. ad-hoc curl without dim
// headers, or pre-onboarding traffic) are intentionally excluded —
// they'd otherwise pile up as orphan "unknown" entries that the
// dashboard can't do anything useful with anyway.
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
		uid := l.Metadata["user-id"]
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

// ─── /_plugin/spend/by-agent-user ────────────────────────────────────
//
// Fan-out of by-agent × by-user in a single pass over the same
// 200k-row ceiling as the other rollups. The flowchart canvas needs
// one row per (agent, user) pair — calling by-agent once per user
// would be N round-trips and N×200k row scans; doing it server-side
// here is a single scan that buckets by a compound key.
//
// Same dim-filter contract as spendByAgent: the optional
// ?user_id= / ?agent_name= scope down the data set at the Bifrost
// query layer. With no filter you get the whole swarm matrix.
func (h *observabilityHandlers) spendByAgentUser(w http.ResponseWriter, r *http.Request) {
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
		writeUpstreamError(w, err, "spend.by_agent_user")
		return
	}

	// Compound-key bucket. Using "agent\x00user" as the map key
	// avoids allocating a struct-keyed map (and any collision
	// concern — \x00 is the one byte we know never appears in a
	// metadata value).
	type agg struct {
		agent  string
		user   string
		cost   float64
		tokens int64
		count  int64
	}
	by := map[string]*agg{}
	for _, l := range logs {
		name := l.Metadata["agent-name"]
		uid := l.Metadata["user-id"]
		if name == "" || uid == "" {
			continue
		}
		k := name + "\x00" + uid
		a, ok := by[k]
		if !ok {
			a = &agg{agent: name, user: uid}
			by[k] = a
		}
		a.cost += l.Cost
		a.count++
	}

	out := SpendByAgentUserResponse{
		Window:  window,
		Results: make([]AgentUserSpend, 0, len(by)),
	}
	for _, a := range by {
		out.Results = append(out.Results, AgentUserSpend{
			AgentName:    a.agent,
			UserID:       a.user,
			UserName:     a.user, // user_id == user_name in v2 (see spendByUser note)
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
// Every dim — including user-id — is read from l.Metadata. The
// `customer_id` column is intentionally NOT consulted: it carries
// the virtual key's customer (a Hive UUID in production), which
// disagrees with the human-readable `metadata.user-id` stamped by
// callers. See the source-of-truth note on spendByUser above.
//
// Unknown dimensions return "" — the caller already rejected those
// in parseDimensionParam, so this branch only fires for rows that
// genuinely have no value for the dim.
func dimensionValue(l logstoreLog, dim string) string {
	return l.Metadata[dim]
}

// ─── /_plugin/runs/:run_id ───────────────────────────────────────────
//
// Routed under `/_plugin/runs/` (subtree); the trailing segments
// dispatch by shape:
//
//   /_plugin/runs/{run_id}                       → runDetail (list)
//   /_plugin/runs/{run_id}/calls/{call_id}       → runCallDetail (body)
//
// Phase 6 will add /:id/state and /:id/kill under the same prefix;
// any other shape returns 404. We don't pull in a router library
// since the dispatch fits in a switch.
func (h *observabilityHandlers) runDetail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	const prefix = "/_plugin/runs/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	if rest == "" {
		http.NotFound(w, r)
		return
	}

	// Two valid shapes: "{run_id}" (list) and "{run_id}/calls/{call_id}".
	parts := strings.Split(rest, "/")
	switch {
	case len(parts) == 1:
		h.runDetailList(w, r, parts[0])
	case len(parts) == 3 && parts[1] == "calls" && parts[2] != "":
		h.runCallDetail(w, r, parts[0], parts[2])
	default:
		// Anything else (e.g. /:id/state, /:id/kill, trailing slash,
		// 4-segment paths) is phase-6 territory or malformed; 404.
		http.NotFound(w, r)
	}
}

// runDetailList serves the paginated call-log envelope for a run.
// Extracted from runDetail's dispatch so the URL parsing stays one
// concern and the data path another.
func (h *observabilityHandlers) runDetailList(w http.ResponseWriter, r *http.Request, runID string) {
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

// runCallDetail serves the full content of one LLM call. Backed by
// Bifrost's GET /api/logs/{id}, which (unlike /api/logs) returns
// the parsed input_history / output_message / params / tools /
// error_details / raw_response fields the list endpoint trims.
//
// The {run_id} segment isn't strictly needed to look up the row
// (Bifrost's primary key is the call id), but we verify it matches
// `metadata.run-id` on the row before returning. Two reasons:
//
//   1. Defence in depth — keeps the call-detail URL self-describing
//      and prevents the SPA from being tricked into enumerating
//      call IDs across runs.
//   2. Symmetric with /_plugin/runs/{id} which is already run-scoped.
//      Operators reasonably expect /runs/A/calls/X and /runs/B/calls/X
//      to give 404 for whichever doesn't actually contain X.
func (h *observabilityHandlers) runCallDetail(w http.ResponseWriter, r *http.Request, runID, callID string) {
	log, err := h.logs.findByID(r.Context(), callID)
	if err != nil {
		writeUpstreamError(w, err, "runs.call_detail")
		return
	}
	if log == nil {
		http.NotFound(w, r)
		return
	}
	if log.Metadata["run-id"] != runID {
		// Either the caller guessed an ID from a different run, or
		// the run-id dim was never stamped. Either way we don't
		// want to leak the row's existence — 404 matches the
		// "row doesn't exist under this run" semantic the URL
		// implies.
		http.NotFound(w, r)
		return
	}

	writeJSON(w, http.StatusOK, CallDetailResponse{
		ID:               log.ID,
		RunID:            runID,
		Timestamp:        log.Timestamp,
		Provider:         log.Provider,
		Model:            log.Model,
		Status:           log.Status,
		Cost:             log.Cost,
		Latency:          log.Latency,
		CustomerID:       log.CustomerID,
		Metadata:         log.Metadata,
		StopReason:      log.StopReason,
		Stream:          log.Stream,
		NumberOfRetries: log.NumberOfRetries,
		FallbackIndex:   log.FallbackIndex,
		TokenUsage:      log.TokenUsage,
		CacheDebug:      log.CacheDebug,
		InputHistory:     log.InputHistory,
		OutputMessage:    log.OutputMessage,
		Params:           log.Params,
		Tools:            log.Tools,
		ErrorDetails:     log.ErrorDetails,
		RawRequest:       log.RawRequest,
		RawResponse:      log.RawResponse,
		ContentSummary:   log.ContentSummary,
	})
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
