package adminapi

import (
	"net/http"
	"sort"
	"time"
)

// ─── /_plugin/agents/:name/runs ──────────────────────────────────────
//
// The agent's runs in the window, one row per run-id, most recent
// activity first. Backs the "Recent runs" table on the AgentDetail
// page. Before this endpoint the page derived that table from
// `/histogram/cost?dimension=run-id`, which is not agent-scoped and
// orders by spend — so an operator selecting canvas-agent saw every
// agent's runs, ranked by cost, with the run they had just fired
// nowhere near the top.
//
// Same strategy as the other rollups: page the window out of
// Bifrost's /api/logs filtered by `metadata.agent-name`, then group
// by `metadata.run-id` in Go. Rows without a run-id are dropped (a
// bare call outside any run has nothing to link to). Same 200k-row
// ceiling as the rest of observability.go.

// AgentRunSummary is one row of /_plugin/agents/:name/runs.
type AgentRunSummary struct {
	RunID string `json:"run_id"`
	// UserID is `metadata.user-id` from the run's first row that
	// carries one — the same key the People pages are keyed on, so
	// the dashboard can link straight to /people/:id. Empty when
	// no row was stamped.
	UserID string `json:"user_id,omitempty"`
	// Models the run called, most-used first (ties by name). A run
	// usually has one; a "+N" affordance in the UI covers the rest.
	Models       []string `json:"models"`
	TotalCost    float64  `json:"total_cost"`
	TotalTokens  int64    `json:"total_tokens"`
	RequestCount int64    `json:"request_count"`
	FirstSeen    string   `json:"first_seen,omitempty"`
	LastSeen     string   `json:"last_seen,omitempty"`
}

// AgentRunsResponse is the envelope for /_plugin/agents/:name/runs.
// `total` is the run count in the window before ?limit=/?offset=
// paging, so the UI can say "showing 50 of 120".
type AgentRunsResponse struct {
	AgentName string            `json:"agent_name"`
	Window    string            `json:"window"`
	Total     int               `json:"total"`
	Runs      []AgentRunSummary `json:"runs"`
}

func (h *observabilityHandlers) agentRuns(w http.ResponseWriter, r *http.Request, name string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}
	limit, offset, ok := parsePagination(w, r)
	if !ok {
		return
	}
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"agent-name": name},
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "agents.runs")
		return
	}

	runs := summarizeRuns(logs)
	total := len(runs)
	if offset > len(runs) {
		offset = len(runs)
	}
	runs = runs[offset:]
	if len(runs) > limit {
		runs = runs[:limit]
	}
	writeJSON(w, http.StatusOK, AgentRunsResponse{
		AgentName: name,
		Window:    window,
		Total:     total,
		Runs:      runs,
	})
}

// summarizeRuns groups log rows by `metadata.run-id` and returns one
// summary per run, sorted by last activity, newest first (ties by
// run-id so the order is stable across polls). Rows with no run-id
// are skipped.
func summarizeRuns(logs []logstoreLog) []AgentRunSummary {
	type agg struct {
		user      string
		models    map[string]int64
		cost      float64
		tokens    int64
		count     int64
		first     time.Time
		firstSeen string
		last      time.Time
		lastSeen  string
	}
	byRun := map[string]*agg{}
	for _, l := range logs {
		runID := l.Metadata["run-id"]
		if runID == "" {
			continue
		}
		a, ok := byRun[runID]
		if !ok {
			a = &agg{models: map[string]int64{}}
			byRun[runID] = a
		}
		if a.user == "" {
			a.user = l.Metadata["user-id"]
		}
		if l.Model != "" {
			a.models[l.Model]++
		}
		a.cost += l.Cost
		a.tokens += l.tokens()
		a.count++
		// Compare as times, not strings: RFC3339Nano trims trailing
		// zeros, so "…:00Z" sorts after "…:00.5Z" lexicographically.
		ts := parseLogTimestamp(l.Timestamp)
		if a.firstSeen == "" || ts.Before(a.first) {
			a.first, a.firstSeen = ts, l.Timestamp
		}
		if a.lastSeen == "" || ts.After(a.last) {
			a.last, a.lastSeen = ts, l.Timestamp
		}
	}

	out := make([]AgentRunSummary, 0, len(byRun))
	for id, a := range byRun {
		models := make([]string, 0, len(a.models))
		for m := range a.models {
			models = append(models, m)
		}
		sort.Slice(models, func(i, j int) bool {
			if a.models[models[i]] != a.models[models[j]] {
				return a.models[models[i]] > a.models[models[j]]
			}
			return models[i] < models[j]
		})
		out = append(out, AgentRunSummary{
			RunID:        id,
			UserID:       a.user,
			Models:       models,
			TotalCost:    a.cost,
			TotalTokens:  a.tokens,
			RequestCount: a.count,
			FirstSeen:    a.firstSeen,
			LastSeen:     a.lastSeen,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		ti, tj := parseLogTimestamp(out[i].LastSeen), parseLogTimestamp(out[j].LastSeen)
		if !ti.Equal(tj) {
			return ti.After(tj)
		}
		return out[i].RunID < out[j].RunID
	})
	return out
}

// parseLogTimestamp reads a Bifrost row timestamp. Unparseable
// values sort as the zero time — oldest — rather than being dropped,
// so a malformed row still counts toward the run's totals.
func parseLogTimestamp(s string) time.Time {
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Time{}
	}
	return t
}
