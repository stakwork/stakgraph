package adminapi

import (
	"net/http"
	"sort"
	"strings"
	"time"
)

// Phase-7 session drill-down: `/_plugin/sessions/:id` (the paginated
// call log, run-detail's sibling) and `/_plugin/sessions/:id/summary`
// (one-shot totals + span).
//
// A "session" here is the x-bf-dim-session-id dim — Hive's chat or
// workflow session, which spans runs. It is NOT Bifrost's own
// session_id filter: that one aliases parent_request_id (the
// multi-turn linkage Bifrost's UI draws), so the native
// /api/logs/sessions/{id} routes would answer a different question.
// Both endpoints filter on metadata.session-id, exactly like
// /runs/:id filters on metadata.run-id.

// SessionDetailResponse is the envelope for /_plugin/sessions/:id.
// `logs` is one page (?limit=, ?offset=, newest first); `stats` and
// `total_count` cover the whole session.
type SessionDetailResponse struct {
	SessionID  string        `json:"session_id"`
	Logs       []RunLogEntry `json:"logs"`
	Stats      RunStats      `json:"stats"`
	TotalCount int64         `json:"total_count"`
}

// SessionSummaryResponse is the envelope for
// /_plugin/sessions/:id/summary. Timestamps are RFC3339; duration is
// latest − started, 0 for a single-call session.
type SessionSummaryResponse struct {
	SessionID    string   `json:"session_id"`
	UserID       string   `json:"user_id,omitempty"`
	TotalCost    float64  `json:"total_cost"`
	TotalTokens  int64    `json:"total_tokens"`
	RequestCount int64    `json:"request_count"`
	StartedAt    string   `json:"started_at,omitempty"`
	LatestAt     string   `json:"latest_at,omitempty"`
	DurationMS   int64    `json:"duration_ms"`
	Agents       []string `json:"agents"`
	Runs         []string `json:"runs"`
}

// sessions dispatches the /_plugin/sessions/ subtree:
//
//	/_plugin/sessions/{session_id}          → sessionDetail
//	/_plugin/sessions/{session_id}/summary  → sessionSummary
//
// Anything else 404s.
func (h *observabilityHandlers) sessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	rest := strings.TrimPrefix(r.URL.Path, "/_plugin/sessions/")
	parts := strings.Split(rest, "/")
	switch {
	case len(parts) == 1 && parts[0] != "":
		h.sessionDetail(w, r, parts[0])
	case len(parts) == 2 && parts[0] != "" && parts[1] == "summary":
		h.sessionSummary(w, r, parts[0])
	default:
		http.NotFound(w, r)
	}
}

func (h *observabilityHandlers) sessionDetail(w http.ResponseWriter, r *http.Request, sessionID string) {
	limit, offset, ok := parsePagination(w, r)
	if !ok {
		return
	}
	res, err := h.logs.search(r.Context(), searchOpts{
		Metadata: map[string]string{"session-id": sessionID},
		Limit:    limit,
		Offset:   offset,
		SortBy:   "timestamp",
		Order:    "desc",
	})
	if err != nil {
		writeUpstreamError(w, err, "sessions.detail")
		return
	}
	out := SessionDetailResponse{
		SessionID: sessionID,
		Logs:      make([]RunLogEntry, 0, len(res.Logs)),
		Stats: RunStats{
			TotalRequests: res.Stats.TotalRequests,
			TotalCost:     res.Stats.TotalCost,
			TotalTokens:   res.Stats.TotalTokens,
		},
		TotalCount: res.Pagination.TotalCount,
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

// sessionSummary scans every row of the session (no window — a
// session is finite; the 200k-row ceiling still applies) and folds
// it into totals, the time span, and the distinct agents / runs.
func (h *observabilityHandlers) sessionSummary(w http.ResponseWriter, r *http.Request, sessionID string) {
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		Metadata: map[string]string{"session-id": sessionID},
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "sessions.summary")
		return
	}

	out := SessionSummaryResponse{
		SessionID: sessionID,
		Agents:    []string{},
		Runs:      []string{},
	}
	agents := map[string]bool{}
	runs := map[string]bool{}
	for _, l := range logs {
		out.TotalCost += l.Cost
		out.TotalTokens += l.tokens()
		out.RequestCount++
		if out.StartedAt == "" || l.Timestamp < out.StartedAt {
			out.StartedAt = l.Timestamp
		}
		if l.Timestamp > out.LatestAt {
			out.LatestAt = l.Timestamp
		}
		if out.UserID == "" {
			out.UserID = l.Metadata["user-id"]
		}
		if a := l.Metadata["agent-name"]; a != "" && !agents[a] {
			agents[a] = true
			out.Agents = append(out.Agents, a)
		}
		if rid := l.Metadata["run-id"]; rid != "" && !runs[rid] {
			runs[rid] = true
			out.Runs = append(out.Runs, rid)
		}
	}
	sort.Strings(out.Agents)
	sort.Strings(out.Runs)
	if out.StartedAt != "" && out.LatestAt != "" {
		s, err1 := time.Parse(time.RFC3339Nano, out.StartedAt)
		e, err2 := time.Parse(time.RFC3339Nano, out.LatestAt)
		if err1 == nil && err2 == nil {
			out.DurationMS = e.Sub(s).Milliseconds()
		}
	}
	writeJSON(w, http.StatusOK, out)
}
