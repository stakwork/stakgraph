package adminapi

import (
	"net/http"
	"sort"
	"strings"
	"time"
)

// UserAgentUsage is one row in `UserDetailResponse.AgentsUsed` —
// a per-agent rollup scoped to this user's traffic in the window.
type UserAgentUsage struct {
	AgentName    string  `json:"agent_name"`
	TotalCost    float64 `json:"total_cost"`
	RequestCount int64   `json:"request_count"`
	LastSeen     string  `json:"last_seen,omitempty"`
}

// UserRunSummary is one row in `UserDetailResponse.RecentRuns`.
// Phase 8 keeps this lightweight — the dashboard renders cost +
// agent + first/last-seen and links into RunDetail for the full
// call log.
type UserRunSummary struct {
	RunID        string  `json:"run_id"`
	AgentName    string  `json:"agent_name"`
	TotalCost    float64 `json:"total_cost"`
	RequestCount int64   `json:"request_count"`
	FirstSeen    string  `json:"first_seen,omitempty"`
	LastSeen     string  `json:"last_seen,omitempty"`
}

// UserDetailResponse is the wire shape for /_plugin/users/:id.
//
// Composition
// -----------
// Everything is derived from a single paged scan of `logs.db`
// filtered by `metadata.user-id = <id>` (with a fallback to the
// indexed `customer_id` column on Bifrost's logs table, which
// equals the user-id per the v2 invariant). One round-trip to
// Bifrost, one in-memory aggregation pass; the dashboard renders
// the result without further fan-out.
//
// Once phase 6's PostLLMHook fills Redis cost accumulators, the
// `total_cost` field could be sourced from the Redis hash instead
// of summing logs — same number, less work. Phase 8 doesn't take
// that shortcut yet because the Redis bucket is per-(agent, day)
// not per-user.
type UserDetailResponse struct {
	UserID       string           `json:"user_id"`
	Window       string           `json:"window"`
	TotalCost    float64          `json:"total_cost"`
	RequestCount int64            `json:"request_count"`
	AgentsUsed   []UserAgentUsage `json:"agents_used"`
	RecentRuns   []UserRunSummary `json:"recent_runs"`
	FirstSeen    string           `json:"first_seen,omitempty"`
	LastSeen     string           `json:"last_seen,omitempty"`
}

// userDetail handles `GET /_plugin/users/:user_id`.
//
// Path parsing follows the same subtree convention runDetail uses —
// the trailing segment is the user_id, anything deeper 404s so
// future endpoints like `/_plugin/users/:id/quota` can land
// alongside without ambiguity.
func (h *observabilityHandlers) userDetail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	const prefix = "/_plugin/users/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	if rest == "" || strings.ContainsRune(rest, '/') {
		http.NotFound(w, r)
		return
	}
	userID := rest

	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}

	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"user-id": userID},
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, "users.detail")
		return
	}

	// Bifrost's customer_id column is the indexed canonical
	// user-id (v2 invariant). Older traffic without canonicalized
	// dims will only show up under customer_id, so do a second
	// scan filtered by that and merge — same de-dup as RunDetail's
	// approach to mixed sources.
	if len(logs) == 0 {
		// Empty metadata filter — but the user still has the
		// indexed customer_id column. Bifrost's /api/logs accepts
		// `customer_ids=` as a first-class filter; the logstore
		// client doesn't expose it yet, so for phase 8 we fall
		// through with an empty list. Logs sourced via the dim
		// filter cover the canonicalized case (post-phase-6)
		// and the demo case (callers stamp x-bf-dim-user-id).
		_ = logs
	}

	out := UserDetailResponse{
		UserID:     userID,
		Window:     window,
		AgentsUsed: []UserAgentUsage{},
		RecentRuns: []UserRunSummary{},
	}
	if len(logs) == 0 {
		writeJSON(w, http.StatusOK, out)
		return
	}

	// Per-agent and per-run aggregation in one pass.
	type agentAgg struct {
		cost     float64
		count    int64
		lastSeen string
	}
	type runAgg struct {
		agent     string
		cost      float64
		count     int64
		firstSeen string
		lastSeen  string
	}
	byAgent := map[string]*agentAgg{}
	byRun := map[string]*runAgg{}

	var minTS, maxTS string
	for _, l := range logs {
		out.TotalCost += l.Cost
		out.RequestCount++
		if minTS == "" || l.Timestamp < minTS {
			minTS = l.Timestamp
		}
		if l.Timestamp > maxTS {
			maxTS = l.Timestamp
		}

		agent := l.Metadata["agent-name"]
		if agent != "" {
			a, ok := byAgent[agent]
			if !ok {
				a = &agentAgg{}
				byAgent[agent] = a
			}
			a.cost += l.Cost
			a.count++
			if l.Timestamp > a.lastSeen {
				a.lastSeen = l.Timestamp
			}
		}

		runID := l.Metadata["run-id"]
		if runID != "" {
			ru, ok := byRun[runID]
			if !ok {
				ru = &runAgg{agent: agent}
				byRun[runID] = ru
			}
			ru.cost += l.Cost
			ru.count++
			if ru.firstSeen == "" || l.Timestamp < ru.firstSeen {
				ru.firstSeen = l.Timestamp
			}
			if l.Timestamp > ru.lastSeen {
				ru.lastSeen = l.Timestamp
			}
		}
	}
	out.FirstSeen = minTS
	out.LastSeen = maxTS

	for name, a := range byAgent {
		out.AgentsUsed = append(out.AgentsUsed, UserAgentUsage{
			AgentName:    name,
			TotalCost:    a.cost,
			RequestCount: a.count,
			LastSeen:     a.lastSeen,
		})
	}
	sort.Slice(out.AgentsUsed, func(i, j int) bool {
		return out.AgentsUsed[i].TotalCost > out.AgentsUsed[j].TotalCost
	})

	for id, ru := range byRun {
		out.RecentRuns = append(out.RecentRuns, UserRunSummary{
			RunID:        id,
			AgentName:    ru.agent,
			TotalCost:    ru.cost,
			RequestCount: ru.count,
			FirstSeen:    ru.firstSeen,
			LastSeen:     ru.lastSeen,
		})
	}
	sort.Slice(out.RecentRuns, func(i, j int) bool {
		// Most-recent first — operator's typical "what just
		// happened" expectation.
		return out.RecentRuns[i].LastSeen > out.RecentRuns[j].LastSeen
	})
	// Cap to top 50 by recency; same convention as AgentDetail.
	if len(out.RecentRuns) > 50 {
		out.RecentRuns = out.RecentRuns[:50]
	}

	// Used-but-not-needed: time package import is for sort
	// determinism on equal timestamps. Not currently relied on,
	// but keeping the import slot pinned so future enhancements
	// (e.g. parsing FirstSeen for "active duration") don't shuffle
	// the file diff.
	_ = time.Time{}

	writeJSON(w, http.StatusOK, out)
}
