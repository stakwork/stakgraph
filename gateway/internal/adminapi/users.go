package adminapi

import (
	"errors"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
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

// UserSpendResponse is the envelope for /_plugin/users/:id/spend —
// the user's totals over the window, straight from Bifrost's
// SearchStats (one limit=1 call, no row paging).
type UserSpendResponse struct {
	UserID       string  `json:"user_id"`
	Window       string  `json:"window"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// UserQuotaResponse is the envelope for /_plugin/users/:id/quota:
// the user's Bifrost Customer budget (Hive's reconciler provisions
// one per workspace × user; llm-governance-v2.md "Hive as credential
// broker") blended with the runs the accumulator has indexed for
// them in Redis.
//
// Two independent degradations, both non-fatal:
//   - no Customer for this id ⇒ customer_found=false, budget fields
//     null (pre-reconciler traffic, or an id that was only ever a
//     dim value).
//   - Redis unavailable ⇒ redis_available=false, inflight_runs null.
//
// The budget window is whatever Bifrost's reset_duration says
// (Bifrost duration vocabulary); phase 7's sketch called it
// "daily", but the reconciler decides that, not the plugin.
type UserQuotaResponse struct {
	UserID        string   `json:"user_id"`
	CustomerFound bool     `json:"customer_found"`
	BudgetUSD     *float64 `json:"budget_usd"`
	BudgetWindow  string   `json:"budget_window,omitempty"`
	SpentUSD      float64  `json:"spent_usd"`
	RemainingUSD  *float64 `json:"remaining_usd"`
	// BudgetLastReset is Bifrost's last_reset (RFC3339); the next
	// reset is last_reset + budget_window.
	BudgetLastReset string        `json:"budget_last_reset,omitempty"`
	RedisAvailable  bool          `json:"redis_available"`
	InflightRuns    []InflightRun `json:"inflight_runs"`
}

// InflightRun is one run in a user's quota view: the live
// accumulators and caps from Redis, as /runs/:id/state would report
// them. "In flight" means the macaroon layer hasn't expired; a run
// that finished early still lists until its exp passes.
type InflightRun struct {
	RunID      string   `json:"run_id"`
	AgentName  string   `json:"agent_name,omitempty"`
	CostUSD    float64  `json:"cost_usd"`
	Steps      int64    `json:"steps"`
	MaxCostUSD *float64 `json:"max_cost_usd"`
	MaxSteps   *int64   `json:"max_steps"`
	Exp        string   `json:"exp,omitempty"`
	Killed     bool     `json:"killed"`
}

// maxInflightRuns caps the quota view's run list. The index is
// pruned by exp on read, so this only matters for a user driving an
// unusual number of concurrent runs.
const maxInflightRuns = 50

// userDetail dispatches the /_plugin/users/ subtree:
//
//	/_plugin/users/{user_id}         → the phase-8 rollup below
//	/_plugin/users/{user_id}/spend   → userSpend
//	/_plugin/users/{user_id}/quota   → userQuota
//
// Anything else 404s.
func (h *observabilityHandlers) userDetail(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	const prefix = "/_plugin/users/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	parts := strings.Split(rest, "/")
	switch {
	case len(parts) == 1 && parts[0] != "":
		h.userRollup(w, r, parts[0])
	case len(parts) == 2 && parts[0] != "" && parts[1] == "spend":
		h.userSpend(w, r, parts[0])
	case len(parts) == 2 && parts[0] != "" && parts[1] == "quota":
		h.userQuota(w, r, parts[0])
	default:
		http.NotFound(w, r)
	}
}

// userSpend handles `GET /_plugin/users/:user_id/spend`. Filters on
// metadata.user-id, not Bifrost's customer_id column — see the
// source-of-truth note on spendByUser.
func (h *observabilityHandlers) userSpend(w http.ResponseWriter, r *http.Request, userID string) {
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}
	res, err := h.logs.search(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"user-id": userID},
		Limit:     1,
	})
	if err != nil {
		writeUpstreamError(w, err, "users.spend")
		return
	}
	writeJSON(w, http.StatusOK, UserSpendResponse{
		UserID:       userID,
		Window:       window,
		TotalCost:    res.Stats.TotalCost,
		TotalTokens:  res.Stats.TotalTokens,
		RequestCount: res.Stats.TotalRequests,
	})
}

// userQuota handles `GET /_plugin/users/:user_id/quota`.
func (h *observabilityHandlers) userQuota(w http.ResponseWriter, r *http.Request, userID string) {
	cust, err := h.logs.customer(r.Context(), userID)
	if err != nil {
		writeUpstreamError(w, err, "users.quota")
		return
	}
	out := UserQuotaResponse{UserID: userID}
	if cust != nil {
		out.CustomerFound = true
		// Hive provisions one budget per customer; if there are
		// several, the first is the one the reconciler wrote.
		if len(cust.Budgets) > 0 {
			b := cust.Budgets[0]
			cap := b.MaxLimit
			remaining := b.MaxLimit - b.CurrentUsage
			if remaining < 0 {
				remaining = 0
			}
			out.BudgetUSD = &cap
			out.BudgetWindow = b.ResetDuration
			out.SpentUSD = b.CurrentUsage
			out.RemainingUSD = &remaining
			out.BudgetLastReset = b.LastReset
		}
	}

	// Live portion. Any Redis failure degrades to "unavailable"
	// rather than failing the budget half — the spec's contract for
	// this endpoint is "return the logs-derived portion".
	now := time.Now().UTC()
	ids, err := auth.ListUserRuns(r.Context(), userID, now, maxInflightRuns)
	if err != nil {
		if !errors.Is(err, auth.ErrRedisUnavailable) {
			pluginlog.Warnf("adminapi: users.quota: run index for %s: %v", userID, err)
		}
		writeJSON(w, http.StatusOK, out)
		return
	}
	out.RedisAvailable = true
	out.InflightRuns = make([]InflightRun, 0, len(ids))
	for _, id := range ids {
		st, err := auth.GetRunState(r.Context(), id)
		if err != nil {
			pluginlog.Warnf("adminapi: users.quota: run state %s: %v", id, err)
			continue
		}
		out.InflightRuns = append(out.InflightRuns, InflightRun{
			RunID:      st.RunID,
			AgentName:  st.AgentName,
			CostUSD:    st.CostUSD,
			Steps:      st.Steps,
			MaxCostUSD: capUSD(st),
			MaxSteps:   capSteps(st),
			Exp:        st.Exp,
			Killed:     st.Killed,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// userRollup is the phase-8 `GET /_plugin/users/:user_id` body.
func (h *observabilityHandlers) userRollup(w http.ResponseWriter, r *http.Request, userID string) {
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

	writeJSON(w, http.StatusOK, out)
}
