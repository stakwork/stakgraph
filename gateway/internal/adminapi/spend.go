package adminapi

import (
	"net/http"
	"sort"
)

// Phase-7 spend rollups that phase 8 didn't ship: by-session,
// by-model, and the single-agent total. Same strategy as the
// by-agent / by-user handlers in observability.go — page the window
// out of Bifrost's /api/logs and bucket in Go, because the dims live
// in `metadata` and Bifrost's native group-bys are column-bound.
// Same 200k-row ceiling, same "rows missing the dim are excluded"
// policy.

// SessionSpend is one row of /_plugin/spend/by-session. A session is
// whatever the caller stamped on x-bf-dim-session-id (Hive's chat /
// workflow session); it spans runs, so `run_count` and the first /
// last timestamps are included for a Sessions list page.
type SessionSpend struct {
	SessionID    string  `json:"session_id"`
	UserID       string  `json:"user_id"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
	RunCount     int64   `json:"run_count"`
	FirstSeen    string  `json:"first_seen,omitempty"`
	LastSeen     string  `json:"last_seen,omitempty"`
}

// SpendBySessionResponse is the envelope for /_plugin/spend/by-session.
type SpendBySessionResponse struct {
	Window  string         `json:"window"`
	Results []SessionSpend `json:"results"`
}

// ModelSpend is one row of /_plugin/spend/by-model, keyed on
// (provider, model) — the same model name can be served by two
// providers at different prices.
type ModelSpend struct {
	Model        string  `json:"model"`
	Provider     string  `json:"provider"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// SpendByModelResponse is the envelope for /_plugin/spend/by-model.
type SpendByModelResponse struct {
	Window  string       `json:"window"`
	Results []ModelSpend `json:"results"`
}

// AgentSpendResponse is the envelope for /_plugin/agents/:name/spend —
// one agent's totals over the window.
type AgentSpendResponse struct {
	AgentName    string  `json:"agent_name"`
	Window       string  `json:"window"`
	TotalCost    float64 `json:"total_cost"`
	TotalTokens  int64   `json:"total_tokens"`
	RequestCount int64   `json:"request_count"`
}

// windowedLogs is the shared front half of every rollup: parse
// ?window=, page the matching rows (plus any ?user_id= / ?agent_name=
// scoping) out of Bifrost, and map upstream failure to 502. When ok
// is false the response has been written.
func (h *observabilityHandlers) windowedLogs(w http.ResponseWriter, r *http.Request, where string) (string, []logstoreLog, bool) {
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return "", nil, false
	}
	logs, err := h.logs.searchAll(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  metadataFilterFromQuery(r),
	}, 1000, 200_000)
	if err != nil {
		writeUpstreamError(w, err, where)
		return "", nil, false
	}
	return window, logs, true
}

// ─── /_plugin/spend/by-session ───────────────────────────────────────

func (h *observabilityHandlers) spendBySession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, logs, ok := h.windowedLogs(w, r, "spend.by_session")
	if !ok {
		return
	}

	type agg struct {
		user      string
		cost      float64
		tokens    int64
		count     int64
		runs      map[string]bool
		firstSeen string
		lastSeen  string
	}
	by := map[string]*agg{}
	for _, l := range logs {
		sid := l.Metadata["session-id"]
		if sid == "" {
			continue
		}
		a, ok := by[sid]
		if !ok {
			a = &agg{user: l.Metadata["user-id"], runs: map[string]bool{}}
			by[sid] = a
		}
		a.cost += l.Cost
		a.tokens += l.tokens()
		a.count++
		if rid := l.Metadata["run-id"]; rid != "" {
			a.runs[rid] = true
		}
		if a.firstSeen == "" || l.Timestamp < a.firstSeen {
			a.firstSeen = l.Timestamp
		}
		if l.Timestamp > a.lastSeen {
			a.lastSeen = l.Timestamp
		}
	}

	out := SpendBySessionResponse{
		Window:  window,
		Results: make([]SessionSpend, 0, len(by)),
	}
	for sid, a := range by {
		out.Results = append(out.Results, SessionSpend{
			SessionID:    sid,
			UserID:       a.user,
			TotalCost:    a.cost,
			TotalTokens:  a.tokens,
			RequestCount: a.count,
			RunCount:     int64(len(a.runs)),
			FirstSeen:    a.firstSeen,
			LastSeen:     a.lastSeen,
		})
	}
	sort.Slice(out.Results, func(i, j int) bool {
		if out.Results[i].TotalCost != out.Results[j].TotalCost {
			return out.Results[i].TotalCost > out.Results[j].TotalCost
		}
		return out.Results[i].SessionID < out.Results[j].SessionID
	})
	writeJSON(w, http.StatusOK, out)
}

// ─── /_plugin/spend/by-model ─────────────────────────────────────────
//
// Provider and model are first-class columns on every Bifrost row,
// so unlike the dim rollups nothing is excluded here — an unattributed
// curl still shows up under the model it hit. That makes by-model the
// one rollup whose totals reconcile with Bifrost's own dashboard.

func (h *observabilityHandlers) spendByModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, logs, ok := h.windowedLogs(w, r, "spend.by_model")
	if !ok {
		return
	}

	type agg struct {
		model    string
		provider string
		cost     float64
		tokens   int64
		count    int64
	}
	by := map[string]*agg{}
	for _, l := range logs {
		prov := l.Provider
		if prov == "" {
			prov = "unknown"
		}
		model := l.Model
		if model == "" {
			model = "unknown"
		}
		k := prov + "\x00" + model
		a, ok := by[k]
		if !ok {
			a = &agg{model: model, provider: prov}
			by[k] = a
		}
		a.cost += l.Cost
		a.tokens += l.tokens()
		a.count++
	}

	out := SpendByModelResponse{
		Window:  window,
		Results: make([]ModelSpend, 0, len(by)),
	}
	for _, a := range by {
		out.Results = append(out.Results, ModelSpend{
			Model:        a.model,
			Provider:     a.provider,
			TotalCost:    a.cost,
			TotalTokens:  a.tokens,
			RequestCount: a.count,
		})
	}
	sort.Slice(out.Results, func(i, j int) bool {
		if out.Results[i].TotalCost != out.Results[j].TotalCost {
			return out.Results[i].TotalCost > out.Results[j].TotalCost
		}
		if out.Results[i].Provider != out.Results[j].Provider {
			return out.Results[i].Provider < out.Results[j].Provider
		}
		return out.Results[i].Model < out.Results[j].Model
	})
	writeJSON(w, http.StatusOK, out)
}

// ─── /_plugin/agents/:name/spend ─────────────────────────────────────
//
// One agent's totals. Bifrost computes SearchStats over the whole
// filtered set regardless of the page, so a single limit=1 call
// returns the numbers without paging rows through the plugin.

func (h *observabilityHandlers) agentSpend(w http.ResponseWriter, r *http.Request, name string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	window, start, end, ok := parseWindow(w, r)
	if !ok {
		return
	}
	res, err := h.logs.search(r.Context(), searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"agent-name": name},
		Limit:     1,
	})
	if err != nil {
		writeUpstreamError(w, err, "agents.spend")
		return
	}
	writeJSON(w, http.StatusOK, AgentSpendResponse{
		AgentName:    name,
		Window:       window,
		TotalCost:    res.Stats.TotalCost,
		TotalTokens:  res.Stats.TotalTokens,
		RequestCount: res.Stats.TotalRequests,
	})
}
