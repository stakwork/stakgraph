package adminapi

import (
	"errors"
	"net/http"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Phase-6 hot-state routes: kill switches and Redis snapshots for
// runs and agents. Thin wrappers over auth.KillRun / auth.GetRunState
// et al. — every decision about keys, TTLs and what "killed" means
// lives in gateway/internal/auth/kill.go; this file only parses the
// URL, picks by method, and shapes the JSON.
//
//	POST   /_plugin/runs/:id/kill      → 200 KillRunResponse
//	DELETE /_plugin/runs/:id/kill      → 204
//	GET    /_plugin/runs/:id/state     → 200 RunStateResponse
//	POST   /_plugin/agents/:name/kill  → 200 KillAgentResponse
//	DELETE /_plugin/agents/:name/kill  → 204
//	GET    /_plugin/agents/:name/state → 200 AgentStateResponse  (?window=1d)
//
// All cookie-or-bearer: the operator dashboard drives these from a
// session cookie (with the CSRF header on mutations); Hive uses the
// bearer. Redis unconfigured ⇒ 503, same as the session store.

// KillRunResponse is the wire shape for POST /_plugin/runs/:id/kill.
type KillRunResponse struct {
	RunID    string `json:"run_id"`
	KilledAt string `json:"killed_at"` // RFC3339 UTC
}

// KillAgentResponse is the wire shape for POST /_plugin/agents/:name/kill.
type KillAgentResponse struct {
	AgentName string `json:"agent_name"`
	KilledAt  string `json:"killed_at"` // RFC3339 UTC
}

// RunStateResponse is the wire shape for GET /_plugin/runs/:id/state —
// the run's live phase-6 accumulators. A run that has never made a
// call reads as all-zero with ttl_seconds = -2 (no key), not 404.
//
// The cap fields come from meta:run:<id>, which the accumulator
// stamps from the verified macaroon chain. `null` means the run has
// no state yet or the layer declared no cap — either way there is
// nothing to draw a meter against. `ancestors` walks the parent
// links outward (nearest parent first), one entry per budgeted run
// above this one, so the UI can render a meter per layer.
type RunStateResponse struct {
	RunID   string   `json:"run_id"`
	CostUSD float64  `json:"cost_usd"`
	Steps   int64    `json:"steps"`
	Tools   []string `json:"tools"` // last 10 tool names, most recent first
	Killed  bool     `json:"killed"`
	// TTLSeconds is the remaining lifetime of the cost accumulator:
	// -2 when the run has no state yet, -1 when it has no expiry.
	TTLSeconds int64 `json:"ttl_seconds"`

	MaxCostUSD *float64 `json:"max_cost_usd"`
	MaxSteps   *int64   `json:"max_steps"`
	// Exp is the macaroon layer's expiry (RFC3339); empty when the
	// run has no meta yet.
	Exp string `json:"exp,omitempty"`
	// AgentName / UserID are recorded by the run's own calls. An
	// ancestor that has only been seen through a child's chain has
	// neither.
	AgentName string             `json:"agent_name,omitempty"`
	UserID    string             `json:"user_id,omitempty"`
	Ancestors []RunAncestorState `json:"ancestors"`
}

// RunAncestorState is one budgeted run above the requested one in
// its macaroon chain: the same accumulators and caps, minus the tool
// history. Nearest parent first.
type RunAncestorState struct {
	RunID      string   `json:"run_id"`
	CostUSD    float64  `json:"cost_usd"`
	Steps      int64    `json:"steps"`
	Killed     bool     `json:"killed"`
	MaxCostUSD *float64 `json:"max_cost_usd"`
	MaxSteps   *int64   `json:"max_steps"`
}

// maxAncestorHops bounds the parent walk. Macaroon chains are a
// handful of layers deep in practice; the bound guards against a
// corrupted parent link forming a cycle.
const maxAncestorHops = 8

// AgentStateResponse is the wire shape for GET /_plugin/agents/:name/state.
type AgentStateResponse struct {
	AgentName       string  `json:"agent_name"`
	Window          string  `json:"window"`
	BucketKey       string  `json:"bucket_key"`
	CurrentSpendUSD float64 `json:"current_spend_usd"`
	// ConfiguredCapUSD is null when the agent has no agent_budgets
	// entry; then `window` is informational (?window= or "1d").
	ConfiguredCapUSD *float64 `json:"configured_cap_usd"`
	Killed           bool     `json:"killed"`
}

type hotStateHandlers struct{}

func newHotStateHandlers() *hotStateHandlers { return &hotStateHandlers{} }

func (h *hotStateHandlers) runKill(w http.ResponseWriter, r *http.Request, runID string) {
	switch r.Method {
	case http.MethodPost:
		if err := auth.KillRun(r.Context(), runID); err != nil {
			writeHotStateErr(w, err, "runs.kill")
			return
		}
		pluginlog.Logf("adminapi: run killed run_id=%s", runID)
		writeJSON(w, http.StatusOK, KillRunResponse{
			RunID:    runID,
			KilledAt: time.Now().UTC().Format(time.RFC3339),
		})
	case http.MethodDelete:
		if err := auth.UnkillRun(r.Context(), runID); err != nil {
			writeHotStateErr(w, err, "runs.unkill")
			return
		}
		pluginlog.Logf("adminapi: run unkilled run_id=%s", runID)
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodPost, http.MethodDelete)
	}
}

func (h *hotStateHandlers) runState(w http.ResponseWriter, r *http.Request, runID string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	st, err := auth.GetRunState(r.Context(), runID)
	if err != nil {
		writeHotStateErr(w, err, "runs.state")
		return
	}
	out := RunStateResponse{
		RunID:      st.RunID,
		CostUSD:    st.CostUSD,
		Steps:      st.Steps,
		Tools:      st.Tools,
		Killed:     st.Killed,
		TTLSeconds: st.TTLSeconds,
		MaxCostUSD: capUSD(st),
		MaxSteps:   capSteps(st),
		Exp:        st.Exp,
		AgentName:  st.AgentName,
		UserID:     st.UserID,
		Ancestors:  []RunAncestorState{},
	}

	// Walk the parent links. A missing ancestor (its keys expired
	// before the child's) ends the walk rather than erroring: the
	// meters we can draw are still worth returning.
	seen := map[string]bool{runID: true}
	for parent := st.Parent; parent != "" && !seen[parent] && len(out.Ancestors) < maxAncestorHops; {
		seen[parent] = true
		ps, err := auth.GetRunState(r.Context(), parent)
		if err != nil {
			writeHotStateErr(w, err, "runs.state.ancestor")
			return
		}
		if !ps.HasMeta {
			break
		}
		out.Ancestors = append(out.Ancestors, RunAncestorState{
			RunID:      ps.RunID,
			CostUSD:    ps.CostUSD,
			Steps:      ps.Steps,
			Killed:     ps.Killed,
			MaxCostUSD: capUSD(ps),
			MaxSteps:   capSteps(ps),
		})
		parent = ps.Parent
	}
	writeJSON(w, http.StatusOK, out)
}

// capUSD / capSteps turn the accumulator's "0 = no cap, and also 0 =
// unknown" into the wire's explicit null.
func capUSD(st auth.RunState) *float64 {
	if !st.HasMeta || st.MaxCostUSD <= 0 {
		return nil
	}
	v := st.MaxCostUSD
	return &v
}

func capSteps(st auth.RunState) *int64 {
	if !st.HasMeta || st.MaxSteps <= 0 {
		return nil
	}
	v := st.MaxSteps
	return &v
}

func (h *hotStateHandlers) agentKill(w http.ResponseWriter, r *http.Request, name string) {
	switch r.Method {
	case http.MethodPost:
		if err := auth.KillAgent(r.Context(), name); err != nil {
			writeHotStateErr(w, err, "agents.kill")
			return
		}
		pluginlog.Logf("adminapi: agent killed agent=%s", name)
		writeJSON(w, http.StatusOK, KillAgentResponse{
			AgentName: name,
			KilledAt:  time.Now().UTC().Format(time.RFC3339),
		})
	case http.MethodDelete:
		if err := auth.UnkillAgent(r.Context(), name); err != nil {
			writeHotStateErr(w, err, "agents.unkill")
			return
		}
		pluginlog.Logf("adminapi: agent unkilled agent=%s", name)
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodPost, http.MethodDelete)
	}
}

func (h *hotStateHandlers) agentState(w http.ResponseWriter, r *http.Request, name string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	st, err := auth.GetAgentState(r.Context(), name, r.URL.Query().Get("window"), time.Now().UTC())
	if err != nil {
		writeHotStateErr(w, err, "agents.state")
		return
	}
	writeJSON(w, http.StatusOK, AgentStateResponse{
		AgentName:        st.AgentName,
		Window:           st.Window,
		BucketKey:        st.BucketKey,
		CurrentSpendUSD:  st.CurrentSpendUSD,
		ConfiguredCapUSD: st.ConfiguredCapUSD,
		Killed:           st.Killed,
	})
}

// writeHotStateErr maps auth-package errors onto HTTP: Redis
// unconfigured ⇒ 503 (operator can retry once the link is up),
// anything else ⇒ 400. The auth helpers only return validation
// errors and Redis errors; a Redis I/O failure also lands on 400
// here rather than 500 because the message is safe to show and
// retrying is the right move either way.
func writeHotStateErr(w http.ResponseWriter, err error, op string) {
	if errors.Is(err, auth.ErrRedisUnavailable) {
		http.Error(w, "redis not configured", http.StatusServiceUnavailable)
		return
	}
	pluginlog.Warnf("adminapi: %s: %v", op, err)
	http.Error(w, err.Error(), http.StatusBadRequest)
}
