package auth

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/duration"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Kill-switch and hot-state helpers — the phase-6 admin primitives
// (`KillRun`, `UnkillRun`, `KillAgent`, `UnkillAgent`, `RunState`,
// `AgentState`). Pure Redis logic, no HTTP; the routes that expose
// them live in gateway/internal/adminapi (runs.go / agents.go).
//
// A kill does not invalidate the macaroon — it's an operator saying
// "stop this now". The hot path (CheckRevocations) turns a set kill
// key into a 402 `run_killed` / `agent_killed` on the next LLM call
// from the run (or any descendant) / from the agent. Both keys carry
// fixed TTLs per phase-6 "TTL policy": a kill is hot operational
// state, not configuration. To stop an agent permanently, set its
// agent_budget cap instead.

const (
	// killRunTTL: if the run isn't dead within the hour, something
	// else has gone wrong.
	killRunTTL = 1 * time.Hour
	// killAgentTTL: re-set if the kill needs to persist longer.
	killAgentTTL = 24 * time.Hour
)

// RunState is a snapshot of one run's phase-6 Redis accumulators.
type RunState struct {
	RunID      string
	CostUSD    float64  // HGET cost:run:<id> total (0 when absent)
	Steps      int64    // HGET steps:run:<id> total (0 when absent)
	Tools      []string // LRANGE tools:run:<id> 0 9, most recent first
	Killed     bool     // EXISTS kill:<id>
	TTLSeconds int64    // TTL cost:run:<id>; -2 when the key is absent, -1 when no expiry
}

// AgentState is a snapshot of one agent's current-bucket spend and
// kill state. ConfiguredCapUSD is nil when the agent has no entry in
// agent_budgets.
type AgentState struct {
	AgentName        string
	Window           string
	BucketKey        string
	CurrentSpendUSD  float64
	ConfiguredCapUSD *float64
	Killed           bool
}

// KillRun sets bifrost:kill:<run_id> = "1" EX 1h. Idempotent; a
// re-kill refreshes the TTL.
func KillRun(ctx context.Context, runID string) error {
	if err := validateKillID("run_id", runID); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Set(octx, redisclient.Key(killRunPrefix+runID), "1", killRunTTL).Err()
}

// UnkillRun deletes the per-run kill key. No-op if absent.
func UnkillRun(ctx context.Context, runID string) error {
	if err := validateKillID("run_id", runID); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Del(octx, redisclient.Key(killRunPrefix+runID)).Err()
}

// KillAgent sets bifrost:kill:agent:<name> = "1" EX 24h.
func KillAgent(ctx context.Context, name string) error {
	if err := validateKillID("agent_name", name); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Set(octx, redisclient.Key(killAgentPrefix+name), "1", killAgentTTL).Err()
}

// UnkillAgent deletes the per-agent kill key. No-op if absent.
func UnkillAgent(ctx context.Context, name string) error {
	if err := validateKillID("agent_name", name); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Del(octx, redisclient.Key(killAgentPrefix+name)).Err()
}

// GetRunState reads the run's accumulators + kill flag in one
// pipelined round-trip. Absent keys read as zero / empty / not
// killed — a run that has never made a call is a valid, empty state,
// not an error.
func GetRunState(ctx context.Context, runID string) (RunState, error) {
	st := RunState{RunID: runID, Tools: []string{}}
	if err := validateKillID("run_id", runID); err != nil {
		return st, err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return st, ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()

	costKey := redisclient.Key(costRunPrefix + runID)
	pipe := rdb.Pipeline()
	costCmd := pipe.HGet(octx, costKey, "total")
	stepsCmd := pipe.HGet(octx, redisclient.Key(stepsRunPrefix+runID), "total")
	toolsCmd := pipe.LRange(octx, redisclient.Key(toolsRunPrefix+runID), 0, toolHistoryLen-1)
	killCmd := pipe.Exists(octx, redisclient.Key(killRunPrefix+runID))
	ttlCmd := pipe.TTL(octx, costKey)
	if _, err := pipe.Exec(octx); err != nil && !errors.Is(err, redis.Nil) {
		return st, fmt.Errorf("redis pipeline: %w", err)
	}

	if v, err := costCmd.Float64(); err == nil {
		st.CostUSD = v
	} else if !errors.Is(err, redis.Nil) {
		return st, fmt.Errorf("cost:run: %w", err)
	}
	if v, err := stepsCmd.Int64(); err == nil {
		st.Steps = v
	} else if !errors.Is(err, redis.Nil) {
		return st, fmt.Errorf("steps:run: %w", err)
	}
	if v, err := toolsCmd.Result(); err == nil && v != nil {
		st.Tools = v
	}
	if v, err := killCmd.Result(); err == nil {
		st.Killed = v == 1
	}
	// go-redis reports "no key" as -2ns and "no expiry" as -1ns;
	// pass those sentinels through unchanged in seconds.
	switch d, _ := ttlCmd.Result(); {
	case d < 0:
		st.TTLSeconds = int64(d)
	default:
		st.TTLSeconds = int64(d / time.Second)
	}
	return st, nil
}

// GetAgentState reads the agent's current-bucket spend + kill flag.
// The window comes from agent_budgets when the agent has a
// configured cap (so the reported bucket is the one enforcement
// reads); otherwise `fallbackWindow` (the operator's ?window=, or
// "1d") picks which bucket to report — informational only, since no
// cap applies to it.
func GetAgentState(ctx context.Context, name, fallbackWindow string, now time.Time) (AgentState, error) {
	st := AgentState{AgentName: name}
	if err := validateKillID("agent_name", name); err != nil {
		return st, err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return st, ErrRedisUnavailable
	}

	window := fallbackWindow
	if b, ok := GetConfig().AgentBudgets[name]; ok && b.CapUSD > 0 && b.Window != "" {
		cap := b.CapUSD
		st.ConfiguredCapUSD = &cap
		window = b.Window
	}
	if window == "" {
		window = "1d"
	}
	w, err := duration.Parse(window)
	if err != nil {
		return st, fmt.Errorf("window %q: %w", window, err)
	}
	st.Window = window
	st.BucketKey = w.BucketKey(now)

	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	pipe := rdb.Pipeline()
	spendCmd := pipe.HGet(octx, redisclient.Key(costAgentPrefix+name+":"+st.BucketKey), "total")
	killCmd := pipe.Exists(octx, redisclient.Key(killAgentPrefix+name))
	if _, err := pipe.Exec(octx); err != nil && !errors.Is(err, redis.Nil) {
		return st, fmt.Errorf("redis pipeline: %w", err)
	}
	if v, err := spendCmd.Float64(); err == nil {
		st.CurrentSpendUSD = v
	} else if !errors.Is(err, redis.Nil) {
		return st, fmt.Errorf("cost:agent: %w", err)
	}
	if v, err := killCmd.Result(); err == nil {
		st.Killed = v == 1
	}
	return st, nil
}

// validateKillID rejects ids that would produce a malformed or
// surprising Redis key: empty, absurdly long, or containing
// whitespace / path separators (which would also have been mangled
// by the URL router that delivered them).
func validateKillID(field, v string) error {
	switch {
	case v == "":
		return fmt.Errorf("%s is required", field)
	case len(v) > 256:
		return fmt.Errorf("%s too long (%d > 256)", field, len(v))
	case strings.ContainsAny(v, " \t\r\n/"):
		return fmt.Errorf("%s contains whitespace or '/'", field)
	}
	return nil
}
