package adminapi

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Spend-source selection
// ----------------------
// Bifrost's logging plugin writes the canonical cost to `logs.db` on
// every call. Phase-6's PostLLMHook is *supposed* to also accumulate
// the same value into `bifrost:cost:agent:<name>:<bucket>`, but phase
// 6's hot-path enforcement hasn't landed yet, which means the Redis
// hash is empty in dev today.
//
// To keep the dashboard honest for the phase-8 demo, we prefer the
// Redis bucket when it has a value (phase 6 will fill it eventually),
// and fall back to summing `logs.db` rows for the same window. Same
// number either way — Redis is just the fast path that phase 6 builds.

// AgentBudgetResponse is the wire shape for /_plugin/agents/:name/budget.
// Phase-8.5 read-only view — phase 9 grows the matching PUT/DELETE
// mutations on `/_plugin/config/agent_budgets/:name`.
//
// `CapUSD` and `Window` come from the plugin config (YAML / Redis
// override merged); `SpentUSD` comes from the live phase-6 Redis
// accumulator `bifrost:cost:agent:<name>:<bucket_key>`. Either source
// being absent is fine — the response surfaces nil so the UI can
// render "no budget" or "no spend yet" rather than misleading zeros.
type AgentBudgetResponse struct {
	AgentName string `json:"agent_name"`

	// Cap is the configured maximum spend for the current window.
	// Null when the agent has no entry in `agent_budgets`.
	CapUSD *float64 `json:"cap_usd"`

	// Window is the Bifrost duration string the cap applies to
	// (e.g. "1d", "1h"). Empty when no cap is set.
	Window string `json:"window"`

	// PeriodStart / PeriodEnd bracket the active bucket. Both are
	// RFC3339 strings, UTC. Phase 8 only renders the start; future
	// "resets in: 23h" badges can read end.
	PeriodStart string `json:"period_start,omitempty"`
	PeriodEnd   string `json:"period_end,omitempty"`

	// SpentUSD is the running total against the cap for the
	// current bucket. 0 (not null) when the bucket key simply
	// doesn't exist yet — that's a real-world "fresh window, no
	// calls yet" state, not a missing-data state.
	SpentUSD float64 `json:"spent_usd"`

	// RemainingUSD = max(cap - spent, 0). Null when cap is null.
	RemainingUSD *float64 `json:"remaining_usd"`

	// Ratio = spent / cap, clamped to [0, 1+] (we don't clamp the
	// upper bound so a runaway "150%" reads honestly). Null when
	// cap is null.
	Ratio *float64 `json:"ratio"`
}

// budgetHandlers carries the Redis client and logstore client into
// the per-request scope. Either can be nil — Redis-nil means phase-6
// hasn't filled the hot bucket yet (fall back to logs), logstore-nil
// means Bifrost is unreachable (we surface what Redis has).
type budgetHandlers struct {
	redis    *redis.Client
	logstore *logstoreClient
}

func newBudgetHandlers(logs *logstoreClient) *budgetHandlers {
	return &budgetHandlers{redis: redisclient.Client(), logstore: logs}
}

// budget handles GET /_plugin/agents/:name/budget.
//
// Path parsing is identical to runDetail in observability.go: subtree
// route, single trailing segment is the agent name, anything deeper
// 404s. Net/http's ServeMux gives us prefix routing for "/foo/"; we
// dispatch inside the handler. Cheaper than pulling in a router
// library for one route.
func (h *budgetHandlers) budget(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	const prefix = "/_plugin/agents/"
	rest := strings.TrimPrefix(r.URL.Path, prefix)
	// Expect exactly `<name>/budget`. Anything else (no segment,
	// extra path, missing /budget tail) is phase-9 territory.
	parts := strings.Split(rest, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] != "budget" {
		http.NotFound(w, r)
		return
	}
	name := parts[0]

	cfg := auth.GetConfig()
	out := AgentBudgetResponse{AgentName: name}

	// Cap lookup. Plugin config is the source of truth until
	// phase 9's `/_plugin/config/agent_budgets/:name` mutations
	// land (which write the Redis override that auth.GetConfig
	// would re-merge from). Either way, the merged value flows
	// out of auth.GetConfig().AgentBudgets.
	if b, ok := cfg.AgentBudgets[name]; ok && b.CapUSD > 0 && b.Window != "" {
		cap := b.CapUSD
		out.CapUSD = &cap
		out.Window = b.Window
		start, end, bucketKey, ok := bucketBounds(b.Window, time.Now().UTC())
		if !ok {
			// Malformed window — surface the cap, swallow the
			// bucket. UI will render cap + 0 spent rather than
			// 500ing.
			pluginlog.Warnf("adminapi: budget agent=%s: unrecognized window %q", name, b.Window)
		} else {
			out.PeriodStart = start.Format(time.RFC3339)
			out.PeriodEnd = end.Format(time.RFC3339)
			out.SpentUSD = h.readSpentWithFallback(r.Context(), name, bucketKey, start, end)
		}
	} else {
		// No cap configured. Still emit "spend today" from the
		// 1d bucket so the UI can render context without a cap.
		start, end, bucketKey, ok := bucketBounds("1d", time.Now().UTC())
		if ok {
			out.PeriodStart = start.Format(time.RFC3339)
			out.PeriodEnd = end.Format(time.RFC3339)
			out.SpentUSD = h.readSpentWithFallback(r.Context(), name, bucketKey, start, end)
		}
	}

	if out.CapUSD != nil {
		rem := *out.CapUSD - out.SpentUSD
		if rem < 0 {
			rem = 0
		}
		out.RemainingUSD = &rem
		ratio := 0.0
		if *out.CapUSD > 0 {
			ratio = out.SpentUSD / *out.CapUSD
		}
		out.Ratio = &ratio
	}

	writeJSON(w, http.StatusOK, out)
}

// readSpentWithFallback tries Redis first (phase-6's hot bucket),
// then falls back to summing `logs.db` rows for the same window.
// Either source returns the same value once phase 6 is wired up.
//
// The Redis path is a single `HGET` and beats the log scan on
// latency, so we prefer it when it has data. The log fallback is
// what keeps the demo honest before phase 6 lands.
func (h *budgetHandlers) readSpentWithFallback(
	ctx context.Context, name, bucket string,
	start, end time.Time,
) float64 {
	if v, ok := h.readSpentFromRedis(ctx, name, bucket); ok {
		return v
	}
	return h.readSpentFromLogs(ctx, name, start, end)
}

// readSpentFromRedis reads `total` from
// bifrost:cost:agent:<name>:<bucket>. Returns (value, true) on a
// hit, (0, false) on every kind of "absent" — missing key, empty
// hash, Redis unconfigured, transient error.
func (h *budgetHandlers) readSpentFromRedis(
	ctx context.Context, name, bucket string,
) (float64, bool) {
	if h.redis == nil {
		return 0, false
	}
	key := redisclient.Key("cost:agent:" + name + ":" + bucket)
	val, err := h.redis.HGet(ctx, key, "total").Float64()
	if err == redis.Nil {
		return 0, false
	}
	if err != nil {
		pluginlog.Warnf("adminapi: budget HGet %s: %v", key, err)
		return 0, false
	}
	return val, true
}

// readSpentFromLogs sums `cost` from every logs.db row in the bucket
// window that carries metadata.agent-name = <name>. Pre-phase-6
// fallback path so the dashboard renders real numbers today; once
// phase 6's PostLLMHook fills the Redis bucket this branch never
// fires.
//
// Returns 0 on any upstream failure — same posture as the Redis
// branch. Operators see "no spend yet" instead of a confusing 500.
func (h *budgetHandlers) readSpentFromLogs(
	ctx context.Context, name string, start, end time.Time,
) float64 {
	if h.logstore == nil {
		return 0
	}
	logs, err := h.logstore.searchAll(ctx, searchOpts{
		StartTime: &start,
		EndTime:   &end,
		Metadata:  map[string]string{"agent-name": name},
	}, 1000, 50_000)
	if err != nil {
		pluginlog.Warnf("adminapi: budget log fallback agent=%s: %v", name, err)
		return 0
	}
	var total float64
	for _, l := range logs {
		total += l.Cost
	}
	return total
}

// bucketBounds derives (period_start, period_end, redis_bucket_key)
// for a given window string, anchored to `now`. Mirrors the per-window
// formats in phase-6 §"Bucket-key format by window suffix" — keep this
// in lockstep with the plugin-side PostLLMHook (gateway/internal/auth/
// ttl.go) so the same key written there is read here.
//
// Phase 8 only needs `1h`/`1d`/`1w`/`1M` for the dashboard. Other
// suffixes return ok=false and the handler degrades gracefully. The
// full grammar is phase-6's concern.
func bucketBounds(window string, now time.Time) (start, end time.Time, key string, ok bool) {
	switch window {
	case "1h":
		// Sub-day rolling: bucket starts on the hour boundary.
		start = now.Truncate(time.Hour)
		end = start.Add(time.Hour)
		key = formatEpochBucket(start)
	case "1d":
		// Calendar day, UTC midnight.
		start = time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
		end = start.AddDate(0, 0, 1)
		key = start.Format("2006-01-02")
	case "1w":
		// ISO week starting Monday.
		// Go's Weekday: Sunday=0..Saturday=6; ISO wants Monday=1..Sunday=7.
		wd := int(now.Weekday())
		if wd == 0 {
			wd = 7
		}
		start = time.Date(now.Year(), now.Month(), now.Day()-(wd-1), 0, 0, 0, 0, time.UTC)
		end = start.AddDate(0, 0, 7)
		y, w := start.ISOWeek()
		key = formatISOWeek(y, w)
	case "1M":
		start = time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)
		end = start.AddDate(0, 1, 0)
		key = start.Format("2006-01")
	case "1Y":
		start = time.Date(now.Year(), 1, 1, 0, 0, 0, 0, time.UTC)
		end = start.AddDate(1, 0, 0)
		key = start.Format("2006")
	default:
		return time.Time{}, time.Time{}, "", false
	}
	return start, end, key, true
}

// formatEpochBucket returns the bucket-start unix-epoch as a
// decimal string. Per phase-6 §"Bucket-key format by window suffix":
// sub-day windows use the unix epoch of the window start (rounded
// to the unit), not a human date.
func formatEpochBucket(t time.Time) string {
	return strconv.FormatInt(t.Unix(), 10)
}

// formatISOWeek returns "YYYY-Www" zero-padded — matches the bucket
// key format spelled out in the phase-6 schema table.
func formatISOWeek(year, week int) string {
	return fmt.Sprintf("%04d-W%02d", year, week)
}
