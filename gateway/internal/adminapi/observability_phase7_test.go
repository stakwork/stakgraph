package adminapi

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Phase-7 remainder: by-session / by-model rollups, token and latency
// histograms, session drill-down, user spend + quota, agent spend,
// and the widened window / bucket vocabulary. Same fakeBifrost
// harness as observability_test.go.

// phase7Logs is a fixture with every dim stamped and token usage on
// each row, spanning two sessions, three runs, two users, two models.
// Timestamps are offsets from `now` truncated to a 10-minute epoch
// boundary, so every row sits inside the last hour (window=1h sees
// everything) and the first two rows always share one 10m histogram
// bucket regardless of the wall clock.
func phase7Logs(now time.Time) []fakeLog {
	base := now.Truncate(10 * time.Minute)
	ts := func(minBefore int) string {
		return base.Add(-time.Duration(minBefore) * time.Minute).Format(time.RFC3339Nano)
	}
	tu := func(p, c int64) json.RawMessage {
		return json.RawMessage(`{"prompt_tokens":` + itoa(p) + `,"completion_tokens":` + itoa(c) + `,"total_tokens":` + itoa(p+c) + `}`)
	}
	md := func(agent, run, user, session string) map[string]string {
		return map[string]string{"agent-name": agent, "run-id": run, "user-id": user, "session-id": session}
	}
	return []fakeLog{
		{ID: "1", Timestamp: ts(29), Provider: "anthropic", Model: "claude-3-5-haiku", Status: "success",
			Cost: 0.05, Latency: 800, CustomerID: "u_alice", Metadata: md("coder", "r1", "u_alice", "s1"), TokenUsage: tu(100, 20)},
		{ID: "2", Timestamp: ts(28), Provider: "anthropic", Model: "claude-3-5-haiku", Status: "success",
			Cost: 0.10, Latency: 1200, CustomerID: "u_alice", Metadata: md("coder", "r1", "u_alice", "s1"), TokenUsage: tu(200, 40)},
		{ID: "3", Timestamp: ts(19), Provider: "openai", Model: "gpt-4o-mini", Status: "success",
			Cost: 0.02, Latency: 600, CustomerID: "u_alice", Metadata: md("web-search", "r2", "u_alice", "s1"), TokenUsage: tu(50, 10)},
		{ID: "4", Timestamp: ts(9), Provider: "openai", Model: "gpt-4o-mini", Status: "success",
			Cost: 0.04, Latency: 400, CustomerID: "u_bob", Metadata: md("coder", "r3", "u_bob", "s2"), TokenUsage: tu(80, 20)},
		// Errored call: no latency, no usage. Excluded from the latency
		// histogram and contributes zero tokens.
		{ID: "5", Timestamp: ts(4), Provider: "openai", Model: "gpt-4o-mini", Status: "error",
			Cost: 0, Latency: 0, CustomerID: "u_bob", Metadata: md("coder", "r3", "u_bob", "s2")},
	}
}

func itoa(v int64) string { return json.Number(formatInt(v)).String() }

func formatInt(v int64) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func decodeOK[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out T
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

// withMiniRedis points redisclient at a fresh miniredis for the test
// and returns the handle for seeding. Observability tests otherwise
// run with no Redis, which is exactly the "redis_available=false"
// path the quota endpoint has to survive.
func withMiniRedis(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	mr := miniredis.RunT(t)
	rc := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	redisclient.SetClientForTest(rc)
	t.Cleanup(func() {
		_ = rc.Close()
		redisclient.SetClientForTest(nil)
	})
	return mr
}

// ─── spend.by-session / by-model ─────────────────────────────────────

func TestSpendBySession(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[SpendBySessionResponse](t, bearerGet(t, srv, "/_plugin/spend/by-session?window=1h"))
	if out.Window != "1h" || len(out.Results) != 2 {
		t.Fatalf("unexpected: %+v", out)
	}
	s1 := out.Results[0] // 0.17 > 0.04
	if s1.SessionID != "s1" || s1.UserID != "u_alice" || s1.RequestCount != 3 || s1.RunCount != 2 {
		t.Errorf("s1: %+v", s1)
	}
	if s1.TotalTokens != 120+240+60 || s1.TotalCost < 0.169 || s1.TotalCost > 0.171 {
		t.Errorf("s1 totals: %+v", s1)
	}
	if s1.FirstSeen >= s1.LastSeen {
		t.Errorf("s1 span: first=%s last=%s", s1.FirstSeen, s1.LastSeen)
	}
	if out.Results[1].SessionID != "s2" || out.Results[1].RequestCount != 2 || out.Results[1].RunCount != 1 {
		t.Errorf("s2: %+v", out.Results[1])
	}
}

func TestSpendByModel(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[SpendByModelResponse](t, bearerGet(t, srv, "/_plugin/spend/by-model?window=1h"))
	if len(out.Results) != 2 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.Results[0].Model != "claude-3-5-haiku" || out.Results[0].Provider != "anthropic" ||
		out.Results[0].RequestCount != 2 || out.Results[0].TotalTokens != 360 {
		t.Errorf("haiku: %+v", out.Results[0])
	}
	// gpt-4o-mini: three rows incl. the errored one (by-model excludes nothing).
	if out.Results[1].Model != "gpt-4o-mini" || out.Results[1].RequestCount != 3 || out.Results[1].TotalTokens != 160 {
		t.Errorf("gpt: %+v", out.Results[1])
	}
}

// ─── histogram.tokens / histogram.latency ────────────────────────────

func TestHistogramTokens_ByAgent(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[HistogramTokensResponse](t, bearerGet(t, srv,
		"/_plugin/histogram/tokens?window=1h&bucket=10m&dimension=agent-name"))
	if out.BucketSizeSeconds != 600 || out.Dimension != "agent-name" || len(out.Series) != 2 {
		t.Fatalf("unexpected: %+v", out)
	}
	coder := out.Series[0] // 460 tokens > web-search's 60
	if coder.DimensionValue != "coder" {
		t.Fatalf("series order: %+v", out.Series)
	}
	var total int64
	for _, p := range coder.Points {
		total += p.TotalTokens
		if p.PromptTokens+p.CompletionTokens != p.TotalTokens {
			t.Errorf("point split: %+v", p)
		}
	}
	if total != 120+240+100 {
		t.Errorf("coder tokens = %d", total)
	}
	// Rows 1 and 2 (29 and 28 minutes before the 10m-aligned base)
	// land in the same bucket, so coder has exactly 2 points (that
	// bucket + row 4's).
	if len(coder.Points) != 2 {
		t.Errorf("expected folded buckets, got %d points", len(coder.Points))
	}
	for i := 1; i < len(coder.Points); i++ {
		if coder.Points[i-1].Timestamp >= coder.Points[i].Timestamp {
			t.Errorf("points not ascending: %+v", coder.Points)
		}
	}
}

func TestHistogramLatency_Percentiles(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	// One bucket spanning the whole window so every call folds together.
	out := decodeOK[HistogramLatencyResponse](t, bearerGet(t, srv,
		"/_plugin/histogram/latency?window=1h&bucket=1h&dimension=user-id"))
	if len(out.Series) != 2 || out.Series[0].DimensionValue != "u_alice" {
		t.Fatalf("unexpected: %+v", out)
	}
	// Both of alice's calls may straddle an epoch-aligned hour
	// boundary; sum counts across points and check the percentiles
	// of whichever point holds the most calls.
	var count int64
	var top LatencyHistogramPoint
	for _, p := range out.Series[0].Points {
		count += p.Count
		if p.Count > top.Count {
			top = p
		}
	}
	if count != 3 {
		t.Fatalf("alice latency samples = %d, want 3 (errored rows excluded)", count)
	}
	if top.P50 <= 0 || top.P95 < top.P50 || top.P99 < top.P95 {
		t.Errorf("percentiles not monotone: %+v", top)
	}
	// bob: one successful call at 400ms; the errored one is dropped.
	bob := out.Series[1]
	if bob.DimensionValue != "u_bob" || len(bob.Points) != 1 || bob.Points[0].Count != 1 ||
		bob.Points[0].P50 != 400 || bob.Points[0].P99 != 400 {
		t.Errorf("bob: %+v", bob)
	}
}

func TestPercentile_NearestRank(t *testing.T) {
	s := []float64{100, 200, 300, 400, 500}
	cases := []struct {
		p    float64
		want float64
	}{{0.5, 300}, {0.95, 500}, {0.99, 500}, {0.2, 100}, {0.0, 100}}
	for _, c := range cases {
		if got := percentile(s, c.p); got != c.want {
			t.Errorf("p%v = %v, want %v", c.p, got, c.want)
		}
	}
	if percentile(nil, 0.5) != 0 {
		t.Error("empty sample must be 0")
	}
}

// ─── sessions ────────────────────────────────────────────────────────

func TestSessionDetail_PagesByMetadata(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[SessionDetailResponse](t, bearerGet(t, srv, "/_plugin/sessions/s1?limit=2"))
	if out.SessionID != "s1" || len(out.Logs) != 2 || out.TotalCount != 3 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.Stats.TotalRequests != 3 || out.Stats.TotalTokens != 420 {
		t.Errorf("stats over whole session: %+v", out.Stats)
	}
	for _, l := range out.Logs {
		if l.Metadata["session-id"] != "s1" {
			t.Errorf("leaked row: %+v", l)
		}
	}
}

func TestSessionSummary(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[SessionSummaryResponse](t, bearerGet(t, srv, "/_plugin/sessions/s1/summary"))
	if out.SessionID != "s1" || out.UserID != "u_alice" || out.RequestCount != 3 || out.TotalTokens != 420 {
		t.Fatalf("unexpected: %+v", out)
	}
	if len(out.Agents) != 2 || out.Agents[0] != "coder" || out.Agents[1] != "web-search" {
		t.Errorf("agents: %v", out.Agents)
	}
	if len(out.Runs) != 2 || out.Runs[0] != "r1" || out.Runs[1] != "r2" {
		t.Errorf("runs: %v", out.Runs)
	}
	// base−29m → base−19m: exactly ten minutes.
	if out.DurationMS != 10*60_000 {
		t.Errorf("duration_ms = %d", out.DurationMS)
	}
	if out.StartedAt >= out.LatestAt {
		t.Errorf("span: %s .. %s", out.StartedAt, out.LatestAt)
	}

	// Unknown session: empty, not 404 (same as an empty logs.db).
	empty := decodeOK[SessionSummaryResponse](t, bearerGet(t, srv, "/_plugin/sessions/nope/summary"))
	if empty.RequestCount != 0 || empty.Agents == nil || empty.Runs == nil || empty.DurationMS != 0 {
		t.Errorf("empty summary: %+v", empty)
	}
}

func TestSessions_404OnBadShape(t *testing.T) {
	srv := newObservabilityTestServer(t, newFakeBifrost(t, nil))
	for _, p := range []string{"/_plugin/sessions/", "/_plugin/sessions/s1/bogus", "/_plugin/sessions/s1/summary/x"} {
		resp := bearerGet(t, srv, p)
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("%s: want 404, got %d", p, resp.StatusCode)
		}
	}
}

// ─── users/:id/spend · users/:id/quota ───────────────────────────────

func TestUserSpend(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[UserSpendResponse](t, bearerGet(t, srv, "/_plugin/users/u_alice/spend?window=1h"))
	if out.UserID != "u_alice" || out.Window != "1h" || out.RequestCount != 3 || out.TotalTokens != 420 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.TotalCost < 0.169 || out.TotalCost > 0.171 {
		t.Errorf("cost: %v", out.TotalCost)
	}
	// The rollup at /users/:id must still answer.
	roll := decodeOK[UserDetailResponse](t, bearerGet(t, srv, "/_plugin/users/u_alice?window=1h"))
	if roll.RequestCount != 3 {
		t.Errorf("rollup: %+v", roll)
	}
	resp := bearerGet(t, srv, "/_plugin/users/u_alice/bogus")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("bad subpath: want 404, got %d", resp.StatusCode)
	}
}

func TestUserQuota_BudgetAndInflight(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, phase7Logs(now))
	bf.customers = map[string]fakeCustomer{
		"u_alice": {ID: "u_alice", Name: "alice", Budgets: []fakeBudget{{
			ID: "b1", MaxLimit: 1000, ResetDuration: "1d", CurrentUsage: 12.5,
			LastReset: now.Add(-3 * time.Hour).Format(time.RFC3339),
		}}},
		"u_nobudget": {ID: "u_nobudget", Name: "nobody"},
	}
	srv := newObservabilityTestServer(t, bf)
	mr := withMiniRedis(t)

	// Two indexed runs: one live, one whose exp has passed (pruned).
	future := float64(now.Add(2 * time.Hour).Unix())
	past := float64(now.Add(-2 * time.Hour).Unix())
	mr.ZAdd("bifrost:runs:user:u_alice", future, "r_live")
	mr.ZAdd("bifrost:runs:user:u_alice", past, "r_dead")
	mr.HSet("bifrost:cost:run:r_live", "total", "1.25")
	mr.HSet("bifrost:steps:run:r_live", "total", "7")
	mr.HSet("bifrost:meta:run:r_live", "max_cost_usd", "5", "max_steps", "100",
		"exp", now.Add(2*time.Hour).Format(time.RFC3339), "parent", "", "agent", "coder", "user", "u_alice")
	mr.Set("bifrost:kill:r_live", "1")

	out := decodeOK[UserQuotaResponse](t, bearerGet(t, srv, "/_plugin/users/u_alice/quota"))
	if !out.CustomerFound || out.BudgetUSD == nil || *out.BudgetUSD != 1000 || out.BudgetWindow != "1d" ||
		out.SpentUSD != 12.5 || out.RemainingUSD == nil || *out.RemainingUSD != 987.5 || out.BudgetLastReset == "" {
		t.Fatalf("budget half: %+v", out)
	}
	if !out.RedisAvailable || len(out.InflightRuns) != 1 {
		t.Fatalf("live half: %+v", out)
	}
	run := out.InflightRuns[0]
	if run.RunID != "r_live" || run.AgentName != "coder" || run.CostUSD != 1.25 || run.Steps != 7 ||
		run.MaxCostUSD == nil || *run.MaxCostUSD != 5 || run.MaxSteps == nil || *run.MaxSteps != 100 || !run.Killed {
		t.Errorf("inflight run: %+v", run)
	}
	if mr.Exists("bifrost:runs:user:u_alice") {
		if members, _ := mr.ZMembers("bifrost:runs:user:u_alice"); len(members) != 1 {
			t.Errorf("expired run not pruned: %v", members)
		}
	}

	// Customer without a budget: found, but nothing to draw.
	nb := decodeOK[UserQuotaResponse](t, bearerGet(t, srv, "/_plugin/users/u_nobudget/quota"))
	if !nb.CustomerFound || nb.BudgetUSD != nil || nb.RemainingUSD != nil || !nb.RedisAvailable || len(nb.InflightRuns) != 0 {
		t.Errorf("no-budget customer: %+v", nb)
	}

	// Unknown customer: 404 upstream folds into customer_found=false.
	unk := decodeOK[UserQuotaResponse](t, bearerGet(t, srv, "/_plugin/users/u_ghost/quota"))
	if unk.CustomerFound || unk.BudgetUSD != nil || !unk.RedisAvailable {
		t.Errorf("unknown customer: %+v", unk)
	}
}

func TestUserQuota_RedisOffDegrades(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, phase7Logs(now))
	bf.customers = map[string]fakeCustomer{
		"u_alice": {ID: "u_alice", Budgets: []fakeBudget{{MaxLimit: 10, ResetDuration: "1d", CurrentUsage: 2}}},
	}
	srv := newObservabilityTestServer(t, bf)
	redisclient.SetClientForTest(nil)

	resp := bearerGet(t, srv, "/_plugin/users/u_alice/quota")
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	raw := map[string]any{}
	_ = json.NewDecoder(resp.Body).Decode(&raw)
	if raw["redis_available"] != false || raw["inflight_runs"] != nil || raw["budget_usd"] != 10.0 {
		t.Errorf("degraded quota: %+v", raw)
	}
}

// ─── agents/:name/spend ──────────────────────────────────────────────

func TestAgentSpend(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[AgentSpendResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/spend?window=1h"))
	if out.AgentName != "coder" || out.Window != "1h" || out.RequestCount != 4 || out.TotalTokens != 460 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.TotalCost < 0.189 || out.TotalCost > 0.191 {
		t.Errorf("cost: %v", out.TotalCost)
	}
	none := decodeOK[AgentSpendResponse](t, bearerGet(t, srv, "/_plugin/agents/ghost/spend"))
	if none.RequestCount != 0 || none.Window != "24h" {
		t.Errorf("unknown agent: %+v", none)
	}
}

func TestAgentSpend_404WithoutLogstore(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil) // no logstore in routeDeps
	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/agents/coder/spend", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
}

// ─── window / bucket vocabulary ──────────────────────────────────────

func TestWindowVocabulary(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	for _, w := range []string{"1h", "6h", "24h", "1d", "7d", "1w", "30d", "1M", "1Y"} {
		resp := bearerGet(t, srv, "/_plugin/spend/by-agent?window="+w)
		var out SpendByAgentResponse
		_ = json.NewDecoder(resp.Body).Decode(&out)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK || out.Window != w {
			t.Errorf("window=%s: status %d window %q", w, resp.StatusCode, out.Window)
		}
	}
	for _, w := range []string{"99y", "2Y", "0d", "1", "d", "01h", "1m5"} {
		resp := bearerGet(t, srv, "/_plugin/spend/by-agent?window="+w)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("window=%s: want 400, got %d", w, resp.StatusCode)
		}
	}
}

func TestBucketVocabulary(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	for _, c := range []struct {
		q    string
		want int
	}{
		{"window=1d&bucket=1h", 200},
		{"window=1d&bucket=15m", 200},
		{"window=1w&bucket=1d", 200},
		{"window=1h&bucket=1d", 400},  // bucket > window
		{"window=1h&bucket=30s", 400}, // sub-minute
		{"window=1h&bucket=x", 400},
	} {
		resp := bearerGet(t, srv, "/_plugin/histogram/cost?"+c.q+"&dimension=agent-name")
		resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s: want %d, got %d", c.q, c.want, resp.StatusCode)
		}
	}
}

// ─── tokens now flow through the phase-8 rollups too ─────────────────

func TestSpendByAgent_TokensFilled(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[SpendByAgentResponse](t, bearerGet(t, srv, "/_plugin/spend/by-agent?window=1h"))
	if len(out.Results) != 2 || out.Results[0].AgentName != "coder" || out.Results[0].TotalTokens != 460 {
		t.Fatalf("tokens on by-agent: %+v", out.Results)
	}
}
