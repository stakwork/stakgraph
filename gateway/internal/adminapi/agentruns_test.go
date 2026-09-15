package adminapi

import (
	"net/http"
	"testing"
	"time"
)

// /_plugin/agents/:name/runs — the per-run rollup behind the
// AgentDetail "Recent runs" table. Same fakeBifrost harness as
// observability_test.go; phase7Logs gives coder two runs (r1 by
// u_alice on haiku, r3 by u_bob on gpt-4o-mini) and web-search one
// (r2), which must not leak into coder's list.

func TestAgentRuns_ScopedAndNewestFirst(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	out := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/runs?window=1h"))
	if out.AgentName != "coder" || out.Window != "1h" || out.Total != 2 || len(out.Runs) != 2 {
		t.Fatalf("envelope: %+v", out)
	}
	// r3's last call (4m ago) is newer than r1's (28m ago).
	r3, r1 := out.Runs[0], out.Runs[1]
	if r3.RunID != "r3" || r1.RunID != "r1" {
		t.Fatalf("order: %s, %s", r3.RunID, r1.RunID)
	}
	if r3.UserID != "u_bob" || r3.RequestCount != 2 || r3.TotalTokens != 100 ||
		len(r3.Models) != 1 || r3.Models[0] != "gpt-4o-mini" {
		t.Errorf("r3: %+v", r3)
	}
	if r3.TotalCost < 0.039 || r3.TotalCost > 0.041 {
		t.Errorf("r3 cost: %v", r3.TotalCost)
	}
	if r1.UserID != "u_alice" || r1.RequestCount != 2 || r1.TotalTokens != 360 ||
		len(r1.Models) != 1 || r1.Models[0] != "claude-3-5-haiku" {
		t.Errorf("r1: %+v", r1)
	}
	if r1.TotalCost < 0.149 || r1.TotalCost > 0.151 {
		t.Errorf("r1 cost: %v", r1.TotalCost)
	}
	base := now.Truncate(10 * time.Minute)
	if r1.FirstSeen != base.Add(-29*time.Minute).Format(time.RFC3339Nano) ||
		r1.LastSeen != base.Add(-28*time.Minute).Format(time.RFC3339Nano) {
		t.Errorf("r1 seen: %s .. %s", r1.FirstSeen, r1.LastSeen)
	}
	for _, r := range out.Runs {
		if r.RunID == "r2" {
			t.Fatal("web-search's run leaked into coder's list")
		}
	}

	none := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/ghost/runs"))
	if none.Total != 0 || len(none.Runs) != 0 || none.Window != "24h" {
		t.Errorf("unknown agent: %+v", none)
	}
}

// Models are ordered by call count; the user-id comes from the first
// row that carries one; rows with no run-id are skipped rather than
// crashing the rollup; sub-second timestamps compare as times.
func TestAgentRuns_ModelsUserAndOrphans(t *testing.T) {
	now := time.Now().UTC()
	ts := func(d time.Duration) string { return now.Add(-d).Format(time.RFC3339Nano) }
	md := func(run, user string) map[string]string {
		m := map[string]string{"agent-name": "coder"}
		if run != "" {
			m["run-id"] = run
		}
		if user != "" {
			m["user-id"] = user
		}
		return m
	}
	logs := []fakeLog{
		{ID: "1", Timestamp: ts(3 * time.Minute), Model: "big", Cost: 1, Metadata: md("rx", "")},
		{ID: "2", Timestamp: ts(2 * time.Minute), Model: "small", Cost: 1, Metadata: md("rx", "u_carol")},
		{ID: "3", Timestamp: ts(1 * time.Minute), Model: "small", Cost: 1, Metadata: md("rx", "u_dave")},
		{ID: "4", Timestamp: ts(30 * time.Second), Model: "", Cost: 1, Metadata: md("rx", "")},
		// No run-id: excluded from every run, must not 500.
		{ID: "5", Timestamp: ts(10 * time.Second), Model: "big", Cost: 9, Metadata: md("", "u_carol")},
		// A second run whose only call is a whole second older than
		// rx's newest but has a *lexicographically* larger timestamp
		// ("…:SSZ" vs "…:SS.5Z"). Must sort after rx.
		{ID: "6", Timestamp: now.Add(-31 * time.Second).Truncate(time.Second).Format(time.RFC3339Nano),
			Model: "big", Cost: 1, Metadata: md("ry", "u_erin")},
	}
	srv := newObservabilityTestServer(t, newFakeBifrost(t, logs))

	out := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/runs?window=1h"))
	if out.Total != 2 || len(out.Runs) != 2 {
		t.Fatalf("want 2 runs, got %+v", out)
	}
	rx := out.Runs[0]
	if rx.RunID != "rx" || out.Runs[1].RunID != "ry" {
		t.Fatalf("order: %s, %s", out.Runs[0].RunID, out.Runs[1].RunID)
	}
	if rx.UserID != "u_carol" {
		t.Errorf("user: %q", rx.UserID)
	}
	if len(rx.Models) != 2 || rx.Models[0] != "small" || rx.Models[1] != "big" {
		t.Errorf("models: %v", rx.Models)
	}
	if rx.RequestCount != 4 || rx.TotalCost != 4 {
		t.Errorf("totals: %+v", rx)
	}
	if rx.FirstSeen != ts(3*time.Minute) || rx.LastSeen != ts(30*time.Second) {
		t.Errorf("seen: %s .. %s", rx.FirstSeen, rx.LastSeen)
	}
}

func TestAgentRuns_Pagination(t *testing.T) {
	now := time.Now().UTC()
	srv := newObservabilityTestServer(t, newFakeBifrost(t, phase7Logs(now)))

	page := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/runs?window=1h&limit=1"))
	if page.Total != 2 || len(page.Runs) != 1 || page.Runs[0].RunID != "r3" {
		t.Errorf("limit=1: %+v", page)
	}
	next := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/runs?window=1h&limit=1&offset=1"))
	if next.Total != 2 || len(next.Runs) != 1 || next.Runs[0].RunID != "r1" {
		t.Errorf("offset=1: %+v", next)
	}
	past := decodeOK[AgentRunsResponse](t, bearerGet(t, srv, "/_plugin/agents/coder/runs?window=1h&offset=9"))
	if past.Total != 2 || len(past.Runs) != 0 {
		t.Errorf("offset past end: %+v", past)
	}

	resp := bearerGet(t, srv, "/_plugin/agents/coder/runs?limit=0")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("limit=0: want 400, got %d", resp.StatusCode)
	}
	resp = bearerGet(t, srv, "/_plugin/agents/coder/runs?window=bogus")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("bad window: want 400, got %d", resp.StatusCode)
	}
}

func TestAgentRuns_404WithoutLogstore(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil) // no logstore in routeDeps
	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/agents/coder/runs", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
}

func TestAgentRuns_Upstream502(t *testing.T) {
	now := time.Now().UTC()
	bf := newFakeBifrost(t, phase7Logs(now))
	srv := newObservabilityTestServer(t, bf)
	bf.failNextWith = http.StatusInternalServerError
	resp := bearerGet(t, srv, "/_plugin/agents/coder/runs")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("want 502, got %d", resp.StatusCode)
	}
}
