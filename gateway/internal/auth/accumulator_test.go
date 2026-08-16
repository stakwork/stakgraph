package auth

import (
	"context"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// chainClaims builds a verified-claims shape with an invocation layer
// plus n attenuation layers, exps spaced so per-layer TTLs differ.
func chainClaims(runIDs []string, exps []string) *macaroon.Claims {
	chain := make([]macaroon.ChainLayer, len(runIDs))
	for i := range runIDs {
		chain[i] = macaroon.ChainLayer{
			RunID:      runIDs[i],
			MaxCostUSD: 5,
			MaxSteps:   100,
			Exp:        exps[i],
		}
	}
	leaf := len(runIDs) - 1
	return &macaroon.Claims{
		OrgID:     "org_acme",
		UserID:    testUserID,
		AgentName: "coder",
		RunID:     runIDs[leaf],
		EffectiveCaveats: macaroon.EffectiveCaveats{
			Agents:     []string{"coder"},
			MaxCostUSD: 5,
			MaxSteps:   100,
			Exp:        exps[leaf],
		},
		UANonce: "aaaa000000000000000000000000aaaa",
		UAIAT:   "2026-05-14T09:00:00Z",
		UAExp:   "2026-06-14T09:00:00Z",
		Chain:   chain,
		Nonces:  []string{"aaaa000000000000000000000000aaaa", "bbbb000000000000000000000000bbbb"},
		IAT:     "2026-05-14T10:00:00Z",
	}
}

func testNow() time.Time {
	return time.Date(2026, 5, 14, 10, 5, 0, 0, time.UTC)
}

func TestAccumulate_PerRunChainWalk(t *testing.T) {
	mr := newMiniRedis(t)
	claims := chainClaims(
		[]string{"r_parent", "r_child"},
		[]string{"2026-05-14T18:00:00Z", "2026-05-14T10:30:00Z"},
	)

	if err := accumulate(context.Background(), claims, 0.42, nil, testNow()); err != nil {
		t.Fatal(err)
	}

	for _, r := range []string{"r_parent", "r_child"} {
		if got := mr.HGet("bifrost:cost:run:"+r, "total"); got != "0.42" {
			t.Errorf("cost:run:%s total = %q, want 0.42", r, got)
		}
		if got := mr.HGet("bifrost:steps:run:"+r, "total"); got != "1" {
			t.Errorf("steps:run:%s total = %q, want 1", r, got)
		}
	}

	// Second call accumulates.
	if err := accumulate(context.Background(), claims, 0.08, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if got := mr.HGet("bifrost:cost:run:r_parent", "total"); got != "0.5" {
		t.Errorf("after second call cost:run:r_parent = %q, want 0.5", got)
	}
	if got := mr.HGet("bifrost:steps:run:r_child", "total"); got != "2" {
		t.Errorf("after second call steps:run:r_child = %q, want 2", got)
	}
}

func TestAccumulate_PerLayerTTL(t *testing.T) {
	mr := newMiniRedis(t)
	// Parent exp 8h out, leaf exp 25m out. Parent keys must get the
	// longer TTL (exp-now+1h), leaf keys the 1h floor + 25m.
	claims := chainClaims(
		[]string{"r_parent", "r_child"},
		[]string{"2026-05-14T18:05:00Z", "2026-05-14T10:30:00Z"},
	)
	if err := accumulate(context.Background(), claims, 0.01, nil, testNow()); err != nil {
		t.Fatal(err)
	}

	parentTTL := mr.TTL("bifrost:cost:run:r_parent")
	childTTL := mr.TTL("bifrost:cost:run:r_child")
	if want := 9 * time.Hour; parentTTL != want { // 8h remaining + 1h grace
		t.Errorf("parent TTL = %s, want %s", parentTTL, want)
	}
	if want := 85 * time.Minute; childTTL != want { // 25m remaining + 1h grace
		t.Errorf("child TTL = %s, want %s", childTTL, want)
	}
}

func TestAccumulate_DuplicateRunIDsWrittenOnce(t *testing.T) {
	mr := newMiniRedis(t)
	// An attenuation that narrows caps but keeps the parent's run_id
	// must not double-count the call.
	claims := chainClaims(
		[]string{"r_same", "r_same"},
		[]string{"2026-05-14T18:00:00Z", "2026-05-14T12:00:00Z"},
	)
	if err := accumulate(context.Background(), claims, 0.10, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if got := mr.HGet("bifrost:cost:run:r_same", "total"); got != "0.1" {
		t.Errorf("cost:run:r_same = %q, want 0.1 (single write)", got)
	}
	if got := mr.HGet("bifrost:steps:run:r_same", "total"); got != "1" {
		t.Errorf("steps:run:r_same = %q, want 1 (single write)", got)
	}
}

func TestAccumulate_UAEnvelope_OnlyWhenBudgetSet(t *testing.T) {
	mr := newMiniRedis(t)
	claims := chainClaims([]string{"r_1"}, []string{"2026-05-14T11:00:00Z"})

	// No UABudget → no cost:ua key.
	if err := accumulate(context.Background(), claims, 0.25, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if mr.Exists("bifrost:cost:ua:" + claims.UANonce) {
		t.Fatal("cost:ua written despite nil UABudget")
	}

	// With a budget → accumulate under the UA nonce, TTL from ua.exp
	// (a month out → clamped to the 7d ceiling).
	claims.UABudget = &macaroon.Budget{MaxTotalUSD: 200}
	if err := accumulate(context.Background(), claims, 0.25, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if got := mr.HGet("bifrost:cost:ua:"+claims.UANonce, "total"); got != "0.25" {
		t.Errorf("cost:ua total = %q, want 0.25", got)
	}
	if got, want := mr.TTL("bifrost:cost:ua:"+claims.UANonce), 7*24*time.Hour; got != want {
		t.Errorf("cost:ua TTL = %s, want %s (7d ceiling)", got, want)
	}
}

func TestAccumulate_AgentBucket_OnlyWhenConfigured(t *testing.T) {
	mr := newMiniRedis(t)
	claims := chainClaims([]string{"r_1"}, []string{"2026-05-14T11:00:00Z"})

	// No configured budget for "coder" → no agent key.
	SetConfigForTest(Config{})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	if err := accumulate(context.Background(), claims, 0.30, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if keys := mr.Keys(); len(keys) != 2 { // cost:run + steps:run only
		t.Fatalf("unexpected keys without agent budget: %v", keys)
	}

	// Configured 1d budget → write to the UTC day bucket with 48h TTL.
	SetConfigForTest(Config{AgentBudgets: map[string]AgentBudget{
		"coder": {CapUSD: 500, Window: "1d"},
	}})
	if err := accumulate(context.Background(), claims, 0.30, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	key := "bifrost:cost:agent:coder:2026-05-14"
	if got := mr.HGet(key, "total"); got != "0.3" {
		t.Errorf("%s total = %q, want 0.3", key, got)
	}
	if got, want := mr.TTL(key), 48*time.Hour; got != want {
		t.Errorf("agent bucket TTL = %s, want %s", got, want)
	}
}

func TestAccumulate_ToolHistory_CappedAtTen(t *testing.T) {
	mr := newMiniRedis(t)
	claims := chainClaims([]string{"r_1"}, []string{"2026-05-14T11:00:00Z"})

	for i := 0; i < 6; i++ {
		if err := accumulate(context.Background(), claims, 0, []string{"grep", "read"}, testNow()); err != nil {
			t.Fatal(err)
		}
	}
	list, err := mr.List("bifrost:tools:run:r_1")
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != toolHistoryLen {
		t.Fatalf("tools list length = %d, want %d (LTRIM cap)", len(list), toolHistoryLen)
	}
	// Most recent push is at the head.
	if list[0] != "read" {
		t.Errorf("tools head = %q, want most-recent \"read\"", list[0])
	}
}

func TestAccumulate_ZeroCostStillCountsStep(t *testing.T) {
	mr := newMiniRedis(t)
	claims := chainClaims([]string{"r_1"}, []string{"2026-05-14T11:00:00Z"})

	// Errored call: cost 0, but the step must still increment (phase-6
	// PostLLMHook step 1).
	if err := accumulate(context.Background(), claims, 0, nil, testNow()); err != nil {
		t.Fatal(err)
	}
	if got := mr.HGet("bifrost:steps:run:r_1", "total"); got != "1" {
		t.Errorf("steps after zero-cost call = %q, want 1", got)
	}
	if got := mr.HGet("bifrost:cost:run:r_1", "total"); got != "0" {
		t.Errorf("cost after zero-cost call = %q, want 0", got)
	}
}

func TestApplyToLLMPost_NilClaimsOrNoRedis_NoOp(t *testing.T) {
	// Nil claims: must not panic, must not write.
	mr := newMiniRedis(t)
	ApplyToLLMPost(nil, 1.0, nil)
	time.Sleep(20 * time.Millisecond) // would-be goroutine window
	if keys := mr.Keys(); len(keys) != 0 {
		t.Fatalf("nil claims wrote keys: %v", keys)
	}
}
