package auth

import (
	"context"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

func killTestClaims() *macaroon.Claims {
	return &macaroon.Claims{
		UserID:    testUserID,
		AgentName: "coder",
		RunID:     "r_child",
		Nonces:    []string{"aaaa000000000000000000000000aaaa", "bbbb000000000000000000000000bbbb"},
		IAT:       time.Now().UTC().Format(time.RFC3339),
		Chain: []macaroon.ChainLayer{
			{RunID: "r_parent", MaxCostUSD: 5},
			{RunID: "r_child", MaxCostUSD: 2},
		},
	}
}

func TestCheckRevocations_KillLeafRun(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:kill:r_child", "1")

	err := CheckRevocations(context.Background(), killTestClaims())
	if err == nil || err.Code != "run_killed" {
		t.Fatalf("want run_killed, got %+v", err)
	}
	if err.HTTPStatus != 402 {
		t.Fatalf("want 402, got %d", err.HTTPStatus)
	}
}

func TestCheckRevocations_KillAncestorRun_KillsDescendant(t *testing.T) {
	mr := newMiniRedis(t)
	// Only the parent is killed; the leaf's own key is absent.
	mr.Set("bifrost:kill:r_parent", "1")

	err := CheckRevocations(context.Background(), killTestClaims())
	if err == nil || err.Code != "run_killed" {
		t.Fatalf("want run_killed via ancestor, got %+v", err)
	}
}

func TestCheckRevocations_KillAgent(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:kill:agent:coder", "1")

	err := CheckRevocations(context.Background(), killTestClaims())
	if err == nil || err.Code != "agent_killed" {
		t.Fatalf("want agent_killed, got %+v", err)
	}
	if err.HTTPStatus != 402 {
		t.Fatalf("want 402, got %d", err.HTTPStatus)
	}
}

func TestCheckRevocations_KillAgent_LeafOnly(t *testing.T) {
	mr := newMiniRedis(t)
	// A kill on a different agent name must not match.
	mr.Set("bifrost:kill:agent:web-search", "1")

	if err := CheckRevocations(context.Background(), killTestClaims()); err != nil {
		t.Fatalf("kill on another agent must not match, got %+v", err)
	}
}

func TestCheckRevocations_RevokedBeatsKilled(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:kill:r_child", "1")
	mr.Set("bifrost:revoke:bbbb000000000000000000000000bbbb", "1")

	err := CheckRevocations(context.Background(), killTestClaims())
	if err == nil || err.Code != "macaroon_revoked" {
		t.Fatalf("revocation should win over kill, got %+v", err)
	}
}

func TestCheckRevocations_KillFallsBackToLeafRunID(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:kill:r_solo", "1")

	claims := &macaroon.Claims{
		UserID: testUserID,
		RunID:  "r_solo", // no Chain populated
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    time.Now().UTC().Format(time.RFC3339),
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil || err.Code != "run_killed" {
		t.Fatalf("want run_killed from Claims.RunID fallback, got %+v", err)
	}
}

func TestKillRun_RoundTrip(t *testing.T) {
	mr := newMiniRedis(t)
	ctx := context.Background()

	if err := KillRun(ctx, "r_1"); err != nil {
		t.Fatal(err)
	}
	if !mr.Exists("bifrost:kill:r_1") {
		t.Fatal("kill key not written")
	}
	if ttl := mr.TTL("bifrost:kill:r_1"); ttl != killRunTTL {
		t.Fatalf("ttl: want %v, got %v", killRunTTL, ttl)
	}
	if err := UnkillRun(ctx, "r_1"); err != nil {
		t.Fatal(err)
	}
	if mr.Exists("bifrost:kill:r_1") {
		t.Fatal("kill key not deleted")
	}
}

func TestKillAgent_RoundTrip(t *testing.T) {
	mr := newMiniRedis(t)
	ctx := context.Background()

	if err := KillAgent(ctx, "coder"); err != nil {
		t.Fatal(err)
	}
	if ttl := mr.TTL("bifrost:kill:agent:coder"); ttl != killAgentTTL {
		t.Fatalf("ttl: want %v, got %v", killAgentTTL, ttl)
	}
	if err := UnkillAgent(ctx, "coder"); err != nil {
		t.Fatal(err)
	}
	if mr.Exists("bifrost:kill:agent:coder") {
		t.Fatal("kill key not deleted")
	}
}

func TestKill_Validation(t *testing.T) {
	_ = newMiniRedis(t)
	ctx := context.Background()
	for _, bad := range []string{"", "has space", "a/b", string(make([]byte, 300))} {
		if err := KillRun(ctx, bad); err == nil {
			t.Errorf("KillRun(%q): want validation error", bad)
		}
		if err := KillAgent(ctx, bad); err == nil {
			t.Errorf("KillAgent(%q): want validation error", bad)
		}
	}
}

func TestKill_RedisUnavailable(t *testing.T) {
	redisclient.SetClientForTest(nil)
	if err := KillRun(context.Background(), "r_1"); err != ErrRedisUnavailable {
		t.Fatalf("want ErrRedisUnavailable, got %v", err)
	}
	if _, err := GetRunState(context.Background(), "r_1"); err != ErrRedisUnavailable {
		t.Fatalf("want ErrRedisUnavailable, got %v", err)
	}
}

func TestGetRunState_Populated(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_1", "total", "1.25")
	mr.HSet("bifrost:steps:run:r_1", "total", "7")
	mr.Lpush("bifrost:tools:run:r_1", "read_file")
	mr.Lpush("bifrost:tools:run:r_1", "bash")
	mr.Set("bifrost:kill:r_1", "1")
	mr.SetTTL("bifrost:cost:run:r_1", 90*time.Minute)

	st, err := GetRunState(context.Background(), "r_1")
	if err != nil {
		t.Fatal(err)
	}
	if st.CostUSD != 1.25 || st.Steps != 7 || !st.Killed {
		t.Fatalf("state: %+v", st)
	}
	if len(st.Tools) != 2 || st.Tools[0] != "bash" || st.Tools[1] != "read_file" {
		t.Fatalf("tools (most recent first): %v", st.Tools)
	}
	if st.TTLSeconds != int64((90 * time.Minute).Seconds()) {
		t.Fatalf("ttl: %d", st.TTLSeconds)
	}
}

func TestGetRunState_Empty(t *testing.T) {
	_ = newMiniRedis(t)
	st, err := GetRunState(context.Background(), "r_never_called")
	if err != nil {
		t.Fatal(err)
	}
	if st.CostUSD != 0 || st.Steps != 0 || st.Killed || len(st.Tools) != 0 {
		t.Fatalf("want empty state, got %+v", st)
	}
	if st.TTLSeconds != -2 {
		t.Fatalf("want -2 (no key) ttl sentinel, got %d", st.TTLSeconds)
	}
}

func TestGetAgentState_ConfiguredCapWinsWindow(t *testing.T) {
	mr := newMiniRedis(t)
	SetConfigForTest(Config{AgentBudgets: map[string]AgentBudget{
		"coder": {CapUSD: 5, Window: "1d"},
	}})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	now := time.Now().UTC()
	mr.HSet("bifrost:cost:agent:coder:"+now.Format("2006-01-02"), "total", "2.5")
	mr.Set("bifrost:kill:agent:coder", "1")

	// ?window=1h must be ignored: the configured 1d bucket is the one
	// enforcement reads.
	st, err := GetAgentState(context.Background(), "coder", "1h", now)
	if err != nil {
		t.Fatal(err)
	}
	if st.Window != "1d" || st.CurrentSpendUSD != 2.5 || !st.Killed {
		t.Fatalf("state: %+v", st)
	}
	if st.ConfiguredCapUSD == nil || *st.ConfiguredCapUSD != 5 {
		t.Fatalf("cap: %+v", st.ConfiguredCapUSD)
	}
}

func TestGetAgentState_NoCap_FallbackWindow(t *testing.T) {
	_ = newMiniRedis(t)
	st, err := GetAgentState(context.Background(), "nobudget", "", time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if st.Window != "1d" || st.ConfiguredCapUSD != nil || st.Killed || st.CurrentSpendUSD != 0 {
		t.Fatalf("state: %+v", st)
	}
	if _, err := GetAgentState(context.Background(), "nobudget", "1x", time.Now()); err == nil {
		t.Fatal("want error for unparseable window")
	}
}

// End-to-end through the hook entry point: a real signed macaroon,
// enforce mode, run killed → 402 short-circuit with the stable code.
func TestApplyToLLMPre_EnforceMode_RunKilled402(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	opts := defaultMacaroonOptions(time.Now())
	encoded := buildMacaroon(t, opts)
	mr.Set("bifrost:kill:"+opts.runID, "1")

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)
	sc := ApplyToLLMPre(bctx)
	if sc == nil || sc.Error == nil {
		t.Fatal("expected short-circuit")
	}
	if sc.Error.StatusCode == nil || *sc.Error.StatusCode != 402 {
		t.Fatalf("want 402, got %+v", sc.Error.StatusCode)
	}
	if sc.Error.Error == nil || sc.Error.Error.Code == nil || *sc.Error.Error.Code != "run_killed" {
		t.Fatalf("want run_killed, got %+v", sc.Error.Error)
	}
	if sc.Error.Type == nil || *sc.Error.Type != "enforcement_rejected" {
		t.Fatalf("want enforcement_rejected type, got %+v", sc.Error.Type)
	}
	if pluginctx.VerifiedClaims(bctx) != nil {
		t.Fatal("rejected request must not stamp claims")
	}
}

func TestApplyToLLMPre_ShadowMode_KilledPassesThrough(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	opts := defaultMacaroonOptions(time.Now())
	encoded := buildMacaroon(t, opts)
	mr.Set("bifrost:kill:agent:coder", "1")

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)
	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("shadow mode must not short-circuit, got %+v", sc)
	}
}
