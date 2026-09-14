package auth

import (
	"context"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
)

// capClaims: a two-layer chain (parent $5/100 steps → child $2/40
// steps) under a UA with a $10 envelope, agent "coder".
func capClaims() *macaroon.Claims {
	return &macaroon.Claims{
		OrgID:     testOrgID,
		UserID:    testUserID,
		AgentName: "coder",
		RunID:     "r_child",
		UANonce:   "aaaa000000000000000000000000aaaa",
		UABudget:  &macaroon.Budget{MaxTotalUSD: 10},
		Nonces:    []string{"aaaa000000000000000000000000aaaa", "bbbb000000000000000000000000bbbb"},
		IAT:       time.Now().UTC().Format(time.RFC3339),
		EffectiveCaveats: macaroon.EffectiveCaveats{
			MaxCostUSD: 2,
			MaxSteps:   40,
		},
		Chain: []macaroon.ChainLayer{
			{RunID: "r_parent", MaxCostUSD: 5, MaxSteps: 100},
			{RunID: "r_child", MaxCostUSD: 2, MaxSteps: 40},
		},
	}
}

func check(t *testing.T, claims *macaroon.Claims, realmID string) *AdapterError {
	t.Helper()
	return CheckCaps(context.Background(), claims, realmID, time.Now().UTC())
}

func wantCode(t *testing.T, err *AdapterError, code string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want %s, got nil", code)
	}
	if err.Code != code {
		t.Fatalf("want %s, got %s (%s)", code, err.Code, err.Message)
	}
	if err.HTTPStatus != 402 {
		t.Fatalf("want 402, got %d", err.HTTPStatus)
	}
}

func TestCheckCaps_ObservabilityMode_NoOp(t *testing.T) {
	if err := check(t, capClaims(), ""); err != nil {
		t.Fatalf("no redis ⇒ no-op, got %+v", err)
	}
}

func TestCheckCaps_NothingSpent_Passes(t *testing.T) {
	_ = newMiniRedis(t)
	if err := check(t, capClaims(), ""); err != nil {
		t.Fatalf("fresh run must pass, got %+v", err)
	}
}

func TestCheckCaps_UnderEveryCap_Passes(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_child", "total", "1.99")
	mr.HSet("bifrost:cost:run:r_parent", "total", "4.99")
	mr.HSet("bifrost:steps:run:r_child", "total", "39")
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "9.99")
	if err := check(t, capClaims(), ""); err != nil {
		t.Fatalf("under cap must pass, got %+v", err)
	}
}

func TestCheckCaps_LeafCostExceeded(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_child", "total", "2.00") // exactly at cap ⇒ done
	wantCode(t, check(t, capClaims(), ""), "run_cost_exceeded")
}

func TestCheckCaps_AncestorCostExceeded_KillsChild(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_child", "total", "0.50")  // child fine
	mr.HSet("bifrost:cost:run:r_parent", "total", "5.10") // parent's tree is over
	err := check(t, capClaims(), "")
	wantCode(t, err, "run_cost_exceeded")
	if want := "run r_parent"; !contains(err.Message, want) {
		t.Fatalf("message should name the parent: %q", err.Message)
	}
}

func TestCheckCaps_StepsExceeded(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:steps:run:r_child", "total", "40")
	wantCode(t, check(t, capClaims(), ""), "run_step_exceeded")
}

func TestCheckCaps_AncestorStepsExceeded(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:steps:run:r_parent", "total", "100")
	wantCode(t, check(t, capClaims(), ""), "run_step_exceeded")
}

func TestCheckCaps_UAEnvelopeExceeded(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "10")
	wantCode(t, check(t, capClaims(), ""), "ua_budget_exceeded")
}

func TestCheckCaps_NoUABudget_NoUARead(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "999")
	c := capClaims()
	c.UABudget = nil
	if err := check(t, c, ""); err != nil {
		t.Fatalf("no UA budget ⇒ no UA cap, got %+v", err)
	}
}

func TestCheckCaps_RealmBudget(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "3")
	c := capClaims()
	c.EffectiveCaveats.Budget = &macaroon.Budget{
		RealmBudgets: map[string]macaroon.RealmBudget{
			"w1": {MaxTotalUSD: 3},
			"w2": {MaxTotalUSD: 50},
		},
	}
	// This swarm is w1: $3 of $3 ⇒ realm cap first (before the $10 UA cap).
	wantCode(t, check(t, c, "w1"), "realm_budget_exceeded")
	// Swarm w2 has a $50 realm cap; $3 is fine.
	if err := check(t, c, "w2"); err != nil {
		t.Fatalf("w2 under realm cap, got %+v", err)
	}
	// Single-swarm deployment: no realm cap applies.
	if err := check(t, c, ""); err != nil {
		t.Fatalf("no realm_id ⇒ no realm cap, got %+v", err)
	}
}

func TestCheckCaps_RealmCapOnly_ReadsUACounter(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "3")
	c := capClaims()
	c.UABudget = nil // no org-wide total; only a realm cap
	c.EffectiveCaveats.Budget = &macaroon.Budget{
		RealmBudgets: map[string]macaroon.RealmBudget{"w1": {MaxTotalUSD: 3}},
	}
	wantCode(t, check(t, c, "w1"), "realm_budget_exceeded")
}

func TestCheckCaps_AgentBudget(t *testing.T) {
	mr := newMiniRedis(t)
	SetConfigForTest(Config{AgentBudgets: map[string]AgentBudget{
		"coder": {CapUSD: 1, Window: "1d"},
	}})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	today := time.Now().UTC().Format("2006-01-02")

	mr.HSet("bifrost:cost:agent:coder:"+today, "total", "0.99")
	if err := check(t, capClaims(), ""); err != nil {
		t.Fatalf("under agent cap, got %+v", err)
	}
	mr.HSet("bifrost:cost:agent:coder:"+today, "total", "1.00")
	wantCode(t, check(t, capClaims(), ""), "agent_cost_exceeded")
}

func TestCheckCaps_AgentBudgetZero_Blocks(t *testing.T) {
	_ = newMiniRedis(t)
	SetConfigForTest(Config{AgentBudgets: map[string]AgentBudget{
		"coder": {CapUSD: 0, Window: "1d"},
	}})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	wantCode(t, check(t, capClaims(), ""), "agent_cost_exceeded")

	// An unrelated agent is unaffected.
	c := capClaims()
	c.AgentName = "web-search"
	if err := check(t, c, ""); err != nil {
		t.Fatalf("other agent must pass, got %+v", err)
	}
}

func TestCheckCaps_ZeroCaveatCaps_MeanNoCap(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_solo", "total", "1000")
	mr.HSet("bifrost:steps:run:r_solo", "total", "1000")
	c := &macaroon.Claims{UserID: testUserID, RunID: "r_solo"} // no Chain, no caps
	if err := check(t, c, ""); err != nil {
		t.Fatalf("zero caps ⇒ uncapped, got %+v", err)
	}
}

func TestCheckCaps_NoChain_FallsBackToEffectiveCaveats(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_solo", "total", "2")
	c := &macaroon.Claims{
		UserID:           testUserID,
		RunID:            "r_solo",
		EffectiveCaveats: macaroon.EffectiveCaveats{MaxCostUSD: 2},
	}
	wantCode(t, check(t, c, ""), "run_cost_exceeded")
}

func TestCheckCaps_Order_CostBeforeSteps(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("bifrost:cost:run:r_child", "total", "9")
	mr.HSet("bifrost:steps:run:r_child", "total", "99")
	mr.HSet("bifrost:cost:ua:aaaa000000000000000000000000aaaa", "total", "99")
	wantCode(t, check(t, capClaims(), ""), "run_cost_exceeded")
}

func TestCheckCaps_RedisDown_FailsClosed(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Close()
	err := check(t, capClaims(), "")
	wantCode(t, err, "budget_check_unavailable")
}

func TestCheckCaps_KeyNamespace(t *testing.T) {
	mr := newMiniRedis(t)
	mr.HSet("cost:run:r_child", "total", "999") // missing bifrost: prefix
	if err := check(t, capClaims(), ""); err != nil {
		t.Fatalf("un-prefixed key must not match, got %+v", err)
	}
}

// ─── through Evaluate / ApplyToLLMPre ────────────────────────────────

func overCapMacaroon(t *testing.T, mr interface{ HSet(string, ...string) }) string {
	t.Helper()
	opts := defaultMacaroonOptions(time.Now()) // $5 cap
	mr.HSet("bifrost:cost:run:"+opts.runID, "total", "5")
	return buildMacaroon(t, opts)
}

func TestEvaluate_OverCap_ClaimsAndCapErr(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	mr := newMiniRedis(t)

	d := Evaluate(context.Background(), overCapMacaroon(t, mr))
	if d.Err != nil {
		t.Fatalf("auth must pass: %+v", d.Err)
	}
	if d.Claims == nil {
		t.Fatal("claims must be set even when over cap")
	}
	if d.CapErr == nil || d.CapErr.Code != "run_cost_exceeded" {
		t.Fatalf("want run_cost_exceeded, got %+v", d.CapErr)
	}
}

func TestApplyToLLMPre_BudgetsEnforced_OverCap402(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true, EnforceBudgets: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, overCapMacaroon(t, mr))
	sc := ApplyToLLMPre(bctx)
	if sc == nil || sc.Error == nil || sc.Error.StatusCode == nil || *sc.Error.StatusCode != 402 {
		t.Fatalf("want 402 short-circuit, got %+v", sc)
	}
	if *sc.Error.Error.Code != "run_cost_exceeded" {
		t.Fatalf("want run_cost_exceeded, got %s", *sc.Error.Error.Code)
	}
	if pluginctx.VerifiedClaims(bctx) != nil {
		t.Fatal("rejected request must not stamp claims")
	}
}

func TestApplyToLLMPre_BudgetShadow_OverCapPassesAndStamps(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true, EnforceBudgets: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, overCapMacaroon(t, mr))
	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("budget shadow must pass through, got %+v", sc)
	}
	if pluginctx.VerifiedClaims(bctx) == nil {
		t.Fatal("shadow pass-through must stamp claims so the accumulator keeps counting")
	}
}

func TestApplyToLLMPre_BudgetsWithoutMacaroonEnforce_Shadow(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	// enforce_budgets alone is inert — budgets would be bypassable
	// by dropping the macaroon.
	SetConfigForTest(Config{EnforceMacaroons: false, EnforceBudgets: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, overCapMacaroon(t, mr))
	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("enforce_budgets without enforce_macaroons must not reject, got %+v", sc)
	}
}

func TestApplyToLLMPre_BudgetsEnforced_UnderCapPasses(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true, EnforceBudgets: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	mr := newMiniRedis(t)

	opts := defaultMacaroonOptions(time.Now())
	mr.HSet("bifrost:cost:run:"+opts.runID, "total", "4.99")
	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, buildMacaroon(t, opts))
	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("under cap must pass, got %+v", sc)
	}
	if pluginctx.VerifiedClaims(bctx) == nil {
		t.Fatal("claims not stamped")
	}
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
