package auth

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
)

func newBifrostCtx() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
}

func TestEvaluate_Happy(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t) // empty redis — no revocations

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	d := Evaluate(context.Background(), encoded)
	if d.Err != nil {
		t.Fatalf("expected success, got %+v", d.Err)
	}
	if d.Claims == nil {
		t.Fatal("nil claims on success")
	}
	if !d.HadHeader {
		t.Fatal("HadHeader=false despite non-empty input")
	}
}

func TestEvaluate_MissingHeader(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })

	d := Evaluate(context.Background(), "")
	if d.HadHeader {
		t.Fatal("HadHeader=true for empty input")
	}
	if d.Err == nil || d.Err.Code != "macaroon_required" {
		t.Fatalf("want macaroon_required, got %+v", d.Err)
	}
}

func TestEvaluate_RevocationLayered(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	mr := newMiniRedis(t)

	opts := defaultMacaroonOptions(time.Now())
	encoded := buildMacaroon(t, opts)
	// Pre-revoke the invocation nonce. The signature is valid; the
	// nonce check should land second.
	mr.Set("bifrost:revoke:"+opts.invNonce, "1")

	d := Evaluate(context.Background(), encoded)
	if d.Err == nil {
		t.Fatal("expected revocation rejection")
	}
	if d.Err.Code != "macaroon_revoked" {
		t.Fatalf("want macaroon_revoked, got %s", d.Err.Code)
	}
}

func TestApplyToLLMPre_ShadowMode_FailurePassesThrough(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	bctx := newBifrostCtx()
	// No x-macaroon stashed → ApplyToLLMPre should NOT short-circuit
	// in shadow mode.
	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("shadow mode should pass through, got short-circuit: %+v", sc)
	}
	if pluginctx.VerifiedClaims(bctx) != nil {
		t.Fatal("no claims should be stamped on a missing-header path")
	}
}

func TestApplyToLLMPre_ShadowMode_BadMacaroonPassesThrough(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, "this-is-not-a-real-macaroon")

	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("shadow mode should not reject bad macaroon, got %+v", sc)
	}
}

func TestApplyToLLMPre_EnforceMode_RejectsMissingHeader(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	bctx := newBifrostCtx()
	sc := ApplyToLLMPre(bctx)
	if sc == nil {
		t.Fatal("enforce mode should short-circuit on missing header")
	}
	if sc.Error == nil || sc.Error.Error == nil || sc.Error.Error.Code == nil {
		t.Fatalf("short-circuit missing code: %+v", sc)
	}
	if *sc.Error.Error.Code != "macaroon_required" {
		t.Fatalf("want macaroon_required, got %s", *sc.Error.Error.Code)
	}
	if sc.Error.StatusCode == nil || *sc.Error.StatusCode != 401 {
		t.Fatalf("want 401, got %+v", sc.Error.StatusCode)
	}
	if sc.Error.AllowFallbacks == nil || *sc.Error.AllowFallbacks {
		t.Fatal("auth errors must not allow fallbacks")
	}
}

func TestApplyToLLMPre_EnforceMode_RejectsBadMacaroon(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, "not-base64")

	sc := ApplyToLLMPre(bctx)
	if sc == nil {
		t.Fatal("enforce mode should reject malformed macaroon")
	}
	if sc.Error.Error == nil || sc.Error.Error.Code == nil {
		t.Fatal("error.code missing")
	}
}

func TestApplyToLLMPre_Happy_StampsClaims(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)

	sc := ApplyToLLMPre(bctx)
	if sc != nil {
		t.Fatalf("happy path short-circuited: %+v", sc)
	}
	claims := pluginctx.VerifiedClaims(bctx)
	if claims == nil {
		t.Fatal("VerifiedClaims not stamped on success")
	}
	if claims.OrgID != testOrgID || claims.UserID != testUserID {
		t.Fatalf("wrong claims stamped: %+v", claims)
	}
	if pluginctx.LeafRunID(bctx) != claims.RunID {
		t.Fatalf("LeafRunID not stamped: got %q want %q",
			pluginctx.LeafRunID(bctx), claims.RunID)
	}
	if pluginctx.LeafAgent(bctx) != claims.AgentName {
		t.Fatalf("LeafAgent not stamped: got %q want %q",
			pluginctx.LeafAgent(bctx), claims.AgentName)
	}
}

func TestApplyToLLMPre_ShadowMode_StampsClaimsOnSuccess(t *testing.T) {
	// Shadow mode still stamps claims on the success path so
	// downstream hooks (PostLLMHook, future cost accumulators) can
	// observe the verified shape — that's the whole point of shadow.
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)

	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("happy path short-circuited in shadow: %+v", sc)
	}
	if pluginctx.VerifiedClaims(bctx) == nil {
		t.Fatal("shadow mode should still stamp claims on success")
	}
}

func TestApplyToLLMPre_EnforceMode_RedisDownFailsClosed(t *testing.T) {
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: true})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	// Bring miniredis up then close it to simulate an outage. The
	// adapter must reject (fail-closed).
	mr := newMiniRedis(t)
	mr.Close()

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)

	sc := ApplyToLLMPre(bctx)
	if sc == nil {
		t.Fatal("enforce mode must fail-closed on redis outage")
	}
	if sc.Error.Error == nil || sc.Error.Error.Code == nil ||
		*sc.Error.Error.Code != "revocation_check_unavailable" {
		t.Fatalf("want revocation_check_unavailable, got %+v", sc.Error)
	}
}

// ─── phase 11: realm-membership check ────────────────────────────────

// realmOpts produces macaroonOptions whose UA carries a
// realm_budgets map covering the supplied realms. The invocation
// inherits the same set (no narrowing), so claims.PermittedRealms
// equals `realms`.
func realmOpts(now time.Time, realms ...string) macaroonOptions {
	opts := defaultMacaroonOptions(now)
	rb := map[string]macaroon.RealmBudget{}
	for _, r := range realms {
		rb[r] = macaroon.RealmBudget{MaxTotalUSD: 100}
	}
	opts.budget = &macaroon.Budget{RealmBudgets: rb}
	return opts
}

func TestEvaluate_SimpleDeployment_NoRealmsAnywhere(t *testing.T) {
	// Both swarm and macaroon are realm-naive → no membership check.
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	d := Evaluate(context.Background(), encoded)
	if d.Err != nil {
		t.Fatalf("simple deployment should pass: %+v", d.Err)
	}
}

func TestEvaluate_MultiRealmSwarmAndMacaroon_Permitted(t *testing.T) {
	reg := newTestRegistry(t)
	if _, err := reg.SetRealmID("w1"); err != nil {
		t.Fatalf("set realm_id: %v", err)
	}
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, realmOpts(time.Now(), "w1", "w2"))
	d := Evaluate(context.Background(), encoded)
	if d.Err != nil {
		t.Fatalf("permitted realm should pass: %+v", d.Err)
	}
	if d.Claims == nil || len(d.Claims.PermittedRealms) != 2 {
		t.Fatalf("permitted_realms not surfaced: %+v", d.Claims)
	}
}

func TestEvaluate_MultiRealmSwarmAndMacaroon_NotPermitted(t *testing.T) {
	reg := newTestRegistry(t)
	if _, err := reg.SetRealmID("w3"); err != nil {
		t.Fatalf("set realm_id: %v", err)
	}
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t)

	// macaroon authorizes w1+w2 only; swarm claims w3 → reject.
	encoded := buildMacaroon(t, realmOpts(time.Now(), "w1", "w2"))
	d := Evaluate(context.Background(), encoded)
	if d.Err == nil || d.Err.Code != "realm_not_permitted" {
		t.Fatalf("want realm_not_permitted, got %+v", d.Err)
	}
}

func TestEvaluate_SwarmIDOnly_NoMacaroonRealms_Accepts(t *testing.T) {
	// Case 3 from the phase doc: swarm has realm_id but the
	// macaroon didn't scope per-realm — accept and rely on the
	// non-realm caps.
	reg := newTestRegistry(t)
	if _, err := reg.SetRealmID("w1"); err != nil {
		t.Fatalf("set realm_id: %v", err)
	}
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	d := Evaluate(context.Background(), encoded)
	if d.Err != nil {
		t.Fatalf("non-realm macaroon on multi-realm swarm should pass: %+v", d.Err)
	}
}

func TestEvaluate_MacaroonRealmsOnly_NoSwarmID_Misconfigured(t *testing.T) {
	// Case 4: multi-realm macaroon, swarm has no identity to match
	// against — treat as a configuration error.
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	_ = newMiniRedis(t)

	encoded := buildMacaroon(t, realmOpts(time.Now(), "w1", "w2"))
	d := Evaluate(context.Background(), encoded)
	if d.Err == nil || d.Err.Code != "realm_not_configured" {
		t.Fatalf("want realm_not_configured, got %+v", d.Err)
	}
}

func TestApplyToLLMPre_ShadowMode_RedisDownPassesThrough(t *testing.T) {
	// Mirror of the above for shadow: even with redis down + a
	// macaroon present, shadow mode must NOT block the request.
	// This is the rollout-safety property.
	reg := newTestRegistry(t)
	SetTrustRegistry(reg)
	t.Cleanup(func() { SetTrustRegistry(nil) })
	SetConfigForTest(Config{EnforceMacaroons: false})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	mr := newMiniRedis(t)
	mr.Close()

	encoded := buildMacaroon(t, defaultMacaroonOptions(time.Now()))
	bctx := newBifrostCtx()
	pluginctx.SetRawMacaroon(bctx, encoded)

	if sc := ApplyToLLMPre(bctx); sc != nil {
		t.Fatalf("shadow mode should not block on redis outage: %+v", sc)
	}
}
