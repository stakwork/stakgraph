package auth

import (
	"context"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

func TestCheckRevocations_ObservabilityMode_NoOp(t *testing.T) {
	// Ensure no client (in case a prior test set one).
	redisclient.SetClientForTest(nil)
	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    time.Now().UTC().Format(time.RFC3339),
	}
	if err := CheckRevocations(context.Background(), claims); err != nil {
		t.Fatalf("observability mode should be a no-op, got %+v", err)
	}
}

func TestCheckRevocations_AllClear(t *testing.T) {
	mr := newMiniRedis(t)
	_ = mr // just need it spun up; no keys set

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{
			"aaaa000000000000000000000000aaaa",
			"bbbb000000000000000000000000bbbb",
		},
		IAT: time.Now().UTC().Format(time.RFC3339),
	}
	if err := CheckRevocations(context.Background(), claims); err != nil {
		t.Fatalf("expected nil error, got %+v", err)
	}
}

func TestCheckRevocations_UANonceRevoked(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke:aaaa000000000000000000000000aaaa", "1")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{
			"aaaa000000000000000000000000aaaa", // ua nonce — index 0
			"bbbb000000000000000000000000bbbb",
		},
		IAT: time.Now().UTC().Format(time.RFC3339),
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil {
		t.Fatal("expected rejection")
	}
	if err.Code != "user_authorization_revoked" {
		t.Fatalf("want user_authorization_revoked (ua nonce is index 0), got %s", err.Code)
	}
}

func TestCheckRevocations_InvNonceRevoked(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke:bbbb000000000000000000000000bbbb", "1")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{
			"aaaa000000000000000000000000aaaa",
			"bbbb000000000000000000000000bbbb", // inv nonce — index 1
		},
		IAT: time.Now().UTC().Format(time.RFC3339),
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil {
		t.Fatal("expected rejection")
	}
	if err.Code != "macaroon_revoked" {
		t.Fatalf("want macaroon_revoked for non-ua layer, got %s", err.Code)
	}
}

func TestCheckRevocations_AttenuationNonceRevoked(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke:cccc000000000000000000000000cccc", "1")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{
			"aaaa000000000000000000000000aaaa",
			"bbbb000000000000000000000000bbbb",
			"cccc000000000000000000000000cccc", // attenuation nonce
		},
		IAT: time.Now().UTC().Format(time.RFC3339),
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil {
		t.Fatal("expected rejection")
	}
	if err.Code != "macaroon_revoked" {
		t.Fatalf("want macaroon_revoked, got %s", err.Code)
	}
}

func TestCheckRevocations_UserCutoff_RejectsOldUA(t *testing.T) {
	mr := newMiniRedis(t)
	// User offboarded at 2026-01-01 — everything before that is dead.
	mr.Set("bifrost:revoke_user_before:"+testUserID, "2026-01-01T00:00:00Z")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		// UA issued before the cutoff (Claims.IAT is invocation.iat
		// in the current pure verifier, but for revocation purposes
		// we're comparing against the same time the UA's iat would
		// most likely match — see uaIATFromClaims doc comment).
		IAT: "2025-12-01T00:00:00Z",
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil {
		t.Fatal("expected user-level revocation rejection")
	}
	if err.Code != "user_authorization_revoked" {
		t.Fatalf("want user_authorization_revoked, got %s", err.Code)
	}
}

func TestCheckRevocations_UserCutoff_AcceptsNewerUA(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke_user_before:"+testUserID, "2026-01-01T00:00:00Z")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    "2026-06-15T00:00:00Z", // after cutoff
	}
	if err := CheckRevocations(context.Background(), claims); err != nil {
		t.Fatalf("post-cutoff UA should pass, got %+v", err)
	}
}

func TestCheckRevocations_UserCutoff_MalformedFailsOpen(t *testing.T) {
	mr := newMiniRedis(t)
	// Operator typo: stored a garbage cutoff. Phase-4 adapter chooses
	// fail-open on this specific path so one bad write doesn't lock
	// out an entire user — the nonce-level checks still apply.
	mr.Set("bifrost:revoke_user_before:"+testUserID, "not-a-time")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    "2026-06-15T00:00:00Z",
	}
	if err := CheckRevocations(context.Background(), claims); err != nil {
		t.Fatalf("malformed cutoff should fail-open, got %+v", err)
	}
}

func TestCheckRevocations_RedisDown_FailsClosed(t *testing.T) {
	mr := newMiniRedis(t)
	// Simulate Redis going away mid-flight.
	mr.Close()

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    time.Now().UTC().Format(time.RFC3339),
	}
	err := CheckRevocations(context.Background(), claims)
	if err == nil {
		t.Fatal("expected revocation_check_unavailable when redis is dead")
	}
	if err.Code != "revocation_check_unavailable" {
		t.Fatalf("want revocation_check_unavailable, got %s", err.Code)
	}
	if err.HTTPStatus != 401 {
		t.Fatalf("want 401 (fail-closed), got %d", err.HTTPStatus)
	}
}

func TestCheckRevocations_NilClaims(t *testing.T) {
	mr := newMiniRedis(t)
	_ = mr
	err := CheckRevocations(context.Background(), nil)
	if err == nil || err.Code != "verify_internal_error" {
		t.Fatalf("want verify_internal_error, got %+v", err)
	}
}

func TestCheckRevocations_KeyNamespace(t *testing.T) {
	// Operational contract: every key carries the bifrost: prefix.
	// If someone accidentally writes a raw "revoke:..." it MUST NOT
	// match — verifies the namespace plumbing is honored.
	mr := newMiniRedis(t)
	// WRONG key — no bifrost: prefix.
	mr.Set("revoke:aaaa000000000000000000000000aaaa", "1")

	claims := &macaroon.Claims{
		UserID: testUserID,
		Nonces: []string{"aaaa000000000000000000000000aaaa"},
		IAT:    time.Now().UTC().Format(time.RFC3339),
	}
	if err := CheckRevocations(context.Background(), claims); err != nil {
		t.Fatalf("un-prefixed key must not match; got %+v", err)
	}
}
