package auth

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

const adminTestNonce = "deadbeefcafef00d0001020304050607"

func TestRevokeNonce_HappyPath(t *testing.T) {
	mr := newMiniRedis(t)
	if err := RevokeNonce(context.Background(), adminTestNonce, 5*time.Minute); err != nil {
		t.Fatalf("RevokeNonce: %v", err)
	}
	got, err := mr.Get("bifrost:revoke:" + adminTestNonce)
	if err != nil {
		t.Fatalf("mr.Get: %v", err)
	}
	if got != "1" {
		t.Fatalf("expected '1', got %q", got)
	}
}

func TestRevokeNonce_HonorsNamespace(t *testing.T) {
	mr := newMiniRedis(t)
	if err := RevokeNonce(context.Background(), adminTestNonce, time.Hour); err != nil {
		t.Fatalf("RevokeNonce: %v", err)
	}
	// Should not have written under the un-prefixed key.
	if _, err := mr.Get("revoke:" + adminTestNonce); err == nil {
		t.Fatal("write landed without bifrost: prefix — namespace contract broken")
	}
}

func TestRevokeNonce_RejectsMalformedNonce(t *testing.T) {
	_ = newMiniRedis(t) // make sure client is configured so we hit validation, not nil-check
	for _, bad := range []string{"", "short", "ZZZZ" + adminTestNonce[4:], adminTestNonce + "extra"} {
		if err := RevokeNonce(context.Background(), bad, time.Hour); err == nil {
			t.Fatalf("expected error for nonce=%q", bad)
		}
	}
}

func TestRevokeNonce_RedisUnavailable(t *testing.T) {
	redisclient.SetClientForTest(nil)
	t.Cleanup(func() { redisclient.SetClientForTest(nil) })
	err := RevokeNonce(context.Background(), adminTestNonce, time.Hour)
	if !errors.Is(err, ErrRedisUnavailable) {
		t.Fatalf("want ErrRedisUnavailable, got %v", err)
	}
}

func TestUnrevokeNonce_RemovesTombstone(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke:"+adminTestNonce, "1")
	if err := UnrevokeNonce(context.Background(), adminTestNonce); err != nil {
		t.Fatalf("UnrevokeNonce: %v", err)
	}
	if _, err := mr.Get("bifrost:revoke:" + adminTestNonce); err == nil {
		t.Fatal("expected key to be removed")
	}
}

func TestSetUserRevokeCutoff_RoundTrip(t *testing.T) {
	_ = newMiniRedis(t)
	cutoff := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	if err := SetUserRevokeCutoff(context.Background(), "u_alice", cutoff); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, ok, err := GetUserRevokeCutoff(context.Background(), "u_alice")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !ok {
		t.Fatal("ok=false after Set")
	}
	if !got.Equal(cutoff) {
		t.Fatalf("round-trip mismatch: %v vs %v", got, cutoff)
	}
}

func TestGetUserRevokeCutoff_Missing(t *testing.T) {
	_ = newMiniRedis(t)
	_, ok, err := GetUserRevokeCutoff(context.Background(), "u_nobody")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ok {
		t.Fatal("ok=true for missing key")
	}
}

func TestClearUserRevokeCutoff(t *testing.T) {
	mr := newMiniRedis(t)
	mr.Set("bifrost:revoke_user_before:u_alice", "2026-06-15T12:00:00Z")
	if err := ClearUserRevokeCutoff(context.Background(), "u_alice"); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	if _, err := mr.Get("bifrost:revoke_user_before:u_alice"); err == nil {
		t.Fatal("expected key to be cleared")
	}
}

func TestSetUserRevokeCutoff_RequiresUserID(t *testing.T) {
	_ = newMiniRedis(t)
	if err := SetUserRevokeCutoff(context.Background(), "", time.Now()); err == nil {
		t.Fatal("empty user_id should error")
	}
}

func TestSetUserRevokeCutoff_MaintainsIndex(t *testing.T) {
	mr := newMiniRedis(t)
	cutoff := time.Date(2026, 6, 15, 12, 0, 0, 0, time.UTC)
	if err := SetUserRevokeCutoff(context.Background(), "u_alice", cutoff); err != nil {
		t.Fatalf("Set: %v", err)
	}
	score, err := mr.ZScore("bifrost:revoke_users", "u_alice")
	if err != nil {
		t.Fatalf("index member missing: %v", err)
	}
	if int64(score) != cutoff.Unix() {
		t.Fatalf("index score %v, want %d", score, cutoff.Unix())
	}
	if err := ClearUserRevokeCutoff(context.Background(), "u_alice"); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	if indexHas(mr, "u_alice") {
		t.Fatal("index member should be removed on clear")
	}
}

// indexHas reports membership in the revoke_users ZSET. miniredis's
// ZScore returns (0, nil) for a missing member of an existing key, so
// it can't answer this on its own.
func indexHas(mr *miniredis.Miniredis, member string) bool {
	members, err := mr.ZMembers("bifrost:revoke_users")
	if err != nil {
		return false
	}
	for _, m := range members {
		if m == member {
			return true
		}
	}
	return false
}

func TestListUserRevokeCutoffs_NewestFirst_And_SelfHeals(t *testing.T) {
	mr := newMiniRedis(t)
	ctx := context.Background()
	older := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	newer := time.Date(2026, 9, 2, 0, 0, 0, 0, time.UTC)
	if err := SetUserRevokeCutoff(ctx, "u_older", older); err != nil {
		t.Fatal(err)
	}
	if err := SetUserRevokeCutoff(ctx, "u_newer", newer); err != nil {
		t.Fatal(err)
	}

	got, err := ListUserRevokeCutoffs(ctx, 0)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 2 || got[0].UserID != "u_newer" || got[1].UserID != "u_older" {
		t.Fatalf("order: %+v", got)
	}
	if !got[0].Before.Equal(newer) || !got[1].Before.Equal(older) {
		t.Fatalf("cutoffs: %+v", got)
	}

	// Limit is honoured, newest kept.
	got, err = ListUserRevokeCutoffs(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].UserID != "u_newer" {
		t.Fatalf("limit=1: %+v", got)
	}

	// A direct DEL of the string key (Hive / an operator in redis-cli)
	// must drop the entry from the list and prune the index.
	mr.Del("bifrost:revoke_user_before:u_newer")
	got, err = ListUserRevokeCutoffs(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].UserID != "u_older" {
		t.Fatalf("after direct del: %+v", got)
	}
	if indexHas(mr, "u_newer") {
		t.Fatal("stale index member not pruned")
	}
	if !indexHas(mr, "u_older") {
		t.Fatal("live index member should survive the prune")
	}
}

func TestListUserRevokeCutoffs_EmptyAndUnavailable(t *testing.T) {
	_ = newMiniRedis(t)
	got, err := ListUserRevokeCutoffs(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("empty index: %+v", got)
	}
	redisclient.SetClientForTest(nil)
	if _, err := ListUserRevokeCutoffs(context.Background(), 0); !errors.Is(err, ErrRedisUnavailable) {
		t.Fatalf("no redis: want ErrRedisUnavailable, got %v", err)
	}
}

func TestSetUserRevokeCutoff_RejectsMalformedUserID(t *testing.T) {
	_ = newMiniRedis(t)
	if err := SetUserRevokeCutoff(context.Background(), "has space", time.Now()); err == nil {
		t.Fatal("user_id with whitespace should error")
	}
}
