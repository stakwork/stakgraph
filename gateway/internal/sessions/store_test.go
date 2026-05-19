package sessions

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// newTestStore stands up a miniredis-backed Store. Tests drive
// real Redis-protocol calls through go-redis so we exercise the
// pipeline / expiry / set semantics, not a mock.
func newTestStore(t *testing.T) (*Store, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	c := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() {
		_ = c.Close()
		redisclient.SetClientForTest(nil)
	})
	redisclient.SetClientForTest(c)
	return NewStore(), mr
}

func TestNewStore_ObservabilityMode(t *testing.T) {
	// No client wired in — package returns nil so callers can
	// detect "no session store available" and refuse to log in.
	redisclient.SetClientForTest(nil)
	if s := NewStore(); s != nil {
		t.Fatalf("expected nil store when redis unconfigured, got %v", s)
	}
}

func TestCreate_PersistsAndIndexes(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()

	id, sess, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if len(id) < 40 {
		t.Fatalf("session id looks too short: %q", id)
	}
	if sess.User != "alice" {
		t.Fatalf("user mismatch: %q", sess.User)
	}

	// Hash present with the right user.
	if got := mr.HGet("bifrost:session:"+id, "user"); got != "alice" {
		t.Fatalf("HGET user: %q", got)
	}
	// Index entry present.
	if ok, _ := mr.SIsMember("bifrost:sessions:alice", id); !ok {
		t.Fatalf("user index missing session id")
	}
	// TTL set.
	ttl := mr.TTL("bifrost:session:" + id)
	if ttl <= 0 || ttl > SessionTTL {
		t.Fatalf("unexpected TTL %s", ttl)
	}
}

func TestGet_RoundTrip(t *testing.T) {
	s, _ := newTestStore(t)
	ctx := context.Background()
	id, _, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}

	sess, err := s.Get(ctx, id)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if sess.User != "alice" {
		t.Fatalf("user: %q", sess.User)
	}
	if sess.IssuedAt.IsZero() || sess.LastSeen.IsZero() {
		t.Fatalf("timestamps zero: %+v", sess)
	}
}

func TestGet_NotFound(t *testing.T) {
	s, _ := newTestStore(t)
	ctx := context.Background()
	if _, err := s.Get(ctx, "nonexistent-id"); err != ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
	if _, err := s.Get(ctx, ""); err != ErrNotFound {
		t.Fatalf("empty id should be NotFound, got %v", err)
	}
}

func TestRefresh_WriteSkipping(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()
	id, _, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}

	originalLastSeen := mr.HGet("bifrost:session:"+id, "last_seen")

	// First refresh immediately — should skip the write because
	// `last_seen` is within LastSeenRefreshInterval of now.
	if _, err := s.Refresh(ctx, id); err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if got := mr.HGet("bifrost:session:"+id, "last_seen"); got != originalLastSeen {
		t.Fatalf("write-skipping failed: last_seen changed (%s → %s)", originalLastSeen, got)
	}

	// Rewrite `last_seen` to a value comfortably outside the
	// refresh interval so the next Refresh must take the write
	// path. We rewrite the stored value (rather than FastForward
	// the miniredis clock) because the refresh-interval check
	// compares wall-clock now against the *stored* timestamp, so
	// FastForward alone wouldn't trigger the branch.
	stale := time.Now().Add(-2 * LastSeenRefreshInterval).Unix()
	mr.HSet("bifrost:session:"+id, "last_seen", strconv.FormatInt(stale, 10))
	if _, err := s.Refresh(ctx, id); err != nil {
		t.Fatalf("Refresh after stale: %v", err)
	}
	if got := mr.HGet("bifrost:session:"+id, "last_seen"); got == strconv.FormatInt(stale, 10) {
		t.Fatalf("expected last_seen update after stale rewrite")
	}
}

func TestRefresh_ResetsTTL(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()
	id, _, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}
	// Burn most of the TTL on the miniredis clock.
	mr.FastForward(SessionTTL - LastSeenRefreshInterval)
	// Stale `last_seen` so the refresh branch fires.
	stale := time.Now().Add(-2 * LastSeenRefreshInterval).Unix()
	mr.HSet("bifrost:session:"+id, "last_seen", strconv.FormatInt(stale, 10))

	if _, err := s.Refresh(ctx, id); err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if ttl := mr.TTL("bifrost:session:" + id); ttl < SessionTTL-time.Second {
		t.Fatalf("TTL not reset on refresh: %s", ttl)
	}
}

func TestDelete_Idempotent(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()
	id, _, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}

	if err := s.Delete(ctx, id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if mr.Exists("bifrost:session:" + id) {
		t.Fatalf("session hash not removed")
	}
	if ok, _ := mr.SIsMember("bifrost:sessions:alice", id); ok {
		t.Fatalf("user index still contains deleted id")
	}
	// Second delete is a no-op.
	if err := s.Delete(ctx, id); err != nil {
		t.Fatalf("second Delete returned error: %v", err)
	}
}

func TestDeleteAllForUser(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()

	id1, _, _ := s.Create(ctx, "alice")
	id2, _, _ := s.Create(ctx, "alice")
	idOther, _, _ := s.Create(ctx, "bob")

	if err := s.DeleteAllForUser(ctx, "alice"); err != nil {
		t.Fatalf("DeleteAllForUser: %v", err)
	}
	for _, id := range []string{id1, id2} {
		if mr.Exists("bifrost:session:" + id) {
			t.Fatalf("alice session %s still present", id)
		}
	}
	// bob's session must be untouched.
	if !mr.Exists("bifrost:session:" + idOther) {
		t.Fatalf("bob's session got swept by alice's kick-all")
	}
	// Empty users sweep is a no-op.
	if err := s.DeleteAllForUser(ctx, "alice"); err != nil {
		t.Fatalf("idempotent kick-all: %v", err)
	}
}

func TestTTLExpiry(t *testing.T) {
	s, mr := newTestStore(t)
	ctx := context.Background()
	id, _, err := s.Create(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}

	mr.FastForward(SessionTTL + time.Second)

	if _, err := s.Get(ctx, id); err != ErrNotFound {
		t.Fatalf("session not expired: %v", err)
	}
}
