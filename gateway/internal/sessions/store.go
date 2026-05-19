// Package sessions owns the plugin's browser-session store.
//
// Why this exists
// ---------------
// Phase 8's observability dashboard authenticates browsers with a
// cookie issued by `POST /_plugin/login` and validated on every
// subsequent `/_plugin/*` request. The cookie value is a random
// 32-byte ID; the actual session state (which user, when issued,
// when last seen) lives in Redis keyed by that ID. Server-side
// state — rather than a stateless JWT — buys us real logout, real
// admin-kick, and TTL on inactive sessions; phase 8's plan
// ("Session storage (Redis)") locks this in.
//
// Schema
// ------
//
//	bifrost:session:<random_id>     HASH    { user, iat, last_seen }   TTL 8h
//	bifrost:sessions:<user>         SET     <session_id, ...>          (no TTL)
//
// `last_seen` refreshes at most once per minute per session
// (write-skipping) to avoid a hot key on chatty UIs. Every refresh
// also resets the hash TTL.
//
// Failure modes
// -------------
// All methods return errors verbatim from go-redis. The middleware
// (gateway/internal/adminapi/session.go) treats any error as
// "session not found" and fails closed by redirecting the caller to
// the login page. Logging in itself fails if Redis is unreachable —
// that's the correct posture for an auth subsystem (fail-closed),
// and matches the broader phase-6 contract.
package sessions

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// SessionTTL is how long an idle session lives before Redis expires
// it. Phase 8 settled on 8h — long enough to cover a workday so
// operators aren't re-typing the password constantly, short enough
// that a forgotten browser tab on a shared machine isn't a permanent
// foothold.
const SessionTTL = 8 * time.Hour

// LastSeenRefreshInterval is the minimum gap between successive
// writes that update `last_seen` on the session hash. Without this
// every request would touch Redis just to bump a timestamp, which
// would pin a hot key on a chatty UI. One minute is granular enough
// for "is this session active?" admin views without being expensive.
const LastSeenRefreshInterval = time.Minute

// ErrNotFound is returned by Get / Refresh when the session ID
// doesn't resolve to a Redis hash. Callers — exclusively the
// session middleware in adminapi — treat this as "no session" and
// redirect to /login.
var ErrNotFound = errors.New("session not found")

// ErrStoreUnavailable is returned when the package was loaded with no
// Redis client wired up (observability mode). Login flows refuse to
// issue cookies in this state — there'd be nowhere to persist them.
var ErrStoreUnavailable = errors.New("session store unavailable (redis not configured)")

// Session is the value returned by Get(). The fields mirror the
// Redis hash exactly so callers can read them without a second
// round-trip.
type Session struct {
	ID       string    `json:"id"`
	User     string    `json:"user"`
	IssuedAt time.Time `json:"iat"`
	LastSeen time.Time `json:"last_seen"`
}

// Store is the session API the rest of the plugin consumes. Backed
// by Redis in production; tests inject a miniredis-backed client via
// redisclient.SetClientForTest.
//
// Construction is intentionally trivial — there's nothing to
// configure beyond "talk to whatever Redis the package can find."
type Store struct {
	// client is captured at construction so a later Redis
	// reconnection (which would swap redisclient.Client()) doesn't
	// accidentally rebind an in-flight store to a different
	// instance. In practice the plugin opens Redis once at boot.
	client *redis.Client
}

// NewStore returns a Store wired to the package-level Redis client.
// Returns nil if redisclient ran in observability mode (no env var) —
// callers MUST nil-check; the login handler refuses to start the
// flow when the store is nil so we never write half-state.
func NewStore() *Store {
	c := redisclient.Client()
	if c == nil {
		return nil
	}
	return &Store{client: c}
}

// Create issues a new session for `user`. Returns the random 32-byte
// session ID (base64url, 43 chars) the caller should set as the
// cookie value, plus the populated Session struct.
//
// Implementation notes
// --------------------
// - Two writes happen in a pipeline: HSET the hash, SADD the user's
//   session index. Pipelining cuts the round-trip in half on the
//   hot path (every login).
// - Pipeline failures leave Redis with a partial state at worst (a
//   hash without an index entry, or vice versa). The hash has a
//   TTL, so the orphan is bounded. The index lives forever so a
//   logout-everywhere is still correct. Both kinds of orphan are
//   harmless (no auth bypass; lookups go through the hash).
func (s *Store) Create(ctx context.Context, user string) (string, *Session, error) {
	if s == nil {
		return "", nil, ErrStoreUnavailable
	}
	id, err := newSessionID()
	if err != nil {
		return "", nil, fmt.Errorf("session: id: %w", err)
	}
	now := time.Now().UTC()
	sess := &Session{
		ID:       id,
		User:     user,
		IssuedAt: now,
		LastSeen: now,
	}

	pipe := s.client.Pipeline()
	pipe.HSet(ctx, sessionKey(id), map[string]any{
		"user":      user,
		"iat":       strconv.FormatInt(now.Unix(), 10),
		"last_seen": strconv.FormatInt(now.Unix(), 10),
	})
	pipe.Expire(ctx, sessionKey(id), SessionTTL)
	pipe.SAdd(ctx, userIndexKey(user), id)
	if _, err := pipe.Exec(ctx); err != nil {
		return "", nil, fmt.Errorf("session: create: %w", err)
	}
	return id, sess, nil
}

// Get returns the session for `id`, or ErrNotFound if expired /
// unknown. Does NOT update `last_seen` — call Refresh for that.
// Separating the two means GET-heavy hooks (e.g. a future websocket
// keepalive) don't pay the write tax.
func (s *Store) Get(ctx context.Context, id string) (*Session, error) {
	if s == nil {
		return nil, ErrStoreUnavailable
	}
	if id == "" {
		return nil, ErrNotFound
	}
	vals, err := s.client.HGetAll(ctx, sessionKey(id)).Result()
	if err != nil {
		return nil, fmt.Errorf("session: get: %w", err)
	}
	if len(vals) == 0 {
		return nil, ErrNotFound
	}
	user := vals["user"]
	iat, _ := strconv.ParseInt(vals["iat"], 10, 64)
	ls, _ := strconv.ParseInt(vals["last_seen"], 10, 64)
	if user == "" {
		// Hash exists but is malformed — treat as not found.
		// Don't bother deleting; TTL will sweep it.
		return nil, ErrNotFound
	}
	return &Session{
		ID:       id,
		User:     user,
		IssuedAt: time.Unix(iat, 0).UTC(),
		LastSeen: time.Unix(ls, 0).UTC(),
	}, nil
}

// Refresh bumps `last_seen` and resets the hash's TTL. Skips the
// write if the existing `last_seen` is within LastSeenRefreshInterval
// of now — see the package doc for the rationale.
//
// Returns the (possibly stale) Session. Callers in the middleware
// path use this to populate request context without a separate Get.
func (s *Store) Refresh(ctx context.Context, id string) (*Session, error) {
	sess, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	if now.Sub(sess.LastSeen) < LastSeenRefreshInterval {
		// Skip the write; the existing TTL is still well above
		// any reasonable poll cadence so we're not at risk of
		// expiring an active session.
		return sess, nil
	}
	pipe := s.client.Pipeline()
	pipe.HSet(ctx, sessionKey(id), "last_seen", strconv.FormatInt(now.Unix(), 10))
	pipe.Expire(ctx, sessionKey(id), SessionTTL)
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("session: refresh: %w", err)
	}
	sess.LastSeen = now
	return sess, nil
}

// Delete removes one session. Idempotent — deleting an already-gone
// session is not an error. Removes the index entry too so
// DeleteAllForUser stays consistent.
func (s *Store) Delete(ctx context.Context, id string) error {
	if s == nil {
		return ErrStoreUnavailable
	}
	// Fetch the user first so we can clean the index. If the hash
	// is already gone, skip the SREM (nothing to clean).
	user, err := s.client.HGet(ctx, sessionKey(id), "user").Result()
	if err != nil && err != redis.Nil {
		return fmt.Errorf("session: delete lookup: %w", err)
	}
	pipe := s.client.Pipeline()
	pipe.Del(ctx, sessionKey(id))
	if user != "" {
		pipe.SRem(ctx, userIndexKey(user), id)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("session: delete: %w", err)
	}
	return nil
}

// DeleteAllForUser kicks every session belonging to `user`. Phase 8
// doesn't expose this through any handler, but the primitive needs
// to exist before phase 9's "kick this admin" button can land — and
// it's worth landing here so the storage contract is complete.
func (s *Store) DeleteAllForUser(ctx context.Context, user string) error {
	if s == nil {
		return ErrStoreUnavailable
	}
	ids, err := s.client.SMembers(ctx, userIndexKey(user)).Result()
	if err != nil {
		return fmt.Errorf("session: list user: %w", err)
	}
	if len(ids) == 0 {
		return nil
	}
	pipe := s.client.Pipeline()
	for _, id := range ids {
		pipe.Del(ctx, sessionKey(id))
	}
	pipe.Del(ctx, userIndexKey(user))
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("session: delete user: %w", err)
	}
	return nil
}

// ─── helpers ──────────────────────────────────────────────────────────

func sessionKey(id string) string  { return redisclient.Key("session:" + id) }
func userIndexKey(u string) string { return redisclient.Key("sessions:" + u) }

// newSessionID returns a 43-char base64url-encoded 32-byte random
// identifier. The base64url alphabet is cookie-safe (no padding,
// no `/` or `+`) so the value can be dropped straight into
// Set-Cookie without escaping.
func newSessionID() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b[:]), nil
}
