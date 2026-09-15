package auth

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Admin-side helpers for managing revocation state. The HTTP routes
// that expose these (`/_plugin/revoke/nonce/:nonce`,
// `/_plugin/revoke/user/:user_id`, `/_plugin/revoke/users`) live in
// gateway/internal/adminapi (revoke.go) and call into this file. The
// kill-switch and per-run / per-agent state primitives live next door
// in kill.go.
//
// These helpers exist so a swarm operator (or Hive, fanning out) can:
//
//   - Revoke a specific nonce (any layer: UA, invocation, attenuation)
//     before its natural exp. Use case: an employee leaves mid-week
//     and the org leader pushes a kill before re-issuing.
//   - Set a user-level cutoff (revoke_user_before) so every macaroon
//     issued before time T for that user is rejected. This is the
//     dashboard's third kill axis next to run and agent: every
//     in-flight run and every new spawn under the user's current
//     authorization stops on its next LLM call, until Hive issues a
//     fresh authorization. Per swarm, like the other two.
//
// Every helper is a thin wrapper around the redis ops described in
// gateway/plans/phases/phase-6-plugin-enforcement.md "Redis schema".

// revokeUsersIndexKey is the ZSET that lists every user with a cutoff
// set: member = user_id, score = cutoff as unix seconds. Maintained by
// SetUserRevokeCutoff / ClearUserRevokeCutoff so the dashboard's
// People list and Hive's reconcile sweep can read "who is revoked on
// this swarm" in one round trip instead of scanning the keyspace. The
// per-user string key stays the source of truth — the hot path never
// reads the index, and ListUserRevokeCutoffs re-reads the string keys
// and prunes members whose key is gone (a direct Redis DEL, or a
// cutoff written before this index existed and cleared since).
const revokeUsersIndexKey = "revoke_users"

// adminTimeout bounds a single admin Redis op. These run outside the
// request hot path so we can afford to be more generous than
// pipelineTimeout.
const adminTimeout = 3 * time.Second

// ErrRedisUnavailable is returned by every admin helper when
// redisclient.Client() is nil. Distinguished from generic errors so
// the HTTP layer can return 503 (not 500) — the operator can re-try
// after fixing the Redis link without changing their request.
var ErrRedisUnavailable = errors.New("redis client not configured")

// RevokeNonce writes a tombstone for `nonce` with the given TTL. The
// caller is expected to compute the TTL from the macaroon layer's
// exp — see runKeyTTL. Idempotent: re-revoking is a no-op (SET with
// the same value).
//
// Phase-6 schema: bifrost:revoke:<nonce> = "1", TTL = layer.exp.
func RevokeNonce(ctx context.Context, nonce string, ttl time.Duration) error {
	if err := validateNonce(nonce); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Set(octx, redisclient.Key(revokePrefix+nonce), "1", ttl).Err()
}

// RevocationTTL computes the tombstone TTL for a nonce whose layer
// expires at `layerExp` — the same clamp(exp-now+1h, 1h, 7d) formula
// the accumulators use, so a revoke outlives the macaroon by the
// grace hour and never lingers past the 7d ceiling. Exported for the
// adminapi revoke handler.
func RevocationTTL(layerExp, now time.Time) time.Duration {
	return runKeyTTL(layerExp, now)
}

// UnrevokeNonce removes a revocation tombstone. Mostly for operator
// recovery from a mistakenly-pressed revoke button; revocations are
// supposed to be permanent within the macaroon's lifetime.
func UnrevokeNonce(ctx context.Context, nonce string) error {
	if err := validateNonce(nonce); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Del(octx, redisclient.Key(revokePrefix+nonce)).Err()
}

// SetUserRevokeCutoff writes the revoke_user_before:<user_id> key.
// Any user_authorization with iat strictly before `cutoff` will be
// rejected from now on. No TTL — user-level revocation is a
// permanent state until explicitly cleared (phase-6 schema).
//
// `cutoff` is normalized to UTC and serialized as RFC 3339.
//
// Also records the user in the revoke_users index (score = cutoff)
// so ListUserRevokeCutoffs can enumerate without a keyspace scan.
func SetUserRevokeCutoff(ctx context.Context, userID string, cutoff time.Time) error {
	if err := validateKillID("user_id", userID); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	cutoff = cutoff.UTC()
	pipe := rdb.Pipeline()
	pipe.Set(octx,
		redisclient.Key(revokeUserBeforePrefix+userID),
		cutoff.Format(time.RFC3339),
		0,
	)
	pipe.ZAdd(octx, redisclient.Key(revokeUsersIndexKey), redis.Z{
		Score:  float64(cutoff.Unix()),
		Member: userID,
	})
	_, err := pipe.Exec(octx)
	return err
}

// ClearUserRevokeCutoff removes the revoke_user_before:<user_id>
// entry, re-allowing macaroons issued before the previous cutoff.
func ClearUserRevokeCutoff(ctx context.Context, userID string) error {
	if err := validateKillID("user_id", userID); err != nil {
		return err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	pipe := rdb.Pipeline()
	pipe.Del(octx, redisclient.Key(revokeUserBeforePrefix+userID))
	pipe.ZRem(octx, redisclient.Key(revokeUsersIndexKey), userID)
	_, err := pipe.Exec(octx)
	return err
}

// GetUserRevokeCutoff returns the currently configured cutoff for
// `userID`, or zero time + ok=false if none is set. Useful for the
// admin dashboard and for reconciler verification.
func GetUserRevokeCutoff(ctx context.Context, userID string) (time.Time, bool, error) {
	if err := validateKillID("user_id", userID); err != nil {
		return time.Time{}, false, err
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return time.Time{}, false, ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	raw, err := rdb.Get(octx, redisclient.Key(revokeUserBeforePrefix+userID)).Result()
	if errors.Is(err, redis.Nil) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, err
	}
	t, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return time.Time{}, false, fmt.Errorf("malformed cutoff %q: %w", raw, err)
	}
	return t, true, nil
}

// UserRevokeCutoff is one entry of the revoke_users index, re-read
// from its authoritative string key.
type UserRevokeCutoff struct {
	UserID string
	Before time.Time
}

// maxUserRevokeList caps ListUserRevokeCutoffs. A swarm with more
// revoked users than this has an offboarding problem the dashboard
// isn't the tool for; Hive can page by user instead.
const maxUserRevokeList = 500

// ListUserRevokeCutoffs returns every user with a cutoff set on this
// swarm, newest cutoff first, capped at `limit` (≤ 0 ⇒ the default
// cap). Reads the revoke_users index for the members, then the
// per-user string keys for the values: the string key is what the
// hot path enforces, so it wins. Members whose string key is gone
// are dropped from the result and pruned from the index in the same
// call, which keeps the index honest against direct Redis writes
// without a separate sweeper.
//
// Absent index ⇒ empty list, not an error. Users whose cutoff was
// written before the index existed are not listed until re-set.
func ListUserRevokeCutoffs(ctx context.Context, limit int64) ([]UserRevokeCutoff, error) {
	rdb := redisclient.Client()
	if rdb == nil {
		return nil, ErrRedisUnavailable
	}
	if limit <= 0 || limit > maxUserRevokeList {
		limit = maxUserRevokeList
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()

	indexKey := redisclient.Key(revokeUsersIndexKey)
	ids, err := rdb.ZRevRange(octx, indexKey, 0, limit-1).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, fmt.Errorf("revoke_users index: %w", err)
	}
	out := make([]UserRevokeCutoff, 0, len(ids))
	if len(ids) == 0 {
		return out, nil
	}

	keys := make([]string, len(ids))
	for i, id := range ids {
		keys[i] = redisclient.Key(revokeUserBeforePrefix + id)
	}
	vals, err := rdb.MGet(octx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("revoke_user_before mget: %w", err)
	}

	var stale []any
	for i, id := range ids {
		raw, ok := vals[i].(string)
		if !ok {
			stale = append(stale, id)
			continue
		}
		// The hot path fails open on a malformed cutoff; the list is
		// a diagnostic surface, so a key that exists but won't parse
		// is still listed, with a zero Before, rather than hidden.
		t, _ := time.Parse(time.RFC3339, raw)
		out = append(out, UserRevokeCutoff{UserID: id, Before: t})
	}
	if len(stale) > 0 {
		// Best effort: a failed prune leaves the index slightly
		// over-full, which the next list call retries.
		_ = rdb.ZRem(octx, indexKey, stale...).Err()
	}
	return out, nil
}

// validateNonce rejects obviously-malformed nonces before they reach
// Redis. Phase-4 nonces are 16 random bytes hex-encoded to 32 chars
// (lowercase). Anything else is operator typo territory.
func validateNonce(n string) error {
	if len(n) != 32 {
		return fmt.Errorf("nonce must be 32 hex chars, got %d", len(n))
	}
	if _, err := hex.DecodeString(n); err != nil {
		return fmt.Errorf("nonce is not valid hex: %w", err)
	}
	return nil
}
