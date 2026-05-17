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
// that expose these (POST /_plugin/admin/revoke, etc.) live in
// gateway/internal/adminapi and call into this file. Phase-6 will
// grow the kill-switch and per-run/state endpoints alongside these.
//
// These helpers exist now so a swarm operator can:
//
//   - Revoke a specific nonce (any layer: UA, invocation, attenuation)
//     before its natural exp. Use case: an employee leaves mid-week
//     and the org leader pushes a kill before re-issuing.
//   - Set a user-level cutoff (revoke_user_before) so every macaroon
//     issued before time T for that user is rejected. Use case: full
//     user offboarding regardless of how many UAs are out.
//
// Every helper is a thin wrapper around the redis ops described in
// gateway/plans/phases/phase-6-plugin-enforcement.md "Redis schema".

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
func SetUserRevokeCutoff(ctx context.Context, userID string, cutoff time.Time) error {
	if userID == "" {
		return errors.New("user_id is required")
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Set(octx,
		redisclient.Key(revokeUserBeforePrefix+userID),
		cutoff.UTC().Format(time.RFC3339),
		0,
	).Err()
}

// ClearUserRevokeCutoff removes the revoke_user_before:<user_id>
// entry, re-allowing macaroons issued before the previous cutoff.
func ClearUserRevokeCutoff(ctx context.Context, userID string) error {
	if userID == "" {
		return errors.New("user_id is required")
	}
	rdb := redisclient.Client()
	if rdb == nil {
		return ErrRedisUnavailable
	}
	octx, cancel := context.WithTimeout(ctx, adminTimeout)
	defer cancel()
	return rdb.Del(octx, redisclient.Key(revokeUserBeforePrefix+userID)).Err()
}

// GetUserRevokeCutoff returns the currently configured cutoff for
// `userID`, or zero time + ok=false if none is set. Useful for the
// admin dashboard and for reconciler verification.
func GetUserRevokeCutoff(ctx context.Context, userID string) (time.Time, bool, error) {
	if userID == "" {
		return time.Time{}, false, errors.New("user_id is required")
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
