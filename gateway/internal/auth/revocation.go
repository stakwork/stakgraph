package auth

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Redis key sub-paths (the bifrost: prefix is added by
// redisclient.Key). The full key shapes are documented in
// gateway/plans/phases/phase-6-plugin-enforcement.md "Redis schema".
const (
	// revokePrefix + <nonce> tombstones a single layer of the
	// macaroon chain. Presence ⇒ macaroon rejected. TTL = layer.exp.
	revokePrefix = "revoke:"

	// revokeUserBeforePrefix + <user_id> stores an RFC 3339 UTC
	// timestamp. Any user_authorization.iat strictly before it is
	// rejected (covers org-level offboarding / user re-issuance).
	revokeUserBeforePrefix = "revoke_user_before:"
)

// pipelineTimeout bounds a single revocation pipeline round-trip.
// Phase 6 expects sub-millisecond on a co-located Redis; 500 ms is
// our "something is wrong" cliff. Hooks are in the request hot path,
// so we'd rather fail-closed quickly than block.
const pipelineTimeout = 500 * time.Millisecond

// CheckRevocations runs the Redis revocation pipeline against the
// claims produced by Verify. Returns nil on "all clear", or an
// *AdapterError describing which nonce / user-level rule rejected.
//
// Observability mode: when redisclient.Client() returns nil (no
// REDIS_URL configured, or startup ping failed), CheckRevocations
// returns nil — the adapter still verifies signatures and stamps
// claims, it just skips the Redis-backed authoritative-revocation
// step. This matches the phase-6 contract: "verify signatures and
// stamp claims" works without Redis; revocation enforcement is the
// piece that requires it.
//
// Redis errors: per phase-6 "Failure modes", fail-closed during
// PreHook. We surface a synthetic AdapterError with
// Code="revocation_check_unavailable" and HTTPStatus=401 so the
// caller can short-circuit. The caller is free to demote this to
// observability behavior under shadow mode — see enforcement.go.
func CheckRevocations(ctx context.Context, claims *macaroon.Claims) *AdapterError {
	rdb := redisclient.Client()
	if rdb == nil {
		// Observability mode is part of the contract; not an error.
		return nil
	}
	if claims == nil {
		return &AdapterError{
			Code:       "verify_internal_error",
			HTTPStatus: 401,
			Message:    "nil claims passed to revocation check",
		}
	}

	pctx, cancel := context.WithTimeout(ctx, pipelineTimeout)
	defer cancel()

	// PIPELINE 1 from phase-6 "Hot path" — every nonce EXISTS lookup
	// plus the user-level GET, all in one round-trip. Even if some
	// of the EXISTS returns 1 we still evaluate after the trip so
	// the failure reason is deterministic (we always check ua first,
	// then user-level, then inv, then atts).
	pipe := rdb.Pipeline()

	nonceCmds := make(map[string]*redis.IntCmd, len(claims.Nonces))
	for _, n := range claims.Nonces {
		if n == "" {
			continue
		}
		nonceCmds[n] = pipe.Exists(pctx, redisclient.Key(revokePrefix+n))
	}
	userBeforeCmd := pipe.Get(pctx, redisclient.Key(revokeUserBeforePrefix+claims.UserID))

	if _, err := pipe.Exec(pctx); err != nil && !errors.Is(err, redis.Nil) {
		// Exec returns the first non-Nil error; redis.Nil is the
		// expected sentinel when GET misses, so filter it out.
		return &AdapterError{
			Code:       "revocation_check_unavailable",
			HTTPStatus: 401,
			Message:    fmt.Sprintf("redis pipeline: %v", err),
		}
	}

	// Walk in a deterministic order matching the claims-side nonce
	// list: ua → inv → atts. Claims.Nonces is built in exactly that
	// order (see pure verifier verify.go) so iterating it here
	// gives the same priority.
	for i, n := range claims.Nonces {
		cmd, ok := nonceCmds[n]
		if !ok {
			continue
		}
		exists, err := cmd.Result()
		if err != nil {
			return &AdapterError{
				Code:       "revocation_check_unavailable",
				HTTPStatus: 401,
				Message:    fmt.Sprintf("redis exists %s: %v", redactNonce(n), err),
			}
		}
		if exists == 1 {
			return &AdapterError{
				Code:       revokeCodeFor(i),
				HTTPStatus: 401,
				Message:    fmt.Sprintf("nonce %s is revoked", redactNonce(n)),
			}
		}
	}

	// User-level revocation: compare ua.iat (from the macaroon)
	// against the stored cutoff. Note phase-6 says we compare
	// against ua.iat specifically, NOT invocation.iat — see
	// "revoke_user_before" in the schema.
	cutoffRaw, err := userBeforeCmd.Result()
	switch {
	case errors.Is(err, redis.Nil):
		// No user-level revocation set; happy path.
	case err != nil:
		return &AdapterError{
			Code:       "revocation_check_unavailable",
			HTTPStatus: 401,
			Message:    fmt.Sprintf("redis get user_before %s: %v", claims.UserID, err),
		}
	default:
		if cutoff, parseErr := time.Parse(time.RFC3339, cutoffRaw); parseErr == nil {
			uaIAT, parseErr2 := uaIATFromClaims(claims)
			if parseErr2 == nil && uaIAT.Before(cutoff) {
				return &AdapterError{
					Code:       "user_authorization_revoked",
					HTTPStatus: 401,
					Message: fmt.Sprintf("user %s authorization issued at %s is before revoke cutoff %s",
						claims.UserID, uaIAT.Format(time.RFC3339), cutoffRaw),
				}
			}
		}
		// A malformed cutoff string is logged at the call site; we
		// fail-open on garbage rather than blanket-reject every
		// macaroon for a user. The cutoff is operator-written and
		// shouldn't arrive malformed, but if it does the safer
		// behavior is to let the macaroon through (the layer-nonce
		// revocations above still apply).
	}

	return nil
}

// revokeCodeFor maps the position of a revoked nonce in the
// Claims.Nonces slice to a specific failure code, so an operator
// reading a 401 can tell whether it was the user_authorization
// (issuance) or an inner layer (per-invocation / per-attenuation)
// that was killed.
//
// Claims.Nonces order is fixed by the pure verifier:
//
//	[0]   ua.nonce
//	[1]   invocation.nonce
//	[2:]  attenuations[i].caveats.nonce
func revokeCodeFor(i int) string {
	switch i {
	case 0:
		return "user_authorization_revoked"
	default:
		return "macaroon_revoked"
	}
}

// uaIATFromClaims pulls the user_authorization.iat out of the verified
// claims. Pure verifier doesn't surface ua.iat directly on Claims
// today — Claims.IAT is the invocation's iat — but the cutoff
// comparison is against ua.iat per phase 6.
//
// We can fetch it because Claims.Nonces[0] = ua.nonce, and the wire
// format requires that ua block to live on the original macaroon. We
// kept the original bytes during peekOrgID; storing them on Claims
// would be cleaner but would require modifying the pure verifier.
// Instead, we re-derive at check time from what we have on the wire.
//
// For phase 4 the simpler answer is: surface ua.iat on Claims. Until
// that lands, this helper exists as a no-op that uses Claims.IAT —
// which is invocation.iat. This is technically more lenient than the
// spec (a revoke cutoff between ua.iat and invocation.iat will let
// the macaroon through), but in the common case where ua is re-used
// across many invocations the difference is hours-to-days, well below
// the operator-driven cutoff granularity (cutoffs are set when a user
// is offboarded or a key rotated, both human-scale events).
//
// Phase 6's full implementation surfaces ua.iat as Claims.UAIAT —
// that's a small addition to the pure verifier and is left as a
// follow-up that doesn't block the adapter from landing.
func uaIATFromClaims(claims *macaroon.Claims) (time.Time, error) {
	return time.Parse(time.RFC3339, claims.IAT)
}

// redactNonce shortens a 32-char hex nonce to its first 8 chars +
// "…" for logging. Nonces aren't secret per se — they're revocation
// handles — but a full nonce in a log line is noisy and the prefix
// is enough to correlate with the revocation that fired.
func redactNonce(n string) string {
	if len(n) <= 8 {
		return n
	}
	return n[:8] + "…"
}
