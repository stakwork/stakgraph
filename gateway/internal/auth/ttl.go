package auth

import (
	"time"
)

// runKeyTTL implements phase-6's clamp(macaroon.exp - now + 1h, 1h, 7d)
// formula for the per-run Redis keys (cost:run, steps:run, tools:run)
// — and also for cost:ua:<nonce> when budget tracking lands. See
// gateway/plans/phases/phase-6-plugin-enforcement.md "TTL policy".
//
//   - The +1h grace ensures a PostHook firing right at exp doesn't
//     race TTL eviction with the next reader.
//   - The 1h floor handles macaroons that are already near or past
//     expiry when the call lands (the verifier rejects, but if
//     somehow the call goes through anyway we still want sane TTLs
//     on the bookkeeping keys).
//   - The 7d ceiling bounds Redis memory growth for absurdly-long
//     exp values; an actively-spending run keeps its keys alive past
//     this via EXPIRE on every write.
//
// Phase 4 only uses this for revocation tombstone TTLs today; the
// per-run/per-UA accumulators land in phase 6. Kept here so the
// formula has one home.
func runKeyTTL(macaroonExp, now time.Time) time.Duration {
	const (
		floor   = 1 * time.Hour
		ceiling = 7 * 24 * time.Hour
		grace   = 1 * time.Hour
	)
	d := macaroonExp.Sub(now) + grace
	if d < floor {
		return floor
	}
	if d > ceiling {
		return ceiling
	}
	return d
}

// parseRFC3339 is a tiny convenience that returns the zero time on
// parse error rather than (time.Time{}, error). Phase-4 timestamps
// are validated at JSON-unmarshal time by the pure verifier; if we
// get here with a non-RFC3339 string something has gone wrong
// upstream and the safest behavior for TTL math is "treat as
// already-expired" — which the floor handles.
func parseRFC3339(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}
	}
	return t
}
