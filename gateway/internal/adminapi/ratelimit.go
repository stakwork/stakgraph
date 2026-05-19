package adminapi

import (
	"context"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// loginAttemptWindow is the rolling window over which failed login
// attempts are counted. Long enough to defeat slow brute force,
// short enough that a single fat-fingered password streak from a
// real user clears on its own without an operator intervention.
const loginAttemptWindow = 15 * time.Minute

// maxLoginAttempts is the per-IP failure budget. Hits 10 fails →
// 429 with Retry-After until the window's TTL ticks down.
const maxLoginAttempts = 10

// loginLimiter is the per-IP failed-attempt counter for
// `POST /_plugin/login`. Phase 8 deliberately limits per IP, not
// per username, because per-username locking is an account-
// enumeration oracle (an attacker can probe which usernames are
// valid by triggering locks).
//
// All operations are best-effort: if Redis is unreachable, allow()
// returns true and increment() / reset() log and move on. The login
// handler is still gated by the constant-time password check so a
// degraded limiter doesn't unlock the door, it only removes the
// extra friction.
type loginLimiter struct {
	client *redis.Client // may be nil ⇒ limiter disabled (always allows)
}

func newLoginLimiter() *loginLimiter {
	return &loginLimiter{client: redisclient.Client()}
}

// allow reports whether the given IP is currently under the failure
// limit. If false, returns the remaining TTL (for Retry-After).
func (l *loginLimiter) allow(ctx context.Context, ip string) (ok bool, retryAfter time.Duration) {
	if l == nil || l.client == nil || ip == "" {
		return true, 0
	}
	key := redisclient.Key("login_attempts:" + ip)
	val, err := l.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return true, 0
	}
	if err != nil {
		pluginlog.Warnf("ratelimit: get %s: %v", key, err)
		return true, 0
	}
	n, _ := strconv.Atoi(val)
	if n < maxLoginAttempts {
		return true, 0
	}
	ttl, err := l.client.TTL(ctx, key).Result()
	if err != nil || ttl < 0 {
		ttl = loginAttemptWindow
	}
	return false, ttl
}

// recordFailure bumps the counter and ensures the TTL is set. INCR
// followed by EXPIRE-only-on-first-hit is the textbook pattern; we
// use EXPIRE unconditionally because the cost of an extra command
// is negligible against the simplicity of not having to read-modify.
func (l *loginLimiter) recordFailure(ctx context.Context, ip string) {
	if l == nil || l.client == nil || ip == "" {
		return
	}
	key := redisclient.Key("login_attempts:" + ip)
	pipe := l.client.Pipeline()
	pipe.Incr(ctx, key)
	pipe.Expire(ctx, key, loginAttemptWindow)
	if _, err := pipe.Exec(ctx); err != nil {
		pluginlog.Warnf("ratelimit: incr %s: %v", key, err)
	}
}

// reset clears the counter for a successful login so a user who
// fat-fingered a few times doesn't carry a "near limit" balance
// around for 15 minutes after they finally get it right.
func (l *loginLimiter) reset(ctx context.Context, ip string) {
	if l == nil || l.client == nil || ip == "" {
		return
	}
	key := redisclient.Key("login_attempts:" + ip)
	if err := l.client.Del(ctx, key).Err(); err != nil {
		pluginlog.Warnf("ratelimit: del %s: %v", key, err)
	}
}
