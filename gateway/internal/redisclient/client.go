// Package redisclient owns the gateway plugin's single Redis client.
//
// Why this exists
// ---------------
// Phase-6 macaroon enforcement (revocation tombstones, per-run cost
// accumulators, kill switches, tool-loop history) all live in Redis.
// The plugin shares the swarm's redis.sphinx instance with Jarvis;
// every key the plugin reads or writes is prefixed with `bifrost:` to
// keep the keyspaces partitioned. See
// gateway/plans/phases/phase-6-plugin-enforcement.md ("Namespace") for
// the contract.
//
// Status
// ------
// Phase 4/5 ships only the connection bring-up and a startup smoke
// ping (DBSIZE). The actual enforcement reads/writes (PreLLMHook
// pipelines, PostLLMHook accumulators) land with the phase-4 adapter
// PR in gateway/internal/auth/. This package exists so that adapter
// has a single, already-initialised client to import.
//
// Observability mode
// ------------------
// When BIFROST_PLUGIN_REDIS_URL is unset, Init() is a no-op and
// Client() returns nil. Downstream code MUST treat nil as
// "observability only — skip Redis-backed enforcement" rather than
// crash. This matches the phase-5 stance that the trust registry is
// wired but enforcement is off until phase 4's adapter lands.
package redisclient

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// KeyPrefix is the fixed namespace every plugin-owned Redis key
// carries. See phase-6 "Namespace" — operators rely on
// `SCAN MATCH bifrost:*` to enumerate plugin state in a shared Redis
// instance, so this is part of the operational contract, not an
// internal detail.
const KeyPrefix = "bifrost:"

// pingTimeout bounds the startup DBSIZE call. If Redis is reachable
// at all this is plenty; if it isn't, we'd rather fail-soft into
// observability mode than block the plugin from coming up.
const pingTimeout = 2 * time.Second

var (
	mu     sync.RWMutex
	client *redis.Client
)

// Init parses BIFROST_PLUGIN_REDIS_URL, opens a client, and issues a
// single DBSIZE call as a connectivity smoke test. Logs both the
// configured URL (with credentials stripped) and the DBSIZE result.
//
// Returns nil in two cases:
//
//   - Env var unset: observability mode, no client constructed.
//   - Env var set but ping fails: client is closed and discarded, and
//     a loud warning is logged. The plugin still starts; downstream
//     enforcement that needs Redis will fail-closed at the call site
//     per the phase-6 "Failure modes" contract.
//
// Returns a non-nil error only when the URL itself is malformed —
// that's a config bug that should keep the plugin from coming up so
// the operator notices.
func Init() error {
	url, ok := env.RedisURLValue()
	if !ok {
		pluginlog.Logf("redis: BIFROST_PLUGIN_REDIS_URL unset — running in observability mode (no enforcement)")
		return nil
	}

	opts, err := redis.ParseURL(url)
	if err != nil {
		return fmt.Errorf("redis: parse %s: %w", env.RedisURL, err)
	}

	c := redis.NewClient(opts)

	// Smoke test. Doubles as the visible "yes the plugin can reach
	// redis" signal in docker-compose / swarm logs.
	ctx, cancel := context.WithTimeout(context.Background(), pingTimeout)
	defer cancel()

	size, err := c.DBSize(ctx).Result()
	if err != nil {
		// Fail-soft: log loudly, discard the client, keep going.
		// The phase-4 adapter will refuse enforcement when client()
		// is nil, which is the correct posture during a Redis outage.
		_ = c.Close()
		pluginlog.Warnf(
			"redis: connect to %s failed: %v — falling back to observability mode",
			redactURL(url), err,
		)
		return nil
	}

	pluginlog.Logf(
		"redis: connected url=%s db=%d dbsize=%d prefix=%s",
		redactURL(url), opts.DB, size, KeyPrefix,
	)

	mu.Lock()
	client = c
	mu.Unlock()
	return nil
}

// Client returns the initialised Redis client, or nil if Init() ran
// in observability mode (env unset) or fell back from a failed ping.
// Callers MUST nil-check; see package doc.
func Client() *redis.Client {
	mu.RLock()
	defer mu.RUnlock()
	return client
}

// SetClientForTest overrides the package-level client. Tests use
// this to inject a miniredis-backed *redis.Client without exporting
// the mutex. Production code MUST go through Init. Pass nil to
// restore observability mode for the next test.
func SetClientForTest(c *redis.Client) {
	mu.Lock()
	client = c
	mu.Unlock()
}

// Close shuts down the client if any. Called from main.Cleanup().
func Close() error {
	mu.Lock()
	defer mu.Unlock()
	if client == nil {
		return nil
	}
	err := client.Close()
	client = nil
	return err
}

// Key prepends the gateway plugin's namespace to a sub-key. Use this
// instead of string concatenation at every call site so the prefix
// stays consistent and greppable.
//
//	Key("revoke:" + nonce)              → "bifrost:revoke:<nonce>"
//	Key("cost:run:" + runID)            → "bifrost:cost:run:<run_id>"
//	Key("cost:ua:" + uaNonce)           → "bifrost:cost:ua:<ua_nonce>"
//
// Sub-key shapes are defined in phase-6-plugin-enforcement.md "Redis
// schema"; this helper is the only place that knows about the prefix.
func Key(subkey string) string { return KeyPrefix + subkey }

// redactURL strips the password (if any) from a redis:// URL for
// logging. Public hostnames are fine to log; credentials are not.
func redactURL(raw string) string {
	opts, err := redis.ParseURL(raw)
	if err != nil {
		return "<unparseable>"
	}
	// Reconstruct without the password. ParseURL returned the
	// canonical pieces; we emit the shape operators will recognise.
	host := opts.Addr
	user := ""
	if opts.Username != "" {
		user = opts.Username + "@"
	} else if opts.Password != "" {
		// Credentials present but no username — note that without
		// leaking the password itself.
		user = "***@"
	}
	return fmt.Sprintf("redis://%s%s/%d", user, host, opts.DB)
}
