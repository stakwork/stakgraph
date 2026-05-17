// Package env is the single source of truth for environment-variable
// reads inside the gateway plugin.
//
// Centralising this means:
//   - One place to document what each variable does.
//   - One place to enforce defaults / minimum lengths.
//   - One place to grep when the swarm / docker-compose / k8s side
//     changes a variable name.
//
// We deliberately do NOT cache env reads. The plugin reads each
// variable exactly once at startup (during Init), and the rest of the
// code just holds the returned strings. If something needs hot-reload,
// it grows its own watcher; until then KISS.
package env

import (
	"os"
)

// Variable names. Kept as constants so a typo in a getter doesn't
// produce a silent "missing env" result.
const (
	// AdminUser is the username Bifrost authenticates the dashboard
	// and /api/* endpoints with. Configured in data/config.json's
	// auth_config block as `env.BIFROST_ADMIN_USER`.
	AdminUser = "BIFROST_ADMIN_USER"

	// AdminPass is the password partner to AdminUser. Bifrost
	// bcrypt-hashes the value at load time; the plaintext only ever
	// lives in this env var and in memory.
	AdminPass = "BIFROST_ADMIN_PASS"

	// ProvisioningToken is the shared secret /_plugin/* routes
	// authenticate against. In swarm this is the same value as
	// boltwall.stakwork_secret so Hive already has it.
	ProvisioningToken = "BIFROST_PROVISIONING_TOKEN"

	// PluginPort is where the in-process plugin HTTP server listens.
	// Bound to PluginBind (loopback by default) — the wrapper is the
	// only intended client.
	PluginPort = "BIFROST_PLUGIN_PORT"

	// PluginBind controls the bind address for the plugin HTTP
	// server. Default 127.0.0.1; only change for diagnostics.
	PluginBind = "BIFROST_PLUGIN_BIND"

	// TrustInline is an inline-JSON seed for the trust registry —
	// used by self-host / declarative deployments. Ignored on the
	// sphinx-swarm path where Hive populates the registry via the
	// admin API. See gateway/plans/phases/phase-5-trust-registry.md
	// ("Three configuration sources").
	TrustInline = "BIFROST_PLUGIN_TRUST"

	// TrustFile is a path to a JSON file with the same shape as
	// TrustInline. Mutually exclusive with TrustInline; if both are
	// set TrustInline wins (and a warning is logged).
	TrustFile = "BIFROST_PLUGIN_TRUST_FILE"

	// TrustReconcile selects the behaviour when persisted state and
	// the env-supplied seed both exist and disagree:
	//
	//   - "ignore"    (default): use persisted, log a warning
	//   - "overwrite": replace persisted with env, log loudly
	//   - "refuse":    exit non-zero
	//
	// The default protects the most common upgrade path — someone
	// set the env var months ago, the registry evolved via the API,
	// and a restart shouldn't silently revert.
	TrustReconcile = "BIFROST_PLUGIN_TRUST_RECONCILE"

	// TrustPath is the on-disk location of the canonical trust
	// registry. Defaults to /app/data/trust.json so it sits next to
	// logs.db on the same data volume.
	TrustPath = "BIFROST_PLUGIN_TRUST_PATH"

	// RedisURL is the connection string for the macaroon-enforcement
	// Redis. In sphinx-swarm this points at the shared redis.sphinx
	// instance; in docker-compose it points at the sidecar `redis`
	// service. Expected format: redis://host:port/db
	//
	// Empty / unset ⇒ plugin runs in observability mode (verifies
	// macaroon signatures but skips Redis-backed revocation / budget
	// checks). See gateway/plans/phases/phase-6-plugin-enforcement.md
	// "Namespace" for the keyspace contract.
	RedisURL = "BIFROST_PLUGIN_REDIS_URL"
)

// Defaults that apply when an env var is unset.
const (
	DefaultPluginPort = "8189"
	DefaultPluginBind = "127.0.0.1"
	DefaultTrustPath  = "/app/data/trust.json"

	// DefaultTrustReconcile is the safe default — see TrustReconcile.
	DefaultTrustReconcile = "ignore"
)

// Get reads `name` and returns its value or "" if unset.
func Get(name string) string { return os.Getenv(name) }

// GetOr reads `name` and returns its value, or `fallback` if unset
// or empty.
func GetOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

// PluginAddr returns the host:port the plugin HTTP server should bind
// to. Composed from PluginBind + PluginPort with sensible defaults.
func PluginAddr() string {
	return GetOr(PluginBind, DefaultPluginBind) + ":" + GetOr(PluginPort, DefaultPluginPort)
}

// AdminCreds returns (user, pass, ok). `ok` is false if either value
// is unset — callers should treat that as "auth not configured" and
// usually skip whatever subsystem needed them.
func AdminCreds() (user, pass string, ok bool) {
	user = os.Getenv(AdminUser)
	pass = os.Getenv(AdminPass)
	return user, pass, user != "" && pass != ""
}

// ProvisioningTokenValue returns (token, ok). Same contract as
// AdminCreds.
func ProvisioningTokenValue() (string, bool) {
	t := os.Getenv(ProvisioningToken)
	return t, t != ""
}

// TrustPathValue returns the persisted-trust file path, falling back
// to DefaultTrustPath.
func TrustPathValue() string { return GetOr(TrustPath, DefaultTrustPath) }

// TrustReconcileValue returns the configured reconcile mode, falling
// back to DefaultTrustReconcile. The caller is responsible for
// validating it against the known set ("ignore", "overwrite",
// "refuse") — keeping the env package free of policy logic.
func TrustReconcileValue() string { return GetOr(TrustReconcile, DefaultTrustReconcile) }

// RedisURLValue returns (url, ok). `ok` is false when unset — callers
// should treat that as "observability mode" and skip wiring the
// Redis-dependent enforcement path. The plugin remains fully
// functional as a signature verifier without it.
func RedisURLValue() (string, bool) {
	u := os.Getenv(RedisURL)
	return u, u != ""
}

// TrustSeed returns the env-supplied registry seed and where it came
// from. Exactly one of the two env vars is honoured per
// "Env-var preset shape" in the phase-5 doc:
//
//   - inline=true  : raw JSON from BIFROST_PLUGIN_TRUST
//   - inline=false : path from BIFROST_PLUGIN_TRUST_FILE; caller reads it
//   - ok=false     : neither set
//
// If both env vars are set, inline wins. Callers should log a warning
// in that case — env.go intentionally doesn't log so it stays a leaf
// dependency.
func TrustSeed() (value string, inline bool, ok bool) {
	if v := os.Getenv(TrustInline); v != "" {
		return v, true, true
	}
	if v := os.Getenv(TrustFile); v != "" {
		return v, false, true
	}
	return "", false, false
}
