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
)

// Defaults that apply when an env var is unset.
const (
	DefaultPluginPort = "8189"
	DefaultPluginBind = "127.0.0.1"
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
