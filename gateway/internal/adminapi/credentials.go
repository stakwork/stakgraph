package adminapi

import (
	"encoding/json"
	"net/http"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// adminCredentialsHandler returns the handler for
// `GET /_plugin/admin-credentials`.
//
// The endpoint echoes the admin user/password the plugin saw in its
// env at boot. Hive calls it ONCE per swarm to bootstrap itself with
// the Bifrost admin credentials (the swarm super-admin generates the
// password randomly per-instance — see gateway/plans/phases/
// phase-3-swarm-handoff.md).
//
// Idempotency
// -----------
// The body is pre-encoded at handler construction time and never
// changes during the process lifetime. Re-calling the endpoint always
// returns the same values — so Hive can call it any number of times
// (e.g. after losing the encrypted password column) and recover.
//
// Auth
// ----
// Bearer token (BIFROST_PROVISIONING_TOKEN) — applied by the
// requireBearerToken middleware at the router level. By the time this
// handler runs, the token check has already passed.
func adminCredentialsHandler(adminUser, adminPass string) http.HandlerFunc {
	// Pre-encode once. Avoids re-marshalling on every request and
	// makes the closure capture explicit.
	body, err := json.Marshal(map[string]string{
		"admin_username": adminUser,
		"admin_password": adminPass,
	})
	if err != nil {
		// Marshalling a 2-string map can't actually fail, but log it
		// anyway — if we ever change the body shape and break this,
		// we want a loud signal not a silent panic.
		pluginlog.Errf("adminapi: pre-encode credentials: %v", err)
		return func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "credentials unavailable", http.StatusInternalServerError)
		}
	}

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}
}
