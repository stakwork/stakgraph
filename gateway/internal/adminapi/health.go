package adminapi

import "net/http"

// healthHandler responds with a tiny JSON body indicating the plugin
// server is up. No auth — returns no sensitive data and is used by:
//
//   - The wrapper's readiness probe (gateway/wrapper/main.go), to wait
//     until the plugin has bound its socket before opening the public
//     port.
//   - External health checks (swarm / k8s) if anyone wants to probe
//     the plugin specifically rather than going through Bifrost.
//
// Bound at `/_plugin/health`.
func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true}`))
}
