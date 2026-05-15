// Package adminapi is the gateway plugin's in-process HTTP server,
// hosting the `/_plugin/*` route namespace.
//
// The wrapper binary (gateway/wrapper) reverse-proxies traffic on
// `/_plugin/*` to this server (loopback only). Why a separate server
// instead of routes registered through Bifrost's router: Bifrost
// plugins (.so) cannot register arbitrary HTTP routes through
// Bifrost's own router, and we also want routes that aren't behind
// Bifrost's AuthMiddleware so Hive can bootstrap on a fresh swarm.
//
// Lifecycle
// ---------
//   - Start() is called from the plugin's Init().
//   - Stop() is called from the plugin's Cleanup() with a small grace
//     period.
//
// One server per process. Re-calling Start() is a no-op.
package adminapi

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

var (
	startOnce sync.Once
	srv       *http.Server
)

// Start brings up the plugin HTTP server in a goroutine.
//
// If required env vars (BIFROST_ADMIN_USER/PASS/PROVISIONING_TOKEN)
// are missing the server is NOT started — this matches the
// stand-alone-dev use case where someone wants to play with the image
// without configuring auth at all.
//
// Returns nil on success or on "skipped because env was missing".
// Callers in Init don't actually need to do anything with the return
// value today; it exists for symmetry with Stop.
func Start() error {
	startOnce.Do(start)
	return nil
}

// Stop shuts the server down with a short grace period. Safe to call
// even if Start was never called or skipped.
func Stop() error {
	if srv == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		pluginlog.Errf("adminapi: shutdown: %v", err)
		return err
	}
	return nil
}

func start() {
	adminUser, adminPass, hasCreds := env.AdminCreds()
	token, hasToken := env.ProvisioningTokenValue()

	if !hasCreds || !hasToken {
		pluginlog.Warnf(
			"adminapi: server NOT started: missing env "+
				"(have BIFROST_ADMIN_USER=%t BIFROST_ADMIN_PASS=%t BIFROST_PROVISIONING_TOKEN=%t)",
			adminUser != "", adminPass != "", hasToken,
		)
		return
	}

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         adminUser,
		adminPass:         adminPass,
		provisioningToken: token,
	})

	addr := env.PluginAddr()
	srv = &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	go func() {
		pluginlog.Logf("adminapi: listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			pluginlog.Errf("adminapi: server crashed: %v", err)
		}
	}()
}

// routeDeps is the bag of values registerRoutes hands to each route
// builder. Centralizing here avoids individual route files reading env
// directly — that responsibility lives in env/.
type routeDeps struct {
	adminUser         string
	adminPass         string
	provisioningToken string
}

// registerRoutes is in its own function so each route lives in its
// own file (credentials.go, health.go, …) but the routing table is
// readable in one place.
func registerRoutes(mux *http.ServeMux, deps routeDeps) {
	auth := requireBearerToken(deps.provisioningToken)

	mux.HandleFunc("/_plugin/admin-credentials",
		auth(adminCredentialsHandler(deps.adminUser, deps.adminPass)))
	mux.HandleFunc("/_plugin/health", healthHandler)
}
