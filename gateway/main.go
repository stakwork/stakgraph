// Package main is the Bifrost gateway plugin for stakgraph's LLM governance
// layer (see mcp/docs/plans/llm-governance.md).
//
// This is the v0 boilerplate: every hook just logs that it was called along
// with the request/response metadata. No macaroon verification, no Redis,
// no budget enforcement yet. The point is to prove the plugin loads into
// bifrost-http and that all hook entry points fire in the expected order.
//
// Bifrost loads this as a Go plugin (`-buildmode=plugin`), so the symbols
// below MUST be exported with these exact names and signatures. See
// https://docs.getbifrost.ai/plugins/writing-go-plugin .
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// PluginName is the system identifier reported via GetName().
const PluginName = "stakgraph-gateway"

// Default port for the plugin's in-process HTTP server, which hosts
// the `/_plugin/*` route namespace. The container's public listener
// is owned by a thin Go wrapper (see gateway/wrapper) that reverse-
// proxies most traffic to bifrost-http and routes `/_plugin/*` here.
// Because the wrapper is the only client, this server binds to
// loopback only and is unreachable from outside the container.
//
// Overridable via `BIFROST_PLUGIN_PORT` for local dev.
const defaultPluginPort = "8189"

// Context keys used to pass values between hooks for a single request.
// Keys live in the BifrostContext for the lifetime of one request.
const (
	ctxKeyRequestID schemas.BifrostContextKey = "stakgraph-gateway/request-id"
	ctxKeyStartTime schemas.BifrostContextKey = "stakgraph-gateway/start-time"
)

// pluginConfig is what Init() receives from the `config` block in
// bifrost's config.json. Nothing required yet; logged at startup so we
// can confirm wiring works.
type pluginConfig struct {
	LogLevel string `json:"log_level,omitempty"`
}

var cfg pluginConfig

// pluginSrv is the in-process HTTP server that exposes the `/_plugin/*`
// route namespace. Initialised in Init, shut down in Cleanup. Kept in
// a package var so Cleanup can close it.
//
// First route is `/_plugin/admin-credentials`, which lets Hive bootstrap
// itself with Bifrost's admin user/password on a fresh swarm. Future
// routes will expose plugin-specific governance reads (per-agent cost,
// run-id rollups, etc.) that Bifrost's built-in `/api/logs` doesn't
// surface in the shape we need.
var (
	pluginOnce sync.Once
	pluginSrv  *http.Server
)

// Init is called once when bifrost-http loads the plugin.
func Init(config any) error {
	// bifrost passes config as a map[string]interface{} (decoded JSON).
	if m, ok := config.(map[string]any); ok {
		if v, ok := m["log_level"].(string); ok {
			cfg.LogLevel = v
		}
	}
	fmt.Fprintf(os.Stderr, "[%s] Init called config=%+v\n", PluginName, cfg)

	// Start the plugin HTTP server on first Init only. bifrost-http
	// calls Init exactly once per process today, but defending against
	// future hot-reload behaviour costs nothing.
	pluginOnce.Do(startPluginServer)

	return nil
}

// GetName returns the plugin's system identifier. Must be stable across
// restarts — bifrost uses it as a map key.
func GetName() string {
	return PluginName
}

// Cleanup is called on bifrost shutdown.
func Cleanup() error {
	fmt.Fprintf(os.Stderr, "[%s] Cleanup called\n", PluginName)
	if pluginSrv != nil {
		// Best-effort shutdown — give in-flight requests a couple of
		// seconds, but don't block the bifrost shutdown path forever.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := pluginSrv.Shutdown(shutdownCtx); err != nil {
			fmt.Fprintf(os.Stderr,
				"[%s] plugin server shutdown error: %v\n",
				PluginName, err)
		}
	}
	return nil
}

// HTTPTransportPreHook fires at the HTTP transport layer, before the
// request enters bifrost core. This is where we'll eventually inspect
// `x-macaroon` and verify it. The attribution dimensions
// (`x-bf-dim-*`) are extracted by Bifrost itself and stored at
// schemas.BifrostContextKeyDimensions, which Bifrost's built-in logging
// plugin then merges into logs.metadata — no plugin code required for
// observability. Here we just read them for log-line purposes.
func HTTPTransportPreHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	start := time.Now()
	ctx.SetValue(ctxKeyStartTime, start)

	macaroon := req.CaseInsensitiveHeaderLookup("x-macaroon")

	// Bifrost's transport already extracted x-bf-dim-* into this map.
	dims, _ := ctx.Value(schemas.BifrostContextKeyDimensions).(map[string]string)
	runID := dims["run-id"]
	sessionID := dims["session-id"]
	agentName := dims["agent-name"]
	workspaceID := dims["workspace-id"]
	userID := dims["user-id"]

	ctx.SetValue(ctxKeyRequestID, runID)

	logf("HTTPTransportPreHook method=%s path=%s body_bytes=%d macaroon=%s run_id=%s session_id=%s agent=%s workspace=%s user=%s",
		req.Method,
		req.Path,
		len(req.Body),
		redact(macaroon),
		runID,
		sessionID,
		agentName,
		workspaceID,
		userID,
	)
	ctx.Log(schemas.LogLevelInfo, fmt.Sprintf("PreHook %s %s", req.Method, req.Path))

	// nil, nil = continue pipeline unchanged.
	return nil, nil
}

// HTTPTransportPostHook fires after the upstream provider call, for
// non-streaming responses. This is where the v2 plugin will compute
// cost and increment Redis counters.
func HTTPTransportPostHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, resp *schemas.HTTPResponse) error {
	elapsed := elapsedSince(ctx)
	runID, _ := ctx.Value(ctxKeyRequestID).(string)

	logf("HTTPTransportPostHook path=%s status=%d body_bytes=%d run_id=%s elapsed_ms=%d",
		req.Path,
		resp.StatusCode,
		len(resp.Body),
		runID,
		elapsed.Milliseconds(),
	)
	return nil
}

// HTTPTransportStreamChunkHook fires once per streamed chunk. PostHook
// does NOT fire for streaming responses, so cost accounting on streams
// will eventually hook in here (on the final chunk that carries usage).
func HTTPTransportStreamChunkHook(ctx *schemas.BifrostContext, req *schemas.HTTPRequest, chunk *schemas.BifrostStreamChunk) (*schemas.BifrostStreamChunk, error) {
	if chunk == nil {
		return chunk, nil
	}
	// Keep this quiet — one log line per chunk would flood the output.
	// We only log when the chunk has usage data or is an error, which
	// roughly corresponds to "interesting boundary" chunks.
	if chunk.BifrostError != nil {
		logf("StreamChunk error path=%s err=%v", req.Path, chunk.BifrostError.Error)
	}
	return chunk, nil
}

// PreLLMHook fires after bifrost has parsed the request into its
// internal BifrostRequest type. We see the provider, model, and
// (eventually) message content here.
func PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	provider, model, _ := req.GetRequestFields()
	runID, _ := ctx.Value(ctxKeyRequestID).(string)

	logf("PreLLMHook provider=%s model=%s request_type=%s run_id=%s",
		provider,
		model,
		req.RequestType,
		runID,
	)
	return req, nil, nil
}

// PostLLMHook fires after the upstream provider call (or after a
// short-circuit). For streaming requests this fires once when the
// response is set up; the actual chunks go through StreamChunkHook.
func PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	elapsed := elapsedSince(ctx)
	runID, _ := ctx.Value(ctxKeyRequestID).(string)

	var (
		hadResp = resp != nil
		hadErr  = bifrostErr != nil
	)

	// Try to pull usage/cost if the provider populated it (Chat responses).
	var (
		promptTokens, completionTokens, totalTokens int
	)
	if hadResp && resp.ChatResponse != nil && resp.ChatResponse.Usage != nil {
		u := resp.ChatResponse.Usage
		promptTokens = u.PromptTokens
		completionTokens = u.CompletionTokens
		totalTokens = u.TotalTokens
	}

	logf("PostLLMHook run_id=%s had_resp=%t had_err=%t prompt_tokens=%d completion_tokens=%d total_tokens=%d elapsed_ms=%d",
		runID,
		hadResp,
		hadErr,
		promptTokens,
		completionTokens,
		totalTokens,
		elapsed.Milliseconds(),
	)
	return resp, bifrostErr, nil
}

// --- plugin HTTP server ------------------------------------------------
//
// Bifrost plugins (the .so we are) can't register arbitrary HTTP routes
// through Bifrost's router — HTTPTransportPreHook only fires on inference
// paths, and unknown URLs 404 before any middleware can see them. We
// also need routes that AREN'T behind Bifrost's auth middleware so a
// fresh swarm can bootstrap itself (chicken-and-egg if the route to
// fetch the admin password needs the admin password).
//
// So the plugin runs its own HTTP server on loopback. The container's
// public listener is owned by a thin Go wrapper (gateway/wrapper) which
// reverse-proxies `/_plugin/*` to this server and everything else to
// bifrost-http. End result: one public port (Traefik-friendly), but
// the plugin owns a clean `/_plugin/*` namespace it can grow over time.
//
// Currently the only route is `/_plugin/admin-credentials`, used by
// Hive to bootstrap on a fresh swarm via the shared boltwall stakwork
// secret. Planned future routes:
//   - GET /_plugin/metrics/agent-cost?since=…  (per-agent rollups)
//   - GET /_plugin/metrics/run/{run_id}        (run-level breakdown)
//   - GET /_plugin/health                      (combined plugin+bifrost
//                                               readiness)
//
// All routes under /_plugin require Bearer auth with the shared token
// (BIFROST_PROVISIONING_TOKEN, same value as boltwall.stakwork_secret).

// startPluginServer brings up the in-process plugin HTTP server.
// Returns immediately; the server runs in a goroutine.
//
// Required env (server is skipped if any are missing — see logf below):
//   - BIFROST_ADMIN_USER, BIFROST_ADMIN_PASS: the admin credentials
//     bifrost-http was started with (resolved from auth_config in
//     config.json). The plugin only ECHOES them; bifrost is the
//     source of truth for the actual hashed credential.
//   - BIFROST_PROVISIONING_TOKEN: shared secret Hive presents as
//     `Authorization: Bearer <token>`.
//
// Optional env:
//   - BIFROST_PLUGIN_PORT (default 8189).
//   - BIFROST_PLUGIN_BIND (default 127.0.0.1 — loopback only because
//     the wrapper is the only intended client).
//
// If any required env var is missing the server is NOT started and a
// warning is logged. This keeps the standalone-docker case (no swarm)
// from failing hard — developers without a provisioning token simply
// don't get the bootstrap endpoint.
func startPluginServer() {
	adminUser := os.Getenv("BIFROST_ADMIN_USER")
	adminPass := os.Getenv("BIFROST_ADMIN_PASS")
	token := os.Getenv("BIFROST_PROVISIONING_TOKEN")
	port := os.Getenv("BIFROST_PLUGIN_PORT")
	if port == "" {
		port = defaultPluginPort
	}
	bind := os.Getenv("BIFROST_PLUGIN_BIND")
	if bind == "" {
		bind = "127.0.0.1"
	}

	if adminUser == "" || adminPass == "" || token == "" {
		logf("plugin server NOT started: missing env "+
			"(have BIFROST_ADMIN_USER=%t BIFROST_ADMIN_PASS=%t BIFROST_PROVISIONING_TOKEN=%t)",
			adminUser != "", adminPass != "", token != "")
		return
	}

	// Pre-encode the credentials body once. The values can't change at
	// runtime (env is read at process start) so re-marshalling on every
	// request would be pointless.
	credsBody, err := json.Marshal(map[string]string{
		"admin_username": adminUser,
		"admin_password": adminPass,
	})
	if err != nil {
		logf("plugin server NOT started: marshal credentials: %v", err)
		return
	}

	// Constant-time compare to avoid leaking token length / prefix via
	// response-time differences. Closed-over so we don't expose the
	// token bytes via a package var.
	tokenBytes := []byte(token)
	authValid := func(presented string) bool {
		// Strip optional "Bearer " prefix; accept either case.
		if len(presented) > 7 && (presented[:7] == "Bearer " || presented[:7] == "bearer ") {
			presented = presented[7:]
		}
		// subtle.ConstantTimeCompare returns 1 only when lengths AND
		// bytes match; mismatched lengths short-circuit to 0.
		return subtle.ConstantTimeCompare([]byte(presented), tokenBytes) == 1
	}

	// requireToken wraps a handler with Bearer auth. Use for every
	// route under /_plugin except /_plugin/health.
	requireToken := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if !authValid(r.Header.Get("Authorization")) {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			next(w, r)
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/_plugin/admin-credentials", requireToken(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(credsBody)
	}))
	// Liveness probe — used by the wrapper's readiness check and
	// useful for swarm/k8s. No auth (returns no sensitive data).
	mux.HandleFunc("/_plugin/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	})

	addr := bind + ":" + port
	pluginSrv = &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	go func() {
		logf("plugin server listening on %s", addr)
		if err := pluginSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logf("plugin server crashed: %v", err)
		}
	}()
}

// --- helpers -----------------------------------------------------------

// logf writes a single line to stderr prefixed with the plugin name and
// a wallclock timestamp, so plugin output is greppable in bifrost logs.
func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[%s] %s "+format+"\n",
		append([]any{PluginName, time.Now().UTC().Format(time.RFC3339Nano)}, args...)...)
}

// elapsedSince reads the start time stashed in ctx by the PreHook and
// returns time since. Returns 0 if not set (e.g. for hooks that fire
// without a PreHook predecessor, like SDK-mode usage).
func elapsedSince(ctx *schemas.BifrostContext) time.Duration {
	start, ok := ctx.Value(ctxKeyStartTime).(time.Time)
	if !ok {
		return 0
	}
	return time.Since(start)
}

// redact returns a short fingerprint of a secret-ish header value.
// Avoids dumping the full macaroon to logs while still letting you
// confirm "yep, the header was set" at a glance.
func redact(s string) string {
	if s == "" {
		return "<unset>"
	}
	if len(s) <= 12 {
		n := len(s)
		if n > 4 {
			n = 4
		}
		return "set(" + s[:n] + "…)"
	}
	return fmt.Sprintf("set(%s…%s,len=%d)", s[:4], s[len(s)-4:], len(s))
}
