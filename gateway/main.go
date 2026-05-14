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
	"fmt"
	"os"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// PluginName is the system identifier reported via GetName().
const PluginName = "stakgraph-gateway"

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

// Init is called once when bifrost-http loads the plugin.
func Init(config any) error {
	// bifrost passes config as a map[string]interface{} (decoded JSON).
	if m, ok := config.(map[string]any); ok {
		if v, ok := m["log_level"].(string); ok {
			cfg.LogLevel = v
		}
	}
	fmt.Fprintf(os.Stderr, "[%s] Init called config=%+v\n", PluginName, cfg)
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
