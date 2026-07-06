package adminapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// hivecallback.go wires the gateway's one outbound relationship with
// Hive.
//
// Why it exists
// -------------
// The Evals tab lets an operator create / edit / delete eval sets and
// *run* them from inside the gateway dashboard. Reads come straight
// out of neo4j (see evals.go), but every WRITE and every RUN is
// delegated back to Hive:
//
//   - eval nodes are Jarvis-authored (schema + node_key + Data_Bank +
//     namespace machinery). The gateway must not forge them with raw
//     Cypher, so it asks Hive — which already owns the Jarvis write
//     path — to do it.
//   - runs dispatch a Stakwork workflow with Bifrost creds + the
//     Stakwork API key, all of which live Hive-side.
//
// Config source
// -------------
// Hive pushes its callback base URL + a workspace-scoped API key to
// POST /_plugin/hive-callback (bearer-only) during agent-catalog
// reconciliation. We stash both in Redis so they survive across
// requests without an env redeploy and can be rotated by simply
// re-pushing. When Redis is in observability mode (no URL) the config
// can't be persisted and the write/run endpoints answer 503 — reads
// still work.
//
// Auth model of the callback
// ---------------------------
// The gateway calls Hive with `Authorization: Bearer <workspace_key>`.
// Hive validates the key, resolves the workspace itself, and performs
// the Jarvis / Stakwork action scoped to THAT workspace. The gateway
// never sends a workspace id — the key IS the scope. This mirrors how
// Hive already authenticates its error-ingest + MCP surfaces.

// Redis keys for the pushed Hive callback config. Namespaced under the
// shared `bifrost:` prefix like every other plugin key.
const (
	hiveCallbackURLKey = "hive:callback:url"
	hiveCallbackKeyKey = "hive:callback:api_key"
)

// hiveCallbackTimeout bounds a single gateway→Hive round-trip. Runs
// dispatch a Stakwork project (Hive returns as soon as the project is
// accepted, not when the eval finishes), and CRUD is one Jarvis write,
// so a few seconds is plenty.
const hiveCallbackTimeout = 12 * time.Second

// ─── inbound: POST /_plugin/hive-callback (store config) ─────────────

// hiveCallbackConfigRequest is what Hive pushes during reconciliation.
type hiveCallbackConfigRequest struct {
	// HiveURL is Hive's base origin (no trailing slash needed — we
	// normalise). All callback paths are appended to it.
	HiveURL string `json:"hive_url"`
	// APIKey is a workspace-scoped Hive API key (`hive_…`). It is the
	// bearer the gateway presents on every callback; Hive resolves the
	// workspace from it.
	APIKey string `json:"api_key"`
}

type hiveCallbackConfigResponse struct {
	Stored bool `json:"stored"`
}

// hiveCallbackHandlers persists + reads the callback config and issues
// the outbound calls. redis may be nil (observability mode) — then the
// config can't be stored and callers degrade to 503.
type hiveCallbackHandlers struct {
	http *http.Client
}

func newHiveCallbackHandlers() *hiveCallbackHandlers {
	return &hiveCallbackHandlers{
		http: &http.Client{Timeout: hiveCallbackTimeout},
	}
}

// storeConfig handles POST /_plugin/hive-callback — bearer-only, the
// same server-to-server posture as the catalog push. Idempotent:
// re-pushing overwrites, which is how key rotation works.
func (h *hiveCallbackHandlers) storeConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	rc := redisclient.Client()
	if rc == nil {
		writeError(w, http.StatusServiceUnavailable, "callback_unavailable",
			"hive callback config store not available (redis unset on this swarm)")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<16)
	var req hiveCallbackConfigRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	url := strings.TrimRight(strings.TrimSpace(req.HiveURL), "/")
	key := strings.TrimSpace(req.APIKey)
	if url == "" || key == "" {
		writeError(w, http.StatusBadRequest, "missing_field", "hive_url and api_key are required")
		return
	}

	ctx := r.Context()
	if err := rc.Set(ctx, redisclient.Key(hiveCallbackURLKey), url, 0).Err(); err != nil {
		pluginlog.Errf("adminapi: hive callback store url: %v", err)
		writeError(w, http.StatusBadGateway, "callback_store_failed", "redis write failed")
		return
	}
	if err := rc.Set(ctx, redisclient.Key(hiveCallbackKeyKey), key, 0).Err(); err != nil {
		pluginlog.Errf("adminapi: hive callback store key: %v", err)
		writeError(w, http.StatusBadGateway, "callback_store_failed", "redis write failed")
		return
	}
	writeJSON(w, http.StatusOK, hiveCallbackConfigResponse{Stored: true})
}

// ─── outbound: gateway → Hive ────────────────────────────────────────

// config reads the pushed base URL + api key from Redis. ok is false
// when Redis is down or the config was never pushed — callers surface
// that as 503 so the UI can prompt "connect this swarm to Hive".
func (h *hiveCallbackHandlers) config(ctx context.Context) (baseURL, apiKey string, ok bool) {
	rc := redisclient.Client()
	if rc == nil {
		return "", "", false
	}
	vals, err := rc.MGet(ctx,
		redisclient.Key(hiveCallbackURLKey),
		redisclient.Key(hiveCallbackKeyKey),
	).Result()
	if err != nil || len(vals) != 2 {
		return "", "", false
	}
	u, _ := vals[0].(string)
	k, _ := vals[1].(string)
	if u == "" || k == "" {
		return "", "", false
	}
	return u, k, true
}

// hiveError carries the status + parsed error code from a failed
// callback so the gateway handler can relay a faithful status to the
// SPA rather than a flat 502.
type hiveError struct {
	status int
	code   string
	msg    string
}

func (e *hiveError) Error() string { return fmt.Sprintf("hive %d %s: %s", e.status, e.code, e.msg) }

// call issues one JSON request to Hive and decodes the response into
// `out` (may be nil to ignore the body). method/path are appended to
// the pushed base URL; `body` is JSON-encoded when non-nil.
//
// Errors:
//   - *hiveError when the config is missing (status 503) or Hive
//     answers non-2xx (its status + code relayed).
//   - a plain error for transport/timeouts.
func (h *hiveCallbackHandlers) call(ctx context.Context, method, path string, body, out any) error {
	baseURL, apiKey, ok := h.config(ctx)
	if !ok {
		return &hiveError{status: http.StatusServiceUnavailable, code: "hive_not_connected",
			msg: "this swarm is not connected to Hive yet (no callback config pushed)"}
	}

	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("hivecallback: marshal: %w", err)
		}
		reader = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, reader)
	if err != nil {
		return fmt.Errorf("hivecallback: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := h.http.Do(req)
	if err != nil {
		return fmt.Errorf("hivecallback: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		code, msg := "hive_error", strings.TrimSpace(string(raw))
		// Hive returns `{ "error": "…" }`; surface it when present.
		var env struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &env) == nil && env.Error != "" {
			msg = env.Error
		}
		return &hiveError{status: resp.StatusCode, code: code, msg: msg}
	}
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("hivecallback: decode response: %w", err)
		}
	}
	return nil
}

// relayHiveError translates a callback failure into an HTTP response
// on the SPA-facing side, preserving Hive's status when it was an
// *hiveError so a 404 stays a 404 (not a blanket 502).
func relayHiveError(w http.ResponseWriter, err error) {
	if he, ok := err.(*hiveError); ok {
		writeError(w, he.status, he.code, he.msg)
		return
	}
	pluginlog.Errf("adminapi: hive callback: %v", err)
	writeError(w, http.StatusBadGateway, "hive_unreachable", "could not reach Hive")
}
