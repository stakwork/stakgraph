package pluginctx

import (
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// dimHeaderPrefix matches Bifrost's own extraction logic in
// transports/bifrost-http/lib/ctx.go — anything case-insensitively
// starting with this prefix becomes a dimension whose key is the
// suffix.
const dimHeaderPrefix = "x-bf-dim-"

// Well-known dimension names. We use string constants (not enum-style
// types) because dim names are user-controllable headers; the constant
// list is just convenience for the common cases.
//
// Add new names here when a new dim earns first-class treatment in
// the plugin (e.g. a rate-limit lookup key). Anonymous dims still
// flow through fine — they just aren't exposed as named getters.
const (
	DimRunID       = "run-id"
	DimSessionID   = "session-id"
	DimAgentName   = "agent-name"
	DimWorkspaceID = "workspace-id"
	DimUserID      = "user-id"
)

// reservedDimNames are derived by Bifrost from the request itself, so
// we never accept them from `x-bf-dim-*` headers. Mirror of the list
// in Bifrost's extractor.
var reservedDimNames = map[string]struct{}{
	"path":   {},
	"method": {},
}

// ExtractDims walks the request headers and pulls every `x-bf-dim-*`
// into a map keyed by the suffix (lower-cased, prefix stripped). Skips
// reserved names (`path`, `method`) per Bifrost's own contract.
//
// Returns a non-nil map even when there are zero matching headers, so
// callers can index it without nil-checks.
//
// Why we do this ourselves (instead of reading
// `schemas.BifrostContextKeyDimensions`):
//
// At HTTPTransportPreHook time Bifrost has NOT yet populated its own
// dimensions map. That happens inside ConvertToBifrostContext, which
// each handler calls AFTER the TransportInterceptorMiddleware in which
// our PreHook runs. The Bifrost source acknowledges this in
// handlers/middlewares.go: "Root HTTP span starts before
// ConvertToBifrostContext, so read x-bf-dim-* directly."
//
// So PreHook extracts; SetDims stashes; later hooks call Dims to read
// — see ctx.go for the storage key.
func ExtractDims(headers map[string]string) map[string]string {
	dims := make(map[string]string, len(headers))
	for k, v := range headers {
		lk := strings.ToLower(k)
		suffix, ok := strings.CutPrefix(lk, dimHeaderPrefix)
		if !ok || suffix == "" {
			continue
		}
		if _, reserved := reservedDimNames[suffix]; reserved {
			continue
		}
		dims[suffix] = v
	}
	return dims
}

// SetDims stashes the result of ExtractDims on the context. Idempotent
// — call multiple times and the last one wins. Hooks downstream of
// PreHook should never call this; they should only read via Dims.
func SetDims(ctx *schemas.BifrostContext, dims map[string]string) {
	ctx.SetValue(keyDimensions, dims)
}

// Dims returns the dim map stashed by PreHook, or an empty (non-nil)
// map if it wasn't (e.g. SDK-mode usage where the transport hooks
// never fire, or a request type Bifrost routes without going through
// TransportInterceptorMiddleware).
//
// Always safe to index — never returns nil.
func Dims(ctx *schemas.BifrostContext) map[string]string {
	if dims, ok := ctx.Value(keyDimensions).(map[string]string); ok && dims != nil {
		return dims
	}
	return map[string]string{}
}
