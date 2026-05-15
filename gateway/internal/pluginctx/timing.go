package pluginctx

import (
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// MarkStart records the wall-clock instant a request entered the
// plugin's PreHook. Used by later hooks to compute elapsed time. Idempotent
// — call multiple times and the most recent wins, which lets in-pipeline
// timers refer to "elapsed since the most recent stage" if we ever care.
func MarkStart(ctx *schemas.BifrostContext) {
	ctx.SetValue(keyStartTime, time.Now())
}

// Elapsed returns the duration since MarkStart was called, or 0 if no
// start time was recorded (e.g. SDK-mode usage where PreHook never fired).
func Elapsed(ctx *schemas.BifrostContext) time.Duration {
	start, ok := ctx.Value(keyStartTime).(time.Time)
	if !ok {
		return 0
	}
	return time.Since(start)
}

// SetRequestID stashes a request-correlation identifier for downstream
// hooks. Today we set this to x-bf-dim-run-id; once macaroons land it
// becomes the verified run identifier (the same string, but
// cryptographically attested rather than user-supplied).
func SetRequestID(ctx *schemas.BifrostContext, id string) {
	ctx.SetValue(keyRequestID, id)
}

// RequestID retrieves the value stashed by SetRequestID, or "" if
// unset.
func RequestID(ctx *schemas.BifrostContext) string {
	id, _ := ctx.Value(keyRequestID).(string)
	return id
}
