// Package pluginctx is the typed wrapper around the per-request data
// the gateway plugin stashes on `schemas.BifrostContext`.
//
// Why a dedicated package
// -----------------------
// BifrostContext is a free-form key/value bag — every plugin lives in
// the same map and any string key could collide. The pattern Bifrost
// itself uses (see schemas/bifrost.go) is to define typed constants of
// type `schemas.BifrostContextKey` and only access values through
// helpers that assert the right type. This package does that for our
// keys.
//
// What's in here today
// --------------------
//   - Dimensions: the x-bf-dim-* map we extract in PreHook (see
//     dims.go).
//   - Request ID + start time: scratch values shared between hooks of
//     the same request.
//
// What's coming next
// ------------------
//   - VerifiedClaims (struct holding macaroon caveats after verify).
//   - RateLimitDecision (cached during PreHook, read in PostLLMHook).
//   - BudgetSnapshot (Redis read once, decrement once).
//
// Each of those gets its own file in this package, follows the same
// "private key + typed setter + typed getter" idiom, and never leaks
// out to callers as a raw context value.
package pluginctx

import "github.com/maximhq/bifrost/core/schemas"

// Context keys. Private to this package — callers go through the
// typed Set/Get helpers below.
//
// String values are namespaced under "stakgraph-gateway/..." so they
// can never collide with bifrost-core or another plugin's keys.
const (
	keyRequestID  schemas.BifrostContextKey = "stakgraph-gateway/request-id"
	keyStartTime  schemas.BifrostContextKey = "stakgraph-gateway/start-time"
	keyDimensions schemas.BifrostContextKey = "stakgraph-gateway/dimensions"
)
