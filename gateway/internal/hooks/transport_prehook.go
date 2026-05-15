package hooks

import (
	"fmt"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// TransportPre is the body of HTTPTransportPreHook. It fires at the
// HTTP transport layer, before the request enters Bifrost core. This
// is the earliest stage at which we can inspect any request, and the
// canonical place for cryptographic auth (macaroon verification) and
// rate-limit pre-checks once those land.
//
// Responsibilities today
// ----------------------
//  1. Mark the start time so later hooks can compute elapsed.
//  2. Extract x-bf-dim-* headers ourselves (Bifrost doesn't populate
//     its dimensions context value until the handler runs, AFTER this
//     hook — see pluginctx/dims.go for the full rationale).
//  3. Stash extracted dims so PreLLMHook / PostLLMHook /
//     StreamChunkHook can read them via pluginctx.Dims without
//     re-parsing.
//  4. Log a structured-ish single line for greppability.
//
// Future responsibilities (NOT here yet)
// --------------------------------------
//   - Macaroon verification: read req's x-macaroon, verify HMAC +
//     caveats against the trust registry, short-circuit with 401 if
//     invalid.
//   - Rate-limit lookup: keyed by (agent-name, user-id, …) — see
//     internal/ratelimit.
//   - Tool-loop detection short-circuit: if the request looks like
//     it's about to push an agent past its tool budget, short-circuit.
//
// Returning (nil, nil) lets bifrost continue normally; returning a
// non-nil *schemas.HTTPResponse short-circuits without ever hitting
// bifrost core (and provider).
func TransportPre(ctx *schemas.BifrostContext, req *schemas.HTTPRequest) (*schemas.HTTPResponse, error) {
	pluginctx.MarkStart(ctx)

	macaroon := req.CaseInsensitiveHeaderLookup("x-macaroon")

	dims := pluginctx.ExtractDims(req.Headers)
	pluginctx.SetDims(ctx, dims)
	pluginctx.SetRequestID(ctx, dims[pluginctx.DimRunID])

	pluginlog.Logf(
		"HTTPTransportPreHook method=%s path=%s body_bytes=%d macaroon=%s run_id=%s session_id=%s agent=%s workspace=%s user=%s dims_count=%d",
		req.Method,
		req.Path,
		len(req.Body),
		redact(macaroon),
		dims[pluginctx.DimRunID],
		dims[pluginctx.DimSessionID],
		dims[pluginctx.DimAgentName],
		dims[pluginctx.DimWorkspaceID],
		dims[pluginctx.DimUserID],
		len(dims),
	)
	ctx.Log(schemas.LogLevelInfo, fmt.Sprintf("PreHook %s %s", req.Method, req.Path))

	// nil, nil = continue pipeline unchanged.
	return nil, nil
}
