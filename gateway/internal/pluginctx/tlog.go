package pluginctx

import "github.com/maximhq/bifrost/core/schemas"

// Phase-12 transparency-log handoffs. Like the macaroon keys these
// are private; the hooks go through the typed helpers.
const (
	keyRequestHash  schemas.BifrostContextKey = "stakgraph-gateway/request-sha256"
	keyRequestModel schemas.BifrostContextKey = "stakgraph-gateway/request-model"
)

// SetRequestHash stashes the lowercase-hex SHA-256 of the raw request
// body as HTTPTransportPreHook received it. The body itself is not
// retained. Callers skip this when the body is empty (bifrost's
// large-payload mode leaves req.Body empty), so RequestHash returning
// "" means "no transcript commitment for this call" and the leaf
// carries request_sha256: null.
func SetRequestHash(ctx *schemas.BifrostContext, hexDigest string) {
	ctx.SetValue(keyRequestHash, hexDigest)
}

// RequestHash returns the digest stashed by SetRequestHash, or "".
func RequestHash(ctx *schemas.BifrostContext) string {
	if v, ok := ctx.Value(keyRequestHash).(string); ok {
		return v
	}
	return ""
}

type requestModel struct {
	provider string
	model    string
}

// SetRequestModel records the provider and wire model bifrost parsed
// the request to (PreLLMHook is the first place they are known). The
// leaf prefers the resolved model/provider on the response when there
// is one; these are the fallback for the error paths, where the
// response carries neither.
func SetRequestModel(ctx *schemas.BifrostContext, provider, model string) {
	ctx.SetValue(keyRequestModel, requestModel{provider: provider, model: model})
}

// RequestModel returns the values stashed by SetRequestModel, or two
// empty strings.
func RequestModel(ctx *schemas.BifrostContext) (provider, model string) {
	if v, ok := ctx.Value(keyRequestModel).(requestModel); ok {
		return v.provider, v.model
	}
	return "", ""
}
