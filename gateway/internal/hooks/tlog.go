package hooks

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/maximhq/bifrost/core/schemas"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/tlog"
)

// appendTlogLeaf turns one accounted call into a transparency-log
// leaf and appends it. Called from exactly the sites that call
// auth.ApplyToLLMPost, under the same claims-and-MarkAccounted gate,
// so there is one leaf per call and none for a call the macaroon
// adapter did not verify. See
// gateway/plans/phases/phase-12-transparency-log.md "Hot path".
//
// Identity comes from the verified claims. The macaroon hash binds
// the leaf to the exact authorization chain; the request hash (when
// TransportPre saw a body) commits the gateway to the transcript.
// Model and provider prefer what the response resolved to and fall
// back to what PreLLMHook parsed, which is all an errored call has.
//
// Failure is logged and swallowed: the provider call still returns.
// A missing leaf is exactly what a witness cannot distinguish from a
// call that never happened, which is why the error line is loud.
func appendTlogLeaf(
	ctx *schemas.BifrostContext,
	claims *macaroon.Claims,
	call callUsage,
	costUSD float64,
	status string,
) {
	reqProvider, reqModel := pluginctx.RequestModel(ctx)
	model, provider := call.model, call.provider
	if model == "" {
		model = reqModel
	}
	if provider == "" {
		provider = reqProvider
	}
	var promptTokens, completionTokens int
	if call.usage != nil {
		promptTokens, completionTokens = call.usage.PromptTokens, call.usage.CompletionTokens
	}
	leaf := tlog.Leaf{
		OrgID:            claims.OrgID,
		UserID:           claims.UserID,
		RunID:            claims.RunID,
		Agent:            claims.AgentName,
		MacaroonSHA256:   sha256Hex([]byte(pluginctx.RawMacaroon(ctx))),
		RequestSHA256:    optionalString(pluginctx.RequestHash(ctx)),
		Model:            model,
		Provider:         provider,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		CostUSD:          costUSD,
		Status:           status,
	}
	index, err := tlog.Append(leaf)
	if err != nil {
		pluginlog.Errf("tlog: append FAILED run_id=%s agent=%s status=%s — call served but not logged: %v",
			claims.RunID, claims.AgentName, status, err)
		return
	}
	pluginlog.Logf("tlog: leaf index=%d run_id=%s agent=%s status=%s request_committed=%t",
		index, claims.RunID, claims.AgentName, status, leaf.RequestSHA256 != nil)
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// optionalString maps "" to nil so an absent value serializes as an
// explicit JSON null rather than an empty string.
func optionalString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
