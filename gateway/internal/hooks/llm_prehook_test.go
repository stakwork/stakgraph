package hooks

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
)

// enforcingCtxWithoutMacaroon models the worst case for a caller that
// never presents identity: enforce_macaroons on, and the transport
// hook saw no x-macaroon header.
func enforcingCtxWithoutMacaroon(t *testing.T) *schemas.BifrostContext {
	t.Helper()
	auth.SetConfigForTest(auth.Config{EnforceMacaroons: true})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })
	ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(30*time.Second))
	pluginctx.SetRawMacaroon(ctx, "")
	return ctx
}

// GET /v1/models fans out as one list_models request per provider.
// It is a catalogue read, not inference, so it must pass the gate
// even in enforce mode with no macaroon at all.
func TestLLMPre_ListModelsBypassesMacaroonGate(t *testing.T) {
	ctx := enforcingCtxWithoutMacaroon(t)
	req := &schemas.BifrostRequest{
		RequestType:       schemas.ListModelsRequest,
		ListModelsRequest: &schemas.BifrostListModelsRequest{Provider: schemas.Anthropic},
	}

	out, shortCircuit, err := LLMPre(ctx, req)
	if err != nil {
		t.Fatalf("LLMPre returned error: %v", err)
	}
	if shortCircuit != nil {
		t.Fatalf("list_models was short-circuited: %+v", shortCircuit.Error)
	}
	if out != req {
		t.Fatalf("request was replaced; want the same pointer back")
	}
	if pluginctx.VerifiedClaims(ctx) != nil {
		t.Fatalf("no macaroon was presented, yet claims were stamped")
	}
}

// The exemption must be exactly list_models. A billable request with
// no macaroon still gets the enforce-mode 401 with the stable code.
func TestLLMPre_InferenceStillGatedInEnforceMode(t *testing.T) {
	ctx := enforcingCtxWithoutMacaroon(t)
	req := &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionRequest,
		ChatRequest: &schemas.BifrostChatRequest{Provider: schemas.Anthropic, Model: "claude-sonnet-4-5"},
	}

	_, shortCircuit, err := LLMPre(ctx, req)
	if err != nil {
		t.Fatalf("LLMPre returned error: %v", err)
	}
	if shortCircuit == nil || shortCircuit.Error == nil {
		t.Fatalf("chat_completion without a macaroon passed the gate in enforce mode")
	}
	if shortCircuit.Error.StatusCode == nil || *shortCircuit.Error.StatusCode != 401 {
		t.Fatalf("status = %v, want 401", shortCircuit.Error.StatusCode)
	}
	if shortCircuit.Error.Error == nil || shortCircuit.Error.Error.Code == nil ||
		*shortCircuit.Error.Error.Code != "macaroon_required" {
		t.Fatalf("code = %+v, want macaroon_required", shortCircuit.Error.Error)
	}
}

func TestMacaroonExempt_OnlyListModels(t *testing.T) {
	gated := []schemas.RequestType{
		schemas.ChatCompletionRequest,
		schemas.ChatCompletionStreamRequest,
		schemas.ResponsesRequest,
		schemas.ResponsesStreamRequest,
		schemas.EmbeddingRequest,
		schemas.SpeechRequest,
		schemas.TranscriptionRequest,
		schemas.ImageGenerationRequest,
		schemas.VideoGenerationRequest,
	}
	for _, rt := range gated {
		if macaroonExempt(rt) {
			t.Errorf("%s is exempt from the macaroon gate; only list_models should be", rt)
		}
	}
	if !macaroonExempt(schemas.ListModelsRequest) {
		t.Errorf("list_models is not exempt")
	}
}
