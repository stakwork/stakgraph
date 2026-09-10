package hooks

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

func strPtr(s string) *string { return &s }

func TestToolCallNames(t *testing.T) {
	chat := &schemas.BifrostChatResponse{
		Choices: []schemas.BifrostResponseChoice{
			{
				ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
					Message: &schemas.ChatMessage{
						ChatAssistantMessage: &schemas.ChatAssistantMessage{
							ToolCalls: []schemas.ChatAssistantMessageToolCall{
								{Function: schemas.ChatAssistantMessageToolCallFunction{Name: strPtr("grep")}},
								{Function: schemas.ChatAssistantMessageToolCallFunction{Name: strPtr("read")}},
								{Function: schemas.ChatAssistantMessageToolCallFunction{Name: nil}},
							},
						},
					},
				},
			},
			// Stream-shaped choice (no ChatNonStreamResponseChoice):
			// must be skipped without panicking on the nil embed.
			{},
		},
	}
	got := toolCallNames(chat)
	if len(got) != 2 || got[0] != "grep" || got[1] != "read" {
		t.Fatalf("toolCallNames = %v, want [grep read]", got)
	}
	if toolCallNames(nil) != nil {
		t.Fatal("nil chat must yield nil names")
	}
	if toolCallNames(&schemas.BifrostChatResponse{}) != nil {
		t.Fatal("no choices must yield nil names")
	}
}

func TestResolveCost_Precedence(t *testing.T) {
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-3-5-haiku-latest": {InputPerMTok: 1.0, OutputPerMTok: 10.0},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })

	dims := map[string]string{}

	// 1. Provider-computed cost wins over the table.
	chat := &schemas.BifrostChatResponse{Model: "claude-3-5-haiku-latest"}
	usage := &schemas.BifrostLLMUsage{
		PromptTokens:     1000,
		CompletionTokens: 1000,
		TotalTokens:      2000,
		Cost:             &schemas.BifrostCost{TotalCost: 0.5},
	}
	if got := chatCost(chat, usage, dims); got != 0.5 {
		t.Fatalf("provider cost should win, got %v", got)
	}

	// 2. Falls back to the pricing table: 1000*1/1e6 + 1000*10/1e6 = 0.011.
	usage.Cost = nil
	if got := chatCost(chat, usage, dims); got != 0.011 {
		t.Fatalf("table cost = %v, want 0.011", got)
	}

	// 3. Unpriced model → 0.
	chat.Model = "mystery-model"
	if got := chatCost(chat, usage, dims); got != 0 {
		t.Fatalf("unpriced model cost = %v, want 0", got)
	}

	// Nil usage → 0.
	if got := chatCost(chat, nil, dims); got != 0 {
		t.Fatalf("nil usage cost = %v, want 0", got)
	}
}

func TestResolveCost_PrefersResolvedModel(t *testing.T) {
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-3-5-haiku-latest": {InputPerMTok: 1.0, OutputPerMTok: 1.0},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })

	chat := &schemas.BifrostChatResponse{
		Model: "some-alias",
		ExtraFields: schemas.BifrostResponseExtraFields{
			ResolvedModelUsed: "anthropic/claude-3-5-haiku-latest",
		},
	}
	usage := &schemas.BifrostLLMUsage{PromptTokens: 500, CompletionTokens: 500, TotalTokens: 1000}
	if got := chatCost(chat, usage, map[string]string{}); got != 0.001 {
		t.Fatalf("resolved-model cost = %v, want 0.001", got)
	}
}

// The xAI case: bifrost resolves the wire model bare ("grok-4") and
// reports the provider on RoutingInfo; the datasheet keys the row as
// "xai/grok-4". The hook must hand the provider to the price lookup
// or every Grok call accumulates at $0.
func TestResolveCost_ProviderNamespacedCatalog(t *testing.T) {
	auth.SetConfigForTest(auth.Config{})
	pricing.SetTableForTest(map[string]pricing.Price{
		"xai/grok-4": {InputPerMTok: 1.0, OutputPerMTok: 1.0},
	})
	t.Cleanup(func() {
		pricing.SetTableForTest(nil)
		auth.SetConfigForTest(auth.Config{})
	})

	usage := &schemas.BifrostLLMUsage{PromptTokens: 500, CompletionTokens: 500, TotalTokens: 1000}

	chat := &schemas.BifrostChatResponse{
		Model: "grok-4",
		ExtraFields: schemas.BifrostResponseExtraFields{
			ResolvedModelUsed: "grok-4",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.XAI, Model: "grok-4"},
		},
	}
	if got := chatCost(chat, usage, map[string]string{}); got != 0.001 {
		t.Fatalf("RoutingInfo provider cost = %v, want 0.001", got)
	}

	// Older chunks only carry the deprecated ExtraFields.Provider.
	chat.ExtraFields.RoutingInfo = schemas.RoutingInfo{}
	chat.ExtraFields.Provider = schemas.XAI
	if got := chatCost(chat, usage, map[string]string{}); got != 0.001 {
		t.Fatalf("deprecated provider field cost = %v, want 0.001", got)
	}

	// No provider anywhere → the namespaced row is unreachable → $0.
	chat.ExtraFields.Provider = ""
	if got := chatCost(chat, usage, map[string]string{}); got != 0 {
		t.Fatalf("provider-less cost = %v, want 0", got)
	}
}

// chatCost prices a chat-shaped response the way LLMPost does, with
// the usage supplied separately so the precedence cases can vary it.
func chatCost(chat *schemas.BifrostChatResponse, usage *schemas.BifrostLLMUsage, dims map[string]string) float64 {
	call := chatCallUsage(chat)
	call.usage = usage
	return resolveCost(call, dims)
}
