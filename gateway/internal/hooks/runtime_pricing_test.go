package hooks

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// A Maple call: what bifrost hands the hook for a custom provider
// "maple" (base_provider_type openai) asked for "maple/gemma4-31b".
// The provider is the custom provider's name; the model is bare.
func mapleChatResponse() *schemas.BifrostResponse {
	return &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{
		Model: "gemma4-31b",
		Usage: &schemas.BifrostLLMUsage{PromptTokens: 100_000, CompletionTokens: 50_000, TotalTokens: 150_000},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType:       schemas.ChatCompletionRequest,
			ResolvedModelUsed: "gemma4-31b",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.ModelProvider("maple"), Model: "gemma4-31b"},
		},
	}}
}

// cappedClaimsContext is newClaimsContext with a $1 cap on the run,
// so the cap walk has something to bite on.
func cappedClaimsContext(runID string) (*schemas.BifrostContext, *macaroon.Claims) {
	exp := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	claims := &macaroon.Claims{
		OrgID:     "org_test",
		UserID:    "u_test",
		AgentName: "chat-agent",
		RunID:     runID,
		EffectiveCaveats: macaroon.EffectiveCaveats{
			Agents:     []string{"chat-agent"},
			MaxCostUSD: 1,
			Exp:        exp,
		},
		Chain: []macaroon.ChainLayer{{RunID: runID, MaxCostUSD: 1, Exp: exp}},
	}
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	pluginctx.SetVerifiedClaims(ctx, claims)
	return ctx, claims
}

// captureStderr runs fn with os.Stderr redirected and returns what
// it wrote: pluginlog writes there.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stderr
	os.Stderr = w
	done := make(chan string)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	defer func() { os.Stderr = orig }()
	fn()
	_ = w.Close()
	return <-done
}

// Before a runtime row: the Maple call counts $0, says so in the log,
// and the run's $1 cap never bites. After PUT /_plugin/pricing's row:
// the same call is priced, the accumulator holds the dollars, and the
// cap walk refuses the run's next call.
func TestLLMPost_CustomProvider_RuntimePricing(t *testing.T) {
	mr := newMiniRedis(t)
	auth.SetConfigForTest(auth.Config{})
	pricing.SetTableForTest(nil)
	auth.LoadRuntimePricing(filepath.Join(t.TempDir(), "pricing.json"))
	t.Cleanup(func() {
		auth.ResetRuntimePricingForTest("")
		auth.SetConfigForTest(auth.Config{})
	})

	// Unpriced: $0, with the log line.
	ctx, claims := cappedClaimsContext("r_maple_unpriced")
	logs := captureStderr(t, func() {
		if _, _, err := LLMPost(ctx, mapleChatResponse(), nil); err != nil {
			t.Fatal(err)
		}
	})
	if !strings.Contains(logs, `accounting: no price for provider="maple" model="gemma4-31b"`) ||
		!strings.Contains(logs, "150000 tokens accumulated as $0") {
		t.Fatalf("missing the unpriced log line; got:\n%s", logs)
	}
	waitFor(t, "steps:run", func() bool { return mr.HGet("bifrost:steps:run:r_maple_unpriced", "total") == "1" })
	if got := hashFloat(mr, "bifrost:cost:run:r_maple_unpriced"); got != 0 {
		t.Fatalf("unpriced cost:run = %v, want 0", got)
	}
	if err := auth.CheckCaps(context.Background(), claims, "", time.Now().UTC()); err != nil {
		t.Fatalf("a $0 call must not trip the $1 cap: %+v", err)
	}

	// The row stadeum would push from strut's catalog.
	if err := auth.SetRuntimePricing(map[string]auth.ModelPrice{
		"maple/gemma4-31b": {InputPerMTok: 4, OutputPerMTok: 16},
	}); err != nil {
		t.Fatal(err)
	}

	// Priced: 100k × $4/M + 50k × $16/M = 0.4 + 0.8 = $1.20, no log line.
	ctx, claims = cappedClaimsContext("r_maple_priced")
	logs = captureStderr(t, func() {
		if _, _, err := LLMPost(ctx, mapleChatResponse(), nil); err != nil {
			t.Fatal(err)
		}
	})
	if strings.Contains(logs, "no price for") {
		t.Fatalf("priced call still logged as unpriced:\n%s", logs)
	}
	if !strings.Contains(logs, "cost_usd=1.200000") {
		t.Fatalf("PostLLMHook line should carry cost_usd=1.200000; got:\n%s", logs)
	}
	waitFor(t, "cost:run", func() bool {
		got := hashFloat(mr, "bifrost:cost:run:r_maple_priced")
		return got > 1.2-1e-9 && got < 1.2+1e-9
	})
	capErr := auth.CheckCaps(context.Background(), claims, "", time.Now().UTC())
	if capErr == nil || capErr.Code != "run_cost_exceeded" || capErr.HTTPStatus != 402 {
		t.Fatalf("cap walk after a $1.20 call on a $1 run = %+v, want 402 run_cost_exceeded", capErr)
	}
}

// Streaming: the usage-bearing final chunk of a custom provider's
// stream is priced from the runtime table the same way.
func TestStreamChunk_CustomProvider_RuntimePricing(t *testing.T) {
	mr := newMiniRedis(t)
	auth.SetConfigForTest(auth.Config{})
	pricing.SetTableForTest(nil)
	auth.LoadRuntimePricing(filepath.Join(t.TempDir(), "pricing.json"))
	t.Cleanup(func() {
		auth.ResetRuntimePricingForTest("")
		auth.SetConfigForTest(auth.Config{})
	})
	if err := auth.SetRuntimePricing(map[string]auth.ModelPrice{
		"maple/gemma4-31b": {InputPerMTok: 1000, OutputPerMTok: 1000},
	}); err != nil {
		t.Fatal(err)
	}

	ctx := newClaimsContext()
	chunk := &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{
		Model: "gemma4-31b",
		Usage: &schemas.BifrostLLMUsage{PromptTokens: 100, CompletionTokens: 400, TotalTokens: 500},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType: schemas.ChatCompletionStreamRequest,
			RoutingInfo: schemas.RoutingInfo{Provider: schemas.ModelProvider("maple"), Model: "gemma4-31b"},
		},
	}}
	if _, err := StreamChunk(ctx, &schemas.HTTPRequest{Path: "/openai/v1/chat/completions"}, chunk); err != nil {
		t.Fatal(err)
	}
	// 500 tokens × $1000/M.
	waitForAccounting(t, mr, 0.5, nil)
}
