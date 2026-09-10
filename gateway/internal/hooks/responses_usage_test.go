package hooks

import (
	"context"
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/redis/go-redis/v9"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// Bifrost core v1.6 routes the Anthropic-native /anthropic/v1/messages
// integration as RequestType "responses", so every Claude call made
// with the Anthropic SDK arrives here with ChatResponse nil and usage
// on ResponsesResponse. Before this coverage the hooks read only the
// chat shape and accounted such calls at prompt=0 completion=0
// cost=$0 (observed on swarm38, 2026-09-10).

func msgType(t schemas.ResponsesMessageType) *schemas.ResponsesMessageType { return &t }

func functionCallItem(name string) schemas.ResponsesMessage {
	return schemas.ResponsesMessage{
		Type: msgType(schemas.ResponsesMessageTypeFunctionCall),
		ResponsesToolMessage: &schemas.ResponsesToolMessage{
			Name:   strPtr(name),
			CallID: strPtr("call_" + name),
		},
	}
}

func TestResponsesUsage_MapsOntoChatShape(t *testing.T) {
	if responsesUsage(nil) != nil {
		t.Fatal("nil usage must stay nil")
	}
	cost := &schemas.BifrostCost{TotalCost: 0.25}
	got := responsesUsage(&schemas.ResponsesResponseUsage{
		InputTokens: 120, OutputTokens: 30, TotalTokens: 150, Cost: cost,
	})
	if got.PromptTokens != 120 || got.CompletionTokens != 30 || got.TotalTokens != 150 {
		t.Fatalf("token mapping = %+v, want prompt=120 completion=30 total=150", *got)
	}
	if got.Cost != cost {
		t.Fatal("provider-computed cost must carry over verbatim")
	}
	// A provider that leaves total_tokens unset still bills the sum.
	got = responsesUsage(&schemas.ResponsesResponseUsage{InputTokens: 7, OutputTokens: 3})
	if got.TotalTokens != 10 {
		t.Fatalf("total fallback = %d, want 10", got.TotalTokens)
	}
}

func TestResponsesToolCallNames(t *testing.T) {
	output := []schemas.ResponsesMessage{
		{Type: msgType(schemas.ResponsesMessageTypeReasoning)},
		functionCallItem("grep"),
		{Type: msgType(schemas.ResponsesMessageTypeMessage)},
		// Server-side tool: not a client tool invocation.
		{
			Type:                 msgType(schemas.ResponsesMessageTypeWebSearchCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{Name: strPtr("web_search")},
		},
		functionCallItem("read"),
		// function_call without the tool embed / without a name:
		// skipped, never a nil deref.
		{Type: msgType(schemas.ResponsesMessageTypeFunctionCall)},
		{Type: msgType(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{}},
		{},
	}
	got := responsesToolCallNames(output)
	if len(got) != 2 || got[0] != "grep" || got[1] != "read" {
		t.Fatalf("responsesToolCallNames = %v, want [grep read]", got)
	}
	if responsesToolCallNames(nil) != nil {
		t.Fatal("no output must yield nil names")
	}
}

func TestCallUsageOf_ResponsesShape(t *testing.T) {
	resp := &schemas.BifrostResponse{ResponsesResponse: &schemas.BifrostResponsesResponse{
		Model:  "some-alias",
		Usage:  &schemas.ResponsesResponseUsage{InputTokens: 1000, OutputTokens: 1000, TotalTokens: 2000},
		Output: []schemas.ResponsesMessage{functionCallItem("grep")},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType:       schemas.ResponsesRequest,
			ResolvedModelUsed: "claude-sonnet-5",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.Anthropic, Model: "claude-sonnet-5"},
		},
	}}
	call := callUsageOf(resp)
	if call.usage == nil || call.usage.PromptTokens != 1000 || call.usage.CompletionTokens != 1000 || call.usage.TotalTokens != 2000 {
		t.Fatalf("usage = %+v, want 1000/1000/2000", call.usage)
	}
	if call.model != "claude-sonnet-5" || call.provider != "anthropic" {
		t.Fatalf("model/provider = %q/%q, want claude-sonnet-5/anthropic", call.model, call.provider)
	}
	if len(call.tools) != 1 || call.tools[0] != "grep" {
		t.Fatalf("tools = %v, want [grep]", call.tools)
	}
	if isStreamRequest(resp) {
		t.Fatal(`"responses" is not a stream request type`)
	}

	if callUsageOf(nil).usage != nil || callUsageOf(&schemas.BifrostResponse{}).usage != nil {
		t.Fatal("nil / empty response must yield nil usage")
	}
}

func TestResolveCost_ResponsesShape(t *testing.T) {
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 1.0, OutputPerMTok: 10.0},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })

	r := &schemas.BifrostResponsesResponse{
		Model: "claude-sonnet-5",
		Usage: &schemas.ResponsesResponseUsage{
			InputTokens: 1000, OutputTokens: 1000, TotalTokens: 2000,
			Cost: &schemas.BifrostCost{TotalCost: 0.5},
		},
	}
	// Provider-computed cost wins, exactly as on the chat path.
	if got := resolveCost(responsesCallUsage(r, &r.ExtraFields), nil); got != 0.5 {
		t.Fatalf("provider cost should win, got %v", got)
	}
	// Then the pricing table: 1000*1/1e6 + 1000*10/1e6 = 0.011.
	r.Usage.Cost = nil
	if got := resolveCost(responsesCallUsage(r, &r.ExtraFields), nil); got != 0.011 {
		t.Fatalf("table cost = %v, want 0.011", got)
	}
}

func TestStreamCallUsage_OnlyTerminalResponsesEvents(t *testing.T) {
	usage := &schemas.ResponsesResponseUsage{InputTokens: 100, OutputTokens: 1, TotalTokens: 101}
	chunk := func(typ schemas.ResponsesStreamResponseType, r *schemas.BifrostResponsesResponse) *schemas.BifrostStreamChunk {
		return &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
			Type: typ, Response: r,
		}}
	}
	// Non-terminal events carry usage too (Anthropic forwards
	// message_start's on response.created) and must be ignored.
	for _, typ := range []schemas.ResponsesStreamResponseType{
		schemas.ResponsesStreamResponseTypeCreated,
		schemas.ResponsesStreamResponseTypeInProgress,
		schemas.ResponsesStreamResponseTypeOutputTextDelta,
	} {
		if got := streamCallUsage(chunk(typ, &schemas.BifrostResponsesResponse{Usage: usage})); got.usage != nil {
			t.Fatalf("%s must not be accountable, got %+v", typ, got.usage)
		}
	}
	for _, typ := range []schemas.ResponsesStreamResponseType{
		schemas.ResponsesStreamResponseTypeCompleted,
		schemas.ResponsesStreamResponseTypeIncomplete,
		schemas.ResponsesStreamResponseTypeFailed,
	} {
		if got := streamCallUsage(chunk(typ, &schemas.BifrostResponsesResponse{Usage: usage})); got.usage == nil || got.usage.TotalTokens != 101 {
			t.Fatalf("%s must be accountable, got %+v", typ, got.usage)
		}
	}
	// Terminal event with no nested response / no usage: nothing to bill.
	if got := streamCallUsage(chunk(schemas.ResponsesStreamResponseTypeCompleted, nil)); got.usage != nil {
		t.Fatal("completed without a response must not be accountable")
	}
	if got := streamCallUsage(chunk(schemas.ResponsesStreamResponseTypeCompleted, &schemas.BifrostResponsesResponse{})); got.usage != nil {
		t.Fatal("completed without usage must not be accountable")
	}
	if got := streamCallUsage(&schemas.BifrostStreamChunk{}); got.usage != nil {
		t.Fatal("empty chunk must not be accountable")
	}
}

// --- end-to-end through the hook bodies, against miniredis ---------

// newMiniRedis mirrors the auth package's helper: points redisclient
// at an in-process miniredis so ApplyToLLMPost has somewhere to write.
func newMiniRedis(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	mr := miniredis.RunT(t)
	c := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	redisclient.SetClientForTest(c)
	t.Cleanup(func() {
		redisclient.SetClientForTest(nil)
		_ = c.Close()
	})
	return mr
}

// newClaimsContext returns a request context carrying a single-layer
// verified chain for run r_resp / agent chat-agent.
func newClaimsContext() *schemas.BifrostContext {
	exp := time.Now().Add(time.Hour).UTC().Format(time.RFC3339)
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	pluginctx.SetVerifiedClaims(ctx, &macaroon.Claims{
		OrgID:     "org_test",
		UserID:    "u_test",
		AgentName: "chat-agent",
		RunID:     "r_resp",
		EffectiveCaveats: macaroon.EffectiveCaveats{
			Agents: []string{"chat-agent"},
			Exp:    exp,
		},
		Chain: []macaroon.ChainLayer{{RunID: "r_resp", Exp: exp}},
	})
	return ctx
}

// waitFor polls until cond holds: ApplyToLLMPost writes on a
// goroutine, so the accumulators land shortly after the hook returns.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func hashFloat(mr *miniredis.Miniredis, key string) float64 {
	f, err := strconv.ParseFloat(mr.HGet(key, "total"), 64)
	if err != nil {
		return math.NaN()
	}
	return f
}

func waitForAccounting(t *testing.T, mr *miniredis.Miniredis, wantCost float64, wantTools []string) {
	t.Helper()
	waitFor(t, "cost:run", func() bool {
		return math.Abs(hashFloat(mr, "bifrost:cost:run:r_resp")-wantCost) < 1e-9
	})
	waitFor(t, "tools:run", func() bool {
		got, _ := mr.List("bifrost:tools:run:r_resp")
		return len(got) == len(wantTools)
	})
	if got := mr.HGet("bifrost:steps:run:r_resp", "total"); got != "1" {
		t.Fatalf("steps:run = %q, want 1", got)
	}
	// LPUSH order: most recent tool first.
	got, _ := mr.List("bifrost:tools:run:r_resp")
	for i, want := range wantTools {
		if got[i] != want {
			t.Fatalf("tools:run = %v, want %v", got, wantTools)
		}
	}
}

func TestLLMPost_ResponsesShape_Accounts(t *testing.T) {
	mr := newMiniRedis(t)
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 1000, OutputPerMTok: 10000},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })
	ctx := newClaimsContext()

	resp := &schemas.BifrostResponse{ResponsesResponse: &schemas.BifrostResponsesResponse{
		Model:  "claude-sonnet-5",
		Usage:  &schemas.ResponsesResponseUsage{InputTokens: 1000, OutputTokens: 1000, TotalTokens: 2000},
		Output: []schemas.ResponsesMessage{functionCallItem("grep"), functionCallItem("read")},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType: schemas.ResponsesRequest,
			RoutingInfo: schemas.RoutingInfo{Provider: schemas.Anthropic, Model: "claude-sonnet-5"},
		},
	}}
	if _, _, err := LLMPost(ctx, resp, nil); err != nil {
		t.Fatal(err)
	}

	// 1000*1000/1e6 + 1000*10000/1e6 = 1 + 10.
	waitForAccounting(t, mr, 11, []string{"read", "grep"})
	if pluginctx.MarkAccounted(ctx) {
		t.Fatal("LLMPost must have consumed the request's accounting slot")
	}
}

// The set-up firing for a Responses stream carries no usage; it must
// leave the accounting slot for the terminal chunk.
func TestLLMPost_ResponsesStreamSetup_DefersToChunks(t *testing.T) {
	ctx := newClaimsContext()
	resp := &schemas.BifrostResponse{ResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeCreated,
		ExtraFields: schemas.BifrostResponseExtraFields{RequestType: schemas.ResponsesStreamRequest},
	}}
	if _, _, err := LLMPost(ctx, resp, nil); err != nil {
		t.Fatal(err)
	}
	if !pluginctx.MarkAccounted(ctx) {
		t.Fatal("stream set-up firing must not consume the accounting slot")
	}
}

func TestStreamChunk_ResponsesTerminalEvent_Accounts(t *testing.T) {
	mr := newMiniRedis(t)
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 1000, OutputPerMTok: 1000},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })
	ctx := newClaimsContext()
	req := &schemas.HTTPRequest{Path: "/anthropic/v1/messages"}

	// Core stamps routing on every chunk's own ExtraFields; the
	// nested Response's block stays blank.
	ef := schemas.BifrostResponseExtraFields{
		RequestType:       schemas.ResponsesStreamRequest,
		ResolvedModelUsed: "claude-sonnet-5",
		RoutingInfo:       schemas.RoutingInfo{Provider: schemas.Anthropic, Model: "claude-sonnet-5"},
	}
	event := func(typ schemas.ResponsesStreamResponseType, r *schemas.BifrostResponsesResponse) *schemas.BifrostStreamChunk {
		return &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
			Type: typ, Response: r, ExtraFields: ef,
		}}
	}
	// response.created forwards message_start's input-only usage
	// (100 in / 1 out → would price at 0.101). It must not be billed.
	created := event(schemas.ResponsesStreamResponseTypeCreated, &schemas.BifrostResponsesResponse{
		Model: "claude-sonnet-5",
		Usage: &schemas.ResponsesResponseUsage{InputTokens: 100, OutputTokens: 1, TotalTokens: 101},
	})
	inProgress := event(schemas.ResponsesStreamResponseTypeInProgress, &schemas.BifrostResponsesResponse{})
	delta := event(schemas.ResponsesStreamResponseTypeOutputTextDelta, nil)
	completed := event(schemas.ResponsesStreamResponseTypeCompleted, &schemas.BifrostResponsesResponse{
		Model:  "claude-sonnet-5",
		Usage:  &schemas.ResponsesResponseUsage{InputTokens: 100, OutputTokens: 400, TotalTokens: 500},
		Output: []schemas.ResponsesMessage{functionCallItem("grep")},
	})
	// A duplicate terminal event must not double-count.
	for _, c := range []*schemas.BifrostStreamChunk{created, inProgress, delta, completed, completed} {
		if _, err := StreamChunk(ctx, req, c); err != nil {
			t.Fatal(err)
		}
	}

	// 100*1000/1e6 + 400*1000/1e6 = 0.1 + 0.4 — the completed
	// event's usage, once.
	waitForAccounting(t, mr, 0.5, []string{"grep"})
	if pluginctx.MarkAccounted(ctx) {
		t.Fatal("StreamChunk must have consumed the request's accounting slot")
	}
}

// Regression guard for the chat stream shape the refactor shares
// code with: the usage-bearing final chunk still accounts.
func TestStreamChunk_ChatFinalChunk_StillAccounts(t *testing.T) {
	mr := newMiniRedis(t)
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"gpt-x": {InputPerMTok: 1000, OutputPerMTok: 1000},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })
	ctx := newClaimsContext()
	req := &schemas.HTTPRequest{Path: "/v1/chat/completions"}

	chunk := &schemas.BifrostStreamChunk{BifrostChatResponse: &schemas.BifrostChatResponse{
		Model: "gpt-x",
		Usage: &schemas.BifrostLLMUsage{PromptTokens: 100, CompletionTokens: 400, TotalTokens: 500},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType: schemas.ChatCompletionStreamRequest,
			RoutingInfo: schemas.RoutingInfo{Provider: schemas.OpenAI, Model: "gpt-x"},
		},
	}}
	if _, err := StreamChunk(ctx, req, chunk); err != nil {
		t.Fatal(err)
	}
	waitForAccounting(t, mr, 0.5, nil)
}
