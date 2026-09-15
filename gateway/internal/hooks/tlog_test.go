package hooks

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/tlog"
)

// Phase-12 hook wiring: one leaf per accounted call, from exactly the
// accounting sites, carrying the verified identity and the hashes.

func newTestTlog(t *testing.T) *tlog.Log {
	t.Helper()
	l, err := tlog.Open(filepath.Join(t.TempDir(), "leaves.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	tlog.SetDefaultForTest(l)
	t.Cleanup(func() {
		tlog.SetDefaultForTest(nil)
		_ = l.Close()
	})
	return l
}

func leavesOf(t *testing.T, l *tlog.Log) []tlog.Leaf {
	t.Helper()
	page, err := l.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]tlog.Leaf, len(page.Leaves))
	for i, raw := range page.Leaves {
		if err := json.Unmarshal(raw, &out[i]); err != nil {
			t.Fatalf("leaf %d: %v", i, err)
		}
	}
	return out
}

// tlogContext is newClaimsContext (org_test / u_test / chat-agent /
// r_resp) plus what TransportPre and LLMPre stash for the leaf.
func tlogContext() *schemas.BifrostContext {
	ctx := newClaimsContext()
	pluginctx.SetRawMacaroon(ctx, "the-raw-macaroon-header")
	pluginctx.SetRequestModel(ctx, "anthropic", "claude-sonnet-5")
	return ctx
}

func chatResponse(requestType schemas.RequestType) *schemas.BifrostResponse {
	return &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{
		Model: "claude-sonnet-5",
		Usage: &schemas.BifrostLLMUsage{PromptTokens: 1000, CompletionTokens: 100, TotalTokens: 1100},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType:       requestType,
			ResolvedModelUsed: "claude-sonnet-5",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.Anthropic, Model: "claude-sonnet-5"},
		},
	}}
}

var streamReq = &schemas.HTTPRequest{Path: "/v1/chat/completions"}

func priceSonnet(t *testing.T) {
	t.Helper()
	auth.SetConfigForTest(auth.Config{ModelPricing: map[string]auth.ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 1, OutputPerMTok: 10},
	}})
	t.Cleanup(func() { auth.SetConfigForTest(auth.Config{}) })
}

func TestTransportPre_StashesBodyHash(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-5","messages":[]}`)
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	if _, err := TransportPre(ctx, &schemas.HTTPRequest{
		Method: "POST", Path: "/anthropic/v1/messages", Headers: map[string]string{}, Body: body,
	}); err != nil {
		t.Fatal(err)
	}
	if got := pluginctx.RequestHash(ctx); got != sha256Hex(body) {
		t.Fatalf("request hash = %q, want sha256 of the body", got)
	}

	// Empty body (bifrost's large-payload mode): nothing stashed, so
	// the leaf carries request_sha256: null rather than sha256("").
	ctx = schemas.NewBifrostContext(context.Background(), time.Time{})
	if _, err := TransportPre(ctx, &schemas.HTTPRequest{Method: "POST", Path: "/x", Headers: map[string]string{}}); err != nil {
		t.Fatal(err)
	}
	if got := pluginctx.RequestHash(ctx); got != "" {
		t.Fatalf("empty body must stash no hash, got %q", got)
	}
}

func TestLLMPost_NonStream_AppendsOneLeaf(t *testing.T) {
	l := newTestTlog(t)
	priceSonnet(t)
	ctx := tlogContext()
	pluginctx.SetRequestHash(ctx, sha256Hex([]byte("the body")))

	resp := chatResponse(schemas.ChatCompletionRequest)
	LLMPost(ctx, resp, nil)
	LLMPost(ctx, resp, nil) // a second firing is deduped by MarkAccounted

	leaves := leavesOf(t, l)
	if len(leaves) != 1 {
		t.Fatalf("want 1 leaf, got %d", len(leaves))
	}
	leaf := leaves[0]
	if leaf.V != tlog.LeafVersion || len(leaf.LeafID) != 32 {
		t.Errorf("v/leaf_id = %d/%q", leaf.V, leaf.LeafID)
	}
	if _, err := time.Parse(time.RFC3339Nano, leaf.TS); err != nil {
		t.Errorf("ts = %q: %v", leaf.TS, err)
	}
	if leaf.OrgID != "org_test" || leaf.UserID != "u_test" || leaf.RunID != "r_resp" || leaf.Agent != "chat-agent" {
		t.Errorf("identity = %s/%s/%s/%s, want claims", leaf.OrgID, leaf.UserID, leaf.RunID, leaf.Agent)
	}
	if leaf.MacaroonSHA256 != sha256Hex([]byte("the-raw-macaroon-header")) {
		t.Errorf("macaroon_sha256 = %s", leaf.MacaroonSHA256)
	}
	if leaf.RequestSHA256 == nil || *leaf.RequestSHA256 != sha256Hex([]byte("the body")) {
		t.Errorf("request_sha256 = %v", leaf.RequestSHA256)
	}
	if leaf.Model != "claude-sonnet-5" || leaf.Provider != "anthropic" {
		t.Errorf("model/provider = %s/%s", leaf.Model, leaf.Provider)
	}
	if leaf.PromptTokens != 1000 || leaf.CompletionTokens != 100 {
		t.Errorf("tokens = %d/%d", leaf.PromptTokens, leaf.CompletionTokens)
	}
	// 1000*1/1e6 + 100*10/1e6
	if leaf.CostUSD != 0.002 {
		t.Errorf("cost_usd = %v, want 0.002", leaf.CostUSD)
	}
	if leaf.Status != tlog.StatusOK {
		t.Errorf("status = %s", leaf.Status)
	}
	if leaf.ResponseSHA256 != nil || leaf.AgentRequestSig != nil {
		t.Error("reserved fields must be null")
	}
}

func TestLLMPost_Error_AppendsErrorLeaf(t *testing.T) {
	l := newTestTlog(t)
	ctx := tlogContext() // no request hash: the leaf must say null

	LLMPost(ctx, nil, &schemas.BifrostError{Error: &schemas.ErrorField{Message: "upstream 500"}})

	leaves := leavesOf(t, l)
	if len(leaves) != 1 {
		t.Fatalf("want 1 leaf, got %d", len(leaves))
	}
	leaf := leaves[0]
	if leaf.Status != tlog.StatusError || leaf.CostUSD != 0 || leaf.PromptTokens != 0 || leaf.CompletionTokens != 0 {
		t.Errorf("error leaf = %+v", leaf)
	}
	// No response to name the model: fall back to what PreLLMHook saw.
	if leaf.Model != "claude-sonnet-5" || leaf.Provider != "anthropic" {
		t.Errorf("model/provider = %s/%s, want the PreLLMHook fallback", leaf.Model, leaf.Provider)
	}
	if leaf.RequestSHA256 != nil {
		t.Errorf("request_sha256 = %q, want null when no body hash was stashed", *leaf.RequestSHA256)
	}
	if leaf.RunID != "r_resp" {
		t.Errorf("run_id = %s", leaf.RunID)
	}
}

func TestStream_SetupFiringSkipped_UsageChunkLogsOnce(t *testing.T) {
	l := newTestTlog(t)
	priceSonnet(t)
	ctx := tlogContext()

	// PostLLMHook fires once at stream set-up with no usage: no leaf.
	LLMPost(ctx, chatResponse(schemas.ChatCompletionStreamRequest), nil)
	if l.Size() != 0 {
		t.Fatalf("stream set-up firing appended %d leaves", l.Size())
	}
	chunk := &schemas.BifrostStreamChunk{
		BifrostChatResponse: chatResponse(schemas.ChatCompletionStreamRequest).ChatResponse,
	}
	if _, err := StreamChunk(ctx, streamReq, chunk); err != nil {
		t.Fatal(err)
	}
	// A provider that repeats usage on a later chunk must not double-log.
	if _, err := StreamChunk(ctx, streamReq, chunk); err != nil {
		t.Fatal(err)
	}
	leaves := leavesOf(t, l)
	if len(leaves) != 1 {
		t.Fatalf("want 1 leaf, got %d", len(leaves))
	}
	if leaves[0].Status != tlog.StatusOK || leaves[0].PromptTokens != 1000 || leaves[0].CostUSD != 0.002 {
		t.Errorf("stream leaf = %+v", leaves[0])
	}
	if leaves[0].Model != "claude-sonnet-5" || leaves[0].Provider != "anthropic" {
		t.Errorf("model/provider = %s/%s", leaves[0].Model, leaves[0].Provider)
	}
}

func TestStreamChunk_Error_AppendsErrorLeafOnce(t *testing.T) {
	l := newTestTlog(t)
	ctx := tlogContext()

	errChunk := &schemas.BifrostStreamChunk{BifrostError: &schemas.BifrostError{Error: &schemas.ErrorField{Message: "boom"}}}
	if _, err := StreamChunk(ctx, streamReq, errChunk); err != nil {
		t.Fatal(err)
	}
	// Anything after the error on the same request is already accounted.
	usage := &schemas.BifrostStreamChunk{BifrostChatResponse: chatResponse(schemas.ChatCompletionStreamRequest).ChatResponse}
	if _, err := StreamChunk(ctx, streamReq, usage); err != nil {
		t.Fatal(err)
	}
	leaves := leavesOf(t, l)
	if len(leaves) != 1 || leaves[0].Status != tlog.StatusError {
		t.Fatalf("leaves = %+v", leaves)
	}
}

func TestNoClaims_NoLeaf(t *testing.T) {
	l := newTestTlog(t)
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	pluginctx.SetRequestModel(ctx, "anthropic", "claude-sonnet-5")

	LLMPost(ctx, chatResponse(schemas.ChatCompletionRequest), nil)
	LLMPost(ctx, nil, &schemas.BifrostError{Error: &schemas.ErrorField{Message: "x"}})
	_, _ = StreamChunk(ctx, streamReq, &schemas.BifrostStreamChunk{BifrostError: &schemas.BifrostError{}})
	if l.Size() != 0 {
		t.Fatalf("unverified calls must not be logged, got %d leaves", l.Size())
	}
}

// Append failure is logged and swallowed: the hook must return the
// provider's response untouched.
func TestAppend_FailureDoesNotBreakTheCall(t *testing.T) {
	tlog.SetDefaultForTest(nil) // ErrNotInitialized on every append
	t.Cleanup(func() { tlog.SetDefaultForTest(nil) })
	ctx := tlogContext()
	resp := chatResponse(schemas.ChatCompletionRequest)
	got, gotErr, err := LLMPost(ctx, resp, nil)
	if got != resp || gotErr != nil || err != nil {
		t.Fatalf("LLMPost altered the response on tlog failure: %v %v %v", got, gotErr, err)
	}
}
