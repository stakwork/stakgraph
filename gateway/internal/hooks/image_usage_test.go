package hooks

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// gpt-image-1-mini's datasheet rates (getbifrost.ai/datasheet,
// 2026-11-12): text in, image in, image out, per token.
const (
	miniTextIn   = 2e-6
	miniImageIn  = 2.5e-6
	miniImageOut = 8e-6
)

// fakeCatalog stands in for Bifrost's model catalog
// (framework/modelcatalog), which bifrost-http stamps on every
// request's context and ctx.CalculateCost reads. It prices
// gpt-image-1-mini the way the catalog's computeImageCost does for a
// token-priced model and records what it was handed, so a test can
// see that the call reached it with its usage and routing intact.
type fakeCatalog struct {
	calls int
	seen  *schemas.BifrostResponse
}

func (f *fakeCatalog) GetModelInfo(schemas.ModelProvider, string) *schemas.Model { return nil }

func (f *fakeCatalog) CalculateRequestCost(_ *schemas.BifrostContext, resp *schemas.BifrostResponse) float64 {
	f.calls++
	f.seen = resp
	ef := resp.GetExtraFields()
	if ef == nil || ef.RoutingInfo.Provider != schemas.OpenAI || ef.RoutingInfo.Model != "gpt-image-1-mini" {
		return 0
	}
	var u *schemas.ImageUsage
	switch {
	case resp.ImageGenerationResponse != nil:
		u = resp.ImageGenerationResponse.Usage
	case resp.ImageGenerationStreamResponse != nil:
		u = resp.ImageGenerationStreamResponse.Usage
	}
	if u == nil {
		return 0
	}
	var textIn, imageIn int
	if d := u.InputTokensDetails; d != nil {
		textIn, imageIn = d.TextTokens, d.ImageTokens
	} else {
		textIn = u.InputTokens
	}
	return float64(textIn)*miniTextIn + float64(imageIn)*miniImageIn + float64(u.OutputTokens)*miniImageOut
}

func withCatalog(ctx *schemas.BifrostContext, c schemas.ModelInfoProvider) *schemas.BifrostContext {
	ctx.SetValue(schemas.BifrostContextKeyModelCatalog, c)
	return ctx
}

// The probe call of 2026-11-12: gpt-image-1-mini, low quality,
// 1024×1024. OpenAI reports tokens and no cost.
func openAIImageResponse() *schemas.BifrostResponse {
	return &schemas.BifrostResponse{ImageGenerationResponse: &schemas.BifrostImageGenerationResponse{
		Model: "gpt-image-1-mini",
		Data:  []schemas.ImageData{{B64JSON: "iVBORw0KGgo="}},
		Usage: &schemas.ImageUsage{
			InputTokens:         18,
			InputTokensDetails:  &schemas.ImageTokenDetails{TextTokens: 18},
			OutputTokens:        272,
			OutputTokensDetails: &schemas.ImageTokenDetails{ImageTokens: 272},
			TotalTokens:         290,
		},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType:       schemas.ImageGenerationRequest,
			ResolvedModelUsed: "gpt-image-1-mini",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.OpenAI, Model: "gpt-image-1-mini"},
		},
	}}
}

// The xAI probe call: grok-imagine-image reports its own cost, as
// cost_in_usd_ticks (10^10 a dollar), and no tokens.
func xaiImageResponse(normalized bool) *schemas.BifrostResponse {
	ticks := int64(200_000_000)
	usage := &schemas.ImageUsage{CostInUsdTicks: &ticks}
	if normalized {
		usage.NormalizeProviderCost() // what core's xAI provider does
	}
	return &schemas.BifrostResponse{ImageGenerationResponse: &schemas.BifrostImageGenerationResponse{
		Model: "grok-imagine-image",
		Data:  []schemas.ImageData{{URL: "https://imgen.x.ai/xai-imgen/1.jpeg"}},
		Usage: usage,
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType:       schemas.ImageGenerationRequest,
			ResolvedModelUsed: "grok-imagine-image",
			RoutingInfo:       schemas.RoutingInfo{Provider: schemas.XAI, Model: "grok-imagine-image"},
		},
	}}
}

// isolatePricing clears every per-token price source, so an image
// call can only be priced the image way.
func isolatePricing(t *testing.T) {
	t.Helper()
	auth.SetConfigForTest(auth.Config{})
	pricing.SetTableForTest(nil)
	auth.LoadRuntimePricing(filepath.Join(t.TempDir(), "pricing.json"))
	t.Cleanup(func() {
		auth.ResetRuntimePricingForTest("")
		auth.SetConfigForTest(auth.Config{})
	})
}

// xAI: the provider's own $0.02 is what the run is charged, whether
// core normalized the ticks into Cost or not, and the catalog is
// never asked.
func TestLLMPost_Image_ProviderCost(t *testing.T) {
	for _, normalized := range []bool{true, false} {
		mr := newMiniRedis(t)
		isolatePricing(t)
		cat := &fakeCatalog{}
		ctx := withCatalog(newClaimsContext(), cat)
		logs := captureStderr(t, func() {
			if _, _, err := LLMPost(ctx, xaiImageResponse(normalized), nil); err != nil {
				t.Fatal(err)
			}
		})
		if !strings.Contains(logs, "cost_usd=0.020000") {
			t.Fatalf("normalized=%t: PostLLMHook line should carry cost_usd=0.020000; got:\n%s", normalized, logs)
		}
		if strings.Contains(logs, "no price for") {
			t.Fatalf("normalized=%t: a priced image call logged as unpriced:\n%s", normalized, logs)
		}
		waitForAccounting(t, mr, 0.02, nil)
		if cat.calls != 0 {
			t.Fatalf("normalized=%t: the catalog was asked %d times, want 0: the provider's cost wins", normalized, cat.calls)
		}
	}
}

// OpenAI: no cost on the response, so the catalog prices the tokens:
// 18 × $2/M + 272 × $8/M = $0.002212.
func TestLLMPost_Image_CatalogCost(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	// A per-token row for the model must NOT price the image: it
	// would bill image tokens at a text rate.
	if err := auth.SetRuntimePricing(map[string]auth.ModelPrice{
		"gpt-image-1-mini": {InputPerMTok: 1000, OutputPerMTok: 1000},
	}); err != nil {
		t.Fatal(err)
	}
	cat := &fakeCatalog{}
	ctx := withCatalog(newClaimsContext(), cat)
	logs := captureStderr(t, func() {
		if _, _, err := LLMPost(ctx, openAIImageResponse(), nil); err != nil {
			t.Fatal(err)
		}
	})
	const want = 0.002212
	if !strings.Contains(logs, "cost_usd=0.002212") {
		t.Fatalf("PostLLMHook line should carry cost_usd=0.002212; got:\n%s", logs)
	}
	if !strings.Contains(logs, "prompt_tokens=18 completion_tokens=272 total_tokens=290") {
		t.Fatalf("PostLLMHook line should carry the image's tokens; got:\n%s", logs)
	}
	waitForAccounting(t, mr, want, nil)
	if cat.calls != 1 || cat.seen == nil || cat.seen.ImageGenerationResponse == nil {
		t.Fatalf("the catalog should price the image response once; calls=%d seen=%+v", cat.calls, cat.seen)
	}
}

// An image the provider did not price and the catalog does not know
// is $0, said loudly — and still a step against the run.
func TestLLMPost_Image_Unpriced(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	resp := openAIImageResponse()
	resp.ImageGenerationResponse.ExtraFields.RoutingInfo.Model = "mystery-image"
	resp.ImageGenerationResponse.ExtraFields.ResolvedModelUsed = "mystery-image"
	for name, ctx := range map[string]*schemas.BifrostContext{
		"no catalog":      newClaimsContext(),
		"catalog unknown": withCatalog(newClaimsContext(), &fakeCatalog{}),
	} {
		mr.FlushAll()
		logs := captureStderr(t, func() {
			if _, _, err := LLMPost(ctx, resp, nil); err != nil {
				t.Fatal(err)
			}
		})
		if !strings.Contains(logs, `accounting: no price for image call provider="openai" model="mystery-image"`) {
			t.Fatalf("%s: missing the unpriced image line; got:\n%s", name, logs)
		}
		waitForAccounting(t, mr, 0, nil)
	}
}

// The run's cap sees image spend: fifty-one $0.02 images on a $1 run
// and the next call is refused. Before, every image counted $0 and
// the cap never bit.
func TestLLMPost_Image_TripsRunCap(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	for range 51 {
		ctx, _ := cappedClaimsContext("r_images")
		if _, _, err := LLMPost(ctx, xaiImageResponse(true), nil); err != nil {
			t.Fatal(err)
		}
	}
	waitFor(t, "cost:run", func() bool {
		return math.Abs(hashFloat(mr, "bifrost:cost:run:r_images")-1.02) < 1e-9
	})
	_, claims := cappedClaimsContext("r_images")
	capErr := auth.CheckCaps(context.Background(), claims, "", time.Now().UTC())
	if capErr == nil || capErr.Code != "run_cost_exceeded" || capErr.HTTPStatus != 402 {
		t.Fatalf("cap walk after $1.02 of images on a $1 run = %+v, want 402 run_cost_exceeded", capErr)
	}
}

// A streamed image: partial images are not accounted; the completed
// event is, through the catalog, once.
func TestStreamChunk_Image(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	cat := &fakeCatalog{}
	ctx := withCatalog(newClaimsContext(), cat)
	ef := schemas.BifrostResponseExtraFields{
		RequestType:       schemas.ImageGenerationStreamRequest,
		ResolvedModelUsed: "gpt-image-1-mini",
		RoutingInfo:       schemas.RoutingInfo{Provider: schemas.OpenAI, Model: "gpt-image-1-mini"},
	}
	req := &schemas.HTTPRequest{Path: "/openai/v1/images/generations"}

	partial := &schemas.BifrostStreamChunk{BifrostImageGenerationStreamResponse: &schemas.BifrostImageGenerationStreamResponse{
		Type:        schemas.ImageGenerationEventTypePartial,
		B64JSON:     "iVBORw0KGgo=",
		ExtraFields: ef,
	}}
	if _, err := StreamChunk(ctx, req, partial); err != nil {
		t.Fatal(err)
	}
	if cat.calls != 0 {
		t.Fatalf("a partial image was priced (%d calls)", cat.calls)
	}

	completed := &schemas.BifrostStreamChunk{BifrostImageGenerationStreamResponse: &schemas.BifrostImageGenerationStreamResponse{
		Type:    schemas.ImageGenerationEventTypeCompleted,
		B64JSON: "iVBORw0KGgo=",
		Usage: &schemas.ImageUsage{
			InputTokens:  18,
			OutputTokens: 272,
			TotalTokens:  290,
		},
		ExtraFields: ef,
	}}
	logs := captureStderr(t, func() {
		for range 2 { // a repeated completed event is not billed twice
			if _, err := StreamChunk(ctx, req, completed); err != nil {
				t.Fatal(err)
			}
		}
	})
	if !strings.Contains(logs, "StreamChunk accounted") || !strings.Contains(logs, "cost_usd=0.002212") {
		t.Fatalf("missing the accounted line with cost_usd=0.002212; got:\n%s", logs)
	}
	waitForAccounting(t, mr, 0.002212, nil)
	if cat.calls != 1 {
		t.Fatalf("catalog calls = %d, want 1", cat.calls)
	}
}

// A model priced per image reports no tokens; its completed event is
// still accounted, and a stream the provider priced bills that.
func TestStreamChunk_Image_NoTokens(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	ticks := int64(400_000_000)
	ctx := newClaimsContext()
	chunk := &schemas.BifrostStreamChunk{BifrostImageGenerationStreamResponse: &schemas.BifrostImageGenerationStreamResponse{
		Type:  schemas.ImageGenerationEventTypeCompleted,
		URL:   "https://example.invalid/1.png",
		Usage: &schemas.ImageUsage{CostInUsdTicks: &ticks},
		ExtraFields: schemas.BifrostResponseExtraFields{
			RequestType: schemas.ImageGenerationStreamRequest,
			RoutingInfo: schemas.RoutingInfo{Provider: schemas.ModelProvider("replicate"), Model: "flux-schnell"},
		},
	}}
	if _, err := StreamChunk(ctx, &schemas.HTTPRequest{Path: "/v1/images/generations"}, chunk); err != nil {
		t.Fatal(err)
	}
	waitForAccounting(t, mr, 0.04, nil)
}

// A stream LLMPost sees is left to StreamChunk.
func TestLLMPost_ImageStream_Skipped(t *testing.T) {
	mr := newMiniRedis(t)
	isolatePricing(t)
	cat := &fakeCatalog{}
	ctx := withCatalog(newClaimsContext(), cat)
	resp := openAIImageResponse()
	resp.ImageGenerationResponse.ExtraFields.RequestType = schemas.ImageGenerationStreamRequest
	if _, _, err := LLMPost(ctx, resp, nil); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if cat.calls != 0 || mr.HGet("bifrost:steps:run:r_resp", "total") != "" {
		t.Fatalf("a stream's set-up firing was accounted (catalog calls %d)", cat.calls)
	}
}
