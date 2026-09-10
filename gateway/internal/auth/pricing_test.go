package auth

import (
	"math"
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

func TestPriceCall(t *testing.T) {
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"claude-3-5-haiku-latest": {InputPerMTok: 0.80, OutputPerMTok: 4.00},
	}})
	t.Cleanup(func() { SetConfigForTest(Config{}) })

	// 1000 in + 500 out = 0.0008 + 0.002 = 0.0028.
	got, ok := PriceCall("anthropic", "claude-3-5-haiku-latest", 1000, 500)
	if !ok || got != 0.0028 {
		t.Fatalf("PriceCall = (%v, %v), want (0.0028, true)", got, ok)
	}

	// Provider-prefixed form resolves to the same entry.
	got, ok = PriceCall("anthropic", "anthropic/claude-3-5-haiku-latest", 1000, 500)
	if !ok || got != 0.0028 {
		t.Fatalf("prefixed PriceCall = (%v, %v), want (0.0028, true)", got, ok)
	}

	// Unpriced model: (0, false), caller logs.
	if _, ok := PriceCall("openai", "gpt-4o", 10, 10); ok {
		t.Fatal("unpriced model must return ok=false")
	}

	// Empty table: never ok.
	SetConfigForTest(Config{})
	if _, ok := PriceCall("anthropic", "claude-3-5-haiku-latest", 10, 10); ok {
		t.Fatal("empty pricing table must return ok=false")
	}
}

func TestPriceCall_CatalogFallback(t *testing.T) {
	pricing.SetTableForTest(map[string]pricing.Price{
		"claude-sonnet-5": {InputPerMTok: 2.0, OutputPerMTok: 10.0},
	})
	t.Cleanup(func() {
		pricing.SetTableForTest(nil)
		SetConfigForTest(Config{})
	})

	// No config entry → catalog prices it: 1000*2/1e6 + 500*10/1e6.
	SetConfigForTest(Config{})
	got, ok := PriceCall("anthropic", "claude-sonnet-5", 1000, 500)
	if !ok || got != 0.007 {
		t.Fatalf("catalog PriceCall = (%v, %v), want (0.007, true)", got, ok)
	}

	// Config entry for the same model overrides the catalog.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 4.0, OutputPerMTok: 20.0},
	}})
	got, ok = PriceCall("anthropic", "claude-sonnet-5", 1000, 500)
	if !ok || got != 0.014 {
		t.Fatalf("config-over-catalog PriceCall = (%v, %v), want (0.014, true)", got, ok)
	}

	// Neither source knows the model → (0, false).
	if _, ok := PriceCall("anthropic", "mystery-model", 10, 10); ok {
		t.Fatal("model absent from both sources must return ok=false")
	}
}

func TestPriceCall_ProviderNamespaced(t *testing.T) {
	// Catalog keyed the way bifrost's datasheet keys xAI and
	// OpenRouter rows; bifrost reports the wire model bare.
	pricing.SetTableForTest(map[string]pricing.Price{
		"xai/grok-4":                         {InputPerMTok: 3.0, OutputPerMTok: 15.0},
		"openrouter/moonshotai/kimi-k2-0905": {InputPerMTok: 1.0, OutputPerMTok: 1.0},
	})
	t.Cleanup(func() {
		pricing.SetTableForTest(nil)
		SetConfigForTest(Config{})
	})
	SetConfigForTest(Config{})

	// 1000*3/1e6 + 500*15/1e6 = 0.003 + 0.0075 (float sum, so a
	// tolerance rather than an exact decimal).
	got, ok := PriceCall("xai", "grok-4", 1000, 500)
	if !ok || math.Abs(got-0.0105) > 1e-12 {
		t.Fatalf("PriceCall(xai, grok-4) = (%v, %v), want (≈0.0105, true)", got, ok)
	}
	got, ok = PriceCall("openrouter", "moonshotai/kimi-k2-0905", 1000, 1000)
	if !ok || got != 0.002 {
		t.Fatalf("PriceCall(openrouter, …kimi) = (%v, %v), want (0.002, true)", got, ok)
	}
	// No provider → no namespaced candidate → the bare row is absent.
	if _, ok := PriceCall("", "grok-4", 10, 10); ok {
		t.Fatal("bare grok-4 without a provider must miss the namespaced catalog")
	}

	// The operator table accepts either spelling and still wins.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"grok-4": {InputPerMTok: 1.0, OutputPerMTok: 1.0},
	}})
	if got, ok := PriceCall("xai", "grok-4", 1000, 1000); !ok || got != 0.002 {
		t.Fatalf("bare config key over catalog = (%v, %v), want (0.002, true)", got, ok)
	}
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"xai/grok-4": {InputPerMTok: 2.0, OutputPerMTok: 2.0},
	}})
	if got, ok := PriceCall("xai", "grok-4", 1000, 1000); !ok || got != 0.004 {
		t.Fatalf("namespaced config key over catalog = (%v, %v), want (0.004, true)", got, ok)
	}
}
