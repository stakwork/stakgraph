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
	got, ok := PriceCall("anthropic", "claude-3-5-haiku-latest", pricing.Usage{Prompt: 1000, Completion: 500})
	if !ok || got != 0.0028 {
		t.Fatalf("PriceCall = (%v, %v), want (0.0028, true)", got, ok)
	}

	// Provider-prefixed form resolves to the same entry.
	got, ok = PriceCall("anthropic", "anthropic/claude-3-5-haiku-latest", pricing.Usage{Prompt: 1000, Completion: 500})
	if !ok || got != 0.0028 {
		t.Fatalf("prefixed PriceCall = (%v, %v), want (0.0028, true)", got, ok)
	}

	// Unpriced model: (0, false), caller logs.
	if _, ok := PriceCall("openai", "gpt-4o", pricing.Usage{Prompt: 10, Completion: 10}); ok {
		t.Fatal("unpriced model must return ok=false")
	}

	// Empty table: never ok.
	SetConfigForTest(Config{})
	if _, ok := PriceCall("anthropic", "claude-3-5-haiku-latest", pricing.Usage{Prompt: 10, Completion: 10}); ok {
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
	got, ok := PriceCall("anthropic", "claude-sonnet-5", pricing.Usage{Prompt: 1000, Completion: 500})
	if !ok || got != 0.007 {
		t.Fatalf("catalog PriceCall = (%v, %v), want (0.007, true)", got, ok)
	}

	// Config entry for the same model overrides the catalog.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 4.0, OutputPerMTok: 20.0},
	}})
	got, ok = PriceCall("anthropic", "claude-sonnet-5", pricing.Usage{Prompt: 1000, Completion: 500})
	if !ok || got != 0.014 {
		t.Fatalf("config-over-catalog PriceCall = (%v, %v), want (0.014, true)", got, ok)
	}

	// Neither source knows the model → (0, false).
	if _, ok := PriceCall("anthropic", "mystery-model", pricing.Usage{Prompt: 10, Completion: 10}); ok {
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
	got, ok := PriceCall("xai", "grok-4", pricing.Usage{Prompt: 1000, Completion: 500})
	if !ok || math.Abs(got-0.0105) > 1e-12 {
		t.Fatalf("PriceCall(xai, grok-4) = (%v, %v), want (≈0.0105, true)", got, ok)
	}
	got, ok = PriceCall("openrouter", "moonshotai/kimi-k2-0905", pricing.Usage{Prompt: 1000, Completion: 1000})
	if !ok || got != 0.002 {
		t.Fatalf("PriceCall(openrouter, …kimi) = (%v, %v), want (0.002, true)", got, ok)
	}
	// No provider → no namespaced candidate → the bare row is absent.
	if _, ok := PriceCall("", "grok-4", pricing.Usage{Prompt: 10, Completion: 10}); ok {
		t.Fatal("bare grok-4 without a provider must miss the namespaced catalog")
	}

	// The operator table accepts either spelling and still wins.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"grok-4": {InputPerMTok: 1.0, OutputPerMTok: 1.0},
	}})
	if got, ok := PriceCall("xai", "grok-4", pricing.Usage{Prompt: 1000, Completion: 1000}); !ok || got != 0.002 {
		t.Fatalf("bare config key over catalog = (%v, %v), want (0.002, true)", got, ok)
	}
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"xai/grok-4": {InputPerMTok: 2.0, OutputPerMTok: 2.0},
	}})
	if got, ok := PriceCall("xai", "grok-4", pricing.Usage{Prompt: 1000, Completion: 1000}); !ok || got != 0.004 {
		t.Fatalf("namespaced config key over catalog = (%v, %v), want (0.004, true)", got, ok)
	}
}

// Cached prompt tokens price at their own rates from either source.
// Bifrost folds them into the prompt count, so before this the whole
// count billed at the input rate — 10× over on Claude cache reads.
func TestPriceCall_CacheRates(t *testing.T) {
	pricing.SetTableForTest(map[string]pricing.Price{
		"claude-sonnet-5": {InputPerMTok: 2, OutputPerMTok: 10, CacheReadPerMTok: 0.2, CacheWritePerMTok: 2.5, CacheWrite1hPerMTok: 4},
	})
	t.Cleanup(func() {
		pricing.SetTableForTest(nil)
		SetConfigForTest(Config{})
	})
	SetConfigForTest(Config{})

	// A typical agent turn: 87k prompt of which 80k cache reads and
	// 5k cache writes (1k on the 1h TTL), 1k out.
	turn := pricing.Usage{Prompt: 87000, Completion: 1000, CacheRead: 80000, CacheWrite: 5000, CacheWrite1h: 1000}

	// Catalog: 2000×2 + 80000×0.2 + 4000×2.5 + 1000×4 + 1000×10 = 0.044.
	got, ok := PriceCall("anthropic", "claude-sonnet-5", turn)
	if !ok || math.Abs(got-0.044) > 1e-12 {
		t.Fatalf("catalog cache-aware PriceCall = (%v, %v), want (0.044, true)", got, ok)
	}

	// An operator row with its own cache rates: everything doubled.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 4, OutputPerMTok: 20, CacheReadPerMTok: 0.4, CacheWritePerMTok: 5, CacheWrite1hPerMTok: 8},
	}})
	got, ok = PriceCall("anthropic", "claude-sonnet-5", turn)
	if !ok || math.Abs(got-0.088) > 1e-12 {
		t.Fatalf("config cache-aware PriceCall = (%v, %v), want (0.088, true)", got, ok)
	}

	// An operator row WITHOUT cache rates falls back to its own input
	// rate (bifrost's rule), not to the catalog's cache rates:
	// 87000×4 + 1000×20 = 0.368.
	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"claude-sonnet-5": {InputPerMTok: 4, OutputPerMTok: 20},
	}})
	got, ok = PriceCall("anthropic", "claude-sonnet-5", turn)
	if !ok || math.Abs(got-0.368) > 1e-12 {
		t.Fatalf("config flat-rate PriceCall = (%v, %v), want (0.368, true)", got, ok)
	}
}
