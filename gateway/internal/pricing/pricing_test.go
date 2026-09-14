package pricing

import (
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	retryBaseDelay = time.Millisecond
	os.Exit(m.Run())
}

// sampleSheet mirrors the real datasheet shape: dollars per token,
// extra fields the gateway ignores, and a free row that must be
// dropped.
const sampleSheet = `{
  "claude-sonnet-5": {
    "provider": "anthropic", "mode": "chat",
    "input_cost_per_token": 2e-06,
    "output_cost_per_token": 1e-05,
    "cache_read_input_token_cost": 2e-07,
    "cache_creation_input_token_cost": 2.5e-06,
    "cache_creation_input_token_cost_above_1hr": 4e-06
  },
  "gpt-5.2": {
    "provider": "openai", "mode": "chat",
    "input_cost_per_token": 1.75e-06,
    "output_cost_per_token": 1.4e-05
  },
  "some-free-model": {
    "provider": "misc", "mode": "chat"
  }
}`

func TestParseDatasheet_ConvertsToPerMTok(t *testing.T) {
	m, err := parseDatasheet([]byte(sampleSheet))
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 {
		t.Fatalf("parsed %d models, want 2 (free row dropped)", len(m))
	}
	sonnet := m["claude-sonnet-5"]
	if sonnet.InputPerMTok != 2.0 || sonnet.OutputPerMTok != 10.0 {
		t.Fatalf("claude-sonnet-5 = %+v, want {2 10 …}", sonnet)
	}
	// Per-token → per-Mtok multiplication is float math; compare with
	// a tolerance rather than an exact decimal.
	if diff := math.Abs(sonnet.CacheReadPerMTok - 0.2); diff > 1e-9 {
		t.Fatalf("claude-sonnet-5 cache read = %v, want ≈0.2", sonnet.CacheReadPerMTok)
	}
	if diff := math.Abs(sonnet.CacheWritePerMTok - 2.5); diff > 1e-9 {
		t.Fatalf("claude-sonnet-5 cache write = %v, want ≈2.5", sonnet.CacheWritePerMTok)
	}
	if diff := math.Abs(sonnet.CacheWrite1hPerMTok - 4.0); diff > 1e-9 {
		t.Fatalf("claude-sonnet-5 cache write 1h = %v, want ≈4.0", sonnet.CacheWrite1hPerMTok)
	}
	// A row without cache fields parses to zero rates, which Cost
	// resolves through bifrost's fallbacks.
	gpt := m["gpt-5.2"]
	if gpt.CacheReadPerMTok != 0 || gpt.CacheWritePerMTok != 0 || gpt.CacheWrite1hPerMTok != 0 {
		t.Fatalf("gpt-5.2 cache rates = %+v, want zero", gpt)
	}
}

func TestPrice_Cost(t *testing.T) {
	approx := func(t *testing.T, what string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("%s = %v, want %v", what, got, want)
		}
	}
	// Sonnet 5 list rates.
	sonnet := Price{InputPerMTok: 2, OutputPerMTok: 10, CacheReadPerMTok: 0.2, CacheWritePerMTok: 2.5, CacheWrite1hPerMTok: 4}

	// No cache detail: prompt×input + completion×output, as before.
	approx(t, "plain", sonnet.Cost(Usage{Prompt: 1000, Completion: 500}), 0.007)

	// A typical agent turn: 87k prompt of which 80k cache reads and
	// 5k cache writes (1k on the 1h TTL), 1k out.
	//   2000×2 + 80000×0.2 + 4000×2.5 + 1000×4 + 1000×10 = 0.044
	turn := Usage{Prompt: 87000, Completion: 1000, CacheRead: 80000, CacheWrite: 5000, CacheWrite1h: 1000}
	approx(t, "cache-aware", sonnet.Cost(turn), 0.044)
	// The same turn with no cache rates known is what the gateway
	// billed before: 87000×2 + 1000×10 = 0.184, 4.2× over.
	approx(t, "flat", Price{InputPerMTok: 2, OutputPerMTok: 10}.Cost(turn), 0.184)

	// Fallback chain, per bifrost: no 1h rate ⇒ the write rate; no
	// write rate ⇒ input; no read rate ⇒ input.
	approx(t, "1h→write", Price{InputPerMTok: 2, OutputPerMTok: 10, CacheWritePerMTok: 2.5}.Cost(Usage{Prompt: 1000, CacheWrite: 1000, CacheWrite1h: 1000}), 0.0025)
	approx(t, "write→input", Price{InputPerMTok: 2, OutputPerMTok: 10, CacheReadPerMTok: 0.2}.Cost(Usage{Prompt: 1000, CacheWrite: 1000}), 0.002)
	approx(t, "read→input", Price{InputPerMTok: 2, OutputPerMTok: 10, CacheWritePerMTok: 2.5}.Cost(Usage{Prompt: 1000, CacheRead: 1000}), 0.002)

	// Clamps: cached counts never exceed what the prompt holds, in
	// bifrost's order — reads, then writes, then the 1h subset.
	approx(t, "read clamp", sonnet.Cost(Usage{Prompt: 100, CacheRead: 500}), 100*0.2/1e6)
	// read=60, write=min(60, 40)=40, 1h=min(100, 40)=40 ⇒ 60×0.2 + 40×4.
	approx(t, "write/1h clamp", sonnet.Cost(Usage{Prompt: 100, CacheRead: 60, CacheWrite: 60, CacheWrite1h: 100}), (60*0.2+40*4)/1e6)
	// Negative garbage never bills below the fresh-prompt figure.
	approx(t, "negative", sonnet.Cost(Usage{Prompt: 100, CacheRead: -5, CacheWrite: -5}), 100*2/1e6)
}

func TestParseDatasheet_RejectsGarbage(t *testing.T) {
	if _, err := parseDatasheet([]byte("not json")); err == nil {
		t.Fatal("expected parse error")
	}
	if _, err := parseDatasheet([]byte(`{"only-free": {"provider":"x"}}`)); err == nil {
		t.Fatal("a sheet with zero priced models must be rejected, not swapped in")
	}
}

func TestKeys_Order(t *testing.T) {
	cases := []struct {
		provider, model string
		want            []string
	}{
		{"anthropic", "claude-sonnet-5", []string{"claude-sonnet-5", "anthropic/claude-sonnet-5"}},
		{"xai", "grok-4", []string{"grok-4", "xai/grok-4"}},
		{"", "grok-4", []string{"grok-4"}},
		// Already provider-prefixed: no double prefix, base form last.
		{"anthropic", "anthropic/claude-sonnet-5", []string{"anthropic/claude-sonnet-5", "claude-sonnet-5"}},
		// Nested vendor path (OpenRouter): provider form before base.
		{"openrouter", "moonshotai/kimi-k2-0905", []string{"moonshotai/kimi-k2-0905", "openrouter/moonshotai/kimi-k2-0905", "kimi-k2-0905"}},
		{"xai", "", nil},
	}
	for _, c := range cases {
		got := Keys(c.provider, c.model)
		if len(got) != len(c.want) {
			t.Fatalf("Keys(%q,%q) = %v, want %v", c.provider, c.model, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("Keys(%q,%q) = %v, want %v", c.provider, c.model, got, c.want)
			}
		}
	}
}

func TestLookup_ProviderNamespaced(t *testing.T) {
	SetTableForTest(map[string]Price{
		"claude-sonnet-5":                    {InputPerMTok: 2, OutputPerMTok: 10},
		"xai/grok-4":                         {InputPerMTok: 3, OutputPerMTok: 15},
		"openrouter/moonshotai/kimi-k2-0905": {InputPerMTok: 0.5, OutputPerMTok: 2},
	})
	t.Cleanup(func() { SetTableForTest(nil) })

	if _, ok := Lookup("anthropic", "claude-sonnet-5"); !ok {
		t.Fatal("exact lookup failed")
	}
	if _, ok := Lookup("anthropic", "anthropic/claude-sonnet-5"); !ok {
		t.Fatal("prefix-stripped lookup failed")
	}
	// The xAI regression: bifrost reports the wire model bare, the
	// datasheet keys it under the provider.
	if p, ok := Lookup("xai", "grok-4"); !ok || p.InputPerMTok != 3 {
		t.Fatalf("Lookup(xai, grok-4) = (%+v, %v), want the xai/grok-4 row", p, ok)
	}
	if _, ok := Lookup("xai", "xai/grok-4"); !ok {
		t.Fatal("already-prefixed model must not be double-prefixed into a miss")
	}
	if _, ok := Lookup("", "grok-4"); ok {
		t.Fatal("without a provider the bare grok row must miss (there is none)")
	}
	if p, ok := Lookup("openrouter", "moonshotai/kimi-k2-0905"); !ok || p.OutputPerMTok != 2 {
		t.Fatalf("Lookup(openrouter, moonshotai/kimi-k2-0905) = (%+v, %v)", p, ok)
	}
	if _, ok := Lookup("anthropic", "unknown-model"); ok {
		t.Fatal("unknown model must miss")
	}
	if _, ok := Lookup("anthropic", ""); ok {
		t.Fatal("empty model must miss")
	}
}

func TestFetch_SwapsAndPersists(t *testing.T) {
	SetTableForTest(nil)
	t.Cleanup(func() { SetTableForTest(nil) })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sampleSheet))
	}))
	defer srv.Close()

	cachePath := filepath.Join(t.TempDir(), "sheet.json")
	FetchNowForTest(srv.URL, cachePath)

	if p, ok := Lookup("openai", "gpt-5.2"); !ok || p.InputPerMTok != 1.75 {
		t.Fatalf("post-fetch Lookup(gpt-5.2) = (%+v, %v)", p, ok)
	}
	raw, err := os.ReadFile(cachePath)
	if err != nil {
		t.Fatalf("cache not persisted: %v", err)
	}
	if string(raw) != sampleSheet {
		t.Fatal("persisted cache must be the raw sheet bytes")
	}
}

func TestFetch_FailureKeepsLastGood(t *testing.T) {
	SetTableForTest(map[string]Price{"claude-sonnet-5": {InputPerMTok: 2, OutputPerMTok: 10}})
	t.Cleanup(func() { SetTableForTest(nil) })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	}))
	defer srv.Close()

	FetchNowForTest(srv.URL, filepath.Join(t.TempDir(), "sheet.json"))

	if _, ok := Lookup("anthropic", "claude-sonnet-5"); !ok {
		t.Fatal("failed fetch must keep the last-good table")
	}
}

func TestFetch_BadBodyKeepsLastGood(t *testing.T) {
	SetTableForTest(map[string]Price{"claude-sonnet-5": {InputPerMTok: 2, OutputPerMTok: 10}})
	t.Cleanup(func() { SetTableForTest(nil) })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("<html>captive portal</html>"))
	}))
	defer srv.Close()

	cachePath := filepath.Join(t.TempDir(), "sheet.json")
	FetchNowForTest(srv.URL, cachePath)

	if _, ok := Lookup("anthropic", "claude-sonnet-5"); !ok {
		t.Fatal("unparseable body must keep the last-good table")
	}
	if _, err := os.Stat(cachePath); err == nil {
		t.Fatal("unparseable body must not be persisted over the cache")
	}
}
