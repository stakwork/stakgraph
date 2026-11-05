package auth

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/pricing"
)

// freshRuntimePricing points the runtime table at a file in a temp
// dir, loads it (absent: empty), and clears every price source.
func freshRuntimePricing(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pricing.json")
	LoadRuntimePricing(path)
	SetConfigForTest(Config{})
	pricing.SetTableForTest(nil)
	t.Cleanup(func() {
		ResetRuntimePricingForTest("")
		SetConfigForTest(Config{})
		pricing.SetTableForTest(nil)
	})
	return path
}

var gemmaUsage = pricing.Usage{Prompt: 1000, Completion: 500}

// A custom provider's model is in no source until a row is pushed;
// then it prices, under either spelling of the key.
func TestPriceCall_RuntimeRow(t *testing.T) {
	freshRuntimePricing(t)

	if _, ok := PriceCall("maple", "gemma4-31b", gemmaUsage); ok {
		t.Fatal("maple/gemma4-31b is in no source yet: want ok=false")
	}

	if err := SetRuntimePricing(map[string]ModelPrice{
		"maple/gemma4-31b": {InputPerMTok: 2, OutputPerMTok: 8},
	}); err != nil {
		t.Fatal(err)
	}
	// 1000*2/1e6 + 500*8/1e6 = 0.002 + 0.004.
	for _, call := range []struct{ provider, model string }{
		{"maple", "gemma4-31b"},       // what bifrost reports for a custom provider
		{"maple", "maple/gemma4-31b"}, // the prefixed form
		{"", "maple/gemma4-31b"},      // no provider, prefixed model
	} {
		got, ok := PriceCall(call.provider, call.model, gemmaUsage)
		if !ok || got != 0.006 {
			t.Fatalf("PriceCall(%q, %q) = (%v, %v), want (0.006, true)", call.provider, call.model, got, ok)
		}
	}
	// Another provider's model of the same name does not match the
	// namespaced row.
	if _, ok := PriceCall("numes", "gemma4-31b", gemmaUsage); ok {
		t.Fatal("numes/gemma4-31b must not match the maple row")
	}
}

// Config model_pricing beats the runtime table, which beats the
// datasheet.
func TestPriceCall_RuntimePrecedence(t *testing.T) {
	freshRuntimePricing(t)
	pricing.SetTableForTest(map[string]pricing.Price{
		"grok-4": {InputPerMTok: 1, OutputPerMTok: 1},
	})
	usage := pricing.Usage{Prompt: 1000, Completion: 1000}

	if got, _ := PriceCall("xai", "grok-4", usage); got != 0.002 {
		t.Fatalf("catalog only = %v, want 0.002", got)
	}

	if err := SetRuntimePricing(map[string]ModelPrice{
		"xai/grok-4": {InputPerMTok: 2, OutputPerMTok: 2},
	}); err != nil {
		t.Fatal(err)
	}
	if got, _ := PriceCall("xai", "grok-4", usage); got != 0.004 {
		t.Fatalf("runtime over catalog = %v, want 0.004", got)
	}

	SetConfigForTest(Config{ModelPricing: map[string]ModelPrice{
		"grok-4": {InputPerMTok: 3, OutputPerMTok: 3},
	}})
	if got, _ := PriceCall("xai", "grok-4", usage); got != 0.006 {
		t.Fatalf("config over runtime = %v, want 0.006", got)
	}
}

// A zero row is a price: the model is free, PriceCall says so, and
// the hook stops logging it as unpriced.
func TestPriceCall_RuntimeZeroRowIsFree(t *testing.T) {
	freshRuntimePricing(t)
	if err := SetRuntimePricing(map[string]ModelPrice{"numes/llama-free": {}}); err != nil {
		t.Fatal(err)
	}
	got, ok := PriceCall("numes", "llama-free", gemmaUsage)
	if !ok || got != 0 {
		t.Fatalf("free row = (%v, %v), want (0, true)", got, ok)
	}
}

// Runtime rows price cache tokens with the same formula as the
// other sources.
func TestPriceCall_RuntimeCacheRates(t *testing.T) {
	freshRuntimePricing(t)
	if err := SetRuntimePricing(map[string]ModelPrice{
		"acme/claude-ish": {InputPerMTok: 2000, OutputPerMTok: 10000, CacheReadPerMTok: 200, CacheWritePerMTok: 2500},
	}); err != nil {
		t.Fatal(err)
	}
	// fresh 1000 × 2000 + read 8000 × 200 + write 1000 × 2500 + out 500 × 10000, per 1e6:
	// 2 + 1.6 + 2.5 + 5 = 11.1.
	got, ok := PriceCall("acme", "claude-ish", pricing.Usage{Prompt: 10000, Completion: 500, CacheRead: 8000, CacheWrite: 1000})
	if !ok || got < 11.1-1e-9 || got > 11.1+1e-9 {
		t.Fatalf("cache-aware runtime price = (%v, %v), want (11.1, true)", got, ok)
	}
}

// The table is on disk after a PUT and comes back on the next boot.
func TestRuntimePricing_SurvivesRestart(t *testing.T) {
	path := freshRuntimePricing(t)
	rows := map[string]ModelPrice{
		"maple/gemma4-31b": {InputPerMTok: 2, OutputPerMTok: 8},
		"numes/free-one":   {},
	}
	if err := SetRuntimePricing(rows); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("table not on disk: %v", err)
	}
	var onDisk map[string]ModelPrice
	if err := json.Unmarshal(raw, &onDisk); err != nil || len(onDisk) != 2 {
		t.Fatalf("on disk = %s (%v), want both rows", raw, err)
	}

	// "Restart": forget the table, then boot from the file.
	ResetRuntimePricingForTest("")
	if _, ok := PriceCall("maple", "gemma4-31b", gemmaUsage); ok {
		t.Fatal("reset must forget the table")
	}
	LoadRuntimePricing(path)
	if got, ok := PriceCall("maple", "gemma4-31b", gemmaUsage); !ok || got != 0.006 {
		t.Fatalf("after reload = (%v, %v), want (0.006, true)", got, ok)
	}
	if got := RuntimePricing(); len(got) != 2 {
		t.Fatalf("RuntimePricing after reload = %v, want 2 rows", got)
	}
}

// A PUT replaces the table: rows it leaves out are gone, {} clears.
func TestRuntimePricing_Replaces(t *testing.T) {
	freshRuntimePricing(t)
	if err := SetRuntimePricing(map[string]ModelPrice{
		"maple/a": {InputPerMTok: 1}, "maple/b": {InputPerMTok: 1},
	}); err != nil {
		t.Fatal(err)
	}
	if err := SetRuntimePricing(map[string]ModelPrice{"maple/b": {InputPerMTok: 3}}); err != nil {
		t.Fatal(err)
	}
	got := RuntimePricing()
	if len(got) != 1 || got["maple/b"].InputPerMTok != 3 {
		t.Fatalf("after replace = %v, want only maple/b at 3", got)
	}
	if err := SetRuntimePricing(map[string]ModelPrice{}); err != nil {
		t.Fatal(err)
	}
	if got := RuntimePricing(); got == nil || len(got) != 0 {
		t.Fatalf("after {} = %#v, want an empty, non-nil table", got)
	}
}

// A refused table leaves the old one in memory and on disk.
func TestRuntimePricing_RejectsBadTables(t *testing.T) {
	path := freshRuntimePricing(t)
	good := map[string]ModelPrice{"maple/gemma4-31b": {InputPerMTok: 2, OutputPerMTok: 8}}
	if err := SetRuntimePricing(good); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)

	for name, bad := range map[string]map[string]ModelPrice{
		"nil":            nil,
		"empty key":      {"": {InputPerMTok: 1}},
		"space in key":   {"maple/gemma 4": {InputPerMTok: 1}},
		"negative input": {"maple/x": {InputPerMTok: -1}},
		"negative cache": {"maple/x": {InputPerMTok: 1, CacheWrite1hPerMTok: -0.1}},
	} {
		err := SetRuntimePricing(bad)
		if !errors.Is(err, ErrInvalidPrice) {
			t.Fatalf("%s: err = %v, want ErrInvalidPrice", name, err)
		}
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatalf("refused tables changed the file:\nbefore %s\nafter  %s", before, after)
	}
	if got, ok := PriceCall("maple", "gemma4-31b", gemmaUsage); !ok || got != 0.006 {
		t.Fatalf("refused tables changed the live table: (%v, %v)", got, ok)
	}
}

// When the file cannot be written the PUT fails and nothing changes.
func TestRuntimePricing_PersistFailureKeepsOldTable(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	freshRuntimePricing(t)
	if err := SetRuntimePricing(map[string]ModelPrice{"maple/a": {InputPerMTok: 1}}); err != nil {
		t.Fatal(err)
	}

	ResetRuntimePricingForTest(filepath.Join(blocker, "pricing.json"))
	runtimeTable.Store(&map[string]ModelPrice{"maple/a": {InputPerMTok: 1}})
	err := SetRuntimePricing(map[string]ModelPrice{"maple/b": {InputPerMTok: 1}})
	if err == nil || errors.Is(err, ErrInvalidPrice) {
		t.Fatalf("err = %v, want a persist error", err)
	}
	if got := RuntimePricing(); len(got) != 1 || got["maple/a"].InputPerMTok != 1 {
		t.Fatalf("live table after failed persist = %v, want maple/a only", got)
	}
}

// A corrupt or invalid file on disk starts the table empty rather
// than failing the plugin.
func TestLoadRuntimePricing_BadFileStartsEmpty(t *testing.T) {
	freshRuntimePricing(t)
	for name, body := range map[string]string{
		"not json":      `{"maple/a": `,
		"negative rate": `{"maple/a": {"input_per_mtok": -1}}`,
	} {
		path := filepath.Join(t.TempDir(), "pricing.json")
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		LoadRuntimePricing(path)
		if got := RuntimePricing(); len(got) != 0 {
			t.Fatalf("%s: table = %v, want empty", name, got)
		}
		// And the next PUT still works, replacing the bad file.
		if err := SetRuntimePricing(map[string]ModelPrice{"maple/a": {InputPerMTok: 1}}); err != nil {
			t.Fatalf("%s: PUT after bad file: %v", name, err)
		}
	}
}
