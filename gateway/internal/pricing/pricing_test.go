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
    "cache_read_input_token_cost": 2e-07
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
}

func TestParseDatasheet_RejectsGarbage(t *testing.T) {
	if _, err := parseDatasheet([]byte("not json")); err == nil {
		t.Fatal("expected parse error")
	}
	if _, err := parseDatasheet([]byte(`{"only-free": {"provider":"x"}}`)); err == nil {
		t.Fatal("a sheet with zero priced models must be rejected, not swapped in")
	}
}

func TestLookup_PrefixStrip(t *testing.T) {
	SetTableForTest(map[string]Price{"claude-sonnet-5": {InputPerMTok: 2, OutputPerMTok: 10}})
	t.Cleanup(func() { SetTableForTest(nil) })

	if _, ok := Lookup("claude-sonnet-5"); !ok {
		t.Fatal("exact lookup failed")
	}
	if _, ok := Lookup("anthropic/claude-sonnet-5"); !ok {
		t.Fatal("prefix-stripped lookup failed")
	}
	if _, ok := Lookup("unknown-model"); ok {
		t.Fatal("unknown model must miss")
	}
	if _, ok := Lookup(""); ok {
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

	if p, ok := Lookup("gpt-5.2"); !ok || p.InputPerMTok != 1.75 {
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

	if _, ok := Lookup("claude-sonnet-5"); !ok {
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

	if _, ok := Lookup("claude-sonnet-5"); !ok {
		t.Fatal("unparseable body must keep the last-good table")
	}
	if _, err := os.Stat(cachePath); err == nil {
		t.Fatal("unparseable body must not be persisted over the cache")
	}
}
