// Package pricing maintains the model-price catalog the phase-6
// accumulator uses to turn token usage into dollars.
//
// Source of truth: https://getbifrost.ai/datasheet — the SAME file
// bifrost-http's framework syncs (framework/modelcatalog) to compute
// the `cost` column in logs.db. Pricing enforcement from the same
// datasheet means the gateway's Redis accumulators and Bifrost's own
// reporting agree to the cent, with no reconciliation drift. Keys in
// the datasheet are model names exactly as Bifrost resolves them
// ("claude-sonnet-5", "gpt-5.2", "us.anthropic.claude-opus-5", …).
//
// Lifecycle (mirrors the trust registry's fail-soft posture):
//
//  1. Init loads the last persisted datasheet from disk, if any, so
//     a restart during a network outage still prices from the last
//     known-good table.
//  2. A background goroutine fetches the datasheet immediately and
//     then every 24h (the same cadence bifrost itself uses). Each
//     successful fetch atomically swaps the in-memory table and
//     persists the raw bytes for the next boot.
//  3. Fetch failures log a warning and keep the last-good table.
//
// The hot path (Lookup) is a map read behind an atomic pointer — no
// locks, no I/O, no allocation.
//
// Precedence is decided by the caller (internal/auth.PriceCall):
// provider-computed cost → operator model_pricing config → this
// catalog → $0 with a loud log. The config layer stays as the
// operator override and the air-gapped fallback.
package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Price is one model's rates in dollars per million tokens — the
// unit providers publish prices in. The three cache rates are
// optional in both sources (datasheet and operator config); a zero
// rate falls back the way bifrost's own cost calculator does — see
// Cost.
type Price struct {
	InputPerMTok  float64
	OutputPerMTok float64

	// CacheReadPerMTok prices prompt tokens served from the
	// provider's prompt cache (Anthropic cache_read_input_tokens,
	// OpenAI cached_tokens). Zero ⇒ InputPerMTok.
	CacheReadPerMTok float64
	// CacheWritePerMTok prices prompt tokens written into the cache
	// with the default 5-minute TTL (Anthropic
	// cache_creation_input_tokens). Zero ⇒ InputPerMTok.
	CacheWritePerMTok float64
	// CacheWrite1hPerMTok prices the subset of cache writes made
	// with the 1-hour TTL, which Anthropic bills higher. Zero ⇒
	// CacheWritePerMTok.
	CacheWrite1hPerMTok float64
}

// Usage is one call's token breakdown as Cost consumes it. It follows
// Bifrost's convention, where Prompt already INCLUDES every cached
// token: CacheRead and CacheWrite are subsets of Prompt, and
// CacheWrite1h is the subset of CacheWrite made with the 1-hour TTL.
// Callers pass the counts as the provider reported them; Cost clamps
// inconsistent payloads rather than billing negative tokens.
type Usage struct {
	Prompt       int
	Completion   int
	CacheRead    int
	CacheWrite   int
	CacheWrite1h int
}

// Cost prices one call in dollars. The formula mirrors bifrost's own
// computeTextCost (framework/modelcatalog/datasheet/cost.go at the
// transports tag the Dockerfile builds) so the accumulator and the
// logs.db cost column agree:
//
//	fresh prompt      × input
//	cache reads       × cache read     (input when the rate is unknown)
//	5m cache writes   × cache write    (input when unknown)
//	1h cache writes   × cache write 1h (cache write when unknown)
//	completion        × output
//
// Before this the whole prompt count priced at the input rate, which
// billed Claude cache reads at ten times their real cost and tripped
// per-run caps several times early on cache-heavy runs.
//
// Cached counts are clamped to what Prompt can hold, in the order
// bifrost clamps them: reads first, then writes, then the 1h subset.
func (p Price) Cost(u Usage) float64 {
	const mtok = 1_000_000
	read := clamp(u.CacheRead, 0, u.Prompt)
	write := clamp(u.CacheWrite, 0, u.Prompt-read)
	write1h := clamp(u.CacheWrite1h, 0, write)
	fresh := u.Prompt - read - write

	cost := float64(fresh)*p.InputPerMTok/mtok + float64(u.Completion)*p.OutputPerMTok/mtok
	if read > 0 {
		cost += float64(read) * p.cacheReadRate() / mtok
	}
	if write-write1h > 0 {
		cost += float64(write-write1h) * p.cacheWriteRate() / mtok
	}
	if write1h > 0 {
		cost += float64(write1h) * p.cacheWrite1hRate() / mtok
	}
	return cost
}

func (p Price) cacheReadRate() float64 {
	if p.CacheReadPerMTok > 0 {
		return p.CacheReadPerMTok
	}
	return p.InputPerMTok
}

func (p Price) cacheWriteRate() float64 {
	if p.CacheWritePerMTok > 0 {
		return p.CacheWritePerMTok
	}
	return p.InputPerMTok
}

func (p Price) cacheWrite1hRate() float64 {
	if p.CacheWrite1hPerMTok > 0 {
		return p.CacheWrite1hPerMTok
	}
	return p.cacheWriteRate()
}

func clamp(v, lo, hi int) int {
	if hi < lo {
		hi = lo
	}
	return min(max(v, lo), hi)
}

const (
	fetchTimeout  = 45 * time.Second // matches bifrost's DefaultPricingTimeout
	fetchRetries  = 3                // attempts per cycle: 1 + retries
	syncInterval  = 24 * time.Hour   // matches bifrost's DefaultSyncInterval
	maxSheetBytes = 64 << 20         // hard cap on the response body (sheet is ~5MB today)
)

var (
	table    atomic.Pointer[map[string]Price]
	stopLoop context.CancelFunc

	// retryBaseDelay is the first retry's backoff (doubling per
	// attempt). Package-level so tests can shrink it.
	retryBaseDelay = time.Second
)

// Keys returns the lookup candidates for a (provider, model) pair,
// most specific first:
//
//  1. model as given — the datasheet keys most first-party rows
//     bare ("claude-sonnet-5", "gpt-5.2"), which is also the wire
//     model Bifrost reports in ResolvedModelUsed.
//  2. "<provider>/<model>" — the datasheet namespaces some
//     providers' rows under the provider id ("xai/grok-4",
//     "openrouter/moonshotai/kimi-k2-0905") while Bifrost still
//     reports the wire model bare ("grok-4"). Without this step
//     every Grok and OpenRouter call priced at $0.
//  3. the last path segment of model — callers occasionally hold a
//     prefixed form ("anthropic/claude-sonnet-5").
//
// Shared by the catalog and the operator model_pricing table so the
// two sources can't disagree on what a model name means.
func Keys(provider, model string) []string {
	if model == "" {
		return nil
	}
	keys := []string{model}
	if provider != "" && !strings.HasPrefix(model, provider+"/") {
		keys = append(keys, provider+"/"+model)
	}
	if i := strings.LastIndexByte(model, '/'); i >= 0 && i+1 < len(model) {
		keys = append(keys, model[i+1:])
	}
	return keys
}

// Lookup returns the catalog price for a model, trying each Keys
// candidate in order. provider is the Bifrost provider id that
// served the call ("xai", "openrouter", …); pass "" when unknown
// and only the bare and prefix-stripped forms are tried. ok=false
// when the catalog has no entry — the caller falls through to its
// next price source.
func Lookup(provider, model string) (Price, bool) {
	m := table.Load()
	if m == nil {
		return Price{}, false
	}
	for _, k := range Keys(provider, model) {
		if p, ok := (*m)[k]; ok {
			return p, true
		}
	}
	return Price{}, false
}

// Init loads the persisted datasheet (if any) and starts the
// background fetch/refresh loop. Never fatal: with no cache, no
// network, or BIFROST_PLUGIN_PRICING_URL=off, the plugin runs with
// an empty catalog and pricing falls through to the config table.
// Call Stop from plugin Cleanup.
func Init() {
	cachePath := env.PricingCachePath()
	if raw, err := os.ReadFile(cachePath); err != nil {
		// First boot (or a fresh volume): make the empty-catalog
		// window visible — until the first fetch lands, unpriced
		// models accumulate at $0 unless model_pricing covers them.
		pluginlog.Logf("pricing: no persisted datasheet at %s — catalog empty until first fetch", cachePath)
	} else if m, err := parseDatasheet(raw); err != nil {
		pluginlog.Warnf("pricing: persisted datasheet %s unparseable (%v) — waiting for fetch", cachePath, err)
	} else {
		table.Store(&m)
		pluginlog.Logf("pricing: loaded %d models from %s", len(m), cachePath)
	}

	url := env.PricingURLValue()
	if url == "" {
		pluginlog.Logf("pricing: fetch disabled (%s=off) — catalog is config/cache only", env.PricingURL)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	stopLoop = cancel
	go refreshLoop(ctx, url, cachePath)
}

// Stop cancels the refresh loop. Safe to call without Init.
func Stop() {
	if stopLoop != nil {
		stopLoop()
	}
}

func refreshLoop(ctx context.Context, url, cachePath string) {
	fetchAndSwap(ctx, url, cachePath)
	ticker := time.NewTicker(syncInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fetchAndSwap(ctx, url, cachePath)
		}
	}
}

// fetchAndSwap runs one fetch cycle with retries. On success it swaps
// the in-memory table and persists the raw sheet; on failure it logs
// and leaves the last-good table in place.
func fetchAndSwap(ctx context.Context, url, cachePath string) {
	var lastErr error
	for attempt := 0; attempt <= fetchRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<(attempt-1)) * retryBaseDelay
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
		}
		raw, err := fetchOnce(ctx, url)
		if err != nil {
			lastErr = err
			continue
		}
		m, err := parseDatasheet(raw)
		if err != nil {
			lastErr = err
			continue
		}
		table.Store(&m)
		pluginlog.Logf("pricing: datasheet refreshed models=%d source=%s", len(m), url)
		persist(cachePath, raw)
		return
	}
	pluginlog.Warnf("pricing: datasheet fetch failed after %d attempts (%v) — keeping last-good table", fetchRetries+1, lastErr)
}

func fetchOnce(ctx context.Context, url string) ([]byte, error) {
	fctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(fctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxSheetBytes))
}

// datasheetEntry is the subset of bifrost's PricingEntry the gateway
// consumes. Costs are dollars PER TOKEN in the sheet; we convert to
// per-Mtok at parse time.
type datasheetEntry struct {
	InputCostPerToken    float64 `json:"input_cost_per_token"`
	OutputCostPerToken   float64 `json:"output_cost_per_token"`
	CacheReadPerToken    float64 `json:"cache_read_input_token_cost"`
	CacheWritePerToken   float64 `json:"cache_creation_input_token_cost"`
	CacheWrite1hPerToken float64 `json:"cache_creation_input_token_cost_above_1hr"`
}

func parseDatasheet(raw []byte) (map[string]Price, error) {
	var sheet map[string]datasheetEntry
	if err := json.Unmarshal(raw, &sheet); err != nil {
		return nil, err
	}
	m := make(map[string]Price, len(sheet))
	for model, e := range sheet {
		if e.InputCostPerToken == 0 && e.OutputCostPerToken == 0 {
			continue // free/embedding/no-price rows carry no signal for cost caps
		}
		const mtok = 1_000_000
		m[model] = Price{
			InputPerMTok:        e.InputCostPerToken * mtok,
			OutputPerMTok:       e.OutputCostPerToken * mtok,
			CacheReadPerMTok:    e.CacheReadPerToken * mtok,
			CacheWritePerMTok:   e.CacheWritePerToken * mtok,
			CacheWrite1hPerMTok: e.CacheWrite1hPerToken * mtok,
		}
	}
	if len(m) == 0 {
		return nil, fmt.Errorf("datasheet parsed to zero priced models")
	}
	return m, nil
}

// persist writes the raw sheet atomically (tmp + rename, same pattern
// as trust persistence) so a crash mid-write can't corrupt the cache.
// Failure is a warning, not an error — the cache only matters on the
// next boot, and /app/data may not exist in local dev.
func persist(path string, raw []byte) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		pluginlog.Warnf("pricing: persist mkdir %s: %v", filepath.Dir(path), err)
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		pluginlog.Warnf("pricing: persist write %s: %v", tmp, err)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		pluginlog.Warnf("pricing: persist rename %s: %v", path, err)
	}
}

// SetTableForTest replaces the in-memory table. Pass nil to clear.
// Production code MUST go through Init.
func SetTableForTest(m map[string]Price) {
	if m == nil {
		table.Store(nil)
		return
	}
	table.Store(&m)
}

// FetchNowForTest runs one synchronous fetch cycle against the given
// URL, persisting to cachePath. Exposed so tests can exercise the
// fetch/parse/persist path without the background loop.
func FetchNowForTest(url, cachePath string) {
	fetchAndSwap(context.Background(), url, cachePath)
}
