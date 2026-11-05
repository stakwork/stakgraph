package auth

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// The runtime price table: rows pushed through PUT /_plugin/pricing
// while the gateway runs, for models neither the config block's
// model_pricing nor bifrost's datasheet knows. Custom providers are
// the case: a Maple, Numes or other OpenAI-compatible provider added
// through bifrost's /api/providers has no datasheet row, and
// model_pricing is baked into the image's config.json, so without
// this table their calls accumulate $0 and no cap bites.
//
// Rows have the shape of model_pricing rows (ModelPrice) and are
// keyed the same way (pricing.Keys): "maple/gemma4-31b" or the bare
// model name. A PUT replaces the whole table: the pusher owns the
// list and sends all of it each time, so a model it drops loses its
// row. A row of zeros is a real price: the model is free, and its
// calls stop logging as unpriced.
//
// The table is one JSON file on the data volume (PricingPath), the
// same volume as trust.json and logs.db, written before the
// in-memory swap so what PriceCall reads is always what the next
// boot loads. The hot path is an atomic pointer read, like the
// catalog's.

// ErrInvalidPrice marks a table the gateway refuses: an empty or
// whitespace-bearing model key, or a negative rate.
var ErrInvalidPrice = errors.New("invalid price table")

var (
	runtimeTable atomic.Pointer[map[string]ModelPrice]

	// runtimeMu serializes writers so the file on disk and the
	// in-memory table always hold the same, last-written table.
	runtimeMu   sync.Mutex
	runtimePath string
)

// LoadRuntimePricing sets where the table lives and loads it. Called
// once from plugin Init. Never fatal: a missing file is a first boot,
// and an unreadable or invalid one logs an ERROR and starts empty
// (the next PUT replaces it) — dropping the plugin over a price table
// would serve inference with no governance at all.
func LoadRuntimePricing(path string) {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	runtimePath = path
	empty := map[string]ModelPrice{}
	runtimeTable.Store(&empty)

	raw, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		pluginlog.Logf("pricing: no runtime price table at %s — PUT /_plugin/pricing to add rows", path)
		return
	case err != nil:
		pluginlog.Errf("pricing: read runtime price table %s: %v — starting empty", path, err)
		return
	}
	var rows map[string]ModelPrice
	if err := json.Unmarshal(raw, &rows); err != nil {
		pluginlog.Errf("pricing: runtime price table %s unparseable (%v) — starting empty", path, err)
		return
	}
	if err := validateRuntimePricing(rows); err != nil {
		pluginlog.Errf("pricing: runtime price table %s: %v — starting empty", path, err)
		return
	}
	rows = cloneRows(rows)
	runtimeTable.Store(&rows)
	pluginlog.Logf("pricing: loaded %d runtime price rows from %s", len(rows), path)
}

// SetRuntimePricing replaces the table: validates it, writes it to
// disk, then swaps it in. On any error the old table stays, on disk
// and in memory. Validation failures wrap ErrInvalidPrice.
func SetRuntimePricing(rows map[string]ModelPrice) error {
	if rows == nil {
		return fmt.Errorf("%w: the table must be a JSON object", ErrInvalidPrice)
	}
	if err := validateRuntimePricing(rows); err != nil {
		return err
	}
	rows = cloneRows(rows)
	raw, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return err
	}

	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if runtimePath == "" {
		return errors.New("runtime price table not loaded")
	}
	if err := writeFileAtomic(runtimePath, raw); err != nil {
		return fmt.Errorf("persist runtime price table: %w", err)
	}
	runtimeTable.Store(&rows)
	return nil
}

// RuntimePricing returns a copy of the table, empty (never nil)
// before Init or after a PUT of {}.
func RuntimePricing() map[string]ModelPrice {
	m := runtimeTable.Load()
	if m == nil {
		return map[string]ModelPrice{}
	}
	return cloneRows(*m)
}

func runtimePrice(keys []string) (ModelPrice, bool) {
	m := runtimeTable.Load()
	if m == nil || len(*m) == 0 {
		return ModelPrice{}, false
	}
	for _, k := range keys {
		if entry, ok := (*m)[k]; ok {
			return entry, true
		}
	}
	return ModelPrice{}, false
}

func validateRuntimePricing(rows map[string]ModelPrice) error {
	for key, p := range rows {
		if key == "" || strings.IndexFunc(key, isSpace) >= 0 {
			return fmt.Errorf("%w: model key %q must be non-empty with no spaces", ErrInvalidPrice, key)
		}
		if p.InputPerMTok < 0 || p.OutputPerMTok < 0 || p.CacheReadPerMTok < 0 ||
			p.CacheWritePerMTok < 0 || p.CacheWrite1hPerMTok < 0 {
			return fmt.Errorf("%w: %q has a negative rate", ErrInvalidPrice, key)
		}
	}
	return nil
}

func isSpace(r rune) bool { return r == ' ' || r == '\t' || r == '\n' || r == '\r' }

func cloneRows(rows map[string]ModelPrice) map[string]ModelPrice {
	out := make(map[string]ModelPrice, len(rows))
	for k, v := range rows {
		out[k] = v
	}
	return out
}

// writeFileAtomic writes to a temp file in the same directory, syncs
// it, and renames it over path, so a crash leaves the old table or
// the new one, never half of either.
func writeFileAtomic(path string, raw []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp) // no-op once the rename has happened
	if _, err := f.Write(raw); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ResetRuntimePricingForTest points the table at path and empties it
// in memory, without reading the file. Production code MUST go
// through LoadRuntimePricing.
func ResetRuntimePricingForTest(path string) {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	runtimePath = path
	runtimeTable.Store(nil)
}
