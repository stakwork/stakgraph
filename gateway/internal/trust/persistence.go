package trust

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Reconcile modes — keep these as typed constants so the validate
// function and the LoadFromEnv switch don't drift from each other.
const (
	ReconcileIgnore    = "ignore"
	ReconcileOverwrite = "overwrite"
	ReconcileRefuse    = "refuse"
)

// ErrReconcileRefused is returned when the configured reconcile mode
// is "refuse" and persisted state diverges from the env seed. main()
// is expected to log this and exit non-zero — exactly what an
// operator wants when the spec is meant to be source of truth.
var ErrReconcileRefused = errors.New("trust: persisted state diverges from env seed (BIFROST_PLUGIN_TRUST_RECONCILE=refuse)")

// LoadFromEnv is the single startup entry point. It applies the
// precedence rules from phase-5-trust-registry.md ("Precedence:
// persisted is canonical, env seeds only when empty").
//
// Returns a ready-to-use *Registry. The only failure modes that
// return a non-nil error are:
//
//  1. The persisted file exists but is unparseable. Per the doc the
//     plugin must exit non-zero rather than silently start with an
//     empty registry — main() handles the exit.
//  2. The env seed is set but invalid JSON / fails validation.
//  3. Reconcile mode is "refuse" and the two sources disagree.
//
// Everything else (file missing, env unset, modes "ignore" /
// "overwrite") yields a working Registry.
func LoadFromEnv() (*Registry, error) {
	path := env.TrustPathValue()
	r := &Registry{path: path, orgs: map[string]Org{}}

	persisted, hadPersisted, err := loadPersisted(path)
	if err != nil {
		return nil, err
	}
	if hadPersisted {
		r.absorbFile(persisted)
	}

	seedRaw, inline, hasSeed := env.TrustSeed()
	if !hasSeed {
		return r, nil
	}

	seed, err := parseSeed(seedRaw, inline)
	if err != nil {
		return nil, fmt.Errorf("trust: parse env seed: %w", err)
	}

	// Validate every org in the seed up front. Otherwise a bad
	// entry only surfaces when we try to write it, which is too
	// late for the "refuse" mode.
	for i := range seed.Orgs {
		if err := seed.Orgs[i].validate(); err != nil {
			return nil, fmt.Errorf("trust: env seed orgs[%d]: %w", i, err)
		}
	}
	// Validate the seed's optional realm_id the same way. Empty is
	// fine; whitespace / slashes are rejected up front so a
	// "refuse" deployment fails fast rather than at first request.
	seedRealmID, err := validateRealmID(seed.RealmID)
	if err != nil {
		return nil, fmt.Errorf("trust: env seed realm_id: %w", err)
	}
	seed.RealmID = seedRealmID

	if !hadPersisted || len(r.orgs) == 0 {
		// Empty persisted state → seed from env (write to disk).
		r.seedSource = SeedSourceEnv
		r.lastModified = nowRFC3339()
		r.realmID = seed.RealmID
		for _, o := range seed.Orgs {
			r.orgs[o.OrgID] = cloneOrg(o)
		}
		if err := r.persistLocked(); err != nil {
			return nil, err
		}
		pluginlog.Logf("trust: seeded from env (%d orgs, realm_id=%q)",
			len(seed.Orgs), seed.RealmID)
		return r, nil
	}

	// Both persisted and seed present — compare and apply the
	// configured reconcile mode.
	if seedMatchesRegistry(seed, r) {
		pluginlog.Logf("trust: env seed matches persisted (%d orgs)", len(r.orgs))
		return r, nil
	}

	mode := env.TrustReconcileValue()
	switch mode {
	case ReconcileIgnore, "": // empty falls back here, matches doc default
		pluginlog.Warnf("trust: env seed differs from persisted; using persisted (set BIFROST_PLUGIN_TRUST_RECONCILE=overwrite or =refuse to change)")
		return r, nil
	case ReconcileOverwrite:
		pluginlog.Warnf("trust: env seed overwrites persisted (BIFROST_PLUGIN_TRUST_RECONCILE=overwrite)")
		r.orgs = map[string]Org{}
		for _, o := range seed.Orgs {
			r.orgs[o.OrgID] = cloneOrg(o)
		}
		r.seedSource = SeedSourceEnv
		r.lastModified = nowRFC3339()
		r.realmID = seed.RealmID
		if err := r.persistLocked(); err != nil {
			return nil, err
		}
		return r, nil
	case ReconcileRefuse:
		return nil, ErrReconcileRefused
	default:
		return nil, fmt.Errorf("trust: unknown BIFROST_PLUGIN_TRUST_RECONCILE=%q (want %q|%q|%q)",
			mode, ReconcileIgnore, ReconcileOverwrite, ReconcileRefuse)
	}
}

// absorbFile copies File contents into r without persisting — caller
// already holds the on-disk state. Assumes the caller is in
// LoadFromEnv (i.e. no concurrent readers yet) so no lock is taken.
func (r *Registry) absorbFile(f File) {
	for _, o := range f.Orgs {
		r.orgs[o.OrgID] = cloneOrg(o)
	}
	r.seedSource = f.SeedSource
	r.lastModified = f.LastModified
	r.realmID = f.RealmID
}

// loadPersisted reads the on-disk trust file. Returns (file, true, nil)
// on success, (zero, false, nil) when the file doesn't exist, and
// (zero, false, err) when the file exists but is unreadable or
// unparseable. The latter is intentionally fatal — see LoadFromEnv.
func loadPersisted(path string) (File, bool, error) {
	if path == "" {
		return File{}, false, nil
	}
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return File{}, false, nil
	}
	if err != nil {
		return File{}, false, fmt.Errorf("trust: read %s: %w", path, err)
	}
	var f File
	if err := json.Unmarshal(raw, &f); err != nil {
		return File{}, false, fmt.Errorf("trust: parse %s: %w (refusing to start with inconsistent registry; fix or remove the file)", path, err)
	}
	if f.Version != SchemaVersion {
		// We only have version 1 today; reject anything else
		// rather than guess. Future migrations branch here.
		return File{}, false, fmt.Errorf("trust: %s has version=%d, this build supports %d", path, f.Version, SchemaVersion)
	}
	// Defensive: ensure all persisted entries round-trip through
	// validate(). Catches a hand-edited file with bad pubkeys.
	for i := range f.Orgs {
		if err := f.Orgs[i].validate(); err != nil {
			return File{}, false, fmt.Errorf("trust: %s orgs[%d]: %w", path, i, err)
		}
	}
	return f, true, nil
}

// parseSeed handles both inline JSON (BIFROST_PLUGIN_TRUST) and
// file-path (BIFROST_PLUGIN_TRUST_FILE). The bool comes straight from
// env.TrustSeed.
func parseSeed(value string, inline bool) (Seed, error) {
	var raw []byte
	if inline {
		raw = []byte(value)
	} else {
		b, err := os.ReadFile(value)
		if err != nil {
			return Seed{}, fmt.Errorf("read %s: %w", value, err)
		}
		raw = b
	}
	var s Seed
	if err := json.Unmarshal(raw, &s); err != nil {
		return Seed{}, fmt.Errorf("unmarshal: %w", err)
	}
	return s, nil
}

// seedMatchesRegistry returns true iff the env seed describes the
// same orgs (by OrgID + the "user-supplied" fields — same set as
// orgsEquivalent) as the registry. Order-insensitive.
//
// Grace state isn't compared: an operator who rotated keys via the
// API will have GracePubkeys/GraceUntil populated, but the env seed
// won't, and that mismatch should NOT trigger "overwrite" / "refuse"
// behaviour. Grace state is a runtime concern.
func seedMatchesRegistry(seed Seed, r *Registry) bool {
	if len(seed.Orgs) != len(r.orgs) {
		return false
	}
	if seed.RealmID != r.realmID {
		return false
	}
	for _, o := range seed.Orgs {
		existing, ok := r.orgs[o.OrgID]
		if !ok {
			return false
		}
		// Normalise the seed entry through validate so we compare
		// canonicalised forms (lowercase hex pubkeys, etc.). Ignore
		// errors here — they'd have been caught earlier in
		// LoadFromEnv.
		probe := o
		_ = probe.validate()
		if !orgsEquivalent(existing, probe) {
			return false
		}
	}
	return true
}

// persistLocked writes the registry to disk atomically. Caller must
// hold r.mu (write mode). Skips the write when r.path is empty — used
// by tests that don't want filesystem side effects.
//
// Atomicity: write to <path>.tmp, fsync, rename. Rename on POSIX is
// atomic within a filesystem, so readers either see the previous full
// content or the new full content — never a partial.
func (r *Registry) persistLocked() error {
	if r.path == "" {
		return nil
	}
	snap := r.snapshot()
	raw, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return fmt.Errorf("trust: marshal: %w", err)
	}

	dir := filepath.Dir(r.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("trust: mkdir %s: %w", dir, err)
	}

	tmp, err := os.CreateTemp(dir, ".trust.json.*")
	if err != nil {
		return fmt.Errorf("trust: tempfile in %s: %w", dir, err)
	}
	tmpPath := tmp.Name()
	// On any error path below we must remove the temp file.
	defer func() {
		// If the rename succeeded the file no longer exists at
		// tmpPath; ignore the resulting ENOENT.
		_ = os.Remove(tmpPath)
	}()

	if _, err := tmp.Write(raw); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("trust: write tempfile: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("trust: fsync tempfile: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("trust: close tempfile: %w", err)
	}
	if err := os.Rename(tmpPath, r.path); err != nil {
		return fmt.Errorf("trust: rename %s -> %s: %w", tmpPath, r.path, err)
	}
	return nil
}

// FileEqual is a test helper exposed for the test suite — compares
// two File values ignoring LastModified (which is wall-clock and
// makes assertions painful). Kept in production code so tests in
// other packages can import it; cheaper than a build tag.
//
// Not for use on the hot path.
func FileEqual(a, b File) bool {
	a.LastModified, b.LastModified = "", ""
	sort.Slice(a.Orgs, func(i, j int) bool { return a.Orgs[i].OrgID < a.Orgs[j].OrgID })
	sort.Slice(b.Orgs, func(i, j int) bool { return b.Orgs[i].OrgID < b.Orgs[j].OrgID })
	return reflect.DeepEqual(a, b)
}
