package trust

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
)

// Test fixtures: two deterministic secp256k1 keys. We derive these
// from fixed 32-byte scalars so the resulting compressed pubkeys are
// stable across runs and humans can read the hex if a test fails.
var (
	testPriv1 = mustHex("1111111111111111111111111111111111111111111111111111111111111111")
	testPriv2 = mustHex("2222222222222222222222222222222222222222222222222222222222222222")
	testPub1  = pubFromPriv(testPriv1)
	testPub2  = pubFromPriv(testPriv2)
)

func mustHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func pubFromPriv(priv []byte) string {
	p := secp256k1.PrivKeyFromBytes(priv)
	return hex.EncodeToString(p.PubKey().SerializeCompressed())
}

func validOrg(orgID, pub string) Org {
	return Org{
		OrgID:                 orgID,
		Pubkey:                pub,
		IssuerURL:             "https://hive.example.com",
		RevocationPollSeconds: 60,
	}
}

// ─── validate ─────────────────────────────────────────────────────────

func TestOrgValidate_AcceptsCanonicalEntry(t *testing.T) {
	o := validOrg("org_acme", testPub1)
	if err := o.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
}

func TestOrgValidate_AcceptsHexPrefix(t *testing.T) {
	o := validOrg("org_acme", "0x"+testPub1)
	if err := o.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if o.Pubkey != testPub1 {
		t.Fatalf("pubkey not canonicalised: got %s want %s", o.Pubkey, testPub1)
	}
}

func TestOrgValidate_RejectsBadPubkey(t *testing.T) {
	cases := map[string]string{
		"empty":      "",
		"not-hex":    "nothex",
		"wrong-len":  "abcd",
		"uncompr-65": "04" + testPub1 + testPub1[2:], // 65 bytes uncompressed-ish
		// 33 bytes, valid prefix, but x-coord is all zeros — not on
		// the secp256k1 curve, so ParsePubKey must reject it.
		"not-on-curve": "02" + strings.Repeat("00", 32),
	}
	for name, pk := range cases {
		t.Run(name, func(t *testing.T) {
			o := validOrg("org_acme", pk)
			if err := o.validate(); err == nil {
				t.Fatalf("expected error for %s", name)
			}
		})
	}
}

func TestOrgValidate_DefaultsPollInterval(t *testing.T) {
	o := validOrg("org_acme", testPub1)
	o.RevocationPollSeconds = 0
	if err := o.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if o.RevocationPollSeconds != 60 {
		t.Fatalf("poll not defaulted: got %d", o.RevocationPollSeconds)
	}
}

func TestOrgValidate_RejectsBadOrgID(t *testing.T) {
	for _, id := range []string{"", "  ", "has space", "has/slash"} {
		if err := (&Org{OrgID: id, Pubkey: testPub1}).validate(); err == nil {
			t.Fatalf("expected error for org_id=%q", id)
		}
	}
}

// ─── Registry mutators ────────────────────────────────────────────────

func TestRegistry_UpsertGetDelete(t *testing.T) {
	r := New()
	if err := r.Upsert(validOrg("org_acme", testPub1), SeedSourceAPI); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	got, ok := r.Get("org_acme")
	if !ok || got.Pubkey != testPub1 {
		t.Fatalf("get: ok=%v pubkey=%s", ok, got.Pubkey)
	}
	if err := r.Delete("org_acme"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, ok := r.Get("org_acme"); ok {
		t.Fatal("get after delete should fail")
	}
}

func TestRegistry_DeleteUnknownReturnsErrNotFound(t *testing.T) {
	r := New()
	if err := r.Delete("org_missing"); err != ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

func TestRegistry_UpsertIsIdempotent(t *testing.T) {
	r := newWithTempPath(t)
	o := validOrg("org_acme", testPub1)

	if err := r.Upsert(o, SeedSourceAPI); err != nil {
		t.Fatalf("upsert 1: %v", err)
	}
	first := r.Status().LastModified

	// Re-insert byte-equal entry → LastModified must not bump.
	if err := r.Upsert(o, SeedSourceAPI); err != nil {
		t.Fatalf("upsert 2: %v", err)
	}
	if r.Status().LastModified != first {
		t.Fatalf("idempotent upsert bumped last_modified: %s → %s", first, r.Status().LastModified)
	}
}

func TestRegistry_Rotate_PromotesOldKey(t *testing.T) {
	r := newWithTempPath(t)
	if err := r.Upsert(validOrg("org_acme", testPub1), SeedSourceAPI); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	resp, err := r.Rotate("org_acme", testPub2, 3600)
	if err != nil {
		t.Fatalf("rotate: %v", err)
	}
	if resp.ActivePubkey != testPub2 {
		t.Fatalf("active key not updated: %s", resp.ActivePubkey)
	}
	if len(resp.GracePubkeys) != 1 || resp.GracePubkeys[0] != testPub1 {
		t.Fatalf("grace pubkeys: %v", resp.GracePubkeys)
	}
	if resp.GraceUntil == "" {
		t.Fatal("grace_until should be set")
	}
}

func TestRegistry_Rotate_ZeroGraceDropsHistory(t *testing.T) {
	r := newWithTempPath(t)
	_ = r.Upsert(validOrg("org_acme", testPub1), SeedSourceAPI)
	resp, err := r.Rotate("org_acme", testPub2, 0)
	if err != nil {
		t.Fatalf("rotate: %v", err)
	}
	if len(resp.GracePubkeys) != 0 || resp.GraceUntil != "" {
		t.Fatalf("zero-grace should drop history: %+v", resp)
	}
}

func TestRegistry_Rotate_NotFound(t *testing.T) {
	r := newWithTempPath(t)
	if _, err := r.Rotate("nope", testPub2, 60); err != ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

// ─── persistence ──────────────────────────────────────────────────────

func TestRegistry_PersistsAtomically(t *testing.T) {
	r := newWithTempPath(t)
	if err := r.Upsert(validOrg("org_acme", testPub1), SeedSourceAPI); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	raw, err := os.ReadFile(r.path)
	if err != nil {
		t.Fatalf("read trust file: %v", err)
	}
	var f File
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("parse trust file: %v", err)
	}
	if f.Version != SchemaVersion {
		t.Fatalf("version: %d", f.Version)
	}
	if len(f.Orgs) != 1 || f.Orgs[0].OrgID != "org_acme" {
		t.Fatalf("orgs: %+v", f.Orgs)
	}
}

func TestLoadFromEnv_FreshDisk_NoSeed(t *testing.T) {
	dir := t.TempDir()
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      filepath.Join(dir, "trust.json"),
		"BIFROST_PLUGIN_TRUST":           "",
		"BIFROST_PLUGIN_TRUST_FILE":      "",
		"BIFROST_PLUGIN_TRUST_RECONCILE": "",
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.Status().OrgCount != 0 {
		t.Fatalf("expected empty registry, got %d", r.Status().OrgCount)
	}
}

func TestLoadFromEnv_SeedsWhenEmpty(t *testing.T) {
	dir := t.TempDir()
	seed := mustJSON(Seed{Orgs: []Org{validOrg("org_acme", testPub1)}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      filepath.Join(dir, "trust.json"),
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "",
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.Status().OrgCount != 1 {
		t.Fatalf("expected 1 org, got %d", r.Status().OrgCount)
	}
	if r.Status().SeedSource != SeedSourceEnv {
		t.Fatalf("seed_source: %s", r.Status().SeedSource)
	}
	// Confirm the seeded state was persisted to disk.
	if _, err := os.Stat(filepath.Join(dir, "trust.json")); err != nil {
		t.Fatalf("seed should persist to disk: %v", err)
	}
}

func TestLoadFromEnv_IgnoreModeKeepsPersisted(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")

	// Pre-populate disk with org A.
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		Orgs:         []Org{validOrg("org_A", testPub1)},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))

	// Env says org B — divergent.
	seed := mustJSON(Seed{Orgs: []Org{validOrg("org_B", testPub2)}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      path,
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "ignore",
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.Status().OrgCount != 1 || r.Status().Orgs[0] != "org_A" {
		t.Fatalf("ignore mode should keep persisted: %+v", r.Status())
	}
}

func TestLoadFromEnv_OverwriteModeReplaces(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		Orgs:         []Org{validOrg("org_A", testPub1)},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))

	seed := mustJSON(Seed{Orgs: []Org{validOrg("org_B", testPub2)}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      path,
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "overwrite",
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.Status().OrgCount != 1 || r.Status().Orgs[0] != "org_B" {
		t.Fatalf("overwrite mode should replace: %+v", r.Status())
	}
}

func TestLoadFromEnv_RefuseModeErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		Orgs:         []Org{validOrg("org_A", testPub1)},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))

	seed := mustJSON(Seed{Orgs: []Org{validOrg("org_B", testPub2)}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      path,
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "refuse",
	})
	if _, err := LoadFromEnv(); err != ErrReconcileRefused {
		t.Fatalf("want ErrReconcileRefused, got %v", err)
	}
}

func TestLoadFromEnv_MatchingSeed_NoChange(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	org := validOrg("org_acme", testPub1)
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		Orgs:         []Org{org},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))

	seed := mustJSON(Seed{Orgs: []Org{org}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      path,
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "refuse", // would fail if mismatch detected
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.Status().OrgCount != 1 {
		t.Fatalf("matching seed should be a no-op: %+v", r.Status())
	}
}

func TestLoadFromEnv_BadPersistedFile_Fatal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	must(t, os.WriteFile(path, []byte("{not json"), 0o644))
	withEnv(t, map[string]string{"BIFROST_PLUGIN_TRUST_PATH": path})
	if _, err := LoadFromEnv(); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestLoadFromEnv_InvalidSeedJSON_Fatal(t *testing.T) {
	dir := t.TempDir()
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH": filepath.Join(dir, "trust.json"),
		"BIFROST_PLUGIN_TRUST":      "{garbage",
	})
	if _, err := LoadFromEnv(); err == nil {
		t.Fatal("expected parse error")
	}
}

// ─── phase 11: realm_id (swarm self-identity) ────────────────────────

func TestRegistry_RealmID_DefaultEmpty(t *testing.T) {
	r := New()
	if got := r.RealmID(); got != "" {
		t.Fatalf("default realm_id should be empty, got %q", got)
	}
	if r.Status().RealmID != "" {
		t.Fatalf("default Status.RealmID should be empty, got %q", r.Status().RealmID)
	}
}

func TestRegistry_SetRealmID_RoundTrip(t *testing.T) {
	r := newWithTempPath(t)
	v, err := r.SetRealmID("w1")
	if err != nil {
		t.Fatalf("SetRealmID: %v", err)
	}
	if v != "w1" || r.RealmID() != "w1" || r.Status().RealmID != "w1" {
		t.Fatalf("round-trip: got %q / %q / %q", v, r.RealmID(), r.Status().RealmID)
	}
	// Persisted to disk.
	raw, _ := os.ReadFile(r.path)
	var f File
	must(t, json.Unmarshal(raw, &f))
	if f.RealmID != "w1" {
		t.Fatalf("persisted realm_id: %q", f.RealmID)
	}
}

func TestRegistry_SetRealmID_TrimsAndClears(t *testing.T) {
	r := newWithTempPath(t)
	if _, err := r.SetRealmID("  w2  "); err != nil {
		t.Fatalf("SetRealmID trimmable: %v", err)
	}
	if r.RealmID() != "w2" {
		t.Fatalf("expected trimmed value, got %q", r.RealmID())
	}
	// Empty / whitespace-only clears.
	if _, err := r.SetRealmID("   "); err != nil {
		t.Fatalf("SetRealmID clear: %v", err)
	}
	if r.RealmID() != "" {
		t.Fatalf("expected empty after clear, got %q", r.RealmID())
	}
}

func TestRegistry_SetRealmID_RejectsBadShape(t *testing.T) {
	r := New()
	for _, bad := range []string{"has space", "has/slash", "has\ttab", "has\nnewline"} {
		if _, err := r.SetRealmID(bad); err == nil {
			t.Fatalf("expected error for realm_id=%q", bad)
		}
	}
}

func TestRegistry_SetRealmID_Idempotent(t *testing.T) {
	r := newWithTempPath(t)
	if _, err := r.SetRealmID("w1"); err != nil {
		t.Fatalf("first set: %v", err)
	}
	first := r.Status().LastModified
	if _, err := r.SetRealmID("w1"); err != nil {
		t.Fatalf("second set: %v", err)
	}
	if r.Status().LastModified != first {
		t.Fatalf("idempotent set bumped last_modified: %s -> %s",
			first, r.Status().LastModified)
	}
}

func TestLoadFromEnv_SeedsRealmID(t *testing.T) {
	dir := t.TempDir()
	seed := mustJSON(Seed{
		RealmID: "w1",
		Orgs:    []Org{validOrg("org_acme", testPub1)},
	})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      filepath.Join(dir, "trust.json"),
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "",
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.RealmID() != "w1" {
		t.Fatalf("seed realm_id should land in registry: got %q", r.RealmID())
	}
}

func TestLoadFromEnv_PersistedRealmIDSurvives(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		RealmID:      "w-persisted",
		Orgs:         []Org{validOrg("org_acme", testPub1)},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH": path,
	})
	r, err := LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}
	if r.RealmID() != "w-persisted" {
		t.Fatalf("persisted realm_id not loaded: got %q", r.RealmID())
	}
}

func TestLoadFromEnv_RealmIDMismatchTriggersReconcile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")
	org := validOrg("org_acme", testPub1)
	pre := File{
		Version: SchemaVersion, SeedSource: SeedSourceAPI,
		LastModified: "2025-01-01T00:00:00Z",
		RealmID:      "w-old",
		Orgs:         []Org{org},
	}
	must(t, os.WriteFile(path, []byte(mustJSON(pre)), 0o644))

	// Same orgs but realm_id differs → mismatch under refuse mode.
	seed := mustJSON(Seed{RealmID: "w-new", Orgs: []Org{org}})
	withEnv(t, map[string]string{
		"BIFROST_PLUGIN_TRUST_PATH":      path,
		"BIFROST_PLUGIN_TRUST":           seed,
		"BIFROST_PLUGIN_TRUST_RECONCILE": "refuse",
	})
	if _, err := LoadFromEnv(); err != ErrReconcileRefused {
		t.Fatalf("want ErrReconcileRefused for realm_id mismatch, got %v", err)
	}
}

// ─── helpers ──────────────────────────────────────────────────────────

// newWithTempPath builds a *Registry that persists to a temp file.
// Used by mutator tests that need the disk side effect.
func newWithTempPath(t *testing.T) *Registry {
	t.Helper()
	return &Registry{
		path: filepath.Join(t.TempDir(), "trust.json"),
		orgs: map[string]Org{},
	}
}

// withEnv sets the given env vars for the test and restores them at
// cleanup. Empty values clear the variable.
func withEnv(t *testing.T, kv map[string]string) {
	t.Helper()
	for k, v := range kv {
		prev, had := os.LookupEnv(k)
		if v == "" {
			_ = os.Unsetenv(k)
		} else {
			_ = os.Setenv(k, v)
		}
		t.Cleanup(func() {
			if had {
				_ = os.Setenv(k, prev)
			} else {
				_ = os.Unsetenv(k)
			}
		})
	}
}

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
