package trust

import (
	"errors"
	"sort"
	"sync"
	"time"
)

// ErrNotFound is returned by lookup/mutator methods when an org_id
// isn't in the registry. Admin handlers map it to 404.
var ErrNotFound = errors.New("org not found")

// Registry is the in-memory trust registry. The verifier (later
// phase) and the admin handlers both hold a pointer to the same
// *Registry and synchronise through its RWMutex.
//
// The struct is intentionally NOT a global — main.go constructs one
// in Init() and passes it down. Tests can construct their own with
// New() and exercise the full API without touching the filesystem.
type Registry struct {
	mu sync.RWMutex

	// path is the persisted-file location. Empty ⇒ in-memory only
	// (used in tests). Set via NewWithFile / NewFromEnv.
	path string

	// orgs is keyed by org_id; values are by-value (copies are
	// cheap and the map churn is rare) so callers reading via
	// Get/Status get a stable snapshot without holding the lock.
	orgs map[string]Org

	// seedSource records what originally populated the registry —
	// surfaced via /status, never used for decision-making.
	seedSource SeedSource

	// lastModified is updated on every successful mutator. Stored
	// here (not derived from time.Now on read) so /status reflects
	// the actual last write, even after a reload from disk.
	lastModified string

	// realmID is the swarm's own identity (phase 11). Empty for
	// single-swarm deployments. The hot-path adapter reads this
	// via the RealmID() accessor to drive the per-realm membership
	// check against verified macaroon claims.
	realmID string
}

// New constructs an empty in-memory registry. Useful for tests and
// for callers that want to populate the registry programmatically
// before the admin server starts.
func New() *Registry {
	return &Registry{orgs: map[string]Org{}}
}

// Get returns a copy of the org entry. The boolean is false if no
// such org exists. Returning by value keeps callers from mutating
// registry state through aliased slices.
func (r *Registry) Get(orgID string) (Org, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	o, ok := r.orgs[orgID]
	if !ok {
		return Org{}, false
	}
	// Defensive copies of slice fields — Org is otherwise scalar.
	return cloneOrg(o), true
}

// Status produces the /_plugin/trust/status response body. It walks
// the map once under the read lock, so it's O(n) in the org count
// but n is tiny (operators usually have <100 trusted orgs).
func (r *Registry) Status() StatusResponse {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.orgs))
	for id := range r.orgs {
		ids = append(ids, id)
	}
	sort.Strings(ids) // deterministic output for tests + Hive's diff logic
	return StatusResponse{
		Claimed:      len(ids) > 0,
		OrgCount:     len(ids),
		Orgs:         ids,
		SeedSource:   r.seedSource,
		LastModified: r.lastModified,
		RealmID:      r.realmID,
	}
}

// RealmID returns the swarm's own realm identity, or "" when none is
// configured (single-swarm deployment). Hot-path safe: just a
// read-locked accessor.
//
// The adapter uses this to drive the phase-11 realm-membership
// check against verified macaroon claims. Empty string is a
// meaningful value (no per-realm scoping) — callers should not
// substitute defaults.
func (r *Registry) RealmID() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.realmID
}

// SetRealmID updates the swarm's own realm identity and persists.
// Trimmed/validated via validateRealmID; passing "" clears the
// identity (back to single-swarm mode). Returns ErrInvalidRealmID
// on whitespace / slashes.
//
// Idempotent: setting the same value as currently stored skips the
// disk write (preserves LastModified).
func (r *Registry) SetRealmID(realmID string) (string, error) {
	v, err := validateRealmID(realmID)
	if err != nil {
		return "", err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.realmID == v {
		return v, nil // idempotent no-op
	}
	r.realmID = v
	r.lastModified = nowRFC3339()
	if err := r.persistLocked(); err != nil {
		return "", err
	}
	return v, nil
}

// Upsert validates `o`, inserts or replaces the entry, and persists
// to disk. If the new entry is byte-equal to the existing one, the
// disk write is skipped — the doc requires idempotency on
// (org_id, pubkey, policy) and skipping the write avoids touching
// LastModified spuriously.
//
// `source` is recorded as seedSource only when the registry was
// previously empty; subsequent upserts don't overwrite that field.
// This matches the phase-5 doc's intent: seedSource is informational
// and reflects the _initial_ source.
func (r *Registry) Upsert(o Org, source SeedSource) error {
	if err := o.validate(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if existing, ok := r.orgs[o.OrgID]; ok && orgsEquivalent(existing, o) {
		return nil // idempotent no-op
	}

	r.orgs[o.OrgID] = cloneOrg(o)
	if r.seedSource == SeedSourceUnknown {
		r.seedSource = source
	}
	r.lastModified = nowRFC3339()
	return r.persistLocked()
}

// Delete removes an org. Returns ErrNotFound if the org_id was not
// present (so the admin handler can map to 404).
func (r *Registry) Delete(orgID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.orgs[orgID]; !ok {
		return ErrNotFound
	}
	delete(r.orgs, orgID)
	r.lastModified = nowRFC3339()
	return r.persistLocked()
}

// Rotate adds `newPubkey` as the active key and moves the previous
// active key into `grace_pubkeys` for `graceSeconds`. The previous
// key is dropped from the grace list automatically once the deadline
// passes — but only on the next call that touches the org, since
// phase-5 has no background sweeper. That's acceptable because the
// verifier itself checks GraceUntil at macaroon-verify time.
func (r *Registry) Rotate(orgID, newPubkey string, graceSeconds int) (RotateResponse, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	existing, ok := r.orgs[orgID]
	if !ok {
		return RotateResponse{}, ErrNotFound
	}
	if graceSeconds < 0 {
		return RotateResponse{}, errors.New("grace_seconds must be >= 0")
	}

	// Validate the new pubkey by running it through a synthetic Org
	// — re-uses the canonicalisation logic in validate().
	probe := Org{
		OrgID:                 orgID,
		Pubkey:                newPubkey,
		IssuerURL:             existing.IssuerURL,
		RevocationPollSeconds: existing.RevocationPollSeconds,
	}
	if err := probe.validate(); err != nil {
		return RotateResponse{}, err
	}

	// Promote old active key into the grace list (deduped) and set
	// the rotation deadline.
	old := existing.Pubkey
	updated := existing
	updated.Pubkey = probe.Pubkey
	updated.GracePubkeys = appendUnique(existing.GracePubkeys, old)
	if graceSeconds > 0 {
		updated.GraceUntil = time.Now().UTC().Add(time.Duration(graceSeconds) * time.Second).Format(time.RFC3339)
	} else {
		updated.GraceUntil = ""
		updated.GracePubkeys = nil // immediate cut-over: drop history
	}

	r.orgs[orgID] = updated
	r.lastModified = nowRFC3339()
	if err := r.persistLocked(); err != nil {
		return RotateResponse{}, err
	}
	return RotateResponse{
		OK:           true,
		ActivePubkey: updated.Pubkey,
		GraceUntil:   updated.GraceUntil,
		GracePubkeys: append([]string(nil), updated.GracePubkeys...),
	}, nil
}

// snapshot returns a deep copy of the registry as a File suitable for
// persistence. Caller must hold r.mu (any mode). Kept package-private
// so the only on-disk path is through persistLocked.
func (r *Registry) snapshot() File {
	orgs := make([]Org, 0, len(r.orgs))
	for _, o := range r.orgs {
		orgs = append(orgs, cloneOrg(o))
	}
	// Stable order on disk → diffs in version control / debugging
	// are readable.
	sort.Slice(orgs, func(i, j int) bool { return orgs[i].OrgID < orgs[j].OrgID })
	return File{
		Version:      SchemaVersion,
		SeedSource:   r.seedSource,
		LastModified: r.lastModified,
		RealmID:      r.realmID,
		Orgs:         orgs,
	}
}

// orgsEquivalent compares two Orgs for the idempotency check on
// Upsert. We compare the fields a caller can sensibly supply via
// POST — pubkey, issuer_url, revocation_poll_seconds — and IGNORE
// grace fields, which are managed by Rotate.
func orgsEquivalent(a, b Org) bool {
	return a.OrgID == b.OrgID &&
		a.Pubkey == b.Pubkey &&
		a.IssuerURL == b.IssuerURL &&
		a.RevocationPollSeconds == b.RevocationPollSeconds
}

func cloneOrg(o Org) Org {
	out := o
	if o.GracePubkeys != nil {
		out.GracePubkeys = append([]string(nil), o.GracePubkeys...)
	}
	return out
}

func appendUnique(xs []string, x string) []string {
	for _, v := range xs {
		if v == x {
			return xs
		}
	}
	return append(xs, x)
}
