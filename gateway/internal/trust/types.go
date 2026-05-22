// Package trust holds the gateway plugin's trust registry — the set
// of orgs whose macaroons the plugin will verify. See
// gateway/plans/phases/phase-5-trust-registry.md for the design.
//
// File layout
// -----------
//   - types.go      : on-disk + wire schema (this file)
//   - registry.go   : in-memory state, mutators, lookup
//   - persistence.go: atomic file IO, env-seed precedence
//
// The HTTP surface lives in gateway/internal/adminapi/trust.go and
// only talks to *Registry through this package's exported API — the
// admin layer never reaches into types directly so the wire shape
// stays under our control.
package trust

import (
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
)

// SchemaVersion is the on-disk schema version. Bump on
// backwards-incompatible changes; we'll grow a migration step in
// loadPersisted when the second version lands.
const SchemaVersion = 1

// SeedSource records where the initial registry came from.
//
// "env" or "api" tells operators (via /status) whether to look at
// docker-compose env or at Hive's reconciler when reasoning about
// the current contents.
type SeedSource string

const (
	SeedSourceUnknown SeedSource = ""
	SeedSourceEnv     SeedSource = "env"
	SeedSourceAPI     SeedSource = "api"
)

// Org is the canonical trust entry. One per org_id.
//
// Wire shape matches both:
//   - the on-disk file (with `version`/`last_modified` envelope, see File)
//   - the POST /_plugin/trust request body
//   - the GET  /_plugin/trust/:org_id response body
//
// `GracePubkeys` is populated by /rotate and consumed by the verifier
// during the grace window; admin clients normally don't set it
// directly on POST, but the field is exposed so an operator can
// hand-edit the persisted file in an emergency.
//
// Intentionally NO budget / realm fields: the macaroon already
// carries those (UserAuthorization grants realms, Invocation
// caveats carry budgets), and the trust registry's job is purely
// "do I trust this org's signatures?" — not "what limits should I
// impose on top of theirs?" If the org ever wants to cap user-
// chosen invocation budgets, that belongs in UserAuthorization
// (org-signed), not the per-swarm trust registry.
type Org struct {
	OrgID                 string   `json:"org_id"`
	Pubkey                string   `json:"pubkey"` // hex-encoded, 33-byte compressed secp256k1
	IssuerURL             string   `json:"issuer_url"`
	RevocationPollSeconds int      `json:"revocation_poll_seconds"`
	GracePubkeys          []string `json:"grace_pubkeys,omitempty"` // previous-root keys during rotation grace
	GraceUntil            string   `json:"grace_until,omitempty"`   // RFC3339; empty when no grace active
}

// File is the on-disk shape — Org list plus bookkeeping. We persist
// this verbatim with atomic write-temp-fsync-rename.
//
// RealmID is the swarm's own identity, added in phase 11. Optional
// (empty for single-swarm deployments). When set, the plugin's
// realm-membership check asserts macaroons with realm_budgets list
// this swarm's RealmID as a permitted realm. See
// gateway/plans/phases/phase-11-symmetric-recursive-authorization.md
// for the multi-swarm model.
type File struct {
	Version      int        `json:"version"`
	SeedSource   SeedSource `json:"seed_source"`
	LastModified string     `json:"last_modified"`
	RealmID      string     `json:"realm_id,omitempty"`
	Orgs         []Org      `json:"orgs"`
}

// Seed is the env-supplied JSON shape — just the org list and the
// optional swarm self-identity, no bookkeeping. Used for both
// BIFROST_PLUGIN_TRUST (inline) and BIFROST_PLUGIN_TRUST_FILE
// (path → JSON).
type Seed struct {
	RealmID string `json:"realm_id,omitempty"`
	Orgs    []Org  `json:"orgs"`
}

// StatusResponse is the body of GET /_plugin/trust/status. Defined
// here (not in adminapi) so the admin layer is a thin shell — the
// Registry produces this struct and the handler just JSON-encodes
// it.
//
// RealmID is included so operators and the dashboard's Provenance
// card can show the swarm's own identity. omitempty hides the field
// from single-swarm deployments that never set it.
type StatusResponse struct {
	Claimed      bool       `json:"claimed"`
	OrgCount     int        `json:"org_count"`
	Orgs         []string   `json:"orgs"`
	SeedSource   SeedSource `json:"seed_source"`
	LastModified string     `json:"last_modified"`
	RealmID      string     `json:"realm_id,omitempty"`
}

// RealmIDRequest is the body of PUT /_plugin/trust/realm_id.
// Sending an empty string clears the swarm's self-identity (back
// to single-swarm mode).
type RealmIDRequest struct {
	RealmID string `json:"realm_id"`
}

// RealmIDResponse mirrors the other admin handlers' shape.
type RealmIDResponse struct {
	OK      bool   `json:"ok"`
	RealmID string `json:"realm_id"`
}

// RotateRequest is the body of POST /_plugin/trust/:org_id/rotate.
type RotateRequest struct {
	NewPubkey    string `json:"new_pubkey"`
	GraceSeconds int    `json:"grace_seconds"`
}

// RotateResponse mirrors the phase-5 doc's rotate example.
type RotateResponse struct {
	OK           bool     `json:"ok"`
	ActivePubkey string   `json:"active_pubkey"`
	GraceUntil   string   `json:"grace_until"`
	GracePubkeys []string `json:"grace_pubkeys"`
}

// ─── validation ───────────────────────────────────────────────────────

// ErrInvalidOrg is returned by validate() and surfaced as 400 by the
// admin handlers. Concrete sub-errors are wrapped for context.
var ErrInvalidOrg = errors.New("invalid org entry")

// validate normalises and sanity-checks an Org. It mutates the
// receiver in place (lowercases the pubkey, defaults the poll
// interval) so the persisted shape is canonical regardless of how a
// caller formatted the input.
//
// Validation rules — match the phase-5 doc's "Failure modes" 400 row:
//   - org_id non-empty, no whitespace
//   - pubkey is a valid 33-byte compressed secp256k1 hex string
//   - issuer_url parses as an absolute URL (when non-empty)
//   - revocation_poll_seconds >= 0; defaulted to 60 if zero
func (o *Org) validate() error {
	o.OrgID = strings.TrimSpace(o.OrgID)
	if o.OrgID == "" {
		return fmt.Errorf("%w: org_id is required", ErrInvalidOrg)
	}
	if strings.ContainsAny(o.OrgID, " \t\r\n/") {
		return fmt.Errorf("%w: org_id contains whitespace or slash", ErrInvalidOrg)
	}

	pk, err := decodePubkey(o.Pubkey)
	if err != nil {
		return fmt.Errorf("%w: pubkey: %v", ErrInvalidOrg, err)
	}
	o.Pubkey = hex.EncodeToString(pk) // canonicalise to lowercase hex, no 0x prefix

	if o.IssuerURL != "" {
		u, err := url.Parse(o.IssuerURL)
		if err != nil || !u.IsAbs() {
			return fmt.Errorf("%w: issuer_url must be an absolute URL", ErrInvalidOrg)
		}
	}

	if o.RevocationPollSeconds < 0 {
		return fmt.Errorf("%w: revocation_poll_seconds must be >= 0", ErrInvalidOrg)
	}
	if o.RevocationPollSeconds == 0 {
		o.RevocationPollSeconds = 60
	}

	// Normalise grace pubkeys too — same encoding rules.
	for i, gp := range o.GracePubkeys {
		pk, err := decodePubkey(gp)
		if err != nil {
			return fmt.Errorf("%w: grace_pubkeys[%d]: %v", ErrInvalidOrg, i, err)
		}
		o.GracePubkeys[i] = hex.EncodeToString(pk)
	}

	return nil
}

// decodePubkey accepts either "0x"-prefixed or bare hex (case-
// insensitive) and returns the raw 33-byte compressed secp256k1
// pubkey. We require compression (33 bytes, leading 0x02/0x03) so
// the on-disk form is unambiguous — uncompressed keys are rejected.
func decodePubkey(s string) ([]byte, error) {
	s = strings.TrimSpace(strings.TrimPrefix(strings.ToLower(s), "0x"))
	if s == "" {
		return nil, errors.New("empty")
	}
	raw, err := hex.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("not hex: %w", err)
	}
	if len(raw) != 33 {
		return nil, fmt.Errorf("must be 33-byte compressed secp256k1, got %d bytes", len(raw))
	}
	// Verify the bytes actually parse as a curve point — guards
	// against operators pasting random hex of the right length.
	if _, err := secp256k1.ParsePubKey(raw); err != nil {
		return nil, fmt.Errorf("not a valid secp256k1 point: %w", err)
	}
	return raw, nil
}

// nowRFC3339 is the single timestamp formatter used for
// LastModified / GraceUntil. Centralised so changing the precision
// (e.g. to RFC3339Nano) is a one-line edit.
func nowRFC3339() string { return time.Now().UTC().Format(time.RFC3339) }

// ErrInvalidRealmID is returned by validateRealmID when the supplied
// realm-id has whitespace, slashes, or other characters that would
// confuse downstream consumers (e.g. Redis key suffixes, URLs).
var ErrInvalidRealmID = errors.New("invalid realm_id")

// validateRealmID enforces the same id-shape rules as Org.OrgID:
// non-empty when set, no whitespace, no slashes. Empty string is
// allowed and means "single-swarm deployment, no self-identity".
//
// Returns the trimmed value. The Registry uses the returned string
// rather than the input so an operator sending " w1 " ends up with
// "w1" on disk.
func validateRealmID(s string) (string, error) {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		// Empty is a valid state: it clears the swarm's identity.
		// Surface it as "" so callers don't need to nil-check.
		return "", nil
	}
	if strings.ContainsAny(trimmed, " \t\r\n/") {
		return "", fmt.Errorf("%w: contains whitespace or slash", ErrInvalidRealmID)
	}
	return trimmed, nil
}
