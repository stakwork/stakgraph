package macaroon

// Wire types for the three-layer macaroon. Field names and JSON tags
// match gateway/plans/phases/phase-4-macaroon-shape.md and the
// symmetric-recursive refinement in
// gateway/plans/phases/phase-11-symmetric-recursive-authorization.md.
//
// All binary fields (pubkeys, signatures, HMACs, nonces) are
// lowercase hex strings WITHOUT 0x prefix. All timestamps are RFC
// 3339 / ISO 8601 UTC strings.

// ─── signature envelopes ──────────────────────────────────────────────

// SigAlg is the algorithm tag for a signature or pubkey.
type SigAlg string

const (
	AlgEd25519              SigAlg = "ed25519"
	AlgEcdsaSecp256k1Sha256 SigAlg = "ecdsa-secp256k1-sha256"
	AlgMultisigV1           SigAlg = "multisig-v1"
)

// Sig is the on-wire shape of an org_sig or user_sig. Multisig
// envelopes use Sigs; single-key envelopes use Sig.
//
// We unmarshal into this single shape and dispatch on Alg in the
// verifier rather than carrying a sum type — it keeps json.Unmarshal
// simple and matches what JCS round-trips.
type Sig struct {
	Alg  SigAlg   `json:"alg"`
	Sig  string   `json:"sig,omitempty"`
	Sigs []SubSig `json:"sigs,omitempty"`
}

// SubSig is one entry in a multisig-v1 envelope.
type SubSig struct {
	KeyIndex int    `json:"key_index"`
	Alg      SigAlg `json:"alg"`
	Sig      string `json:"sig"`
}

// ─── pubkeys + policies (trust-registry shape) ────────────────────────

// PubKey is a single public key with its algorithm tag.
type PubKey struct {
	Alg SigAlg `json:"alg"`
	Key string `json:"key"` // hex, no 0x prefix
}

// PolicyType is the kind of trust-registry policy attached to an org.
type PolicyType string

const (
	PolicySingle   PolicyType = "single"
	PolicyMultisig PolicyType = "multisig"
)

// Policy is the on-wire trust-registry entry for one org. Phase 4
// supports two shapes: single-key and m-of-n multisig.
//
// JSON unmarshal of either shape lands here; the verifier dispatches
// on Type. Fields not relevant to the active Type are zero.
type Policy struct {
	Type      PolicyType `json:"type"`
	Key       *PubKey    `json:"key,omitempty"`       // single
	Threshold int        `json:"threshold,omitempty"` // multisig
	Keys      []PubKey   `json:"keys,omitempty"`      // multisig
}

// ─── macaroon layers ──────────────────────────────────────────────────

// RealmBudget is a per-realm spending cap inside a Budget. Phase 11
// adds the realm-budgets map so multi-swarm deployments can pin
// exact per-swarm cumulative caps without relying on the implicit
// "max_total_usd leaks across swarms" behavior.
//
// Only MaxTotalUSD is defined today. Per-realm step / rate caps are
// an open question in phase 11 (see "Open questions" §2) and can be
// added without breaking compatibility — RealmBudget is a struct,
// not a bare number, precisely to leave that door open.
type RealmBudget struct {
	MaxTotalUSD float64 `json:"max_total_usd"`
}

// Budget is the spending envelope carried by any signed layer.
// Phase 11 renamed UserBudget→Budget so the same shape appears on
// UserAuthorization, Invocation, and AttenuationCaveats. Children
// narrow against parents using the rules in verify.go's narrow().
//
// All fields are independently optional. Zero / nil means "no cap on
// this axis" — consistent with the empty-by-default convention used
// for agent_budgets in phase 6. The verifier checks structural
// narrowing (child ≤ parent at each axis); the adapter enforces the
// cumulative caps (MaxTotalUSD, RealmBudgets[r].MaxTotalUSD) against
// Redis at request time.
//
// Pointer + omitempty so layers that don't carry a budget produce
// byte-identical wire bytes to a pre-phase-11 macaroon when present
// only on the UA — important for the "single-swarm operators don't
// pay for any of this" promise.
type Budget struct {
	MaxTotalUSD         float64                `json:"max_total_usd,omitempty"`
	MaxPerInvocationUSD float64                `json:"max_per_invocation_usd,omitempty"`
	RealmBudgets        map[string]RealmBudget `json:"realm_budgets,omitempty"`
}

// UserBudget is a deprecated alias preserved so external callers can
// continue to read the budget field by its phase-4 name during the
// phase-11 cutover. New code should use Budget directly.
//
// Deprecated: use Budget. Kept only for the cutover window.
type UserBudget = Budget

// UserAuthorization is the org-signed envelope. The verifier strips
// OrgSig, JCS-canonicalizes the rest, and verifies that canonical-
// JSON-bytes against the org policy.
//
// Phase 11 changes:
//   - Permissions wrapper is gone; agents lifted to top-level.
//   - The singular realm grant disappears; what realms a UA is
//     permitted on is encoded by Budget.RealmBudgets (opt-in,
//     multi-swarm only).
//   - Budget is the rename of UserBudget; same wire shape extended
//     with realm_budgets.
type UserAuthorization struct {
	UserID     string   `json:"user_id"`
	UserPubkey PubKey   `json:"user_pubkey"`
	Agents     []string `json:"agents"`
	Budget     *Budget  `json:"budget,omitempty"`
	IAT        string   `json:"iat"`
	Exp        string   `json:"exp"`
	Nonce      string   `json:"nonce"`
	OrgSig     Sig      `json:"org_sig"`
}

// Invocation is the user-signed envelope. Strip UserSig and
// canonicalize the rest to get the Ed25519 signing input.
//
// Phase 11: Realm (singular) is gone; the symmetric Budget block is
// new and optional — a single-swarm deployment can leave it absent
// and rely on the UA-level caps alone.
type Invocation struct {
	Agents     []string `json:"agents"`
	RunID      string   `json:"run_id"`
	MaxCostUSD float64  `json:"max_cost_usd"`
	MaxSteps   int      `json:"max_steps"`
	Budget     *Budget  `json:"budget,omitempty"`
	IAT        string   `json:"iat"`
	Exp        string   `json:"exp"`
	Nonce      string   `json:"nonce"`
	UserSig    Sig      `json:"user_sig"`
}

// AttenuationCaveats is the narrowed-scope object a parent agent
// crafts when spawning a sub-agent. The HMAC binds it to the parent's
// signature.
//
// Phase 11: Budget is the new optional block carrying realm_budgets;
// parents on multi-swarm deployments use it to authorize sub-agents
// to spend on specific swarms with locally-attenuated caps (no Hive
// round-trip required to cross realms).
type AttenuationCaveats struct {
	Agents     []string `json:"agents"`
	MaxCostUSD float64  `json:"max_cost_usd"`
	MaxSteps   int      `json:"max_steps"`
	RunID      string   `json:"run_id"`
	Budget     *Budget  `json:"budget,omitempty"`
	Exp        string   `json:"exp"`
	Nonce      string   `json:"nonce"`
}

// Attenuation is one link of the HMAC chain.
type Attenuation struct {
	Caveats AttenuationCaveats `json:"caveats"`
	HMAC    string             `json:"hmac"` // hex
}

// Macaroon is the full assembled object.
type Macaroon struct {
	V                 int               `json:"v"`
	OrgID             string            `json:"org_id"`
	UserAuthorization UserAuthorization `json:"user_authorization"`
	Invocation        Invocation        `json:"invocation"`
	Attenuations      []Attenuation     `json:"attenuations"`
}

// ─── verifier output ──────────────────────────────────────────────────

// EffectiveCaveats is the invocation's caveats narrowed by every
// attenuation in the chain. This is what the plugin enforces.
//
// Budget carries the layer's narrowed Budget block (or nil if no
// budget appears anywhere in the chain). The plugin reads
// Budget.RealmBudgets for the per-realm membership + cap check.
type EffectiveCaveats struct {
	Agents     []string
	MaxCostUSD float64
	MaxSteps   int
	Budget     *Budget
	Exp        string
}

// Claims is the verified output of Verify. Adapters stamp this on
// their plugin context for downstream hooks.
//
// UANonce and UABudget are surfaced separately from the general
// Nonces list because phase-6 cumulative-spend tracking
// (cost:ua:<UANonce>) needs them by themselves, before any
// attenuation processing. UABudget is nil when the UA carried no
// budget — adapters MUST treat nil as "no UA-level cap" rather than
// substituting defaults; absent budget is a design choice, not
// missing data.
//
// Phase 11:
//   - Realm (the singular invocation realm) is gone.
//   - PermittedRealms is the set of realm-ids the verified chain
//     authorizes spend on. Derived from EffectiveCaveats.Budget.
//     RealmBudgets — sorted, deduplicated, or nil when no
//     realm_budgets appears anywhere in the chain (single-swarm
//     deployments). The plugin's realm-membership check uses
//     EffectiveCaveats.Budget.RealmBudgets directly; PermittedRealms
//     is for logging and observability.
type Claims struct {
	OrgID            string
	UserID           string
	AgentName        string // last element of the final agents list
	RunID            string // innermost attenuation's run_id, or invocation's if none
	EffectiveCaveats EffectiveCaveats
	UANonce          string   // ua.nonce; key for cost:ua:<nonce>
	UABudget         *Budget  // nil if UA carried no budget
	PermittedRealms  []string // sorted keys of EffectiveCaveats.Budget.RealmBudgets, or nil
	Nonces           []string // [ua.nonce, inv.nonce, atts[*].caveats.nonce]
	IAT              string   // invocation.iat
}
