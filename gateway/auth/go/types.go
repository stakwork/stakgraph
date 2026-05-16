package macaroon

// Wire types for the three-layer macaroon. Field names and JSON tags
// match gateway/plans/phases/phase-4-macaroon-shape.md exactly.
//
// All binary fields (pubkeys, signatures, HMACs, nonces) are
// lowercase hex strings WITHOUT 0x prefix. All timestamps are RFC
// 3339 / ISO 8601 UTC strings.

// ─── signature envelopes ──────────────────────────────────────────────

// SigAlg is the algorithm tag for a signature or pubkey.
type SigAlg string

const (
	AlgEd25519             SigAlg = "ed25519"
	AlgEcdsaSecp256k1Sha256 SigAlg = "ecdsa-secp256k1-sha256"
	AlgMultisigV1          SigAlg = "multisig-v1"
)

// Sig is the on-wire shape of an org_sig or user_sig. Multisig
// envelopes use Sigs; single-key envelopes use Sig.
//
// We unmarshal into this single shape and dispatch on Alg in the
// verifier rather than carrying a sum type — it keeps json.Unmarshal
// simple and matches what JCS round-trips.
type Sig struct {
	Alg  SigAlg `json:"alg"`
	Sig  string `json:"sig,omitempty"`
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

// UserPermissions is what the org grants the user. Verifier enforces
// invocation.workspace ∈ workspaces and inv.agents ⊆ agents.
type UserPermissions struct {
	Workspaces []string `json:"workspaces"`
	Agents     []string `json:"agents"`
}

// UserAuthorization is the org-signed envelope. The verifier strips
// OrgSig, JCS-canonicalizes the rest, and verifies that canonical-
// JSON-bytes against the org policy.
type UserAuthorization struct {
	UserID      string          `json:"user_id"`
	UserPubkey  PubKey          `json:"user_pubkey"`
	Permissions UserPermissions `json:"permissions"`
	IAT         string          `json:"iat"`
	Exp         string          `json:"exp"`
	Nonce       string          `json:"nonce"`
	OrgSig      Sig             `json:"org_sig"`
}

// Invocation is the user-signed envelope. Strip UserSig and
// canonicalize the rest to get the Ed25519 signing input.
type Invocation struct {
	Workspace   string   `json:"workspace"`
	Agents      []string `json:"agents"`
	RunID       string   `json:"run_id"`
	MaxCostUSD  float64  `json:"max_cost_usd"`
	MaxSteps    int      `json:"max_steps"`
	IAT         string   `json:"iat"`
	Exp         string   `json:"exp"`
	Nonce       string   `json:"nonce"`
	UserSig     Sig      `json:"user_sig"`
}

// AttenuationCaveats is the narrowed-scope object a parent agent
// crafts when spawning a sub-agent. The HMAC binds it to the parent's
// signature.
type AttenuationCaveats struct {
	Agents     []string `json:"agents"`
	MaxCostUSD float64  `json:"max_cost_usd"`
	MaxSteps   int      `json:"max_steps"`
	RunID      string   `json:"run_id"`
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
type EffectiveCaveats struct {
	Agents     []string
	MaxCostUSD float64
	MaxSteps   int
	Exp        string
}

// Claims is the verified output of Verify. Adapters stamp this on
// their plugin context for downstream hooks.
type Claims struct {
	OrgID            string
	UserID           string
	Workspace        string
	AgentName        string // last element of the final agents list
	RunID            string // innermost attenuation's run_id, or invocation's if none
	EffectiveCaveats EffectiveCaveats
	Nonces           []string // [ua.nonce, inv.nonce, atts[*].caveats.nonce]
	IAT              string   // invocation.iat
}
