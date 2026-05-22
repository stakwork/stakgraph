package macaroon

import (
	"crypto/hmac"
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

// Verify checks a macaroon end-to-end against a single org's policy
// and returns the verified claims.
//
// Inputs:
//   - macaroonB64: the base64url-encoded wire form, as carried in the
//     x-macaroon HTTP header.
//   - policy: the trust-registry entry for the macaroon's org.
//   - now: the verifier's current time. Caveats use absolute exp.
//
// Verify is pure: no I/O, no revocation, no Bifrost types. Adapters
// (gateway/internal/auth/) layer revocation + transport on top.
func Verify(macaroonB64 string, policy Policy, now time.Time) (*Claims, error) {
	raw, err := Base64urlToBytes(macaroonB64)
	if err != nil {
		return nil, newError(ErrMacaroonMalformed, fmt.Sprintf("base64url: %v", err))
	}
	return VerifyJSON(raw, policy, now)
}

// VerifyJSON is Verify with a pre-decoded JSON payload. Useful for
// tests and for adapters that want to peek at fields (e.g. org_id)
// before policy lookup.
func VerifyJSON(raw []byte, policy Policy, now time.Time) (*Claims, error) {
	var m Macaroon
	if err := strictUnmarshalMacaroon(raw, &m); err != nil {
		return nil, newError(ErrMacaroonMalformed, err.Error())
	}
	if m.V != 1 {
		return nil, newError(ErrMacaroonUnsupportedVersion, fmt.Sprintf("v=%d", m.V))
	}

	if err := verifyUserAuthorization(&m.UserAuthorization, policy); err != nil {
		return nil, err
	}
	if expBefore(m.UserAuthorization.Exp, now) {
		return nil, newError(ErrUserAuthorizationExpired, m.UserAuthorization.Exp)
	}

	if err := verifyInvocation(&m.Invocation, &m.UserAuthorization); err != nil {
		return nil, err
	}
	if err := enforceInvocationCaveats(&m.Invocation, &m.UserAuthorization, now); err != nil {
		return nil, err
	}

	effective, runID, attNonces, err := walkAttenuations(&m.Invocation, m.UserAuthorization.Budget, m.Attenuations, now)
	if err != nil {
		return nil, err
	}

	agentName := ""
	if n := len(effective.Agents); n > 0 {
		agentName = effective.Agents[n-1]
	}

	nonces := make([]string, 0, 2+len(attNonces))
	nonces = append(nonces, m.UserAuthorization.Nonce, m.Invocation.Nonce)
	nonces = append(nonces, attNonces...)

	return &Claims{
		OrgID:            m.OrgID,
		UserID:           m.UserAuthorization.UserID,
		AgentName:        agentName,
		RunID:            runID,
		EffectiveCaveats: effective,
		UANonce:          m.UserAuthorization.Nonce,
		UABudget:         m.UserAuthorization.Budget,
		PermittedRealms:  permittedRealms(&effective),
		Nonces:           nonces,
		IAT:              m.Invocation.IAT,
	}, nil
}

// strictUnmarshalMacaroon parses the macaroon JSON and rejects unknown
// top-level fields. Wire format v=1 has a fixed top-level shape.
//
// Two-pass: parse to a probe map to check unknown top-level fields,
// then parse to the typed struct.
func strictUnmarshalMacaroon(raw []byte, m *Macaroon) error {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(raw, &probe); err != nil {
		return fmt.Errorf("top-level: %w", err)
	}
	for k := range probe {
		switch k {
		case "v", "org_id", "user_authorization", "invocation", "attenuations":
		default:
			return fmt.Errorf("unknown top-level field: %s", k)
		}
	}
	if err := json.Unmarshal(raw, m); err != nil {
		return fmt.Errorf("typed unmarshal: %w", err)
	}
	return nil
}

// ─── org-signed user_authorization ────────────────────────────────────

func verifyUserAuthorization(ua *UserAuthorization, policy Policy) error {
	msg, err := JCSStripField(ua, "org_sig")
	if err != nil {
		return newError(ErrMacaroonMalformed, fmt.Sprintf("jcs ua: %v", err))
	}

	switch policy.Type {
	case PolicySingle:
		if policy.Key == nil {
			return newError(ErrInvalidUserAuthorization, "policy.single missing key")
		}
		if ua.OrgSig.Alg != AlgEcdsaSecp256k1Sha256 {
			return newError(ErrInvalidUserAuthorization,
				fmt.Sprintf("expected single-key ecdsa, got alg=%s", ua.OrgSig.Alg))
		}
		pub, err := HexToBytes(policy.Key.Key)
		if err != nil {
			return newError(ErrInvalidUserAuthorization, fmt.Sprintf("policy pubkey hex: %v", err))
		}
		sig, err := HexToBytes(ua.OrgSig.Sig)
		if err != nil {
			return newError(ErrInvalidUserAuthorization, fmt.Sprintf("org_sig hex: %v", err))
		}
		if !EcdsaSecp256k1Verify(pub, msg, sig) {
			return newError(ErrInvalidUserAuthorization, "ecdsa verify failed")
		}
		return nil

	case PolicyMultisig:
		if ua.OrgSig.Alg != AlgMultisigV1 {
			return newError(ErrInvalidUserAuthorization,
				fmt.Sprintf("expected multisig-v1, got alg=%s", ua.OrgSig.Alg))
		}
		validCount := 0
		seen := make(map[int]bool)
		for _, entry := range ua.OrgSig.Sigs {
			if seen[entry.KeyIndex] {
				continue // ignore duplicates
			}
			seen[entry.KeyIndex] = true
			if entry.KeyIndex < 0 || entry.KeyIndex >= len(policy.Keys) {
				continue
			}
			pk := policy.Keys[entry.KeyIndex]
			if entry.Alg != pk.Alg || entry.Alg != AlgEcdsaSecp256k1Sha256 {
				continue
			}
			pubBytes, err := HexToBytes(pk.Key)
			if err != nil {
				continue
			}
			sigBytes, err := HexToBytes(entry.Sig)
			if err != nil {
				continue
			}
			if EcdsaSecp256k1Verify(pubBytes, msg, sigBytes) {
				validCount++
			}
		}
		if validCount < policy.Threshold {
			return newError(ErrInvalidUserAuthorization,
				fmt.Sprintf("multisig threshold not met: %d/%d", validCount, policy.Threshold))
		}
		return nil

	default:
		return newError(ErrInvalidUserAuthorization,
			fmt.Sprintf("unknown policy type: %s", policy.Type))
	}
}

// ─── user-signed invocation ───────────────────────────────────────────

func verifyInvocation(inv *Invocation, ua *UserAuthorization) error {
	if inv.UserSig.Alg != AlgEd25519 || ua.UserPubkey.Alg != AlgEd25519 {
		return newError(ErrInvalidInvocationSig,
			"non-ed25519 user signature/pubkey not supported in v=1")
	}
	msg, err := JCSStripField(inv, "user_sig")
	if err != nil {
		return newError(ErrMacaroonMalformed, fmt.Sprintf("jcs inv: %v", err))
	}
	pub, err := HexToBytes(ua.UserPubkey.Key)
	if err != nil {
		return newError(ErrInvalidInvocationSig, fmt.Sprintf("user_pubkey hex: %v", err))
	}
	sig, err := HexToBytes(inv.UserSig.Sig)
	if err != nil {
		return newError(ErrInvalidInvocationSig, fmt.Sprintf("user_sig hex: %v", err))
	}
	if !Ed25519Verify(pub, msg, sig) {
		return newError(ErrInvalidInvocationSig, "ed25519 verify failed")
	}
	return nil
}

// enforceInvocationCaveats is the UA→invocation boundary check. This
// is the one boundary where `agents` narrows (`child ⊆ parent`) — the
// user grants a set of agents on the UA, and the invocation picks a
// subset to actually use this run. Subsequent attenuation boundaries
// extend lineage (`child ⊇ parent`); see narrowAttenuation.
func enforceInvocationCaveats(inv *Invocation, ua *UserAuthorization, now time.Time) error {
	if len(inv.Agents) == 0 {
		return newError(ErrInvocationViolated, "invocation.agents must be non-empty")
	}
	for _, a := range inv.Agents {
		if !containsString(ua.Agents, a) {
			return newError(ErrInvocationViolated,
				fmt.Sprintf("agent %s not in user permissions", a))
		}
	}
	if expBefore(inv.Exp, now) {
		return newError(ErrMacaroonExpired, fmt.Sprintf("invocation expired at %s", inv.Exp))
	}
	// Exp narrowing: invocation must not outlive the UA. Signature-
	// time field comparison, no clock dependency.
	if inv.Exp > ua.Exp {
		return newError(ErrInvocationViolated,
			fmt.Sprintf("invocation exp %s > ua exp %s", inv.Exp, ua.Exp))
	}
	// Per-invocation budget cap. Pure signature-time check — the
	// cumulative cap (MaxTotalUSD) is enforced by the adapter via
	// Redis in PreLLMHook, not here.
	if ua.Budget != nil && ua.Budget.MaxPerInvocationUSD > 0 &&
		inv.MaxCostUSD > ua.Budget.MaxPerInvocationUSD {
		return newError(ErrUAPerInvocationExceeded,
			fmt.Sprintf("invocation max_cost_usd %v > ua.budget.max_per_invocation_usd %v",
				inv.MaxCostUSD, ua.Budget.MaxPerInvocationUSD))
	}
	// Budget narrowing: phase 11's symmetric rule applies between
	// the UA's Budget and the invocation's Budget block. The
	// invocation block, when present, must not widen any axis the UA
	// constrained.
	if err := narrowBudget(ua.Budget, inv.Budget); err != nil {
		return err
	}
	return nil
}

// ─── attenuation chain walk ───────────────────────────────────────────

func walkAttenuations(inv *Invocation, uaBudget *Budget, atts []Attenuation, now time.Time) (EffectiveCaveats, string, []string, error) {
	prevSigBytes, err := HexToBytes(inv.UserSig.Sig)
	if err != nil {
		return EffectiveCaveats{}, "", nil, newError(ErrAttenuationInvalid,
			fmt.Sprintf("invocation user_sig hex: %v", err))
	}
	// Effective Budget at the invocation layer = the invocation's
	// own block when set, else inherit the UA's. This mirrors the
	// "Mixed mode" rule in phase-11: parent has budget, child omits
	// → child inherits unchanged. Without this, realm_budgets set
	// only at the UA wouldn't propagate to the membership check.
	effective := EffectiveCaveats{
		Agents:     append([]string{}, inv.Agents...),
		MaxCostUSD: inv.MaxCostUSD,
		MaxSteps:   inv.MaxSteps,
		Budget:     mergeAttenuationBudget(uaBudget, inv.Budget),
		Exp:        inv.Exp,
	}
	runID := inv.RunID
	nonces := make([]string, 0, len(atts))

	for i, att := range atts {
		expected, err := ComputeAttenuationHMAC(prevSigBytes, att.Caveats)
		if err != nil {
			return effective, runID, nonces, newError(ErrAttenuationInvalid,
				fmt.Sprintf("attenuation[%d] hmac compute: %v", i, err))
		}
		gotHmac, err := HexToBytes(att.HMAC)
		if err != nil {
			return effective, runID, nonces, newError(ErrAttenuationInvalid,
				fmt.Sprintf("attenuation[%d] hmac hex: %v", i, err))
		}
		if !hmac.Equal(expected, gotHmac) {
			return effective, runID, nonces, newError(ErrAttenuationInvalid,
				fmt.Sprintf("attenuation[%d] hmac mismatch", i))
		}
		if err := narrowAttenuation(effective, att.Caveats); err != nil {
			return effective, runID, nonces, err
		}
		if expBefore(att.Caveats.Exp, now) {
			return effective, runID, nonces, newError(ErrMacaroonExpired,
				fmt.Sprintf("attenuation[%d] expired at %s", i, att.Caveats.Exp))
		}
		effective = EffectiveCaveats{
			Agents:     append([]string{}, att.Caveats.Agents...),
			MaxCostUSD: att.Caveats.MaxCostUSD,
			MaxSteps:   att.Caveats.MaxSteps,
			Budget:     mergeAttenuationBudget(effective.Budget, att.Caveats.Budget),
			Exp:        att.Caveats.Exp,
		}
		runID = att.Caveats.RunID
		nonces = append(nonces, att.Caveats.Nonce)
		prevSigBytes = gotHmac
	}

	return effective, runID, nonces, nil
}

// narrowAttenuation is the parent→child check at every attenuation
// boundary. Agents is the lineage-extension axis (child ⊇ parent);
// every other axis is shrink-only (child ≤ parent). This is the half
// of phase 11's symmetric rule that differs from the UA→invocation
// boundary handled in enforceInvocationCaveats.
func narrowAttenuation(parent EffectiveCaveats, child AttenuationCaveats) error {
	// agents: child ⊇ parent — child must include every parent entry,
	// and may add. (The agents list grows; cost/steps/exp shrink.)
	// The last entry remains "the most-specific agent" for billing.
	for _, a := range parent.Agents {
		if !containsString(child.Agents, a) {
			return newError(ErrAttenuationWidened,
				fmt.Sprintf("child agents dropped parent entry: %s", a))
		}
	}
	if child.MaxCostUSD > parent.MaxCostUSD {
		return newError(ErrAttenuationWidened,
			fmt.Sprintf("child max_cost_usd %v > parent %v", child.MaxCostUSD, parent.MaxCostUSD))
	}
	if child.MaxSteps > parent.MaxSteps {
		return newError(ErrAttenuationWidened,
			fmt.Sprintf("child max_steps %d > parent %d", child.MaxSteps, parent.MaxSteps))
	}
	// RFC 3339 UTC strings of the same length and Z suffix are
	// lexicographically ordered the same as their time values.
	if child.Exp > parent.Exp {
		return newError(ErrAttenuationWidened,
			fmt.Sprintf("child exp %s > parent %s", child.Exp, parent.Exp))
	}
	if err := narrowBudget(parent.Budget, child.Budget); err != nil {
		return err
	}
	return nil
}

// narrowBudget enforces the symmetric budget-narrowing rule between
// any parent→child layer boundary (UA→invocation, invocation→
// attenuation, attenuation→attenuation). Returns ErrAttenuationWidened
// when the child widens any axis. nil child means "inherits parent
// unchanged"; nil parent + non-nil child means "child introduces a
// constraint that didn't exist" — that's narrowing, allowed.
//
// Realm-budgets narrowing (rule 4 in phase-11): for each realm-id
// key in child.RealmBudgets, the same key must exist in
// parent.RealmBudgets (if parent set realm_budgets at all), and the
// child's per-realm cap must be ≤ parent's. Child may also OMIT
// realms the parent permitted — that's narrowing, allowed.
func narrowBudget(parent, child *Budget) error {
	if child == nil {
		return nil
	}
	if parent != nil {
		if parent.MaxPerInvocationUSD > 0 && child.MaxPerInvocationUSD > 0 &&
			child.MaxPerInvocationUSD > parent.MaxPerInvocationUSD {
			return newError(ErrAttenuationWidened,
				fmt.Sprintf("child budget.max_per_invocation_usd %v > parent %v",
					child.MaxPerInvocationUSD, parent.MaxPerInvocationUSD))
		}
		if parent.MaxTotalUSD > 0 && child.MaxTotalUSD > 0 &&
			child.MaxTotalUSD > parent.MaxTotalUSD {
			return newError(ErrAttenuationWidened,
				fmt.Sprintf("child budget.max_total_usd %v > parent %v",
					child.MaxTotalUSD, parent.MaxTotalUSD))
		}
	}
	if len(child.RealmBudgets) == 0 {
		return nil
	}
	// If parent set realm_budgets, every child key must appear in it
	// and not widen its cap. If parent did NOT set realm_budgets,
	// the child is introducing per-realm scoping that didn't exist
	// upstream — that's narrowing, allowed.
	if parent == nil || len(parent.RealmBudgets) == 0 {
		return nil
	}
	for r, cb := range child.RealmBudgets {
		pb, ok := parent.RealmBudgets[r]
		if !ok {
			return newError(ErrAttenuationWidened,
				fmt.Sprintf("child budget.realm_budgets[%s] not in parent", r))
		}
		if pb.MaxTotalUSD > 0 && cb.MaxTotalUSD > 0 &&
			cb.MaxTotalUSD > pb.MaxTotalUSD {
			return newError(ErrAttenuationWidened,
				fmt.Sprintf("child budget.realm_budgets[%s].max_total_usd %v > parent %v",
					r, cb.MaxTotalUSD, pb.MaxTotalUSD))
		}
	}
	return nil
}

// mergeAttenuationBudget propagates the effective Budget down the
// chain. The rule mirrors narrowBudget's intent: a child that omits
// budget inherits the parent's; a child that sets budget replaces
// the parent's (already validated as narrowing by narrowBudget).
func mergeAttenuationBudget(parent, child *Budget) *Budget {
	if child == nil {
		return parent
	}
	return child
}

// permittedRealms returns the sorted list of realm-ids the effective
// caveats authorize spend on. Returns nil when no realm_budgets
// appears anywhere in the chain — single-swarm deployments rely on
// this nil-ness to skip the membership check entirely.
func permittedRealms(eff *EffectiveCaveats) []string {
	if eff == nil || eff.Budget == nil || len(eff.Budget.RealmBudgets) == 0 {
		return nil
	}
	out := make([]string, 0, len(eff.Budget.RealmBudgets))
	for r := range eff.Budget.RealmBudgets {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}

// ─── helpers ──────────────────────────────────────────────────────────

func expBefore(iso string, now time.Time) bool {
	t, err := time.Parse(time.RFC3339, iso)
	if err != nil {
		// Malformed timestamps are rejected at parse time; treat as
		// expired here defensively so we never accept them.
		return true
	}
	return t.Before(now)
}

func containsString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
