package auth

import (
	"context"
	"fmt"
	"time"

	"github.com/maximhq/bifrost/core/schemas"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginctx"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

// Decision is the result of running the full verify+revocation path
// against an incoming request. Returned to the hook layer so the same
// outcome can be either rejected (enforce mode) or logged (shadow
// mode) without duplicating the verification work.
//
// One of Claims / Err is non-nil after Evaluate (both nil only on the
// missing-header-shadow-mode path). PostVerifyClaims, when set,
// carries the verified claims even on the rejection path so the
// FAIL log line can include run-id / agent / org context — useful
// for tracking down "which run got blocked by which adapter check?"
// without correlating two log lines by timestamp.
type Decision struct {
	// Claims is the verified output of the pure verifier plus
	// adapter-side revocation + realm-membership checks. Non-nil
	// ONLY on full success. Adapters stamp this onto the request
	// context for downstream hooks to see.
	Claims *macaroon.Claims

	// Err describes why verification failed. Nil on success;
	// non-nil on every other path, including "no macaroon header
	// was present" — callers decide whether that constitutes a
	// rejection based on enforce_macaroons.
	Err *AdapterError

	// HadHeader records whether an x-macaroon header was actually
	// present on the request. Distinguishes "header absent" from
	// "header present but invalid" so shadow-mode logs can stay
	// honest about which path the request took.
	HadHeader bool

	// PostVerifyClaims is the pure-verifier output captured BEFORE
	// any post-verify adapter check (revocation, realm-membership)
	// rejected the request. Used purely for log enrichment — never
	// stamped on the request context, never returned to downstream
	// hooks. nil when the pure verifier itself failed (no claims
	// to enrich with) or when the request had no header at all.
	PostVerifyClaims *macaroon.Claims
}

// Evaluate runs the full verify pipeline (signature + revocation +
// phase-11 realm-membership check) against the macaroon stashed on
// ctx by HTTPTransportPreHook plus the trust registry handed to
// Init. It does NOT decide whether to reject — that's the hook
// caller's job, gated on enforce_macaroons.
//
// Evaluate is safe to call before adapter Init has finished — it
// uses the package-level Registry pointer set by Init, and returns
// "trust registry unavailable" if that's still nil.
//
// Pure side-effect free except for log lines; the caller is
// responsible for stamping claims on context (StampClaims) once it
// decides to proceed.
func Evaluate(ctx context.Context, rawMacaroon string) Decision {
	now := time.Now().UTC()
	d := Decision{HadHeader: rawMacaroon != ""}

	registry := getRegistry()
	claims, adapterErr := Verify(rawMacaroon, registry, now)
	if adapterErr != nil {
		d.Err = adapterErr
		return d
	}

	// Pure verify passed — stash claims for log enrichment in case
	// the revocation or realm-membership check rejects below. We
	// don't promote them to d.Claims yet because the request is
	// still tentatively-rejected until all post-verify checks pass.
	d.PostVerifyClaims = claims

	// Note this is the only Redis touch in the phase-4 adapter;
	// phase 6 adds cost/steps cap-walks and per-agent budget reads
	// here.
	if revErr := CheckRevocations(ctx, claims); revErr != nil {
		d.Err = revErr
		return d
	}

	// Phase-11 realm-membership check. Asserts the swarm's own
	// realm_id (if configured) is in the macaroon's permitted set
	// (if the macaroon scoped per-realm). No-op for single-swarm
	// deployments. See gateway/plans/phases/phase-11-symmetric-recursive-authorization.md
	// "The plugin's realm-membership check".
	if memErr := CheckRealmMembership(claims, registry); memErr != nil {
		d.Err = memErr
		return d
	}

	d.Claims = claims
	return d
}

// CheckRealmMembership implements the phase-11 plugin-side
// membership check. Four cases (per the phase doc):
//
//  1. Swarm has no realm_id AND macaroon has no realm_budgets:
//     simple-deployment mode, no check. Returns nil.
//  2. Swarm has a realm_id AND macaroon has realm_budgets:
//     swarm.realm_id MUST be a key in claims.realm_budgets, else
//     reject with realm_not_permitted.
//  3. Swarm has a realm_id AND macaroon has no realm_budgets:
//     the org didn't scope per-realm; swarm accepts and enforces
//     only the non-realm caps. Returns nil.
//  4. Swarm has no realm_id AND macaroon has realm_budgets:
//     configuration error — a multi-realm macaroon landed on a
//     swarm that doesn't claim an identity. Reject with
//     realm_not_configured.
//
// Exported so adapter tests can exercise it without spinning up
// the full hot path; the plugin code calls it through Evaluate.
func CheckRealmMembership(claims *macaroon.Claims, registry *trust.Registry) *AdapterError {
	if claims == nil {
		return nil
	}
	var realmID string
	if registry != nil {
		realmID = registry.RealmID()
	}
	var realmBudgets map[string]macaroon.RealmBudget
	if b := claims.EffectiveCaveats.Budget; b != nil {
		realmBudgets = b.RealmBudgets
	}

	switch {
	case len(realmBudgets) == 0 && realmID == "":
		// Case 1: simple deployment. Nothing to check.
		return nil
	case len(realmBudgets) == 0:
		// Case 3: org didn't scope per-realm. Non-realm caps apply
		// but the membership check is a no-op.
		return nil
	case realmID == "":
		// Case 4: multi-realm macaroon, swarm has no identity.
		// Treat as a configuration error rather than letting the
		// macaroon through; otherwise a misconfigured swarm would
		// silently accept calls that the org meant to scope away.
		return &AdapterError{
			Code:       "realm_not_configured",
			HTTPStatus: 401,
			Message: "macaroon carries realm_budgets but this swarm has no realm_id configured " +
				"(set via PUT /_plugin/trust/realm_id)",
		}
	}
	// Case 2: both sides set. Membership is required.
	if _, ok := realmBudgets[realmID]; !ok {
		return &AdapterError{
			Code:       "realm_not_permitted",
			HTTPStatus: 401,
			Message: fmt.Sprintf("swarm realm_id %q not in macaroon's permitted_realms %v",
				realmID, claims.PermittedRealms),
		}
	}
	return nil
}

// StampClaims writes the verified claims to the request context so
// downstream hooks (PostLLMHook, future cost accumulator, etc.) can
// see them. Idempotent — re-calling overrides; tests use that to
// inject fixtures.
//
// Separate from Evaluate so the hook layer can decide whether to
// stamp on shadow-mode passthrough (we do: downstream observability
// is the whole point of shadow mode) vs an enforce-mode failure
// (we don't: rejected requests shouldn't stamp anything).
func StampClaims(bctx *schemas.BifrostContext, claims *macaroon.Claims) {
	pluginctx.SetVerifiedClaims(bctx, claims)
}

// ApplyToLLMPre is the canonical "call from PreLLMHook" entry point.
// It glues Evaluate + StampClaims + the shadow/enforce gate into one
// helper so the hook body stays a three-liner.
//
// Returns (shortCircuit, nil) when enforcement is on and the request
// should be rejected; (nil, nil) when the request should continue.
// All paths log a structured line — shadow-mode mismatches in
// particular log loudly so operators can audit rollout readiness.
//
// The returned short-circuit, if any, carries a *schemas.BifrostError
// whose StatusCode matches the AdapterError.HTTPStatus and whose
// Error.Code is the AdapterError.Code (stable across versions).
func ApplyToLLMPre(bctx *schemas.BifrostContext) *schemas.LLMPluginShortCircuit {
	rawMacaroon := pluginctx.RawMacaroon(bctx)
	decision := Evaluate(bctx, rawMacaroon)
	cfg := GetConfig()

	logDecision(decision, cfg.EnforceMacaroons)

	switch {
	case decision.Claims != nil:
		// Happy path. Always stamp claims, in both modes — shadow
		// mode wants downstream hooks to see the verified shape.
		StampClaims(bctx, decision.Claims)
		return nil

	case decision.Err != nil && cfg.EnforceMacaroons:
		return shortCircuitFromError(decision.Err)

	default:
		// Shadow mode with a failure — or enforce mode with no
		// header but enforcement disabled (can't happen since we
		// gated on EnforceMacaroons above). Let the request through;
		// the log line already recorded the mismatch.
		return nil
	}
}

// shortCircuitFromError turns an *AdapterError into the Bifrost
// short-circuit envelope. Plugin developers populate
// LLMPluginShortCircuit.Error with a *BifrostError; AllowFallbacks
// is explicitly false because auth failures must not be retried
// against fallback providers (a fallback provider seeing the same
// bad macaroon would just fail the same way at someone else's cost).
func shortCircuitFromError(e *AdapterError) *schemas.LLMPluginShortCircuit {
	allowFallbacks := false
	code := e.Code
	msg := e.Message
	if msg == "" {
		msg = code
	}
	status := e.HTTPStatus
	if status == 0 {
		status = 401
	}
	errType := "macaroon_verification_failed"
	return &schemas.LLMPluginShortCircuit{
		Error: &schemas.BifrostError{
			IsBifrostError: false, // it's an auth error, not a transport error
			Type:           &errType,
			StatusCode:     &status,
			Error: &schemas.ErrorField{
				Code:    &code,
				Message: msg,
			},
			AllowFallbacks: &allowFallbacks,
		},
	}
}

func logDecision(d Decision, enforce bool) {
	mode := "shadow"
	if enforce {
		mode = "enforce"
	}
	switch {
	case d.Claims != nil:
		pluginlog.Logf(
			"auth: verify ok mode=%s org=%s user=%s agent=%s run_id=%s permitted_realms=%v",
			mode, d.Claims.OrgID, d.Claims.UserID,
			d.Claims.AgentName, d.Claims.RunID, d.Claims.PermittedRealms,
		)
	case d.Err != nil && !d.HadHeader:
		// Missing header is a softer log than "bad macaroon".
		pluginlog.Warnf("auth: no x-macaroon header mode=%s code=%s", mode, d.Err.Code)
	case d.Err != nil:
		// Include run_id / agent / org when we have them — a
		// post-verify rejection (revocation, realm-membership)
		// has fully-verified claims attached for log enrichment
		// even though the request was rejected. Falls back to
		// "(no-claims)" placeholders for pre-verify failures
		// (bad signature, malformed, expired).
		runID, agent, org := "(no-claims)", "(no-claims)", "(no-claims)"
		if c := d.PostVerifyClaims; c != nil {
			runID, agent, org = c.RunID, c.AgentName, c.OrgID
		}
		pluginlog.Warnf(
			"auth: verify FAIL mode=%s code=%s status=%d org=%s agent=%s run_id=%s detail=%q",
			mode, d.Err.Code, d.Err.HTTPStatus, org, agent, runID, d.Err.Message,
		)
	}
}

// ─── package-level wiring ──────────────────────────────────────────────

// The trust registry pointer is wired in from main.go via
// SetTrustRegistry. We mirror the same pattern adminapi uses for the
// same reason: trust loading and adapter Init are independent
// concerns and shouldn't be coupled through one constructor.

var registryRef *trust.Registry

// SetTrustRegistry wires the trust registry the adapter consults on
// every verify. Must be called before the first request reaches the
// hooks. Passing nil disables verification (Verify returns
// "trust_registry_unavailable") — useful in tests that exercise the
// shadow-mode log path without standing up a registry.
func SetTrustRegistry(r *trust.Registry) {
	registryRef = r
}

func getRegistry() *trust.Registry { return registryRef }
