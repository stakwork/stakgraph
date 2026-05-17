package auth

import (
	"context"
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
// One of Claims / Err is always non-nil. When the macaroon header was
// missing AND enforcement is off, both are nil — the hook should
// just continue.
type Decision struct {
	// Claims is the verified output of the pure verifier plus
	// adapter-side revocation checks. Non-nil on full success;
	// nil on any failure (including the missing-header case).
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
}

// Evaluate runs the full verify pipeline (signature + revocation)
// against the macaroon stashed on ctx by HTTPTransportPreHook plus
// the trust registry handed to Init. It does NOT decide whether to
// reject — that's the hook caller's job, gated on enforce_macaroons.
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

	// Signature verified — now revocation. Note this is the only
	// Redis touch in the phase-4 adapter; phase 6 adds cost/steps
	// cap-walks and per-agent budget reads here.
	if revErr := CheckRevocations(ctx, claims); revErr != nil {
		d.Err = revErr
		return d
	}

	d.Claims = claims
	return d
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
			"auth: verify ok mode=%s org=%s user=%s workspace=%s agent=%s run_id=%s",
			mode, d.Claims.OrgID, d.Claims.UserID, d.Claims.Workspace,
			d.Claims.AgentName, d.Claims.RunID,
		)
	case d.Err != nil && !d.HadHeader:
		// Missing header is a softer log than "bad macaroon".
		pluginlog.Warnf("auth: no x-macaroon header mode=%s code=%s", mode, d.Err.Code)
	case d.Err != nil:
		pluginlog.Warnf("auth: verify FAIL mode=%s code=%s status=%d detail=%q",
			mode, d.Err.Code, d.Err.HTTPStatus, d.Err.Message)
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
