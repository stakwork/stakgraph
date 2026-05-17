package macaroon

import "fmt"

// ErrorCode is a machine-readable tag for a verification failure.
// Adapters translate these to bifrost.Error codes (or HTTP status
// codes) at their boundary.
type ErrorCode string

const (
	ErrMacaroonMalformed          ErrorCode = "macaroon_malformed"
	ErrMacaroonUnsupportedVersion ErrorCode = "macaroon_unsupported_version"
	ErrInvalidUserAuthorization   ErrorCode = "invalid_user_authorization"
	ErrUserAuthorizationExpired   ErrorCode = "user_authorization_expired"
	ErrInvalidInvocationSig       ErrorCode = "invalid_invocation_signature"
	ErrInvocationViolated         ErrorCode = "invocation_violated"
	ErrUAPerInvocationExceeded    ErrorCode = "ua_per_invocation_exceeded"
	ErrMacaroonExpired            ErrorCode = "macaroon_expired"
	ErrAttenuationInvalid         ErrorCode = "attenuation_invalid"
	ErrAttenuationWidened         ErrorCode = "attenuation_widened"
)

// VerifyError is the error type returned by [Verify]. Adapters
// dispatch on Code to pick the right transport response.
type VerifyError struct {
	Code   ErrorCode
	Detail string
}

func (e *VerifyError) Error() string {
	if e.Detail == "" {
		return string(e.Code)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Detail)
}

func newError(code ErrorCode, detail string) *VerifyError {
	return &VerifyError{Code: code, Detail: detail}
}
