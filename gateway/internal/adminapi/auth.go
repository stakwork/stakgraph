package adminapi

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// requireBearerToken returns middleware that gates a handler behind
// `Authorization: Bearer <expected>` (or the raw bare token, for
// laxer callers). The comparison runs in constant time so a probing
// client can't learn the token length / prefix from response timings.
//
// The expected value is captured in a closure so it never appears in
// a package-level var — that way logs / panics can never accidentally
// surface it.
func requireBearerToken(expected string) func(http.HandlerFunc) http.HandlerFunc {
	expectedBytes := []byte(expected)
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			presented := r.Header.Get("Authorization")
			// Accept both "Bearer xxx" and "xxx" so clients that
			// pre-strip the scheme (test scripts, for example) still
			// work. Case-insensitive match on the scheme.
			if len(presented) > 7 {
				lower := strings.ToLower(presented[:7])
				if lower == "bearer " {
					presented = presented[7:]
				}
			}
			if subtle.ConstantTimeCompare([]byte(presented), expectedBytes) != 1 {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			next(w, r)
		}
	}
}
