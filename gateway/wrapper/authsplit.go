// Authsplit: pre-bifrost rewrite that unpacks `<vk>.<macaroon>` from
// whichever inbound auth header carries it into the canonical two-
// header shape (VK in the original header + `x-macaroon`).
//
// This exists because some agent harnesses only let callers configure
// a single "API key" field, with no way to add custom headers — but
// our governance model needs both a VK (for Bifrost customer auth)
// and a macaroon (for plugin authorization). Concatenating them into
// the API key field every harness already supports, then splitting
// here before Bifrost sees the request, is the path that works for
// every harness.
//
// See gateway/plans/phases/phase-10-bearer-concat-transport.md for
// the full plan, wire format, and rationale.
//
// Invariants
// ----------
//   - Strictly additive: any request that works today MUST continue
//     to work byte-for-byte after this code runs.
//   - The rewrite triggers ONLY when all five conditions hold:
//     1. Request is on the bifrost-bound proxy branch (caller's
//        responsibility — we don't check the path here).
//     2. One of authHeaderNames is present and non-empty.
//     3. That header's value (with any "Bearer " prefix stripped)
//        contains '.'.
//     4. The left side of the first '.' matches vkPattern.
//     5. No x-macaroon header is already set.
//   - The split function is allocation-light: it returns sub-strings
//     that share the original value's backing bytes.
//
// Nothing here knows anything cryptographic. Macaroon validation
// happens later in gateway/internal/auth/verifier.go after the plugin
// sees x-macaroon. If the right side isn't a valid macaroon, the
// plugin produces the same 401 it would have for a malformed
// x-macaroon today.

package main

import (
	"net/http"
	"net/http/httputil"
	"regexp"
	"strings"
)

// authHeaderNames is the set of inbound headers Bifrost-http accepts
// the VK on. Confirmed against bifrost/core/schemas/plugin_test.go's
// exact-match and case-insensitive lookup tests; no other inbound
// header carries the VK.
//
// Order matters: we stop at the first present header. If a future
// SDK lands that uses a fourth header name, add it here.
var authHeaderNames = []string{
	"Authorization",
	"X-Api-Key",
	"X-Goog-Api-Key",
}

// macaroonHeader is the canonical header the plugin reads. After the
// rewrite, the right side of the split lands here.
const macaroonHeader = "X-Macaroon"

// vkPattern matches a Bifrost virtual key. Anchored at both ends and
// excludes '.', which is what makes '.' an unambiguous separator
// between vk and macaroon.
//
// If Bifrost ever changes VK format to include '.', this phase needs
// revisiting. The test suite pins that assumption.
var vkPattern = regexp.MustCompile(`^sk-bf-[A-Za-z0-9_-]+$`)

// splitVKMacaroon takes a header value (already stripped of any
// "Bearer " prefix by the caller) and returns the VK + macaroon if
// the value matches the concat shape `<vk>.<macaroon>`.
//
// Returns ok=false (and zero strings) for any value that doesn't
// match. In particular: empty input, no '.', empty left side, empty
// right side, and any left side that doesn't match vkPattern. The
// last case is what protects OAuth/JWT bearer tokens from being
// mis-split — they contain '.' but the left side never starts with
// "sk-bf-".
func splitVKMacaroon(raw string) (vk, mac string, ok bool) {
	if raw == "" {
		return "", "", false
	}
	i := strings.IndexByte(raw, '.')
	if i <= 0 || i == len(raw)-1 {
		// No '.', or '.' at index 0 (empty left), or '.' at the very
		// end (empty right). All three are pass-through cases.
		return "", "", false
	}
	vk, mac = raw[:i], raw[i+1:]
	if !vkPattern.MatchString(vk) {
		return "", "", false
	}
	return vk, mac, true
}

// stripBearer removes a leading "Bearer " prefix (case-insensitive)
// from a header value. Returns the stripped value and whether a prefix
// was present, so the caller can re-attach the prefix when writing
// the VK back to the same header.
//
// This handles "Bearer ", "bearer ", "BEARER ", etc. Anything else
// (including no prefix) passes through unchanged with hadPrefix=false.
func stripBearer(v string) (stripped string, hadPrefix bool) {
	const prefix = "Bearer "
	if len(v) >= len(prefix) && strings.EqualFold(v[:len(prefix)], prefix) {
		return v[len(prefix):], true
	}
	return v, false
}

// rewriteAuthHeaders mutates h in place to convert a concat-form
// credential into the canonical two-header shape.
//
// Algorithm (see phase-10 plan §"Wrapper behavior" for the full
// specification):
//
//  1. If x-macaroon is already set, leave the request alone — the
//     caller explicitly used the two-header form.
//  2. Iterate authHeaderNames in order. Take the first non-empty
//     header value.
//  3. Strip any "Bearer " prefix (only meaningful on Authorization).
//  4. Try to split the remainder on the first '.'. If splitVKMacaroon
//     returns ok=false, RETURN without modifying anything — we do NOT
//     fall through to the next header. The convention is "the header
//     carrying the credential is the one to rewrite," and stopping
//     at the first present header locks that in.
//  5. Replace the original header with just the VK (re-adding the
//     "Bearer " prefix if it was there) and set x-macaroon to the
//     right side.
//
// This function exposes Header (not *httputil.ProxyRequest) for
// straightforward unit testing. The proxy wiring is in main.go.
func rewriteAuthHeaders(h http.Header) {
	if h.Get(macaroonHeader) != "" {
		return
	}
	for _, name := range authHeaderNames {
		v := h.Get(name)
		if v == "" {
			continue
		}
		stripped, hadBearer := stripBearer(v)
		vk, mac, ok := splitVKMacaroon(stripped)
		if !ok {
			// First present header carries the credential but isn't
			// in concat form — pass through untouched. Do NOT continue
			// the loop; the credential lives in this header.
			return
		}
		if hadBearer {
			h.Set(name, "Bearer "+vk)
		} else {
			h.Set(name, vk)
		}
		h.Set(macaroonHeader, mac)
		return
	}
}

// installAuthRewrite wraps an existing reverse proxy's Director so
// rewriteAuthHeaders runs on every outgoing request, after the
// Director set by NewSingleHostReverseProxy has finished its own URL
// rewriting work.
//
// Why Director-chaining and not the newer Rewrite hook? The proxy is
// constructed by httputil.NewSingleHostReverseProxy, which installs a
// Director (not a Rewrite). Per Go's docs, Rewrite and Director are
// mutually exclusive — setting Rewrite would silently disable the
// URL-rewriting work NewSingleHostReverseProxy did. Wrapping the
// existing Director is the documented composition pattern.
//
// The wrapped Director mutates req.Header on the outbound request,
// not the original inbound one (httputil.ReverseProxy.ServeHTTP makes
// a shallow request copy before calling Director, so req.Header here
// is already the outbound's). Confirmed against Go's net/http/httputil
// source: ReverseProxy.ServeHTTP at "outreq := req.Clone(ctx)".
func installAuthRewrite(proxy *httputil.ReverseProxy) {
	inner := proxy.Director
	proxy.Director = func(req *http.Request) {
		inner(req)
		rewriteAuthHeaders(req.Header)
	}
}
