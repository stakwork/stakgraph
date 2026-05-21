// Tests for the bearer-concat → two-header transport split.
//
// These tests exhaustively cover the table in
// gateway/plans/phases/phase-10-bearer-concat-transport.md §"Test
// surface" PLUS the backward-compatibility regression cases. They
// run with plain `go test ./...` from this directory (no Bifrost,
// no Docker, no Redis).
//
// What "backward compat" means here: any request shape that worked
// against the wrapper today must produce a byte-identical outbound
// Header map after rewriteAuthHeaders runs. The "Regression" group
// below pins that property.

package main

import (
	"net/http"
	"testing"
)

// hdr is a small helper to build an http.Header from key/value pairs.
// It exists so test cases read like declarative tables.
func hdr(kvs ...string) http.Header {
	if len(kvs)%2 != 0 {
		panic("hdr: need even number of args")
	}
	h := http.Header{}
	for i := 0; i < len(kvs); i += 2 {
		h.Add(kvs[i], kvs[i+1])
	}
	return h
}

// assertHeaderEqual compares two http.Header maps key-by-key. We
// don't use reflect.DeepEqual because it's sensitive to nil vs.
// empty slice on absent keys, and we want a friendlier diff.
func assertHeaderEqual(t *testing.T, got, want http.Header) {
	t.Helper()
	// Every key in want must be present with the same value.
	for k, wantVals := range want {
		gotVals := got.Values(k)
		if len(gotVals) != len(wantVals) {
			t.Errorf("header %q: got %d values %v, want %d values %v",
				k, len(gotVals), gotVals, len(wantVals), wantVals)
			continue
		}
		for i := range wantVals {
			if gotVals[i] != wantVals[i] {
				t.Errorf("header %q[%d]: got %q, want %q",
					k, i, gotVals[i], wantVals[i])
			}
		}
	}
	// Every key in got must be in want (catches unexpected additions
	// like spurious X-Macaroon on a passthrough case).
	for k := range got {
		if _, ok := want[k]; !ok {
			t.Errorf("header %q present in got but not want (values=%v)",
				k, got.Values(k))
		}
	}
}

// ─── splitVKMacaroon: pure function tests ─────────────────────────────

func TestSplitVKMacaroon(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantVK  string
		wantMac string
		wantOK  bool
	}{
		{
			name:    "happy path",
			input:   "sk-bf-abc123.eyJtYWNhcm9vbiI6Indvcmsifg",
			wantVK:  "sk-bf-abc123",
			wantMac: "eyJtYWNhcm9vbiI6Indvcmsifg",
			wantOK:  true,
		},
		{
			// JWTs / OAuth bearer tokens have dots but the left side
			// doesn't match the VK pattern. This is the critical
			// passthrough case that protects every non-Bifrost
			// credential type from being mangled.
			name:   "JWT-shaped value rejected (left side not a VK)",
			input:  "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature",
			wantOK: false,
		},
		{
			name:   "plain VK with no dot",
			input:  "sk-bf-abc123",
			wantOK: false,
		},
		{
			name:   "empty input",
			input:  "",
			wantOK: false,
		},
		{
			name:   "leading dot (empty left side)",
			input:  ".some-macaroon-payload",
			wantOK: false,
		},
		{
			name:   "trailing dot (empty right side)",
			input:  "sk-bf-abc123.",
			wantOK: false,
		},
		{
			name:   "just a dot",
			input:  ".",
			wantOK: false,
		},
		{
			// Left side doesn't start with sk-bf-. The wrapper must
			// not mistake a non-Bifrost token (some provider's
			// dot-bearing key) for the concat form.
			name:   "left side missing sk-bf- prefix",
			input:  "notavk.something",
			wantOK: false,
		},
		{
			// VK alphabet excludes '.', '+', '/', etc. The pattern
			// must reject any character outside [A-Za-z0-9_-] in the
			// VK position.
			name:   "VK with disallowed character",
			input:  "sk-bf-abc+def.macpart",
			wantOK: false,
		},
		{
			// First '.' wins. If the macaroon itself somehow
			// contained additional dots (it shouldn't — base64url
			// doesn't include '.'), we'd still split at the first
			// one. Pinned here so the behavior is documented.
			name:    "first dot wins",
			input:   "sk-bf-abc.first.second.third",
			wantVK:  "sk-bf-abc",
			wantMac: "first.second.third",
			wantOK:  true,
		},
		{
			// Underscores and hyphens are valid in the VK alphabet.
			name:    "VK with underscore and hyphen",
			input:   "sk-bf-a_b-c.macpart",
			wantVK:  "sk-bf-a_b-c",
			wantMac: "macpart",
			wantOK:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vk, mac, ok := splitVKMacaroon(tt.input)
			if ok != tt.wantOK {
				t.Fatalf("ok: got %v want %v (vk=%q mac=%q)", ok, tt.wantOK, vk, mac)
			}
			if ok {
				if vk != tt.wantVK {
					t.Errorf("vk: got %q want %q", vk, tt.wantVK)
				}
				if mac != tt.wantMac {
					t.Errorf("mac: got %q want %q", mac, tt.wantMac)
				}
			}
		})
	}
}

// ─── stripBearer: pure function tests ─────────────────────────────────

func TestStripBearer(t *testing.T) {
	tests := []struct {
		input        string
		wantStripped string
		wantHad      bool
	}{
		{"Bearer sk-bf-x", "sk-bf-x", true},
		{"bearer sk-bf-x", "sk-bf-x", true},
		{"BEARER sk-bf-x", "sk-bf-x", true},
		{"BeArEr sk-bf-x", "sk-bf-x", true},
		// No prefix → returned unchanged. This is the x-api-key
		// shape: Anthropic and Gemini SDKs send raw values without
		// "Bearer ".
		{"sk-bf-x", "sk-bf-x", false},
		// Too short to contain "Bearer ".
		{"abc", "abc", false},
		// Empty stays empty.
		{"", "", false},
		// "Bearer" without trailing space is NOT a prefix match. The
		// space is part of the convention; without it we'd mangle a
		// theoretical credential that started with the letters
		// "Bearer".
		{"BearerNoSpace", "BearerNoSpace", false},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, had := stripBearer(tt.input)
			if got != tt.wantStripped {
				t.Errorf("stripped: got %q want %q", got, tt.wantStripped)
			}
			if had != tt.wantHad {
				t.Errorf("hadPrefix: got %v want %v", had, tt.wantHad)
			}
		})
	}
}

// ─── rewriteAuthHeaders: integration table ────────────────────────────

func TestRewriteAuthHeaders(t *testing.T) {
	const (
		vk  = "sk-bf-abc123"
		mac = "eyJtYWNhcm9vbiI6Indvcmsifg"
	)

	tests := []struct {
		name string
		in   http.Header
		want http.Header
	}{
		// ── New "concat" path ────────────────────────────────────
		{
			name: "Authorization Bearer concat",
			in:   hdr("Authorization", "Bearer "+vk+"."+mac),
			want: hdr(
				"Authorization", "Bearer "+vk,
				"X-Macaroon", mac,
			),
		},
		{
			name: "Authorization lowercase bearer concat",
			in:   hdr("Authorization", "bearer "+vk+"."+mac),
			want: hdr(
				"Authorization", "Bearer "+vk,
				"X-Macaroon", mac,
			),
		},
		{
			name: "x-api-key concat (Anthropic-shaped)",
			in:   hdr("X-Api-Key", vk+"."+mac),
			want: hdr(
				"X-Api-Key", vk,
				"X-Macaroon", mac,
			),
		},
		{
			name: "x-goog-api-key concat (Gemini-shaped)",
			in:   hdr("X-Goog-Api-Key", vk+"."+mac),
			want: hdr(
				"X-Goog-Api-Key", vk,
				"X-Macaroon", mac,
			),
		},
		{
			// Case-insensitive header lookup: Go canonicalizes
			// header keys via textproto.CanonicalMIMEHeaderKey so
			// "x-api-key", "X-Api-Key", and "X-API-KEY" all hit the
			// same map entry. Pin this explicitly so a future
			// refactor that bypasses canonicalization breaks here.
			name: "lowercase header name works (Go canonicalizes)",
			in:   hdr("x-api-key", vk+"."+mac),
			want: hdr(
				"X-Api-Key", vk,
				"X-Macaroon", mac,
			),
		},

		// ── Backward-compat passthrough cases ────────────────────
		{
			name: "Authorization Bearer with no dot (plain VK)",
			in:   hdr("Authorization", "Bearer "+vk),
			want: hdr("Authorization", "Bearer "+vk),
		},
		{
			name: "x-api-key with no dot (plain VK)",
			in:   hdr("X-Api-Key", vk),
			want: hdr("X-Api-Key", vk),
		},
		{
			// The existing two-header form. Some callers explicitly
			// stamp both Authorization (VK) and X-Macaroon. Today
			// they work; after this change they must STILL work
			// unchanged — the existing X-Macaroon takes precedence
			// regardless of what's in Authorization.
			name: "two-header form passes through unchanged",
			in: hdr(
				"Authorization", "Bearer "+vk,
				"X-Macaroon", "existing-macaroon",
			),
			want: hdr(
				"Authorization", "Bearer "+vk,
				"X-Macaroon", "existing-macaroon",
			),
		},
		{
			// Critical edge case: if X-Macaroon is already set, the
			// wrapper must NOT clobber it even if Authorization
			// contains a concat. The caller has explicitly opted
			// into the two-header form.
			name: "existing X-Macaroon wins over concat in Authorization",
			in: hdr(
				"Authorization", "Bearer "+vk+"."+mac,
				"X-Macaroon", "caller-stamped-macaroon",
			),
			want: hdr(
				"Authorization", "Bearer "+vk+"."+mac,
				"X-Macaroon", "caller-stamped-macaroon",
			),
		},
		{
			// OAuth / JWT tokens contain dots but the left side
			// doesn't match the VK pattern. The wrapper must leave
			// them strictly alone. This protects every non-Bifrost
			// bearer-token caller from being broken by this change.
			name: "JWT in Authorization passes through",
			in:   hdr("Authorization", "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature"),
			want: hdr("Authorization", "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature"),
		},
		{
			// Empty right side ("sk-bf-x.") looks superficially
			// concat-shaped but rule 6 rejects it. The header reaches
			// Bifrost untouched and Bifrost returns its own error
			// for the malformed VK — same behavior as today.
			name: "concat-shaped with empty macaroon falls through",
			in:   hdr("Authorization", "Bearer "+vk+"."),
			want: hdr("Authorization", "Bearer "+vk+"."),
		},
		{
			name: "no auth header at all",
			in:   hdr("Content-Type", "application/json"),
			want: hdr("Content-Type", "application/json"),
		},

		// ── Multi-header tie-breaking ────────────────────────────
		{
			// First present header in authHeaderNames order
			// (Authorization) wins. The x-api-key value is left
			// completely alone, even though it also contains a dot.
			// This is the documented "first wins, don't fall
			// through" rule.
			name: "both Authorization and x-api-key present, Authorization wins",
			in: hdr(
				"Authorization", "Bearer "+vk+"."+mac,
				"X-Api-Key", "sk-bf-other.othermac",
			),
			want: hdr(
				"Authorization", "Bearer "+vk,
				"X-Api-Key", "sk-bf-other.othermac", // unchanged
				"X-Macaroon", mac,
			),
		},
		{
			// If Authorization has no dot but x-api-key does, we
			// STILL stop at Authorization. The rule is "the first
			// present header carries the credential" — we don't go
			// hunting for a dot. This is intentional defense against
			// smuggling: a caller can't slip a macaroon in via
			// x-api-key when the real auth lives in Authorization.
			name: "Authorization without dot stops loop, x-api-key not scanned",
			in: hdr(
				"Authorization", "Bearer "+vk,
				"X-Api-Key", "sk-bf-other.othermac",
			),
			want: hdr(
				"Authorization", "Bearer "+vk,
				"X-Api-Key", "sk-bf-other.othermac",
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clone so we can show the before-state in failure msgs.
			got := tt.in.Clone()
			rewriteAuthHeaders(got)
			assertHeaderEqual(t, got, tt.want)
		})
	}
}

// TestRewriteAuthHeaders_AssumesVKShape pins the assumption that no
// real Bifrost VK contains '.'. If Bifrost ever changes VK format,
// this test starts being meaningful — until then it documents the
// invariant that protects the rule-5 regex.
//
// If you're touching this file because a VK with '.' showed up:
//   - Decide whether to keep '.' as the separator (and use a richer
//     parser, e.g. require the macaroon side to be valid base64url).
//   - Or pick a different separator that's still outside both
//     alphabets.
//
// See gateway/plans/phases/phase-10-bearer-concat-transport.md
// §"Wire format" for the rationale.
func TestRewriteAuthHeaders_AssumesVKShape(t *testing.T) {
	// Representative VKs that have actually appeared. Add to this
	// list (or wire it up to fixture data) if a regression ever hits.
	knownVKShapes := []string{
		"sk-bf-abc123",
		"sk-bf-A1B2_C3-D4",
		"sk-bf-" + "x_y-z_" + "ABCDEF0123456789",
	}
	for _, v := range knownVKShapes {
		if !vkPattern.MatchString(v) {
			t.Errorf("vkPattern rejected a known-valid VK shape %q — "+
				"phase-10's '.' separator assumes VKs never contain '.', "+
				"and the pattern must accept every shape Bifrost issues",
				v)
		}
	}
}
