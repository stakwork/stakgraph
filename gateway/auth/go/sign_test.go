package macaroon_test

// Signer-side fixture test (Go).
//
// Loads every gateway/auth/fixtures/*.json, re-signs the unsigned
// layers from the seed private keys, attenuates from the unsigned
// caveats, assembles + encodes, and asserts the resulting bytes match
// what the TS signer produced.
//
// Both Ed25519 (RFC 8032) and ECDSA-secp256k1 (RFC 6979 + BIP 62
// low-s) are deterministic in @noble/curves (TS) and
// github.com/decred/dcrd/dcrec/secp256k1/v4 (Go), so byte-equality is
// the right gate. If either side ever switched to a randomized
// signer, this test would catch it.

import (
	"encoding/json"
	"testing"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

func TestSignerFixtures(t *testing.T) {
	for _, name := range listFixtures(t) {
		name := name
		t.Run(name, func(t *testing.T) {
			runSignerFixture(t, name)
		})
	}
}

func runSignerFixture(t *testing.T, name string) {
	fx := loadFixture(t, name)

	// ─── 1. Parse the unsigned ua/inv into typed structs ──────────
	// These mirror what an issuer holds: a UserAuthorization with
	// OrgSig at its zero value (filled in by the signer), and an
	// Invocation with UserSig at its zero value.
	var uaUnsigned macaroon.UserAuthorization
	if err := json.Unmarshal(fx.Inputs.UAUnsigned, &uaUnsigned); err != nil {
		t.Fatalf("unmarshal ua_unsigned: %v", err)
	}
	var invUnsigned macaroon.Invocation
	if err := json.Unmarshal(fx.Inputs.InvUnsigned, &invUnsigned); err != nil {
		t.Fatalf("unmarshal inv_unsigned: %v", err)
	}

	// ─── 2. Sign user_authorization (single-key OR multisig) ─────
	var uaSigned macaroon.UserAuthorization
	if fx.Inputs.OrgPrivHex != "" {
		// Single-key org.
		orgPriv, err := macaroon.HexToBytes(fx.Inputs.OrgPrivHex)
		if err != nil {
			t.Fatalf("decode org_priv_hex: %v", err)
		}
		uaSigned, err = macaroon.SignUserAuthorizationSingle(uaUnsigned, orgPriv)
		if err != nil {
			t.Fatalf("SignUserAuthorizationSingle: %v", err)
		}
	} else {
		// Multisig org: assemble signers in the order the fixture
		// declares (participating_signers), each with its own privkey
		// keyed by index in org_privs_hex.
		if len(fx.Inputs.Signers) == 0 {
			t.Fatalf("multisig fixture missing participating_signers")
		}
		signers := make([]macaroon.MultisigSigner, 0, len(fx.Inputs.Signers))
		for _, idx := range fx.Inputs.Signers {
			key := indexKey(idx)
			privHex, ok := fx.Inputs.OrgPrivsHex[key]
			if !ok {
				t.Fatalf("multisig fixture missing org_privs_hex[%s]", key)
			}
			privBytes, err := macaroon.HexToBytes(privHex)
			if err != nil {
				t.Fatalf("decode org_privs_hex[%s]: %v", key, err)
			}
			signers = append(signers, macaroon.MultisigSigner{
				KeyIndex: idx,
				PrivKey:  privBytes,
			})
		}
		var err error
		uaSigned, err = macaroon.SignUserAuthorizationMultisig(uaUnsigned, signers)
		if err != nil {
			t.Fatalf("SignUserAuthorizationMultisig: %v", err)
		}
	}

	// ─── 3. Sign invocation ─────────────────────────────────────
	userPriv, err := macaroon.HexToBytes(fx.Inputs.UserPrivHex)
	if err != nil {
		t.Fatalf("decode user_priv_hex: %v", err)
	}
	invSigned, err := macaroon.SignInvocation(invUnsigned, userPriv)
	if err != nil {
		t.Fatalf("SignInvocation: %v", err)
	}

	// ─── 4. Build attenuation chain from unsigned caveats ────────
	prevSigBytes, err := macaroon.InvocationSigBytes(invSigned)
	if err != nil {
		t.Fatalf("InvocationSigBytes: %v", err)
	}
	atts := make([]macaroon.Attenuation, 0, len(fx.Inputs.AttsUnsigned))
	for i, raw := range fx.Inputs.AttsUnsigned {
		var caveats macaroon.AttenuationCaveats
		if err := json.Unmarshal(raw, &caveats); err != nil {
			t.Fatalf("unmarshal atts_unsigned[%d]: %v", i, err)
		}
		att, err := macaroon.Attenuate(prevSigBytes, caveats)
		if err != nil {
			t.Fatalf("Attenuate[%d]: %v", i, err)
		}
		atts = append(atts, att)
		prevSigBytes, err = macaroon.AttenuationSigBytes(att)
		if err != nil {
			t.Fatalf("AttenuationSigBytes[%d]: %v", i, err)
		}
	}

	// ─── 5. Assemble + encode ────────────────────────────────────
	full := macaroon.Macaroon{
		V:                 1,
		OrgID:             fx.Inputs.OrgID,
		UserAuthorization: uaSigned,
		Invocation:        invSigned,
		Attenuations:      atts,
	}
	b64, err := macaroon.EncodeMacaroon(full)
	if err != nil {
		t.Fatalf("EncodeMacaroon: %v", err)
	}
	if b64 != fx.Expected.MacaroonB64url {
		t.Errorf("macaroon_b64url mismatch\n  got:  %s\n  want: %s",
			b64, fx.Expected.MacaroonB64url)
	}

	// ─── 6. Spot-check canonical JSON matches too ────────────────
	// (b64 equality already implies this, but a separate assertion
	// makes the diff readable when something goes wrong.)
	canonical, err := macaroon.JCS(full)
	if err != nil {
		t.Fatalf("JCS(full): %v", err)
	}
	if got := string(canonical); got != fx.Expected.MacaroonCanonicalJSON {
		t.Errorf("macaroon_canonical_json mismatch\n  got:  %s\n  want: %s",
			got, fx.Expected.MacaroonCanonicalJSON)
	}
}

// indexKey maps an integer key_index to its JSON-object string key
// in org_privs_hex (e.g. 0 → "0"). Multisig fixtures store privkeys
// keyed by their stringified position.
func indexKey(i int) string {
	if i < 0 || i > 9 {
		// Phase 4 multisig fixtures use single-digit indices; if this
		// ever needs to grow, use strconv.Itoa.
		return ""
	}
	return string(rune('0' + i))
}
