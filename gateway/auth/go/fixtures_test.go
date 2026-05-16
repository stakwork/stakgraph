package macaroon_test

// Cross-language byte-equivalence test (Go side).
//
// Loads every gateway/auth/fixtures/*.json, asserts that this Go
// package reproduces every deterministic intermediate value the TS
// package recorded, and verifies the assembled macaroon end-to-end.
//
// We don't re-sign on the Go side (signing is the TS package's job in
// production; the Go side is pure verifier). What we DO assert,
// byte-for-byte:
//
//   1. JCS of (ua minus org_sig) → expected.ua_signing_bytes_hex
//   2. JCS of (inv minus user_sig) → expected.inv_signing_bytes_hex
//   3. JCS of (ua minus org_sig) as string → expected.ua_canonical_json
//   4. JCS of (inv minus user_sig) as string → expected.inv_canonical_json
//   5. For each attenuation: JCS of caveats, HMAC input, HMAC output
//      → expected.attenuation_hmac_inputs[i].*
//   6. JCS of the full macaroon → expected.macaroon_canonical_json
//   7. base64url of (6) → expected.macaroon_b64url
//   8. Verify(macaroon_b64url, policy, now) → expected.claims
//
// If TS and Go agree on these, they agree on the wire format.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

const fixturesRelPath = "../fixtures"

// fixtureFile is the on-disk shape of one fixture. Mirrors the TS
// regenerate-fixtures.ts script's output exactly.
type fixtureFile struct {
	Description string `json:"description"`
	Inputs      struct {
		OrgID       string             `json:"org_id"`
		OrgPrivHex  string             `json:"org_priv_hex,omitempty"`
		OrgPrivsHex map[string]string  `json:"org_privs_hex,omitempty"`
		Signers     []int              `json:"participating_signers,omitempty"`
		UserPrivHex string             `json:"user_priv_hex"`
		Policy      macaroon.Policy    `json:"policy"`
		UAUnsigned  json.RawMessage    `json:"ua_unsigned"`
		InvUnsigned json.RawMessage    `json:"inv_unsigned"`
		AttsUnsigned []json.RawMessage `json:"atts_unsigned"`
	} `json:"inputs"`
	Expected struct {
		UASigningBytesHex      string `json:"ua_signing_bytes_hex"`
		InvSigningBytesHex     string `json:"inv_signing_bytes_hex"`
		UACanonicalJSON        string `json:"ua_canonical_json"`
		InvCanonicalJSON       string `json:"inv_canonical_json"`
		MacaroonCanonicalJSON  string `json:"macaroon_canonical_json"`
		MacaroonB64url         string `json:"macaroon_b64url"`
		AttenuationHmacInputs  []struct {
			PrevSigHex            string `json:"prev_sig_hex"`
			CaveatsCanonicalJSON  string `json:"caveats_canonical_json"`
			HmacInputHex          string `json:"hmac_input_hex"`
			HmacOutputHex         string `json:"hmac_output_hex"`
		} `json:"attenuation_hmac_inputs"`
		Claims struct {
			OrgID            string   `json:"org_id"`
			UserID           string   `json:"user_id"`
			Workspace        string   `json:"workspace"`
			AgentName        string   `json:"agent_name"`
			RunID            string   `json:"run_id"`
			EffectiveCaveats struct {
				Agents     []string `json:"agents"`
				MaxCostUSD float64  `json:"max_cost_usd"`
				MaxSteps   int      `json:"max_steps"`
				Exp        string   `json:"exp"`
			} `json:"effective_caveats"`
			Nonces []string `json:"nonces"`
			IAT    string   `json:"iat"`
		} `json:"claims"`
		VerifyResult string `json:"verify_result"`
	} `json:"expected"`
}

func listFixtures(t *testing.T) []string {
	entries, err := os.ReadDir(fixturesRelPath)
	if err != nil {
		t.Fatalf("read fixtures dir: %v", err)
	}
	var out []string
	for _, e := range entries {
		n := e.Name()
		if len(n) >= 7 && n[2] == '-' && filepath.Ext(n) == ".json" {
			out = append(out, n)
		}
	}
	sort.Strings(out)
	return out
}

func loadFixture(t *testing.T, name string) *fixtureFile {
	raw, err := os.ReadFile(filepath.Join(fixturesRelPath, name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	var fx fixtureFile
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture %s: %v", name, err)
	}
	return &fx
}

func TestFixtures(t *testing.T) {
	files := listFixtures(t)
	if len(files) == 0 {
		t.Fatal("no fixture files found")
	}
	for _, f := range files {
		f := f
		t.Run(f, func(t *testing.T) {
			runFixture(t, f)
		})
	}
}

func runFixture(t *testing.T, name string) {
	fx := loadFixture(t, name)

	// ─── 1+3. JCS of (ua minus org_sig) ──────────────────────────
	uaSignBytes, err := jcsStripFromRaw(fx.Inputs.UAUnsigned, "")
	// ua_unsigned does not contain org_sig — JCS it directly.
	if err != nil {
		t.Fatalf("jcs ua_unsigned: %v", err)
	}
	if got := macaroon.BytesToHex(uaSignBytes); got != fx.Expected.UASigningBytesHex {
		t.Errorf("ua_signing_bytes_hex mismatch\n  got:  %s\n  want: %s", got, fx.Expected.UASigningBytesHex)
	}
	if got := string(uaSignBytes); got != fx.Expected.UACanonicalJSON {
		t.Errorf("ua_canonical_json mismatch\n  got:  %s\n  want: %s", got, fx.Expected.UACanonicalJSON)
	}

	// ─── 2+4. JCS of (inv minus user_sig) ────────────────────────
	invSignBytes, err := jcsStripFromRaw(fx.Inputs.InvUnsigned, "")
	if err != nil {
		t.Fatalf("jcs inv_unsigned: %v", err)
	}
	if got := macaroon.BytesToHex(invSignBytes); got != fx.Expected.InvSigningBytesHex {
		t.Errorf("inv_signing_bytes_hex mismatch\n  got:  %s\n  want: %s", got, fx.Expected.InvSigningBytesHex)
	}
	if got := string(invSignBytes); got != fx.Expected.InvCanonicalJSON {
		t.Errorf("inv_canonical_json mismatch\n  got:  %s\n  want: %s", got, fx.Expected.InvCanonicalJSON)
	}

	// ─── 5. attenuation HMAC inputs ──────────────────────────────
	// To reproduce these we need the signed macaroon's user_sig.sig
	// (for the first link's prev_sig) and each subsequent hmac. We
	// pull these from the assembled macaroon in expected.MacaroonB64url.
	signed, err := decodeMacaroonForTest(fx.Expected.MacaroonB64url)
	if err != nil {
		t.Fatalf("decode assembled macaroon: %v", err)
	}
	prevSig, err := macaroon.HexToBytes(signed.Invocation.UserSig.Sig)
	if err != nil {
		t.Fatalf("decode invocation.user_sig.sig: %v", err)
	}
	for i, att := range signed.Attenuations {
		exp := fx.Expected.AttenuationHmacInputs[i]
		if got := macaroon.BytesToHex(prevSig); got != exp.PrevSigHex {
			t.Errorf("att[%d].prev_sig_hex mismatch\n  got:  %s\n  want: %s", i, got, exp.PrevSigHex)
		}
		caveatsCanonical, err := macaroon.JCS(att.Caveats)
		if err != nil {
			t.Fatalf("att[%d] jcs caveats: %v", i, err)
		}
		if got := string(caveatsCanonical); got != exp.CaveatsCanonicalJSON {
			t.Errorf("att[%d].caveats_canonical_json mismatch\n  got:  %s\n  want: %s", i, got, exp.CaveatsCanonicalJSON)
		}
		if got := macaroon.BytesToHex(caveatsCanonical); got != exp.HmacInputHex {
			t.Errorf("att[%d].hmac_input_hex mismatch\n  got:  %s\n  want: %s", i, got, exp.HmacInputHex)
		}
		hmacOut, err := macaroon.ComputeAttenuationHMAC(prevSig, att.Caveats)
		if err != nil {
			t.Fatalf("att[%d] hmac: %v", i, err)
		}
		if got := macaroon.BytesToHex(hmacOut); got != exp.HmacOutputHex {
			t.Errorf("att[%d].hmac_output_hex mismatch\n  got:  %s\n  want: %s", i, got, exp.HmacOutputHex)
		}
		prevSig, err = macaroon.HexToBytes(att.HMAC)
		if err != nil {
			t.Fatalf("att[%d] decode hmac: %v", i, err)
		}
	}

	// ─── 6+7. full macaroon JCS + base64url ──────────────────────
	macaroonCanonical, err := macaroon.JCS(signed)
	if err != nil {
		t.Fatalf("jcs full macaroon: %v", err)
	}
	if got := string(macaroonCanonical); got != fx.Expected.MacaroonCanonicalJSON {
		t.Errorf("macaroon_canonical_json mismatch\n  got:  %s\n  want: %s", got, fx.Expected.MacaroonCanonicalJSON)
	}
	if got := macaroon.BytesToBase64url(macaroonCanonical); got != fx.Expected.MacaroonB64url {
		t.Errorf("macaroon_b64url mismatch\n  got:  %s\n  want: %s", got, fx.Expected.MacaroonB64url)
	}

	// ─── 8. Verify end-to-end ────────────────────────────────────
	// Pick a "now" that's after invocation.iat but before exp. We use
	// invocation.iat + 1s, same as the TS test.
	invIat, err := time.Parse(time.RFC3339, signed.Invocation.IAT)
	if err != nil {
		t.Fatalf("parse invocation.iat: %v", err)
	}
	claims, err := macaroon.Verify(fx.Expected.MacaroonB64url, fx.Inputs.Policy, invIat.Add(time.Second))
	if err != nil {
		t.Fatalf("Verify failed: %v", err)
	}
	if claims.OrgID != fx.Expected.Claims.OrgID {
		t.Errorf("claims.OrgID: got %q want %q", claims.OrgID, fx.Expected.Claims.OrgID)
	}
	if claims.UserID != fx.Expected.Claims.UserID {
		t.Errorf("claims.UserID: got %q want %q", claims.UserID, fx.Expected.Claims.UserID)
	}
	if claims.Workspace != fx.Expected.Claims.Workspace {
		t.Errorf("claims.Workspace: got %q want %q", claims.Workspace, fx.Expected.Claims.Workspace)
	}
	if claims.AgentName != fx.Expected.Claims.AgentName {
		t.Errorf("claims.AgentName: got %q want %q", claims.AgentName, fx.Expected.Claims.AgentName)
	}
	if claims.RunID != fx.Expected.Claims.RunID {
		t.Errorf("claims.RunID: got %q want %q", claims.RunID, fx.Expected.Claims.RunID)
	}
	if claims.IAT != fx.Expected.Claims.IAT {
		t.Errorf("claims.IAT: got %q want %q", claims.IAT, fx.Expected.Claims.IAT)
	}
	if !reflect.DeepEqual(claims.Nonces, fx.Expected.Claims.Nonces) {
		t.Errorf("claims.Nonces: got %v want %v", claims.Nonces, fx.Expected.Claims.Nonces)
	}
	if !reflect.DeepEqual(claims.EffectiveCaveats.Agents, fx.Expected.Claims.EffectiveCaveats.Agents) {
		t.Errorf("claims.EffectiveCaveats.Agents: got %v want %v",
			claims.EffectiveCaveats.Agents, fx.Expected.Claims.EffectiveCaveats.Agents)
	}
	if claims.EffectiveCaveats.MaxCostUSD != fx.Expected.Claims.EffectiveCaveats.MaxCostUSD {
		t.Errorf("claims.EffectiveCaveats.MaxCostUSD: got %v want %v",
			claims.EffectiveCaveats.MaxCostUSD, fx.Expected.Claims.EffectiveCaveats.MaxCostUSD)
	}
	if claims.EffectiveCaveats.MaxSteps != fx.Expected.Claims.EffectiveCaveats.MaxSteps {
		t.Errorf("claims.EffectiveCaveats.MaxSteps: got %v want %v",
			claims.EffectiveCaveats.MaxSteps, fx.Expected.Claims.EffectiveCaveats.MaxSteps)
	}
	if claims.EffectiveCaveats.Exp != fx.Expected.Claims.EffectiveCaveats.Exp {
		t.Errorf("claims.EffectiveCaveats.Exp: got %q want %q",
			claims.EffectiveCaveats.Exp, fx.Expected.Claims.EffectiveCaveats.Exp)
	}
}

// jcsStripFromRaw runs JCS on raw JSON. If field is non-empty, that
// top-level field is removed first. Used to JCS the unsigned
// ua/invocation objects from the fixture (which already lack their
// sig field, so we pass "").
func jcsStripFromRaw(raw json.RawMessage, field string) ([]byte, error) {
	var asMap map[string]json.RawMessage
	if err := json.Unmarshal(raw, &asMap); err != nil {
		return nil, err
	}
	if field != "" {
		delete(asMap, field)
	}
	repacked, err := json.Marshal(asMap)
	if err != nil {
		return nil, err
	}
	// Reuse the package's JCS by unmarshalling to any and calling JCS.
	var v any
	if err := json.Unmarshal(repacked, &v); err != nil {
		return nil, err
	}
	return macaroon.JCS(v)
}

// decodeMacaroonForTest base64url-decodes and JSON-parses a macaroon
// for test inspection (we need to read user_sig.sig and attenuation
// hmacs to drive the HMAC chain re-computation).
func decodeMacaroonForTest(b64 string) (*macaroon.Macaroon, error) {
	raw, err := macaroon.Base64urlToBytes(b64)
	if err != nil {
		return nil, err
	}
	var m macaroon.Macaroon
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return &m, nil
}
