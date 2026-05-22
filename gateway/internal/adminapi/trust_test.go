package adminapi

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"

	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

// Test fixtures are duplicated from trust/registry_test.go so the
// admin tests don't depend on the trust package's test binary
// (which Go won't link across packages). The values are identical.
const (
	testToken = "test-bearer-token-do-not-use-in-prod"
)

var (
	testPriv1 = mustHex("1111111111111111111111111111111111111111111111111111111111111111")
	testPriv2 = mustHex("2222222222222222222222222222222222222222222222222222222222222222")
	testPub1  = pubFromPriv(testPriv1)
	testPub2  = pubFromPriv(testPriv2)
)

func mustHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func pubFromPriv(priv []byte) string {
	p := secp256k1.PrivKeyFromBytes(priv)
	return hex.EncodeToString(p.PubKey().SerializeCompressed())
}

// newTestServer wires the same routeDeps registerRoutes uses in
// production, but against an httptest server so we can drive real
// requests. The registry persists to a temp file under t.TempDir().
func newTestServer(t *testing.T) (*httptest.Server, *trust.Registry) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("BIFROST_PLUGIN_TRUST_PATH", filepath.Join(dir, "trust.json"))
	t.Setenv("BIFROST_PLUGIN_TRUST", "")
	t.Setenv("BIFROST_PLUGIN_TRUST_FILE", "")
	t.Setenv("BIFROST_PLUGIN_TRUST_RECONCILE", "")

	reg, err := trust.LoadFromEnv()
	if err != nil {
		t.Fatalf("LoadFromEnv: %v", err)
	}

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         "admin",
		adminPass:         "secret",
		provisioningToken: testToken,
		trust:             reg,
	})
	return httptest.NewServer(mux), reg
}

func do(t *testing.T, srv *httptest.Server, method, path string, body any, authed bool) *http.Response {
	t.Helper()
	var rdr *bytes.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		rdr = bytes.NewReader(raw)
	} else {
		rdr = bytes.NewReader(nil)
	}
	req, err := http.NewRequest(method, srv.URL+path, rdr)
	if err != nil {
		t.Fatal(err)
	}
	if authed {
		req.Header.Set("Authorization", "Bearer "+testToken)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func decode(t *testing.T, resp *http.Response, into any) {
	t.Helper()
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(into); err != nil {
		t.Fatalf("decode: %v", err)
	}
}

func sampleOrg(orgID, pubkey string) map[string]any {
	return map[string]any{
		"org_id":                  orgID,
		"pubkey":                  pubkey,
		"issuer_url":              "https://hive.example.com",
		"revocation_poll_seconds": 60,
	}
}

// ─── auth ─────────────────────────────────────────────────────────────

func TestTrust_RequiresBearer(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	for _, path := range []string{
		"/_plugin/trust/status",
		"/_plugin/trust",
		"/_plugin/trust/org_acme",
		"/_plugin/trust/org_acme/rotate",
	} {
		resp := do(t, srv, http.MethodGet, path, nil, false)
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s without auth: want 401, got %d", path, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

// ─── status ───────────────────────────────────────────────────────────

func TestTrust_Status_Empty(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	resp := do(t, srv, http.MethodGet, "/_plugin/trust/status", nil, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status: %d", resp.StatusCode)
	}
	var got trust.StatusResponse
	decode(t, resp, &got)
	if got.Claimed || got.OrgCount != 0 {
		t.Fatalf("expected empty: %+v", got)
	}
}

func TestTrust_StatusAfterUpsert_Claimed(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	resp := do(t, srv, http.MethodPost, "/_plugin/trust", sampleOrg("org_acme", testPub1), true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert: %d", resp.StatusCode)
	}
	resp.Body.Close()

	resp = do(t, srv, http.MethodGet, "/_plugin/trust/status", nil, true)
	var got trust.StatusResponse
	decode(t, resp, &got)
	if !got.Claimed || got.OrgCount != 1 || got.Orgs[0] != "org_acme" {
		t.Fatalf("status after upsert: %+v", got)
	}
}

// ─── upsert / get / delete ────────────────────────────────────────────

func TestTrust_FullLifecycle(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	// POST upsert
	resp := do(t, srv, http.MethodPost, "/_plugin/trust", sampleOrg("org_acme", testPub1), true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("upsert: %d", resp.StatusCode)
	}
	resp.Body.Close()

	// GET read
	resp = do(t, srv, http.MethodGet, "/_plugin/trust/org_acme", nil, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("get: %d", resp.StatusCode)
	}
	var got trust.Org
	decode(t, resp, &got)
	if got.OrgID != "org_acme" || got.Pubkey != testPub1 {
		t.Fatalf("read mismatch: %+v", got)
	}

	// DELETE
	resp = do(t, srv, http.MethodDelete, "/_plugin/trust/org_acme", nil, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete: %d", resp.StatusCode)
	}
	resp.Body.Close()

	// GET after delete → 404
	resp = do(t, srv, http.MethodGet, "/_plugin/trust/org_acme", nil, true)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("get after delete: want 404, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTrust_Upsert_RejectsBadPubkey(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	body := sampleOrg("org_acme", "not-hex-at-all")
	resp := do(t, srv, http.MethodPost, "/_plugin/trust", body, true)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTrust_Upsert_RejectsUnknownField(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	body := sampleOrg("org_acme", testPub1)
	body["not_a_real_field"] = "oops"
	resp := do(t, srv, http.MethodPost, "/_plugin/trust", body, true)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTrust_Delete_NotFound(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	resp := do(t, srv, http.MethodDelete, "/_plugin/trust/org_missing", nil, true)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

// ─── rotate ───────────────────────────────────────────────────────────

func TestTrust_Rotate_Succeeds(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	_ = do(t, srv, http.MethodPost, "/_plugin/trust", sampleOrg("org_acme", testPub1), true)

	body := map[string]any{"new_pubkey": testPub2, "grace_seconds": 3600}
	resp := do(t, srv, http.MethodPost, "/_plugin/trust/org_acme/rotate", body, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rotate: %d", resp.StatusCode)
	}
	var got trust.RotateResponse
	decode(t, resp, &got)
	if got.ActivePubkey != testPub2 {
		t.Fatalf("active key not rotated: %s", got.ActivePubkey)
	}
	if len(got.GracePubkeys) != 1 || got.GracePubkeys[0] != testPub1 {
		t.Fatalf("grace pubkeys: %v", got.GracePubkeys)
	}
}

func TestTrust_Rotate_NotFound(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	body := map[string]any{"new_pubkey": testPub2, "grace_seconds": 60}
	resp := do(t, srv, http.MethodPost, "/_plugin/trust/missing/rotate", body, true)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

// ─── method enforcement ───────────────────────────────────────────────

func TestTrust_MethodEnforcement(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	// PUT on /_plugin/trust → 405
	resp := do(t, srv, http.MethodPut, "/_plugin/trust", sampleOrg("org_acme", testPub1), true)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", resp.StatusCode)
	}
	if !strings.Contains(resp.Header.Get("Allow"), "POST") {
		t.Fatalf("Allow header: %q", resp.Header.Get("Allow"))
	}
	resp.Body.Close()
}

// ─── phase 11: realm_id endpoint ──────────────────────────────────────

func TestTrust_RealmID_PutSetsAndStatusReturns(t *testing.T) {
	srv, reg := newTestServer(t)
	defer srv.Close()

	resp := do(t, srv, http.MethodPut, "/_plugin/trust/realm_id",
		trust.RealmIDRequest{RealmID: "w1"}, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("put realm_id: %d", resp.StatusCode)
	}
	var put trust.RealmIDResponse
	decode(t, resp, &put)
	if !put.OK || put.RealmID != "w1" {
		t.Fatalf("put response: %+v", put)
	}
	if reg.RealmID() != "w1" {
		t.Fatalf("registry not updated: %q", reg.RealmID())
	}

	// /status surfaces it.
	resp = do(t, srv, http.MethodGet, "/_plugin/trust/status", nil, true)
	var st trust.StatusResponse
	decode(t, resp, &st)
	if st.RealmID != "w1" {
		t.Fatalf("status realm_id: %+v", st)
	}
}

func TestTrust_RealmID_PutRejectsBadShape(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	resp := do(t, srv, http.MethodPut, "/_plugin/trust/realm_id",
		trust.RealmIDRequest{RealmID: "has space"}, true)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTrust_RealmID_PutClears(t *testing.T) {
	srv, reg := newTestServer(t)
	defer srv.Close()

	_ = do(t, srv, http.MethodPut, "/_plugin/trust/realm_id",
		trust.RealmIDRequest{RealmID: "w1"}, true).Body.Close()

	resp := do(t, srv, http.MethodPut, "/_plugin/trust/realm_id",
		trust.RealmIDRequest{RealmID: ""}, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("clear: %d", resp.StatusCode)
	}
	resp.Body.Close()
	if reg.RealmID() != "" {
		t.Fatalf("expected cleared, got %q", reg.RealmID())
	}
}

func TestTrust_RealmID_MethodEnforced(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	// GET is not allowed on the realm_id endpoint — its value is
	// part of /_plugin/trust/status, so a dedicated GET would just
	// duplicate that surface.
	resp := do(t, srv, http.MethodGet, "/_plugin/trust/realm_id", nil, true)
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTrust_RealmID_RequiresBearer(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()

	// PUT is a mutation → bearer-only via methodMuxedAuth (GET-only
	// goes through cookieOrBearer; everything else goes through
	// bearerOnly). Sending no auth must 401.
	resp := do(t, srv, http.MethodPut, "/_plugin/trust/realm_id",
		trust.RealmIDRequest{RealmID: "w1"}, false)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

// ─── 404 dispatch ─────────────────────────────────────────────────────

func TestTrust_UnknownSubpath_404(t *testing.T) {
	srv, _ := newTestServer(t)
	defer srv.Close()
	_ = do(t, srv, http.MethodPost, "/_plugin/trust", sampleOrg("org_acme", testPub1), true)

	// /_plugin/trust/org_acme/something_unknown
	resp := do(t, srv, http.MethodPost, "/_plugin/trust/org_acme/zzz", nil, true)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}
