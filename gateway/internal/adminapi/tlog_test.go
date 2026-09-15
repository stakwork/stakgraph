package adminapi

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/tlog"
)

func openTestTlog(t *testing.T) *tlog.Log {
	t.Helper()
	l, err := tlog.Open(filepath.Join(t.TempDir(), "leaves.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	tlog.SetDefaultForTest(l)
	t.Cleanup(func() {
		tlog.SetDefaultForTest(nil)
		_ = l.Close()
	})
	return l
}

func newTlogTestServer(t *testing.T) (*httptest.Server, *tlog.Log) {
	t.Helper()
	l := openTestTlog(t)
	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         "admin",
		adminPass:         "secret",
		provisioningToken: testToken,
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, l
}

func strPtr(s string) *string { return &s }

func tlogLeaf(i int, agent string) tlog.Leaf {
	return tlog.Leaf{
		OrgID: "org_acme", UserID: "u_alice", RunID: "r_1", Agent: agent,
		MacaroonSHA256: strings.Repeat("ab", 32),
		RequestSHA256:  strPtr(strings.Repeat("cd", 32)),
		Model:          "claude-sonnet-5", Provider: "anthropic",
		PromptTokens: 10 + i, CompletionTokens: i, CostUSD: 0.001 * float64(i+1),
		Status: tlog.StatusOK,
	}
}

func appendLeaves(t *testing.T, l *tlog.Log, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if _, err := l.Append(tlogLeaf(i, "coder")); err != nil {
			t.Fatal(err)
		}
	}
}

func getSth(t *testing.T, srv *httptest.Server, query string, bearer string) (*http.Response, []byte) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, srv.URL+tlogSthPath+query, nil)
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, body
}

func decodePage(t *testing.T, body []byte) tlog.Page {
	t.Helper()
	var page tlog.Page
	if err := json.Unmarshal(body, &page); err != nil {
		t.Fatalf("decode page: %v\n%s", err, body)
	}
	return page
}

func TestTlogSth_RequiresBearer(t *testing.T) {
	srv, _ := newTlogTestServer(t)
	if resp, _ := getSth(t, srv, "", ""); resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no auth: want 401, got %d", resp.StatusCode)
	}
	if resp, _ := getSth(t, srv, "", "wrong-token"); resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad bearer: want 401, got %d", resp.StatusCode)
	}
	if resp, _ := getSth(t, srv, "", testToken); resp.StatusCode != http.StatusOK {
		t.Fatalf("bearer: want 200, got %d", resp.StatusCode)
	}
}

// A dashboard session must not be able to pull material the witness
// then org-signs: the route is bearerOnly, not cookieOrBearer.
func TestTlogSth_RejectsDashboardCookie(t *testing.T) {
	openTestTlog(t)
	srv, client := newAuthTestServer(t)

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("login: %d", resp.StatusCode)
	}
	// The cookie works on a cookie-or-bearer route…
	resp, err = client.Get(srv.URL + "/_plugin/me")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/me with cookie: %d", resp.StatusCode)
	}
	// …and not on the witness route.
	resp, err = client.Get(srv.URL + tlogSthPath)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("tlog/sth with cookie: want 401, got %d", resp.StatusCode)
	}
}

func TestTlogSth_EmptyTree(t *testing.T) {
	srv, l := newTlogTestServer(t)
	resp, body := getSth(t, srv, "", testToken)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("content-type = %q", ct)
	}
	// Empty collections are [] on the wire, never null.
	for _, want := range []string{`"consistency_proof":[]`, `"leaves":[]`, `"next":0`} {
		if !bytes.Contains(body, []byte(want)) {
			t.Errorf("body lacks %s: %s", want, body)
		}
	}
	page := decodePage(t, body)
	empty := tlog.EmptyRoot()
	if page.STH.TreeSize != 0 || page.STH.RootHash != hex.EncodeToString(empty[:]) {
		t.Errorf("empty sth = %+v", page.STH)
	}
	if page.LogPubkey != l.PubkeyHex() {
		t.Errorf("log_pubkey = %s", page.LogPubkey)
	}
	if !tlog.VerifySTH(page.STH, page.LogPubkey) {
		t.Error("empty STH does not verify")
	}
}

func TestTlogSth_PagesAndProofsVerify(t *testing.T) {
	srv, l := newTlogTestServer(t)
	appendLeaves(t, l, 3)

	// since absent ⇒ 0.
	resp, body := getSth(t, srv, "", testToken)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	page := decodePage(t, body)
	if page.STH.TreeSize != 3 || page.Next != 3 || len(page.Leaves) != 3 || len(page.ConsistencyProof) != 0 {
		t.Fatalf("page(0) = size %d next %d leaves %d proof %d", page.STH.TreeSize, page.Next, len(page.Leaves), len(page.ConsistencyProof))
	}
	if !tlog.VerifySTH(page.STH, page.LogPubkey) {
		t.Error("STH does not verify")
	}
	// The served leaves reproduce the served root.
	var hashes []tlog.Hash
	for _, raw := range page.Leaves {
		hashes = append(hashes, tlog.LeafHash(raw))
	}
	if root := tlog.RootFromLeafHashes(hashes); hex.EncodeToString(root[:]) != page.STH.RootHash {
		t.Error("served leaves do not reproduce the served root")
	}

	// since=1: two leaves and a proof from 1 to 3.
	resp, body = getSth(t, srv, "?since=1", testToken)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	page = decodePage(t, body)
	if page.Next != 3 || len(page.Leaves) != 2 {
		t.Fatalf("page(1) = next %d leaves %d", page.Next, len(page.Leaves))
	}
	var proof []tlog.Hash
	for _, s := range page.ConsistencyProof {
		b, err := hex.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		var h tlog.Hash
		copy(h[:], b)
		proof = append(proof, h)
	}
	old, _ := l.RootAt(1)
	if !tlog.VerifyConsistency(1, 3, proof, old, tlog.RootFromLeafHashes(hashes)) {
		t.Error("served consistency proof does not verify")
	}

	// since == tree_size: caught up.
	_, body = getSth(t, srv, "?since=3", testToken)
	page = decodePage(t, body)
	if page.Next != 3 || len(page.Leaves) != 0 || len(page.ConsistencyProof) != 0 {
		t.Fatalf("page(3) = %+v", page)
	}
}

func TestTlogSth_BadSince(t *testing.T) {
	srv, _ := newTlogTestServer(t)
	for _, q := range []string{"?since=-1", "?since=abc", "?since=1.5", "?since=%2B1", "?since=0x10", "?since=%201"} {
		if resp, body := getSth(t, srv, q, testToken); resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: want 400, got %d (%s)", q, resp.StatusCode, body)
		}
	}
}

func TestTlogSth_SinceAheadIs409(t *testing.T) {
	srv, l := newTlogTestServer(t)
	appendLeaves(t, l, 3)
	resp, body := getSth(t, srv, "?since=4", testToken)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("want 409, got %d: %s", resp.StatusCode, body)
	}
	var ahead TlogAheadResponse
	if err := json.Unmarshal(body, &ahead); err != nil {
		t.Fatal(err)
	}
	if ahead.Error != "since_ahead_of_tree" || ahead.Since != 4 || ahead.TreeSize != 3 {
		t.Fatalf("409 body = %+v", ahead)
	}
}

func TestTlogSth_MethodNotAllowed(t *testing.T) {
	srv, _ := newTlogTestServer(t)
	resp := bearerDo(t, srv, http.MethodPost, tlogSthPath, "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed || resp.Header.Get("Allow") != "GET" {
		t.Fatalf("POST: %d allow=%q", resp.StatusCode, resp.Header.Get("Allow"))
	}
}

func TestTlogSth_UnavailableWhenNotInitializedOrDisabled(t *testing.T) {
	srv, _ := newTlogTestServer(t)

	tlog.SetDefaultForTest(nil)
	resp, body := getSth(t, srv, "", testToken)
	if resp.StatusCode != http.StatusServiceUnavailable || !bytes.Contains(body, []byte("tlog_unavailable")) {
		t.Fatalf("uninitialized: %d %s", resp.StatusCode, body)
	}

	// A log that failed to come up is installed disabled, and the
	// route says why.
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	disabled, err := tlog.Open(filepath.Join(blocker, "leaves.jsonl"))
	if err == nil {
		t.Fatal("expected open to fail")
	}
	tlog.SetDefaultForTest(disabled)
	resp, body = getSth(t, srv, "", testToken)
	if resp.StatusCode != http.StatusServiceUnavailable || !bytes.Contains(body, []byte("log disabled")) {
		t.Fatalf("disabled: %d %s", resp.StatusCode, body)
	}
}

func TestTlogSth_StableAtSameSize_ResignedAfterAppend(t *testing.T) {
	srv, l := newTlogTestServer(t)
	appendLeaves(t, l, 2)
	_, a := getSth(t, srv, "", testToken)
	_, b := getSth(t, srv, "", testToken)
	if !bytes.Equal(a, b) {
		t.Fatalf("two polls at the same size differ:\n%s\n%s", a, b)
	}
	appendLeaves(t, l, 1)
	_, c := getSth(t, srv, "", testToken)
	pa, pc := decodePage(t, a), decodePage(t, c)
	if pc.STH.TreeSize != 3 || pc.STH.Sig == pa.STH.Sig || pc.STH.RootHash == pa.STH.RootHash {
		t.Fatalf("head not re-signed after append: %+v vs %+v", pa.STH, pc.STH)
	}
}

// The leaves on the wire are the bytes that were hashed: no HTML
// escaping of <, >, & on the way out.
func TestTlogSth_LeavesServedVerbatim(t *testing.T) {
	srv, l := newTlogTestServer(t)
	if _, err := l.Append(tlogLeaf(0, `a<&>"b"`)); err != nil {
		t.Fatal(err)
	}
	_, body := getSth(t, srv, "", testToken)
	if !bytes.Contains(body, []byte(`"agent":"a<&>\"b\""`)) {
		t.Fatalf("leaf not served verbatim: %s", body)
	}
	page := decodePage(t, body)
	h := tlog.LeafHash(page.Leaves[0])
	if hex.EncodeToString(h[:]) != page.STH.RootHash {
		t.Fatal("single served leaf does not hash to the root")
	}
}
