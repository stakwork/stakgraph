package adminapi

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
)

// bearerDo issues an arbitrary-method request with the provisioning
// bearer. Body is optional JSON.
func bearerDo(t *testing.T, srv *httptest.Server, method, path, body string) *http.Response {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, _ := http.NewRequest(method, srv.URL+path, rd)
	req.Header.Set("Authorization", "Bearer "+testToken)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func decodeBody(t *testing.T, resp *http.Response, into any) {
	t.Helper()
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(into); err != nil {
		t.Fatalf("decode: %v", err)
	}
}

func TestRunKill_RoundTrip(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)

	resp := bearerDo(t, srv, http.MethodPost, "/_plugin/runs/r_1/kill", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("kill: want 200, got %d", resp.StatusCode)
	}
	var kr KillRunResponse
	decodeBody(t, resp, &kr)
	if kr.RunID != "r_1" || kr.KilledAt == "" {
		t.Fatalf("kill response: %+v", kr)
	}
	if !mr.Exists("bifrost:kill:r_1") {
		t.Fatal("kill key not written")
	}
	if ttl := mr.TTL("bifrost:kill:r_1"); ttl != time.Hour {
		t.Fatalf("ttl: %v", ttl)
	}

	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/runs/r_1/state", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("state: want 200, got %d", resp.StatusCode)
	}
	var st RunStateResponse
	decodeBody(t, resp, &st)
	if !st.Killed || st.RunID != "r_1" {
		t.Fatalf("state after kill: %+v", st)
	}

	resp = bearerDo(t, srv, http.MethodDelete, "/_plugin/runs/r_1/kill", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("unkill: want 204, got %d", resp.StatusCode)
	}
	if mr.Exists("bifrost:kill:r_1") {
		t.Fatal("kill key not deleted")
	}
}

func TestRunState_SeededAccumulators(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)
	mr.HSet("bifrost:cost:run:r_2", "total", "0.75")
	mr.HSet("bifrost:steps:run:r_2", "total", "3")
	mr.Lpush("bifrost:tools:run:r_2", "bash")
	mr.SetTTL("bifrost:cost:run:r_2", 2*time.Hour)

	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/runs/r_2/state", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var st RunStateResponse
	decodeBody(t, resp, &st)
	if st.CostUSD != 0.75 || st.Steps != 3 || st.Killed {
		t.Fatalf("state: %+v", st)
	}
	if len(st.Tools) != 1 || st.Tools[0] != "bash" {
		t.Fatalf("tools: %v", st.Tools)
	}
	if st.TTLSeconds != 7200 {
		t.Fatalf("ttl: %d", st.TTLSeconds)
	}
}

func TestRunState_EmptyRunIsZeroNot404(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/runs/r_never/state", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var st RunStateResponse
	decodeBody(t, resp, &st)
	if st.Tools == nil {
		t.Fatal("tools must serialize as [] not null")
	}
	if st.TTLSeconds != -2 {
		t.Fatalf("ttl sentinel: %d", st.TTLSeconds)
	}
}

func TestAgentKill_RoundTrip(t *testing.T) {
	srv, mr := newBudgetTestServer(t, map[string]auth.AgentBudget{
		"coder": {CapUSD: 5, Window: "1d"},
	})
	today := time.Now().UTC().Format("2006-01-02")
	mr.HSet("bifrost:cost:agent:coder:"+today, "total", "1.5")

	resp := bearerDo(t, srv, http.MethodPost, "/_plugin/agents/coder/kill", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("kill: want 200, got %d", resp.StatusCode)
	}
	var kr KillAgentResponse
	decodeBody(t, resp, &kr)
	if kr.AgentName != "coder" {
		t.Fatalf("kill response: %+v", kr)
	}
	if ttl := mr.TTL("bifrost:kill:agent:coder"); ttl != 24*time.Hour {
		t.Fatalf("ttl: %v", ttl)
	}

	// ?window=1h is ignored because the configured cap pins 1d.
	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/agents/coder/state?window=1h", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("state: want 200, got %d", resp.StatusCode)
	}
	var st AgentStateResponse
	decodeBody(t, resp, &st)
	if !st.Killed || st.Window != "1d" || st.BucketKey != today || st.CurrentSpendUSD != 1.5 {
		t.Fatalf("state: %+v", st)
	}
	if st.ConfiguredCapUSD == nil || *st.ConfiguredCapUSD != 5 {
		t.Fatalf("cap: %+v", st.ConfiguredCapUSD)
	}

	resp = bearerDo(t, srv, http.MethodDelete, "/_plugin/agents/coder/kill", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("unkill: want 204, got %d", resp.StatusCode)
	}
	if mr.Exists("bifrost:kill:agent:coder") {
		t.Fatal("kill key not deleted")
	}
}

func TestAgentState_NoCap_WindowParam(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/agents/free/state?window=1h", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var st AgentStateResponse
	decodeBody(t, resp, &st)
	if st.Window != "1h" || st.ConfiguredCapUSD != nil || st.Killed {
		t.Fatalf("state: %+v", st)
	}

	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/agents/free/state?window=bogus", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bogus window: want 400, got %d", resp.StatusCode)
	}
}

func TestHotState_MethodsAndValidation(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	cases := []struct {
		method, path string
		want         int
	}{
		{http.MethodGet, "/_plugin/runs/r_1/kill", http.StatusMethodNotAllowed},
		{http.MethodPost, "/_plugin/runs/r_1/state", http.StatusMethodNotAllowed},
		{http.MethodGet, "/_plugin/agents/coder/kill", http.StatusMethodNotAllowed},
		{http.MethodPost, "/_plugin/agents/coder/state", http.StatusMethodNotAllowed},
		{http.MethodPost, "/_plugin/runs/has%20space/kill", http.StatusBadRequest},
		{http.MethodPost, "/_plugin/runs//kill", http.StatusNotFound},
	}
	for _, c := range cases {
		resp := bearerDo(t, srv, c.method, c.path, "")
		resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s %s: want %d, got %d", c.method, c.path, c.want, resp.StatusCode)
		}
	}
}

func TestHotState_RedisUnconfigured503(t *testing.T) {
	redisclient.SetClientForTest(nil)
	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser: "admin", adminPass: "hunter2", provisioningToken: testToken,
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	for _, c := range []struct{ method, path string }{
		{http.MethodPost, "/_plugin/runs/r_1/kill"},
		{http.MethodGet, "/_plugin/runs/r_1/state"},
		{http.MethodPost, "/_plugin/agents/coder/kill"},
		{http.MethodGet, "/_plugin/agents/coder/state"},
		{http.MethodPost, "/_plugin/revoke/nonce/aaaa000000000000000000000000aaaa"},
	} {
		resp := bearerDo(t, srv, c.method, c.path, "")
		resp.Body.Close()
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("%s %s: want 503, got %d", c.method, c.path, resp.StatusCode)
		}
	}
}

func TestHotState_RequiresAuth(t *testing.T) {
	srv, _ := newBudgetTestServer(t, nil)
	for _, c := range []struct{ method, path string }{
		{http.MethodPost, "/_plugin/runs/r_1/kill"},
		{http.MethodGet, "/_plugin/runs/r_1/state"},
		{http.MethodPost, "/_plugin/agents/coder/kill"},
		{http.MethodPost, "/_plugin/revoke/nonce/aaaa000000000000000000000000aaaa"},
	} {
		req, _ := http.NewRequest(c.method, srv.URL+c.path, nil)
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s %s: want 401, got %d", c.method, c.path, resp.StatusCode)
		}
	}
}

func TestRevokeNonce_RoundTrip(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)
	const nonce = "bbbb000000000000000000000000bbbb"
	exp := time.Now().UTC().Add(3 * time.Hour).Format(time.RFC3339)

	resp := bearerDo(t, srv, http.MethodPost, "/_plugin/revoke/nonce/"+nonce, `{"exp":"`+exp+`"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("revoke: want 200, got %d", resp.StatusCode)
	}
	var rr RevokeNonceResponse
	decodeBody(t, resp, &rr)
	if rr.Nonce != nonce || rr.ExpiresAt == "" {
		t.Fatalf("response: %+v", rr)
	}
	if !mr.Exists("bifrost:revoke:" + nonce) {
		t.Fatal("tombstone not written")
	}
	// clamp(exp - now + 1h) ≈ 4h.
	if ttl := mr.TTL("bifrost:revoke:" + nonce); ttl < 3*time.Hour+59*time.Minute || ttl > 4*time.Hour+time.Minute {
		t.Fatalf("ttl: %v", ttl)
	}

	resp = bearerDo(t, srv, http.MethodDelete, "/_plugin/revoke/nonce/"+nonce, "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("unrevoke: want 204, got %d", resp.StatusCode)
	}
	if mr.Exists("bifrost:revoke:" + nonce) {
		t.Fatal("tombstone not deleted")
	}
}

func TestRevokeNonce_DefaultsToCeiling_And_Validates(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)
	const nonce = "cccc000000000000000000000000cccc"

	resp := bearerDo(t, srv, http.MethodPost, "/_plugin/revoke/nonce/"+nonce, "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("no body: want 200, got %d", resp.StatusCode)
	}
	if ttl := mr.TTL("bifrost:revoke:" + nonce); ttl != 7*24*time.Hour {
		t.Fatalf("default ttl should be the 7d ceiling, got %v", ttl)
	}

	resp = bearerDo(t, srv, http.MethodPost, "/_plugin/revoke/nonce/not-hex", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad nonce: want 400, got %d", resp.StatusCode)
	}
	resp = bearerDo(t, srv, http.MethodPost, "/_plugin/revoke/nonce/"+nonce, `{"exp":"yesterday"}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad exp: want 400, got %d", resp.StatusCode)
	}
}

func TestRevokeUser_RoundTrip(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)

	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/revoke/user/u_bob", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("unset cutoff: want 404, got %d", resp.StatusCode)
	}

	resp = bearerDo(t, srv, http.MethodPut, "/_plugin/revoke/user/u_bob", `{"before":"2026-09-01T00:00:00Z"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("set: want 200, got %d", resp.StatusCode)
	}
	var ur RevokeUserResponse
	decodeBody(t, resp, &ur)
	if ur.UserID != "u_bob" || ur.Before != "2026-09-01T00:00:00Z" {
		t.Fatalf("response: %+v", ur)
	}
	if got, _ := mr.Get("bifrost:revoke_user_before:u_bob"); got != "2026-09-01T00:00:00Z" {
		t.Fatalf("stored cutoff: %q", got)
	}

	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/revoke/user/u_bob", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("get: want 200, got %d", resp.StatusCode)
	}
	decodeBody(t, resp, &ur)
	if ur.Before != "2026-09-01T00:00:00Z" {
		t.Fatalf("get response: %+v", ur)
	}

	resp = bearerDo(t, srv, http.MethodDelete, "/_plugin/revoke/user/u_bob", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("clear: want 204, got %d", resp.StatusCode)
	}
	if mr.Exists("bifrost:revoke_user_before:u_bob") {
		t.Fatal("cutoff not cleared")
	}
}

func TestRevokeUser_DefaultsToNow(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)
	before := time.Now().UTC()
	resp := bearerDo(t, srv, http.MethodPut, "/_plugin/revoke/user/u_now", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	raw, _ := mr.Get("bifrost:revoke_user_before:u_now")
	got, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		t.Fatalf("stored cutoff not RFC3339: %q", raw)
	}
	if got.Before(before.Truncate(time.Second)) || got.After(time.Now().Add(time.Minute)) {
		t.Fatalf("cutoff %v not ≈ now", got)
	}
}
