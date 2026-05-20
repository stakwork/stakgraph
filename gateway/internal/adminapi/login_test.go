package adminapi

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
	"github.com/stakwork/stakgraph/gateway/internal/sessions"
)

// newAuthTestServer wires sessions + adminapi routes against a fresh
// miniredis. Returns an *httptest.Server and a cookie-jar'd
// *http.Client so tests can drive realistic login -> /me -> logout
// flows without re-encoding the cookie themselves.
func newAuthTestServer(t *testing.T) (*httptest.Server, *http.Client) {
	t.Helper()
	mr := miniredis.RunT(t)
	rc := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() {
		_ = rc.Close()
		redisclient.SetClientForTest(nil)
	})
	redisclient.SetClientForTest(rc)

	store := sessions.NewStore()
	if store == nil {
		t.Fatal("sessions.NewStore returned nil — redis wiring broken")
	}

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         "admin",
		adminPass:         "hunter2",
		provisioningToken: testToken,
		trust:             nil,
		sessions:          store,
		logstore:          nil, // not exercised in login tests
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	jar, err := cookiejar.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	return srv, &http.Client{Jar: jar}
}

func basicHeader(user, pass string) string {
	return "Basic " + base64.StdEncoding.EncodeToString([]byte(user+":"+pass))
}

// ─── /_plugin/login ───────────────────────────────────────────────────

func TestLogin_HappyPath(t *testing.T) {
	srv, client := newAuthTestServer(t)

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("login: %d", resp.StatusCode)
	}
	var body LoginResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.User != "admin" {
		t.Fatalf("user: %q", body.User)
	}
	// Cookie must be HttpOnly + Path=/_plugin. SameSite is Lax in
	// dev (httptest is plain HTTP so the request is non-secure)
	// and None in prod — both are valid; we just assert it isn't
	// Strict, which would block the iframe-embed flow.
	var sc *http.Cookie
	for _, c := range resp.Cookies() {
		if c.Name == sessionCookieName {
			sc = c
		}
	}
	if sc == nil {
		t.Fatal("no session cookie set")
	}
	if !sc.HttpOnly {
		t.Error("cookie not HttpOnly")
	}
	if sc.SameSite == http.SameSiteStrictMode {
		t.Errorf("SameSite must not be Strict (got %v); cookie has to ride iframe requests", sc.SameSite)
	}
	if sc.Path != "/_plugin" {
		t.Errorf("Path: %q", sc.Path)
	}
	if sc.MaxAge <= 0 {
		t.Errorf("MaxAge: %d", sc.MaxAge)
	}
}

func TestLogin_WrongPassword_Returns401(t *testing.T) {
	srv, client := newAuthTestServer(t)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "wrong"))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
	if len(resp.Cookies()) > 0 {
		for _, c := range resp.Cookies() {
			if c.Name == sessionCookieName && c.Value != "" {
				t.Fatal("session cookie set on failed login")
			}
		}
	}
}

func TestLogin_MissingHeader_400(t *testing.T) {
	srv, client := newAuthTestServer(t)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestLogin_RateLimit(t *testing.T) {
	srv, client := newAuthTestServer(t)
	// Hammer with wrong passwords until the limit hits.
	for i := 0; i < maxLoginAttempts; i++ {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
		req.Header.Set("Authorization", basicHeader("admin", "wrong"))
		resp, _ := client.Do(req)
		resp.Body.Close()
	}
	// The next attempt — even with the right password — must 429.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("want 429 after limit, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("missing Retry-After")
	}
}

// ─── /_plugin/me + cookie round-trip ──────────────────────────────────

func TestMe_RequiresAuth(t *testing.T) {
	srv, client := newAuthTestServer(t)
	resp, err := client.Get(srv.URL + "/_plugin/me")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
}

func TestMe_AfterLogin_ReturnsUser(t *testing.T) {
	srv, client := newAuthTestServer(t)
	// Log in (cookie jar handles the rest).
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, _ := client.Do(req)
	resp.Body.Close()

	resp, err := client.Get(srv.URL + "/_plugin/me")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/me after login: %d", resp.StatusCode)
	}
	var me MeResponse
	if err := json.NewDecoder(resp.Body).Decode(&me); err != nil {
		t.Fatal(err)
	}
	if me.User != "admin" || me.IssuedAt == 0 {
		t.Fatalf("me: %+v", me)
	}
}

func TestMe_AcceptsBearer(t *testing.T) {
	srv, _ := newAuthTestServer(t)
	// Use a client with NO cookie jar so cookies don't leak across.
	client := &http.Client{}
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/_plugin/me", nil)
	req.Header.Set("Authorization", "Bearer "+testToken)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("bearer me: %d", resp.StatusCode)
	}
}

// ─── /_plugin/logout ──────────────────────────────────────────────────

func TestLogout_ClearsSession(t *testing.T) {
	srv, client := newAuthTestServer(t)

	// Login.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, _ := client.Do(req)
	resp.Body.Close()

	// Logout.
	resp, err := client.Post(srv.URL+"/_plugin/logout", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("logout: %d", resp.StatusCode)
	}

	// /me must now reject — cookie cleared, session deleted.
	resp, err = client.Get(srv.URL + "/_plugin/me")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("/me after logout: %d", resp.StatusCode)
	}
}

// ─── ui handler ───────────────────────────────────────────────────────
//
// The SPA bundle is served unauthenticated on purpose: the SPA itself
// probes /_plugin/me and redirects to /login on 401. Gating the bundle
// at the server layer would surface "unauthorized" text instead of
// the login form.

func TestUI_PubliclyServed(t *testing.T) {
	srv, _ := newAuthTestServer(t)
	client := &http.Client{} // no cookie, no bearer
	resp, err := client.Get(srv.URL + "/_plugin/ui/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("ui without auth: %d (want 200 — SPA handles auth via /me probe)", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("Content-Type: %q", ct)
	}
	if cc := resp.Header.Get("Cache-Control"); cc != "no-store" {
		t.Errorf("Cache-Control: %q", cc)
	}
}

func TestUI_SPAFallback_DeepLink(t *testing.T) {
	srv, _ := newAuthTestServer(t)
	client := &http.Client{}
	// Deep link the SPA router would handle — must serve index.html
	// rather than 404.
	resp, err := client.Get(srv.URL + "/_plugin/ui/agents/coder")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("deep link: %d", resp.StatusCode)
	}
}
