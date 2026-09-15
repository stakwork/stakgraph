package adminapi

import (
	"net/http"
	"strings"
	"testing"
)

// User revocation is the dashboard's third kill axis, so its routes
// take the session cookie (with the CSRF header on mutations) like
// run and agent kills do. Nonce revocation stays bearer-only. These
// tests pin that split; the bearer round-trips live in
// hotstate_test.go next to the other hot-state routes.

// cookieDo issues a request on the cookie-jar'd client. `csrf`
// toggles the X-Bifrost-CSRF header the SPA always sends.
func cookieDo(t *testing.T, client *http.Client, method, url, body string, csrf bool) *http.Response {
	t.Helper()
	var rd *strings.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	var req *http.Request
	if rd != nil {
		req, _ = http.NewRequest(method, url, rd)
		req.Header.Set("Content-Type", "application/json")
	} else {
		req, _ = http.NewRequest(method, url, nil)
	}
	if csrf {
		req.Header.Set(csrfHeader, "1")
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func loginCookie(t *testing.T, srv string, client *http.Client) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, srv+"/_plugin/login", nil)
	req.Header.Set("Authorization", basicHeader("admin", "hunter2"))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("login: %d", resp.StatusCode)
	}
}

func TestRevokeUser_CookieAuth_WithCSRF(t *testing.T) {
	srv, client := newAuthTestServer(t)
	loginCookie(t, srv.URL, client)

	// Mutation without the CSRF header is refused even with a valid
	// cookie — the same rule the kill routes enforce.
	resp := cookieDo(t, client, http.MethodPut, srv.URL+"/_plugin/revoke/user/u_cookie", "", false)
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("PUT without CSRF: want 403, got %d", resp.StatusCode)
	}

	resp = cookieDo(t, client, http.MethodPut, srv.URL+"/_plugin/revoke/user/u_cookie",
		`{"before":"2026-09-01T00:00:00Z"}`, true)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT with cookie+CSRF: want 200, got %d", resp.StatusCode)
	}
	var ur RevokeUserResponse
	decodeBody(t, resp, &ur)
	if ur.UserID != "u_cookie" || ur.Before != "2026-09-01T00:00:00Z" {
		t.Fatalf("response: %+v", ur)
	}

	// Reads need no CSRF header.
	resp = cookieDo(t, client, http.MethodGet, srv.URL+"/_plugin/revoke/user/u_cookie", "", false)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET with cookie: want 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()
	resp = cookieDo(t, client, http.MethodGet, srv.URL+"/_plugin/revoke/users", "", false)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET users with cookie: want 200, got %d", resp.StatusCode)
	}
	var list RevokeUsersResponse
	decodeBody(t, resp, &list)
	if len(list.Users) != 1 || list.Users[0].UserID != "u_cookie" {
		t.Fatalf("list: %+v", list)
	}

	resp = cookieDo(t, client, http.MethodDelete, srv.URL+"/_plugin/revoke/user/u_cookie", "", true)
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("DELETE with cookie+CSRF: want 204, got %d", resp.StatusCode)
	}
}

func TestRevokeNonce_StaysBearerOnly(t *testing.T) {
	srv, client := newAuthTestServer(t)
	loginCookie(t, srv.URL, client)

	const nonce = "dddd000000000000000000000000dddd"
	for _, method := range []string{http.MethodPost, http.MethodDelete} {
		resp := cookieDo(t, client, method, srv.URL+"/_plugin/revoke/nonce/"+nonce, "", true)
		resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s nonce with cookie: want 401, got %d", method, resp.StatusCode)
		}
	}
}

func TestRevokeUsers_List_NewestFirst_MethodGuard(t *testing.T) {
	srv, mr := newBudgetTestServer(t, nil)

	resp := bearerDo(t, srv, http.MethodGet, "/_plugin/revoke/users", "")
	var list RevokeUsersResponse
	decodeBody(t, resp, &list)
	if resp.StatusCode != http.StatusOK || len(list.Users) != 0 {
		t.Fatalf("empty: %d %+v", resp.StatusCode, list)
	}

	for _, c := range []struct{ id, before string }{
		{"u_a", "2026-09-01T00:00:00Z"},
		{"u_b", "2026-09-03T00:00:00Z"},
		{"u_c", "2026-09-02T00:00:00Z"},
	} {
		resp := bearerDo(t, srv, http.MethodPut, "/_plugin/revoke/user/"+c.id, `{"before":"`+c.before+`"}`)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("set %s: %d", c.id, resp.StatusCode)
		}
	}

	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/revoke/users", "")
	decodeBody(t, resp, &list)
	ids := make([]string, 0, len(list.Users))
	for _, u := range list.Users {
		ids = append(ids, u.UserID)
	}
	if strings.Join(ids, ",") != "u_b,u_c,u_a" {
		t.Fatalf("order: %v", ids)
	}
	if list.Users[0].Before != "2026-09-03T00:00:00Z" {
		t.Fatalf("before: %+v", list.Users[0])
	}

	// Clearing one drops it from the list; a direct Redis DEL of
	// another (index left behind) is healed on read.
	resp = bearerDo(t, srv, http.MethodDelete, "/_plugin/revoke/user/u_b", "")
	resp.Body.Close()
	mr.Del("bifrost:revoke_user_before:u_c")
	resp = bearerDo(t, srv, http.MethodGet, "/_plugin/revoke/users", "")
	decodeBody(t, resp, &list)
	if len(list.Users) != 1 || list.Users[0].UserID != "u_a" {
		t.Fatalf("after clear+del: %+v", list.Users)
	}

	resp = bearerDo(t, srv, http.MethodPost, "/_plugin/revoke/users", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST users: want 405, got %d", resp.StatusCode)
	}
	resp = bearerDo(t, srv, http.MethodPut, "/_plugin/revoke/user/has%20space", "")
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad user id: want 400, got %d", resp.StatusCode)
	}
}
