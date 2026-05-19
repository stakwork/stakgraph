package adminapi

import (
	"context"
	"crypto/subtle"
	"encoding/base64"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/sessions"
)

// loginTimeout caps the full login flow (rate-limit check, password
// compare, session create). 5 seconds is generous for Redis on
// loopback; anything slower than that is a degraded environment and
// the client retry will pick a healthier moment.
const loginTimeout = 5 * time.Second

// LoginResponse is the JSON body returned by `POST /_plugin/login`
// on success. The SPA stashes the username so the topbar can render
// it without an extra `/me` round-trip.
//
// Exported because tygo emits TS bindings from this declaration
// (see gateway/tygo.yaml); rename in lockstep with the frontend.
type LoginResponse struct {
	User string `json:"user"`
}

// MeResponse is `GET /_plugin/me` — the SPA's boot probe to decide
// "am I authenticated?". Unix timestamps (seconds) keep the wire
// format compact and language-agnostic; the frontend converts to
// Date once.
type MeResponse struct {
	User     string `json:"user"`
	IssuedAt int64  `json:"iat"`
	LastSeen int64  `json:"last_seen"`
}

// loginHandlers wraps the dependencies common to /login, /logout,
// /me. Kept as a small struct (vs. closures over each) so tests can
// inject a fake store / fake limiter without rewiring the world.
type loginHandlers struct {
	store     *sessions.Store
	limiter   *loginLimiter
	adminUser string
	adminPass string
}

func newLoginHandlers(store *sessions.Store, limiter *loginLimiter, user, pass string) *loginHandlers {
	return &loginHandlers{store: store, limiter: limiter, adminUser: user, adminPass: pass}
}

// login handles POST /_plugin/login.
//
// Request shape
// -------------
//   Header: Authorization: Basic base64(user:pass)
//   Body:   (ignored)
//
// Success
// -------
//   200 OK
//   Set-Cookie: bifrost_session=...; HttpOnly; Secure; SameSite=Strict;
//               Path=/_plugin; Max-Age=28800
//   {"user":"admin"}
//
// Failures
// --------
//   400 missing/malformed Authorization header
//   401 wrong credentials (rate-limit counter bumped)
//   429 too many failed attempts in window
//   503 session store unavailable (Redis down)
//
// Why Basic header (and not a JSON body)? Browsers don't natively
// send Basic on a fetch unless you opt in, so the SPA assembles the
// header itself; using the same well-known field name (Authorization
// + Basic) keeps the wire shape boring and curl-debuggable.
func (h *loginHandlers) login(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	if h.store == nil {
		http.Error(w, "session store unavailable", http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), loginTimeout)
	defer cancel()

	ip := clientIP(r)

	// Rate-limit check before the password compare so a flood of
	// attempts can't burn CPU on constant-time compares.
	if ok, retry := h.limiter.allow(ctx, ip); !ok {
		w.Header().Set("Retry-After", strconv.Itoa(int(retry.Seconds())))
		writeError(w, http.StatusTooManyRequests, "too_many_login_attempts",
			"too many failed login attempts; try again later")
		return
	}

	user, pass, ok := parseBasicAuth(r.Header.Get("Authorization"))
	if !ok {
		writeError(w, http.StatusBadRequest, "bad_request",
			"missing or malformed Authorization: Basic header")
		return
	}

	// Constant-time compare on BOTH fields so the response timing
	// doesn't leak which one was wrong.
	userMatch := subtle.ConstantTimeCompare([]byte(user), []byte(h.adminUser))
	passMatch := subtle.ConstantTimeCompare([]byte(pass), []byte(h.adminPass))
	if userMatch != 1 || passMatch != 1 {
		h.limiter.recordFailure(ctx, ip)
		// Match the canonical "wrong password" body the SPA can
		// detect without parsing — generic 401 with an error code
		// it recognises.
		writeError(w, http.StatusUnauthorized, "invalid_credentials", "invalid credentials")
		return
	}

	id, _, err := h.store.Create(ctx, user)
	if err != nil {
		pluginlog.Errf("adminapi: login create: %v", err)
		writeError(w, http.StatusServiceUnavailable, "session_store_unavailable",
			"could not create session")
		return
	}

	// Wipe the failure counter — a successful login means whatever
	// scary near-limit balance the user accumulated is moot.
	h.limiter.reset(ctx, ip)

	setSessionCookie(w, r, id)
	writeJSON(w, http.StatusOK, LoginResponse{User: user})
}

// logout handles POST /_plugin/logout. Idempotent — calling without
// a cookie or with a stale cookie still returns 204 so the SPA can
// fire-and-forget on logout without branching on the response.
func (h *loginHandlers) logout(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), loginTimeout)
	defer cancel()

	if c, err := r.Cookie(sessionCookieName); err == nil && c.Value != "" && h.store != nil {
		if err := h.store.Delete(ctx, c.Value); err != nil {
			pluginlog.Warnf("adminapi: logout delete: %v", err)
			// Continue — the cookie still gets cleared client-side.
		}
	}
	clearSessionCookie(w, r)
	w.WriteHeader(http.StatusNoContent)
}

// me handles GET /_plugin/me. Returns 200 with the resolved session
// or 401 if the cookie is missing / stale. The SPA fires this on
// boot to decide between rendering the dashboard or redirecting to
// /login.
func (h *loginHandlers) me(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	sess := sessionFromContext(r)
	if sess == nil {
		// Bearer-authenticated callers (Hive, curl smoke tests)
		// don't have a Session in context. Return a sentinel
		// payload so they can still poke `/me` to confirm the
		// route is wired and the bearer is accepted — without
		// inventing a fake user identity for them.
		writeJSON(w, http.StatusOK, MeResponse{User: "", IssuedAt: 0, LastSeen: 0})
		return
	}
	writeJSON(w, http.StatusOK, MeResponse{
		User:     sess.User,
		IssuedAt: sess.IssuedAt.Unix(),
		LastSeen: sess.LastSeen.Unix(),
	})
}

// parseBasicAuth returns (user, pass, ok). Mirrors the stdlib
// http.Request.BasicAuth but operates on a raw header so the
// handler can decide what to do with malformed input (we return
// 400 rather than 401 so the caller knows their request was the
// problem, not their credentials).
func parseBasicAuth(header string) (user, pass string, ok bool) {
	const prefix = "Basic "
	if len(header) < len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return "", "", false
	}
	raw, err := base64.StdEncoding.DecodeString(header[len(prefix):])
	if err != nil {
		return "", "", false
	}
	s := string(raw)
	idx := strings.IndexByte(s, ':')
	if idx < 0 {
		return "", "", false
	}
	return s[:idx], s[idx+1:], true
}

// writeError emits the phase-7 error envelope used by every
// /_plugin/* JSON endpoint. Centralised so the wire shape doesn't
// drift across handlers.
func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	})
}
