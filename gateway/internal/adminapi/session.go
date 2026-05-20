package adminapi

import (
	"context"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/sessions"
)

// sessionCookieName is the cookie key the SPA reads/writes. Scoped to
// `/_plugin` so it never collides with anything Bifrost-side sets.
const sessionCookieName = "bifrost_session"

// csrfHeader is required on every cookie-authed non-GET request.
// Browsers won't send a custom header on a cross-origin form / image
// / link request, so requiring its presence blocks the classic CSRF
// vectors that SameSite=None opens up. The SPA's apiFetch adds it
// on every non-GET; bearer-authed callers (Hive) are exempt because
// they aren't relying on ambient cookie auth.
const csrfHeader = "X-Bifrost-CSRF"

// sessionLookupTimeout bounds the Redis call the middleware makes on
// every authed request. Generous because Redis is loopback in swarm,
// but tight enough that an unresponsive Redis surfaces as a 401 (and
// a redirect to /login) rather than a hung browser tab.
const sessionLookupTimeout = 2 * time.Second

// ctxKey is the unexported key type for stashing the resolved Session
// on the request context. Handlers that want to know "who is this?"
// pull it back out via sessionFromContext.
type ctxKey struct{}

var sessionContextKey ctxKey

// sessionFromContext returns the Session attached to `r` by the
// session middleware, or nil if the route is bearer-authed (Hive)
// rather than cookie-authed (browser).
func sessionFromContext(r *http.Request) *sessions.Session {
	if v, ok := r.Context().Value(sessionContextKey).(*sessions.Session); ok {
		return v
	}
	return nil
}

// authMode describes how a request must authenticate to reach a
// handler. Exact-anonymous routes (health, login) are handled by
// matching the path in the middleware rather than per-route flags;
// every other handler picks one of these.
type authMode int

const (
	// modeBearerOnly is for Hive's machine-to-plugin calls: trust
	// registry CRUD, admin-credentials echo. Cookies are ignored
	// here so a browser session can never reach a Hive-only route.
	modeBearerOnly authMode = iota

	// modeCookieOrBearer is for routes the dashboard fetches and
	// that Hive may also need: spend, histograms, runs. Cookie is
	// tried first; bearer is the fallback for non-browser callers.
	modeCookieOrBearer

	// modeAnon is for health / login itself. The middleware lets
	// the request through unchecked and the handler decides what
	// extra checks (e.g. basic auth on login) apply.
	modeAnon
)

// sessionGuard is the constructor for the cookie/bearer middleware.
// The plumbing is split this way so server.go can compose per-route
// chains (`auth.bearer(handler)` vs `auth.cookieOrBearer(handler)`)
// without each handler re-implementing the same probe.
type sessionGuard struct {
	store             *sessions.Store // may be nil ⇒ cookie auth disabled
	provisioningToken string
}

func newSessionGuard(store *sessions.Store, token string) *sessionGuard {
	return &sessionGuard{store: store, provisioningToken: token}
}

// bearerOnly returns middleware that only accepts the provisioning
// bearer token. Used for routes Hive owns end-to-end.
func (g *sessionGuard) bearerOnly(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !g.bearerOK(r) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

// cookieOrBearer returns middleware that accepts EITHER a valid
// session cookie OR the provisioning bearer. When a cookie is used,
// the resolved Session is stashed on the request context.
//
// Cookie-authed mutations (any non-GET/HEAD method) additionally
// require the `X-Bifrost-CSRF` header. The SPA's apiFetch adds it
// automatically. Bearer-authed requests skip the check — they aren't
// vulnerable to CSRF because the attacker can't add the header on
// behalf of the victim's bearer.
func (g *sessionGuard) cookieOrBearer(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Cookie first — the common case for the dashboard.
		if sess, ok := g.resolveCookie(r); ok {
			if !csrfOK(r) {
				http.Error(w, "csrf header required", http.StatusForbidden)
				return
			}
			ctx := context.WithValue(r.Context(), sessionContextKey, sess)
			next(w, r.WithContext(ctx))
			return
		}
		if g.bearerOK(r) {
			next(w, r)
			return
		}
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}
}

// csrfOK reports whether the request satisfies the CSRF rule:
// GET / HEAD / OPTIONS are exempt (idempotent reads); every other
// method must carry the `X-Bifrost-CSRF` header. Any non-empty value
// is accepted — the header's presence is the signal; the contents
// don't matter because a cross-origin attacker can't set custom
// headers at all.
func csrfOK(r *http.Request) bool {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return true
	}
	return r.Header.Get(csrfHeader) != ""
}

// resolveCookie returns the Session associated with the inbound
// request's cookie, or (nil, false) if there's no valid cookie. A
// Redis error here is logged and treated as "no session" — failing
// closed is the right posture for an auth check.
func (g *sessionGuard) resolveCookie(r *http.Request) (*sessions.Session, bool) {
	if g.store == nil {
		return nil, false
	}
	c, err := r.Cookie(sessionCookieName)
	if err != nil || c.Value == "" {
		return nil, false
	}
	ctx, cancel := context.WithTimeout(r.Context(), sessionLookupTimeout)
	defer cancel()
	sess, err := g.store.Refresh(ctx, c.Value)
	if err != nil {
		if err != sessions.ErrNotFound && err != sessions.ErrStoreUnavailable {
			pluginlog.Warnf("session: refresh: %v", err)
		}
		return nil, false
	}
	return sess, true
}

// bearerOK is the constant-time bearer check. Equivalent to
// requireBearerToken() but inlined here so it can be one branch of
// the OR check above without two layers of closure.
func (g *sessionGuard) bearerOK(r *http.Request) bool {
	presented := r.Header.Get("Authorization")
	if len(presented) > 7 && strings.EqualFold(presented[:7], "bearer ") {
		presented = presented[7:]
	}
	if presented == "" || g.provisioningToken == "" {
		return false
	}
	// constant-time compare
	if len(presented) != len(g.provisioningToken) {
		return false
	}
	var diff byte
	for i := 0; i < len(presented); i++ {
		diff |= presented[i] ^ g.provisioningToken[i]
	}
	return diff == 0
}

// setSessionCookie writes a fresh session cookie. Centralised so the
// attributes (HttpOnly / Secure / SameSite / Path / Max-Age) stay in
// sync between login (set) and logout (clear).
//
// SameSite=None is required so the cookie rides cross-origin iframe
// requests from Hive. Browsers reject SameSite=None without Secure,
// so Secure is forced on regardless of the inbound request's scheme
// in production. Local dev over plain HTTP still gets Lax so the
// cookie can actually land — `SameSite=None` over HTTP is dropped.
//
// CSRF defence is upgraded to the `X-Bifrost-CSRF` header check on
// every cookie-authed mutation (see session.go's middleware); the
// `frame-ancestors` CSP on the UI shell blocks rogue embedders.
func setSessionCookie(w http.ResponseWriter, r *http.Request, id string) {
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    id,
		Path:     "/_plugin",
		HttpOnly: true,
		Secure:   isSecureRequest(r),
		SameSite: sessionSameSite(r),
		MaxAge:   int(sessions.SessionTTL.Seconds()),
	})
}

// clearSessionCookie writes a cookie with the same attributes as
// setSessionCookie but Max-Age=-1 so browsers discard it immediately.
// Must mirror setSessionCookie's path / SameSite or the browser
// keeps a duplicate cookie around.
func clearSessionCookie(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    "",
		Path:     "/_plugin",
		HttpOnly: true,
		Secure:   isSecureRequest(r),
		SameSite: sessionSameSite(r),
		MaxAge:   -1,
	})
}

// sessionSameSite picks the SameSite attribute for the session cookie.
// Prod always uses None (so the cookie rides inside Hive's iframe);
// dev falls back to Lax because browsers silently drop `None` over
// plain HTTP and we want localhost to keep working.
func sessionSameSite(r *http.Request) http.SameSite {
	if isSecureRequest(r) {
		return http.SameSiteNoneMode
	}
	return http.SameSiteLaxMode
}

// isSecureRequest mirrors the rule used by `setSessionCookie` so a
// test can ask the same question. See that function's doc.
func isSecureRequest(r *http.Request) bool {
	if env.IsProduction() {
		return true
	}
	if strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https") {
		return true
	}
	return r.TLS != nil
}

// clientIP best-effort extracts the caller's address for the
// login-rate-limiter. Trusts `X-Forwarded-For` when present —
// running without an ingress that strips/sets XFF makes the limiter
// degenerate, which is out of phase 8's threat model (documented in
// the plan).
func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// Take the leftmost token (the original client) and trim
		// whitespace. The rest is the proxy chain.
		if comma := strings.IndexByte(xff, ','); comma >= 0 {
			return strings.TrimSpace(xff[:comma])
		}
		return strings.TrimSpace(xff)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
