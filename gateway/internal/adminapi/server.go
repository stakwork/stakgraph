// Package adminapi is the gateway plugin's in-process HTTP server,
// hosting the `/_plugin/*` route namespace.
//
// The wrapper binary (gateway/wrapper) reverse-proxies traffic on
// `/_plugin/*` to this server (loopback only). Why a separate server
// instead of routes registered through Bifrost's router: Bifrost
// plugins (.so) cannot register arbitrary HTTP routes through
// Bifrost's own router, and we also want routes that aren't behind
// Bifrost's AuthMiddleware so Hive can bootstrap on a fresh swarm.
//
// Lifecycle
// ---------
//   - Start() is called from the plugin's Init().
//   - Stop() is called from the plugin's Cleanup() with a small grace
//     period.
//
// One server per process. Re-calling Start() is a no-op.
package adminapi

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/graphclient"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/sessions"
	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

var (
	startOnce sync.Once
	srv       *http.Server

	// registry is the trust registry handed in by main.Init via
	// SetTrustRegistry. Stored at package scope so Start() can pick
	// it up regardless of init order — trust loading and server
	// start are independent concerns and shouldn't be coupled
	// through a single function signature.
	registry *trust.Registry
)

// SetTrustRegistry wires the trust registry the admin server will
// expose under /_plugin/trust/*. Must be called before Start();
// calling after is a no-op (the server has already captured the
// pointer it had at startup).
//
// Passing nil is allowed and skips registering the trust routes —
// useful in tests that don't exercise trust handlers.
func SetTrustRegistry(r *trust.Registry) { registry = r }

// Start brings up the plugin HTTP server in a goroutine.
//
// If required env vars (BIFROST_ADMIN_USER/PASS/PROVISIONING_TOKEN)
// are missing the server is NOT started — this matches the
// stand-alone-dev use case where someone wants to play with the image
// without configuring auth at all.
//
// Returns nil on success or on "skipped because env was missing".
// Callers in Init don't actually need to do anything with the return
// value today; it exists for symmetry with Stop.
func Start() error {
	startOnce.Do(start)
	return nil
}

// Stop shuts the server down with a short grace period. Safe to call
// even if Start was never called or skipped.
func Stop() error {
	if srv == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		pluginlog.Errf("adminapi: shutdown: %v", err)
		return err
	}
	return nil
}

func start() {
	adminUser, adminPass, hasCreds := env.AdminCreds()
	token, hasToken := env.ProvisioningTokenValue()

	if !hasCreds || !hasToken {
		pluginlog.Warnf(
			"adminapi: server NOT started: missing env "+
				"(have BIFROST_ADMIN_USER=%t BIFROST_ADMIN_PASS=%t BIFROST_PROVISIONING_TOKEN=%t)",
			adminUser != "", adminPass != "", hasToken,
		)
		return
	}

	// Session store is best-effort — when Redis is in observability
	// mode (no URL configured), NewStore returns nil and the login
	// handler refuses to start a session rather than half-issue
	// cookies that can't be persisted. Bearer-authed routes still
	// work in that mode, which is what dev / standalone wants.
	store := sessions.NewStore()
	if store == nil {
		pluginlog.Warnf("adminapi: session store disabled (redis not configured); /_plugin/login will 503")
	}

	// Agent-catalog graph client. nil when neo4j isn't configured
	// (NEO4J_PASSWORD unset) — the catalog endpoints then return 503,
	// the same graceful degradation Redis uses for observability mode.
	graph := graphclient.New()
	if graph == nil {
		pluginlog.Warnf("adminapi: agent catalog disabled (neo4j not configured); catalog endpoints will 503")
	}

	mux := http.NewServeMux()
	registerRoutes(mux, routeDeps{
		adminUser:         adminUser,
		adminPass:         adminPass,
		provisioningToken: token,
		trust:             registry,
		sessions:          store,
		logstore:          newLogstoreClient(adminUser, adminPass),
		graph:             graph,
	})

	addr := env.PluginAddr()
	srv = &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	go func() {
		pluginlog.Logf("adminapi: listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			pluginlog.Errf("adminapi: server crashed: %v", err)
		}
	}()
}

// routeDeps is the bag of values registerRoutes hands to each route
// builder. Centralizing here avoids individual route files reading env
// directly — that responsibility lives in env/.
type routeDeps struct {
	adminUser         string
	adminPass         string
	provisioningToken string
	trust             *trust.Registry     // nil ⇒ skip /_plugin/trust/* registration
	sessions          *sessions.Store     // nil ⇒ /_plugin/login returns 503
	logstore          *logstoreClient     // nil ⇒ skip phase-7 observability routes
	graph             *graphclient.Client // nil ⇒ agent-catalog endpoints return 503
}

// methodMuxedAuth picks one of two middleware chains based on the
// request's HTTP method. Used for endpoints whose read path should
// be accessible from the browser dashboard (cookie auth) but whose
// write path stays gated to Hive's machine-to-plugin bearer.
//
//	GET → permissive auth (e.g. cookieOrBearer)
//	*   → strict   auth (e.g. bearerOnly)
//
// Defining this here (rather than calling both wrappers and picking
// in the handler) keeps the middleware chain declarative — server.go
// is the only place that reasons about who-can-do-what.
func methodMuxedAuth(
	permissive, strict func(http.HandlerFunc) http.HandlerFunc,
	readMethod string,
	h http.HandlerFunc,
) http.HandlerFunc {
	permissiveH := permissive(h)
	strictH := strict(h)
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == readMethod {
			permissiveH(w, r)
			return
		}
		strictH(w, r)
	}
}

// registerRoutes is in its own function so each route lives in its
// own file (credentials.go, health.go, …) but the routing table is
// readable in one place.
//
// Auth model
// ----------
//   - `/_plugin/health`, `/_plugin/login`: anonymous.
//   - `/_plugin/admin-credentials`, `/_plugin/trust/*`,
//     `/_plugin/revoke/*`: bearer only (Hive's machine-to-plugin
//     path; cookies are not honoured).
//   - Everything else (observability, kill/state, /me, /logout):
//     cookie OR bearer, with cookie tried first. Cookie-authed
//     mutations (kill, unkill, toggles) need the CSRF header.
//
// The /_plugin/ui/* SPA is also cookie-or-bearer so curl with a
// bearer can pull it for diagnostics, but browsers always reach it
// via a cookie.
func registerRoutes(mux *http.ServeMux, deps routeDeps) {
	guard := newSessionGuard(deps.sessions, deps.provisioningToken)
	bearer := guard.bearerOnly
	cookieOrBearer := guard.cookieOrBearer

	// Anonymous routes.
	mux.HandleFunc("/_plugin/health", healthHandler)

	// Bearer-only routes (Hive's territory).
	mux.HandleFunc("/_plugin/admin-credentials",
		bearer(adminCredentialsHandler(deps.adminUser, deps.adminPass)))

	if deps.trust != nil {
		th := newTrustHandlers(deps.trust)
		// Trust registry reads (status + per-org lookup) are
		// non-sensitive — pubkeys, issuer URLs, and the org-id list
		// are public-by-design (operators paste pubkeys when
		// seeding the registry). Phase-8.5 needs the dashboard's
		// cookie auth to be able to render the Provenance card's
		// "authorized by <org>" badge, so GET-side endpoints
		// accept either cookie OR bearer. Mutations stay strictly
		// bearer-only (Hive's territory).
		mux.HandleFunc(trustStatusPath, methodMuxedAuth(
			cookieOrBearer, bearer, http.MethodGet, th.status,
		))
		// /_plugin/trust (exact): POST upsert ⇒ bearer.
		mux.HandleFunc(trustRootPath, bearer(th.upsert))
		// Prefix /_plugin/trust/<org_id>[/rotate]: GET is read,
		// POST/DELETE are mutations. Dispatch by method so the GET
		// path is reachable from the SPA without leaking the
		// provisioning token to the browser.
		mux.HandleFunc(trustPrefixPath, methodMuxedAuth(
			cookieOrBearer, bearer, http.MethodGet, th.dispatchPrefix,
		))
	}

	// Cookie-or-bearer routes: session lifecycle.
	loginH := newLoginHandlers(deps.sessions, newLoginLimiter(), deps.adminUser, deps.adminPass)
	mux.HandleFunc("/_plugin/login", loginH.login) // anon; rate-limited inside the handler
	mux.HandleFunc("/_plugin/logout", loginH.logout)
	mux.HandleFunc("/_plugin/me", cookieOrBearer(loginH.me))

	// Ticket-based bootstrap for iframe embedding. Hive calls
	// /auth/ticket with its bearer to mint a short-lived single-
	// use ticket, embeds the iframe with `?ticket=<value>`, and
	// the SPA POSTs that to /auth/redeem on boot to receive the
	// session cookie. Admin password never reaches the browser.
	ticketH := newTicketHandlers(deps.sessions, deps.adminUser)
	mux.HandleFunc("/_plugin/auth/ticket", bearer(ticketH.mint))
	mux.HandleFunc("/_plugin/auth/redeem", ticketH.redeem) // anon; ticket IS the proof

	// Phase-6 hot state: kill switches + Redis snapshots. Registered
	// unconditionally (they need Redis, not the logstore) and
	// dispatched from the shared /runs/ and /agents/ subtrees below.
	hot := newHotStateHandlers()

	// Cookie-or-bearer routes: phase-7 observability subset.
	var obs *observabilityHandlers
	if deps.logstore != nil {
		obs = newObservabilityHandlers(deps.logstore)
		mux.HandleFunc("/_plugin/spend/by-agent", cookieOrBearer(obs.spendByAgent))
		mux.HandleFunc("/_plugin/spend/by-user", cookieOrBearer(obs.spendByUser))
		mux.HandleFunc("/_plugin/spend/by-agent-user", cookieOrBearer(obs.spendByAgentUser))
		mux.HandleFunc("/_plugin/histogram/cost", cookieOrBearer(obs.histogramCost))
		// Phase-7 remainder: the session / model rollups, token and
		// latency histograms, and the session drill-down.
		mux.HandleFunc("/_plugin/spend/by-session", cookieOrBearer(obs.spendBySession))
		mux.HandleFunc("/_plugin/spend/by-model", cookieOrBearer(obs.spendByModel))
		mux.HandleFunc("/_plugin/histogram/tokens", cookieOrBearer(obs.histogramTokens))
		mux.HandleFunc("/_plugin/histogram/latency", cookieOrBearer(obs.histogramLatency))
		mux.HandleFunc("/_plugin/sessions/", cookieOrBearer(obs.sessions))
		// /_plugin/users/ subtree: `<id>` is the phase-8 rollup,
		// `<id>/spend` the windowed totals, `<id>/quota` the Bifrost
		// customer budget blended with the user's in-flight runs.
		mux.HandleFunc("/_plugin/users/", cookieOrBearer(obs.userDetail))
	}

	// The `/_plugin/runs/` subtree: `<id>/kill` and `<id>/state` are
	// phase-6 hot state; everything else (`<id>`, `<id>/calls/<cid>`)
	// is the phase-7 logs.db drill-down, which 404s when the logstore
	// isn't configured.
	runsSubtree := func(w http.ResponseWriter, r *http.Request) {
		rest := strings.TrimPrefix(r.URL.Path, "/_plugin/runs/")
		parts := strings.Split(rest, "/")
		if len(parts) == 2 && parts[0] != "" {
			switch parts[1] {
			case "kill":
				hot.runKill(w, r, parts[0])
				return
			case "state":
				hot.runState(w, r, parts[0])
				return
			}
		}
		if obs == nil {
			http.NotFound(w, r)
			return
		}
		obs.runDetail(w, r)
	}
	mux.HandleFunc("/_plugin/runs/", cookieOrBearer(runsSubtree))

	// Revocation admin (bearer-only; issuer territory).
	rv := newRevokeHandlers()
	mux.HandleFunc(revokePrefixPath, bearer(rv.dispatch))

	// Phase-8.5 per-agent budget view. Reads cap from plugin config
	// (auth.GetConfig().AgentBudgets) and current-bucket spend from
	// Redis (`bifrost:cost:agent:<name>:<bucket_key>`) with a
	// fallback to summing `logs.db` rows when phase 6's PostHook
	// hasn't filled the Redis hash yet. Subtree routing on
	// `/_plugin/agents/`; the handler enforces the `<name>/budget`
	// shape and 404s on anything else.
	bgt := newBudgetHandlers(deps.logstore)
	cat := newCatalogHandlers(deps.graph)
	hiveCb := newHiveCallbackHandlers()
	evals := newEvalHandlers(deps.graph, hiveCb)

	// The `/_plugin/agents/` subtree is shared: ServeMux allows only
	// one handler per pattern, so a small dispatcher fans the trailing
	// `<name>/<view>` segment out to the budget (phase-8.5), catalog
	// (agent-catalog), or hot-state (phase-6 kill/state) handler. All
	// cookie-or-bearer. Anything else under the prefix 404s,
	// preserving the budget handler's existing contract.
	agentsSubtree := func(w http.ResponseWriter, r *http.Request) {
		rest := strings.TrimPrefix(r.URL.Path, "/_plugin/agents/")
		parts := strings.Split(rest, "/")
		// `<name>/kill` (POST/DELETE) and `<name>/state` (GET).
		if len(parts) == 2 && parts[0] != "" && parts[1] == "kill" {
			hot.agentKill(w, r, parts[0])
			return
		}
		if len(parts) == 2 && parts[0] != "" && parts[1] == "state" {
			hot.agentState(w, r, parts[0])
			return
		}
		// `<name>/spend` (GET): the agent's windowed totals from
		// logs.db — phase 7. 404 when the logstore isn't configured,
		// like the /runs/ drill-down.
		if len(parts) == 2 && parts[0] != "" && parts[1] == "spend" {
			if obs == nil {
				http.NotFound(w, r)
				return
			}
			obs.agentSpend(w, r, parts[0])
			return
		}
		// `/_plugin/agents/catalog` (single segment) is the catalog
		// list — every registry agent, traffic or not. Distinct from
		// `<name>/catalog` (two segments) which is one agent's detail.
		if len(parts) == 1 && parts[0] == "catalog" {
			cat.list(w, r)
			return
		}
		if len(parts) == 2 && parts[0] != "" && parts[1] == "catalog" {
			cat.read(w, r)
			return
		}
		// `<name>/tools` and `<name>/skills` (PATCH) toggle a child's
		// enabled flag. Operator dashboard actions, so they ride the same
		// cookie-or-bearer auth as the reads (the catalog *push* stays
		// bearer-only).
		if len(parts) == 2 && parts[0] != "" && parts[1] == "skills" {
			cat.toggleSkill(w, r)
			return
		}
		if len(parts) == 2 && parts[0] != "" && parts[1] == "tools" {
			cat.toggleTool(w, r)
			return
		}
		// `<name>/evals` and `<name>/evals/<setId>`: the agent-scoped
		// Evals tab. GET lists the sets linked to the agent; POST
		// creates/links; DELETE unlinks. agentEvals picks by method.
		if len(parts) >= 2 && parts[0] != "" && parts[1] == "evals" {
			evals.agentEvals(w, r, parts[0], parts[2:])
			return
		}
		// Default: the budget handler owns `<name>/budget` and 404s
		// the rest.
		bgt.budget(w, r)
	}
	mux.HandleFunc("/_plugin/agents/", cookieOrBearer(agentsSubtree))

	// Catalog write front door: POST /_plugin/agents (exact, no
	// trailing slash). Bearer-only (server-to-server), same posture as
	// the trust-registry mutations. Registered even when neo4j is
	// unconfigured so the handler can answer 503 rather than ServeMux
	// 301-redirecting to the subtree.
	mux.HandleFunc("/_plugin/agents", bearer(cat.push))

	// Eval set / requirement CRUD + runs. Cookie-or-bearer: these are
	// operator dashboard actions (the CSRF header guards cookie-authed
	// mutations). Reads hit neo4j directly; node mutations + runs are
	// delegated to Hive via the pushed callback config. The exact and
	// subtree patterns both route through one dispatcher.
	mux.HandleFunc("/_plugin/evals", cookieOrBearer(evals.dispatch))
	mux.HandleFunc("/_plugin/evals/", cookieOrBearer(evals.dispatch))

	// Hive callback config push: Hive sends its base URL + a
	// workspace-scoped API key during agent-catalog reconciliation.
	// Bearer-only (server-to-server), same posture as the catalog push.
	mux.HandleFunc("/_plugin/hive-callback", bearer(hiveCb.storeConfig))

	// SPA — served WITHOUT middleware auth. The SPA itself probes
	// /_plugin/me at boot and redirects to /login on 401; gating
	// the static bundle at the server layer would short-circuit
	// that flow and surface "unauthorized" text instead of the
	// login page.
	//
	// Safe because the bundle contains zero secrets — every API
	// call the SPA makes still goes through cookie-or-bearer auth
	// on its respective handler.
	mux.Handle("/_plugin/ui/", uiHandler())
	// `/_plugin/ui` (no trailing slash) is the typical URL users
	// type; redirect them to the canonical /ui/ so the SPA router
	// resolves correctly.
	mux.HandleFunc("/_plugin/ui", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/_plugin/ui/", http.StatusFound)
	})
}
