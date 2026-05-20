package adminapi

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"

	"github.com/stakwork/stakgraph/gateway/internal/env"
)

// uiFS is the compiled SPA bundle, dropped here by the Dockerfile's
// `ui-builder` stage before `go build`. Local `go build` without the
// Vite step produces a placeholder `ui/dist/index.html` so the
// package still compiles outside Docker — see `ui/dist/.gitkeep`
// and the README's "Local plugin build" note.
//
//go:embed all:ui/dist
var uiFS embed.FS

// uiHandler returns the http.Handler that serves `/_plugin/ui/*`.
//
// Two pieces of behaviour worth calling out:
//
//   - `StripPrefix("/_plugin/ui/", …)` lets file-server URLs match
//     the embedded layout (which is rooted at /ui/dist/ → "/"). The
//     prefix in the URL is purely a routing convention; the SPA
//     itself doesn't care what its base path is because
//     `vite.config.ts` sets `base: '/_plugin/ui/'`.
//
//   - The SPA-fallback in spaFallback() returns `index.html` for any
//     path that isn't a real asset. Without this, a hard refresh on
//     a client-side route like `/_plugin/ui/agents/coder` would 404
//     because the static layout has no `agents/coder` file.
//
// The handler is constructed once per process and is concurrency-
// safe — `http.FileServer` and `fs.Sub` both return immutable
// values backed by the embedded read-only FS.
func uiHandler() http.Handler {
	sub, err := fs.Sub(uiFS, "ui/dist")
	if err != nil {
		// Embedding root missing — should never happen at runtime
		// because go:embed would have failed at compile time, but
		// returning a useful 500 here beats panicking when someone
		// strips the dist/ directory by hand in a debugging session.
		return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "ui not built", http.StatusInternalServerError)
		})
	}
	fsrv := http.FileServer(http.FS(sub))
	return withFrameAncestors(http.StripPrefix("/_plugin/ui/", spaFallback(sub, fsrv)))
}

// withFrameAncestors wraps a handler so every response carries a
// `Content-Security-Policy: frame-ancestors 'self' <HiveOrigin>`
// header. This is the only thing standing between the dashboard
// and clickjacking by a rogue site, now that the session cookie is
// SameSite=None. The list is small and constant (Hive + the plugin
// itself, so direct browser sessions also work).
func withFrameAncestors(next http.Handler) http.Handler {
	csp := "frame-ancestors 'self' " + env.HiveOriginValue()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", csp)
		next.ServeHTTP(w, r)
	})
}

// spaFallback wraps a FileServer so that any path that doesn't
// resolve to a real asset returns `index.html` (with no-store cache).
// This is the standard SPA hosting trick — `wouter` (and react-
// router, and every other client-side router) need it to survive
// a hard refresh on a deep link.
//
// We `fs.Stat` rather than rely on FileServer's own 404 because
// FileServer returns text/plain on miss, which would render as
// garbage inside the SPA root. Stat-then-decide gives us a clean
// boundary.
func spaFallback(root fs.FS, fileServer http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/")
		// Root / wouter-client-side routes / unknown paths all
		// serve index.html via the shell path (Cache-Control:
		// no-store). Only paths that map to a real non-index
		// file in dist/ go through FileServer with default caching.
		if path != "" && path != "index.html" {
			if _, err := fs.Stat(root, path); err == nil {
				fileServer.ServeHTTP(w, r)
				return
			}
		}
		// Shell path: serve index.html with no-store so updates
		// roll out on the next page load. A stale shell pointed
		// at stale asset hashes would brick the dashboard until
		// a hard refresh.
		index, err := fs.ReadFile(root, "index.html")
		if err != nil {
			http.Error(w, "ui not built", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(index)
	})
}
