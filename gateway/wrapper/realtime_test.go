package main

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIsRealtimePath(t *testing.T) {
	blocked := []string{
		// WebSocket endpoints (OpenAIRealtimePaths, "" and "/openai" prefixes).
		"/v1/realtime",
		"/realtime",
		"/openai/realtime",
		"/openai/v1/realtime",
		// WebRTC SDP exchange (OpenAIRealtimeWebRTCCallsPaths).
		"/v1/realtime/calls",
		"/realtime/calls",
		"/openai/realtime/calls",
		"/openai/v1/realtime/calls",
		// Ephemeral client-secret / session aliases.
		"/v1/realtime/client_secrets",
		"/v1/realtime/sessions",
		"/openai/v1/realtime/client_secrets",
		// Case-insensitive segment match.
		"/v1/Realtime",
	}
	for _, p := range blocked {
		if !isRealtimePath(p) {
			t.Errorf("isRealtimePath(%q) = false, want true", p)
		}
	}

	allowed := []string{
		"/v1/chat/completions",
		"/v1/responses",
		"/v1/embeddings",
		"/v1/models",
		"/anthropic/v1/messages",
		"/_plugin/health",
		"/health",
		"/",
		// Must not misfire on a substring — only an exact segment counts.
		"/v1/realtimeless",
		"/v1/notrealtime",
		"/v1/realtimely/calls",
	}
	for _, p := range allowed {
		if isRealtimePath(p) {
			t.Errorf("isRealtimePath(%q) = true, want false", p)
		}
	}
}

func TestRealtimeEnabled(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"false", false},
		{"no", false},
		{"off", false},
		{"nonsense", false},
		{"1", true},
		{"true", true},
		{"TRUE", true},
		{"Yes", true},
		{"on", true},
		{"  on  ", true}, // trimmed
	}
	for _, c := range cases {
		t.Setenv("BIFROST_ENABLE_REALTIME", c.val)
		if got := realtimeEnabled(); got != c.want {
			t.Errorf("realtimeEnabled() with %q = %v, want %v", c.val, got, c.want)
		}
	}
}

// markerHandler records that it was hit and writes a recognizable body so
// tests can assert which upstream a request was routed to.
func markerHandler(name string, hit *bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		*hit = true
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, name)
	})
}

func TestNewRouterRealtimeBlocking(t *testing.T) {
	logger := log.New(io.Discard, "", 0)

	newRouterFor := func(blocked bool, bifrostHit, pluginHit *bool) http.Handler {
		return newRouter(
			markerHandler("bifrost", bifrostHit),
			markerHandler("plugin", pluginHit),
			blocked,
			logger,
		)
	}

	t.Run("realtime blocked returns 403 and never reaches bifrost", func(t *testing.T) {
		for _, p := range []string{"/v1/realtime", "/openai/v1/realtime/calls", "/v1/realtime/client_secrets"} {
			var bifrostHit, pluginHit bool
			h := newRouterFor(true, &bifrostHit, &pluginHit)
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, p, nil))
			if rec.Code != http.StatusForbidden {
				t.Errorf("path %q: status = %d, want %d", p, rec.Code, http.StatusForbidden)
			}
			if bifrostHit {
				t.Errorf("path %q: bifrost upstream was hit; realtime should be blocked before it", p)
			}
		}
	})

	t.Run("realtime allowed when enabled reaches bifrost", func(t *testing.T) {
		var bifrostHit, pluginHit bool
		h := newRouterFor(false, &bifrostHit, &pluginHit)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/realtime", nil))
		if !bifrostHit {
			t.Error("realtime enabled: bifrost upstream was not hit")
		}
	})

	t.Run("normal inference reaches bifrost even when realtime blocked", func(t *testing.T) {
		var bifrostHit, pluginHit bool
		h := newRouterFor(true, &bifrostHit, &pluginHit)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
		if !bifrostHit {
			t.Error("chat completions: bifrost upstream was not hit")
		}
		if rec.Code != http.StatusOK {
			t.Errorf("chat completions: status = %d, want %d", rec.Code, http.StatusOK)
		}
	})

	t.Run("plugin path reaches plugin upstream", func(t *testing.T) {
		var bifrostHit, pluginHit bool
		h := newRouterFor(true, &bifrostHit, &pluginHit)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/_plugin/health", nil))
		if !pluginHit {
			t.Error("plugin path: plugin upstream was not hit")
		}
		if bifrostHit {
			t.Error("plugin path: bifrost upstream should not be hit")
		}
	})

	t.Run("plugin path with no plugin server returns 503", func(t *testing.T) {
		var bifrostHit bool
		h := newRouter(markerHandler("bifrost", &bifrostHit), nil, true, logger)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/_plugin/health", nil))
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
	})
}
