// Package pluginlog is the plugin's logging shim.
//
// Everything emits a single greppable line of the form:
//
//	[stakgraph-gateway] <RFC3339Nano UTC> <message>
//
// All logs go to stderr so they interleave naturally with bifrost-http's
// own pretty/json log stream. We don't use slog yet because:
//
//  1. Bifrost-http uses its own logger (zerolog-style) and slog output
//     mixed in would look weird in `docker logs`.
//  2. Plugin code is small enough that structured fields aren't yet a
//     readability win.
//
// When the plugin grows enough that key=value lines become painful to
// grep, swap the body of Logf for slog. Callers should remain the same.
package pluginlog

import (
	"encoding/json"
	"fmt"
	"os"
	"sync/atomic"
	"time"
)

// pluginName is the prefix on every log line. Set once by Init.
var pluginName atomic.Value // string

// configured tracks whether anyone called Init, mostly so Logf has a
// graceful fallback in tests that never bootstrap the package.
var configured atomic.Bool

// Init records the plugin name and (optional) plugin config. It is
// safe to call multiple times — the last caller wins — but in practice
// bifrost-http calls plugin Init exactly once per process.
//
// The `config` parameter is the raw `config` block from bifrost's
// config.json. We don't act on it today beyond echoing it back, but
// future structured fields (e.g. log level, sampling rate) live here.
func Init(name string, config any) {
	pluginName.Store(name)
	configured.Store(true)
	// Marshal config for the boot line so we can confirm what the
	// plugin saw at load time. Best-effort — never panics.
	pretty := mustJSON(config)
	fmt.Fprintf(os.Stderr, "[%s] Init called config=%s\n", name, pretty)
}

// Logf writes one timestamped line to stderr. The first format arg may
// freely contain %s/%d/... — printf semantics apply.
//
// Call sites should be greppable: prefix the format with the hook /
// subsystem name (e.g. "HTTPTransportPreHook ..." or "adminapi: ...")
// so log scrapers don't need to know the source file.
func Logf(format string, args ...any) {
	name := nameOrDefault()
	prefix := fmt.Sprintf("[%s] %s ", name, time.Now().UTC().Format(time.RFC3339Nano))
	fmt.Fprintf(os.Stderr, prefix+format+"\n", args...)
}

// Errf is Logf with an "ERROR " marker — purely for visual triage in
// `docker logs`. No level filtering today; everything always prints.
func Errf(format string, args ...any) {
	Logf("ERROR "+format, args...)
}

// Warnf is Logf with a "WARN " marker. Same caveat as Errf.
func Warnf(format string, args ...any) {
	Logf("WARN "+format, args...)
}

func nameOrDefault() string {
	if v, ok := pluginName.Load().(string); ok && v != "" {
		return v
	}
	return "stakgraph-gateway"
}

// mustJSON returns a JSON string for any value, falling back to %+v if
// marshalling fails. Used only in the Init banner where readability
// matters more than precision.
func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%+v", v)
	}
	return string(b)
}
