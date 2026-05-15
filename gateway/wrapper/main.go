// Command wrapper is the container entrypoint for the stakgraph
// Bifrost gateway image. It owns the single public HTTP port and
// reverse-proxies traffic to two upstreams running on loopback:
//
//	/_plugin/*  ──► 127.0.0.1:8189   (the stakgraph-gateway .so plugin)
//	everything  ──► 127.0.0.1:8080   (bifrost-http itself)
//
// Why a wrapper exists
// --------------------
// Bifrost plugins (.so) cannot register arbitrary HTTP routes through
// Bifrost's own router — HTTPTransportPreHook only fires on inference
// paths, and unknown URLs 404 before any middleware can see them. We
// also need routes that AREN'T behind Bifrost's auth middleware (e.g.
// the bootstrap endpoint that lets Hive learn this Bifrost's admin
// password on a fresh swarm — chicken-and-egg otherwise).
//
// Solution: the plugin runs its own HTTP server on loopback, and this
// wrapper fronts both. End result: one public port (Traefik / k8s
// friendly), but the plugin owns a clean `/_plugin/*` namespace it can
// grow with new metrics / governance routes over time.
//
// Process model
// -------------
// Wrapper exec's bifrost-http as a child process (Stdout/Stderr piped
// through). When the wrapper receives SIGTERM/SIGINT it forwards the
// signal to bifrost-http and waits for it to exit before returning,
// so logs land cleanly and the SQLite WAL gets flushed. The container
// runs tini as PID 1 → wrapper → bifrost-http.
//
// Startup ordering
// ----------------
// Wrapper:
//  1. Starts bifrost-http child process.
//  2. Polls 127.0.0.1:8080 until /health responds (bifrost ready).
//  3. Polls 127.0.0.1:8189 until /_plugin/health responds (plugin
//     ready — skipped if plugin server didn't start, e.g. dev mode
//     without BIFROST_PROVISIONING_TOKEN).
//  4. Opens the public listener.
//
// If bifrost-http exits unexpectedly, the wrapper exits with the same
// status; the container then restarts under docker-compose's
// `restart: unless-stopped`.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

const (
	// defaultPublicAddr is the address the wrapper binds for incoming
	// public traffic. 8181 matches Hive's DEFAULT_BIFROST_PORT and the
	// historical port the MCP tests / smoke scripts hit.
	defaultPublicAddr = ":8181"

	// bifrostUpstream is where bifrost-http is configured to listen
	// (loopback only — see Dockerfile's CMD). The wrapper proxies
	// almost all traffic here.
	bifrostUpstream = "http://127.0.0.1:8080"

	// pluginUpstream is where the .so plugin's in-process HTTP server
	// listens. The wrapper routes `/_plugin/*` here.
	pluginUpstream = "http://127.0.0.1:8189"

	// pluginPathPrefix is what we route to pluginUpstream. Everything
	// else goes to bifrostUpstream. Keep in sync with plugin's
	// registered routes (see gateway/main.go startPluginServer).
	pluginPathPrefix = "/_plugin/"

	// readinessTimeout is how long we wait for bifrost-http to become
	// reachable before giving up and exiting. Generous because Bifrost
	// initialises SQLite, runs migrations, and loads our .so on first
	// boot.
	readinessTimeout = 60 * time.Second

	// readinessInterval is the poll interval used while waiting for
	// upstreams to come up.
	readinessInterval = 250 * time.Millisecond

	// childKillGrace is how long we wait for bifrost-http to exit
	// cleanly after we send SIGTERM. Bifrost flushes the SQLite WAL
	// on shutdown so a too-short grace risks log loss; matches the
	// HEALTHCHECK timeout in the Dockerfile.
	childKillGrace = 10 * time.Second
)

func main() {
	var (
		publicAddr  string
		bifrostBin  string
		bifrostArgs argsFlag
	)
	flag.StringVar(&publicAddr, "listen", defaultPublicAddr,
		"public address to listen on (host:port)")
	flag.StringVar(&bifrostBin, "bifrost-bin", "/app/bifrost-http",
		"path to the bifrost-http binary")
	flag.Var(&bifrostArgs, "bifrost-arg",
		"argument to pass to bifrost-http (repeatable; default arg set used if none given)")
	flag.Parse()

	// If no -bifrost-arg flags were provided, use the same default
	// command line we used to set in the Dockerfile's CMD. This keeps
	// the wrapper a drop-in replacement: people who used to inspect
	// `docker inspect ... .Config.Cmd` see equivalent behaviour.
	if len(bifrostArgs) == 0 {
		bifrostArgs = defaultBifrostArgs()
	}

	logger := log.New(os.Stderr, "[wrapper] ", log.LstdFlags|log.LUTC)

	// Start bifrost-http. The cancelable context is what we use to
	// terminate it cleanly on shutdown.
	rootCtx, cancelRoot := context.WithCancel(context.Background())
	defer cancelRoot()

	cmd := exec.CommandContext(rootCtx, bifrostBin, bifrostArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = nil
	// Inherit env so BIFROST_ADMIN_USER / PASS / PROVISIONING_TOKEN
	// and provider keys flow through.
	cmd.Env = os.Environ()
	// Put bifrost-http in its own process group so we can deliver
	// signals to the whole tree (Bifrost spawns helper goroutines
	// that share the PID, but a separate group lets a future patch
	// add real subprocesses without breaking signal forwarding).
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		logger.Fatalf("failed to start bifrost-http: %v", err)
	}
	logger.Printf("started bifrost-http pid=%d", cmd.Process.Pid)

	// Track child exit. If bifrost crashes/exits on its own, we exit
	// with the same status so docker-compose's restart policy kicks in.
	childDone := make(chan error, 1)
	go func() { childDone <- cmd.Wait() }()

	// Listen for OS signals BEFORE we start waiting on upstreams, so
	// a SIGTERM delivered during slow startup still triggers graceful
	// shutdown rather than blocking on the readiness probe.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Wait for bifrost-http to bind its listener. If it dies during
	// startup, surface that immediately instead of timing out.
	readinessCtx, cancelReadiness := context.WithTimeout(rootCtx, readinessTimeout)
	defer cancelReadiness()
	if err := waitForUpstream(readinessCtx, logger, "bifrost-http",
		bifrostUpstream+"/health", childDone); err != nil {
		shutdownChild(logger, cmd)
		logger.Fatalf("bifrost-http never became ready: %v", err)
	}

	// Plugin server is optional — it only starts if BIFROST_ADMIN_*
	// env vars are set (see gateway/main.go). Best-effort probe with
	// a short timeout so missing-env dev setups don't block startup.
	pluginReady := false
	pluginCtx, cancelPlugin := context.WithTimeout(rootCtx, 5*time.Second)
	if err := waitForUpstream(pluginCtx, logger, "plugin",
		pluginUpstream+"/_plugin/health", childDone); err == nil {
		pluginReady = true
	} else {
		logger.Printf("plugin server not ready (continuing without /_plugin/* routes): %v", err)
	}
	cancelPlugin()

	// Build the proxy and start the public server.
	handler := newProxy(logger, pluginReady)
	server := &http.Server{
		Addr:              publicAddr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
		// No ReadTimeout / WriteTimeout: SSE streams from Bifrost can
		// be arbitrarily long. The upstream is responsible for its
		// own timeouts (bifrost has them) and clients can hang up.
		IdleTimeout: 120 * time.Second,
	}

	serverDone := make(chan error, 1)
	go func() {
		logger.Printf("listening on %s (plugin_routes_enabled=%t)",
			publicAddr, pluginReady)
		serverDone <- server.ListenAndServe()
	}()

	// Wait for either a shutdown signal, the public server to error
	// out, or bifrost-http to exit on its own.
	select {
	case sig := <-sigCh:
		logger.Printf("received %s, shutting down", sig)
	case err := <-childDone:
		logger.Printf("bifrost-http exited unexpectedly: %v", err)
		shutdownPublicServer(server, logger)
		os.Exit(exitCodeFromErr(err))
	case err := <-serverDone:
		if !errors.Is(err, http.ErrServerClosed) {
			logger.Printf("public server failed: %v", err)
		}
		shutdownChild(logger, cmd)
		os.Exit(1)
	}

	// Graceful path: close public listener, then bifrost-http.
	shutdownPublicServer(server, logger)
	shutdownChild(logger, cmd)
	// Drain childDone so we don't leak the goroutine on exit.
	select {
	case <-childDone:
	case <-time.After(childKillGrace):
	}
	logger.Printf("shutdown complete")
}

// defaultBifrostArgs is the canonical command line we used to bake
// into the Dockerfile's CMD. Kept here so the image's behaviour is
// unchanged after the wrapper takes over PID 1.
//
// `-host 127.0.0.1` is the critical change: bifrost-http now binds to
// loopback only; the wrapper is the only public listener.
func defaultBifrostArgs() []string {
	return []string{
		"-app-dir", "/app/data",
		"-host", "127.0.0.1",
		"-port", "8080",
		"-log-level", "info",
		"-log-style", "pretty",
	}
}

// newProxy builds the public HTTP handler. Routes `/_plugin/*` to the
// plugin's loopback server (if reachable) and everything else to
// bifrost-http.
func newProxy(logger *log.Logger, pluginReady bool) http.Handler {
	bifrostURL := mustParseURL(bifrostUpstream)
	bifrostProxy := newSingleHostReverseProxy(bifrostURL, logger, "bifrost")

	var pluginProxy *httputil.ReverseProxy
	if pluginReady {
		pluginURL := mustParseURL(pluginUpstream)
		pluginProxy = newSingleHostReverseProxy(pluginURL, logger, "plugin")
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, pluginPathPrefix) {
			if pluginProxy == nil {
				// Surface a clear error so callers don't think the
				// route is silently being routed to bifrost (which
				// would return Bifrost's generic 404).
				http.Error(w,
					"plugin server is not running (check BIFROST_ADMIN_USER/PASS/PROVISIONING_TOKEN env vars)",
					http.StatusServiceUnavailable)
				return
			}
			pluginProxy.ServeHTTP(w, r)
			return
		}
		bifrostProxy.ServeHTTP(w, r)
	})
}

// newSingleHostReverseProxy builds an httputil.ReverseProxy targeting
// the given URL. We add a custom error handler so upstream failures
// (e.g. connection refused after a crash) become 502s with a useful
// log line instead of a panic.
func newSingleHostReverseProxy(target *url.URL, logger *log.Logger, name string) *httputil.ReverseProxy {
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		logger.Printf("proxy error upstream=%s method=%s path=%s err=%v",
			name, r.Method, r.URL.Path, err)
		http.Error(w, "upstream unavailable", http.StatusBadGateway)
	}
	// FlushInterval=-1 enables immediate flush for streaming responses
	// (SSE / chunked). Bifrost streams /v1/chat/completions with
	// `text/event-stream`, and without this flag chunks would buffer
	// until the connection closed — breaking streaming UX entirely.
	proxy.FlushInterval = -1
	return proxy
}

// waitForUpstream polls the given URL until it returns ANY HTTP
// response (including 4xx/5xx — we just want to know the listener is
// up). Returns nil on success, or the context error / child-exit error.
func waitForUpstream(
	ctx context.Context,
	logger *log.Logger,
	name, probeURL string,
	childDone <-chan error,
) error {
	client := &http.Client{
		Timeout: 1 * time.Second,
		// Don't follow redirects — we just want to confirm the socket
		// answers.
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	deadline := time.Now().Add(readinessTimeout)
	logger.Printf("waiting for %s upstream at %s", name, probeURL)

	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, probeURL, nil)
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			logger.Printf("%s upstream ready (status=%d)", name, resp.StatusCode)
			return nil
		}

		select {
		case <-ctx.Done():
			return fmt.Errorf("%s readiness wait: %w", name, ctx.Err())
		case childErr := <-childDone:
			// Caller treats a nil here as "child died" if they care;
			// we propagate the error to make the failure mode obvious.
			if childErr == nil {
				childErr = errors.New("bifrost-http exited during startup")
			}
			return fmt.Errorf("%s readiness wait aborted: %w", name, childErr)
		case <-time.After(readinessInterval):
			if time.Now().After(deadline) {
				return fmt.Errorf("%s readiness wait: timeout (last error: %v)", name, err)
			}
		}
	}
}

// shutdownPublicServer closes the public listener with a small grace
// period. Existing connections finish; new ones are refused.
func shutdownPublicServer(server *http.Server, logger *log.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Printf("public server shutdown error: %v", err)
	}
}

// shutdownChild sends SIGTERM to bifrost-http and waits up to
// childKillGrace for it to exit. After the grace period it sends
// SIGKILL. Idempotent — safe to call after the child has already
// exited.
func shutdownChild(logger *log.Logger, cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	// ProcessState is non-nil iff Wait() returned, which means the
	// child has already exited. Avoid signaling a zombie.
	if cmd.ProcessState != nil {
		return
	}
	logger.Printf("sending SIGTERM to bifrost-http (pid=%d)", cmd.Process.Pid)
	if err := cmd.Process.Signal(syscall.SIGTERM); err != nil &&
		!errors.Is(err, os.ErrProcessDone) {
		logger.Printf("failed to send SIGTERM: %v", err)
	}

	// Race the grace period against the child actually exiting. We
	// can't Wait() here because main()'s goroutine already does;
	// instead poll ProcessState via Signal(0) which returns
	// os.ErrProcessDone once the child has been reaped.
	deadline := time.Now().Add(childKillGrace)
	for time.Now().Before(deadline) {
		if err := cmd.Process.Signal(syscall.Signal(0)); err != nil {
			// Process gone — Wait() in the other goroutine will have
			// picked it up.
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	logger.Printf("bifrost-http did not exit in %s, sending SIGKILL", childKillGrace)
	_ = cmd.Process.Kill()
}

// exitCodeFromErr extracts the child's exit status from cmd.Wait's
// error, or returns 1 if the child crashed without a clean exit.
func exitCodeFromErr(err error) int {
	if err == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	return 1
}

func mustParseURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		panic(fmt.Sprintf("invalid upstream URL %q: %v", s, err))
	}
	return u
}

// argsFlag is a repeatable string flag for -bifrost-arg.
type argsFlag []string

func (a *argsFlag) String() string { return strings.Join(*a, " ") }
func (a *argsFlag) Set(v string) error {
	*a = append(*a, v)
	return nil
}
