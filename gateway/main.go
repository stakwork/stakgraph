// Package main is the Bifrost gateway plugin for stakgraph's LLM
// governance layer.
//
// What this file is
// -----------------
// Bifrost loads us as a Go plugin (`-buildmode=plugin`). Its loader
// (bifrost/framework/plugins/soloader.go) does `plugin.Lookup("Init")`,
// `plugin.Lookup("GetName")`, `plugin.Lookup("HTTPTransportPreHook")`,
// etc., and the symbols MUST sit on the `main` package of the .so.
// So this file is exactly that — every Bifrost-required symbol, each
// implemented as a one-line delegation into an `internal/` package
// where the real logic lives.
//
// If you want to know what a hook actually DOES, jump into
// `internal/hooks/`. If you want to know what HTTP routes the plugin
// exposes outside Bifrost's router, jump into `internal/adminapi/`.
// If you're auditing what bifrost-http calls into this plugin, this
// file is your one-stop shop.
//
// See gateway/README.md for the architectural overview.
package main

import (
	"github.com/maximhq/bifrost/core/schemas"

	"github.com/stakwork/stakgraph/gateway/internal/adminapi"
	"github.com/stakwork/stakgraph/gateway/internal/auth"
	"github.com/stakwork/stakgraph/gateway/internal/hooks"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
	"github.com/stakwork/stakgraph/gateway/internal/redisclient"
	"github.com/stakwork/stakgraph/gateway/internal/trust"
)

// PluginName is the system identifier reported via GetName(). Bifrost
// uses it as a map key, so it must be stable across restarts.
const PluginName = "stakgraph-gateway"

// Init is called once when bifrost-http loads the plugin.
//
// Order matters:
//
//  1. pluginlog.Init — so subsequent log lines are properly tagged.
//  2. trust.LoadFromEnv — applies the precedence rules from
//     gateway/plans/phases/phase-5-trust-registry.md. Failure here
//     is fatal: an unparseable persisted file or "refuse" reconcile
//     mode with a divergent seed must keep us from starting with an
//     ambiguous registry.
//  3. adminapi.SetTrustRegistry — hands the registry to the admin
//     HTTP server so /_plugin/trust/* routes can register.
//  4. redisclient.Init — opens the macaroon-enforcement Redis
//     connection (DBSIZE smoke test logs the size on success). Only
//     a malformed URL is fatal; an unreachable Redis logs a warning
//     and falls back to observability mode. See
//     gateway/plans/phases/phase-6-plugin-enforcement.md "Namespace".
//  5. auth.Init + auth.SetTrustRegistry — parses the plugin's
//     enforce_macaroons flag from the config block and wires the
//     trust registry into the verifier. See
//     gateway/plans/phases/phase-4-macaroon-shape.md ("Bifrost-
//     plugin adapter").
//  6. adminapi.Start — boots the loopback HTTP server.
func Init(config any) error {
	pluginlog.Init(PluginName, config)

	reg, err := trust.LoadFromEnv()
	if err != nil {
		// Fatal — return the error so bifrost-http's plugin loader
		// reports it and refuses to bring the plugin up. Better
		// than starting with a half-loaded registry.
		pluginlog.Errf("trust: %v", err)
		return err
	}
	adminapi.SetTrustRegistry(reg)

	if err := redisclient.Init(); err != nil {
		// Only malformed URLs reach this branch; an unreachable
		// Redis is non-fatal (handled inside redisclient.Init).
		pluginlog.Errf("redis: %v", err)
		return err
	}

	if err := auth.Init(config); err != nil {
		// Only malformed config JSON reaches this branch — same
		// posture as redisclient.Init: a parse error is operator
		// intent gone wrong, fail fast.
		pluginlog.Errf("auth: %v", err)
		return err
	}
	auth.SetTrustRegistry(reg)
	pluginlog.Logf("auth: macaroon adapter wired enforce=%t", auth.GetConfig().EnforceMacaroons)

	return adminapi.Start()
}

// GetName returns the plugin's system identifier.
func GetName() string { return PluginName }

// Cleanup is called on bifrost shutdown.
//
// Close the Redis client first so any in-flight pipeline gets a
// clean error rather than the goroutine leak go-redis produces on
// abrupt process exit; then stop the admin server.
func Cleanup() error {
	if err := redisclient.Close(); err != nil {
		pluginlog.Warnf("redis: close: %v", err)
	}
	return adminapi.Stop()
}

// HTTPTransportPreHook fires at the HTTP transport layer, before the
// request enters Bifrost core. Earliest place we can short-circuit a
// request (return non-nil *HTTPResponse to do so).
func HTTPTransportPreHook(
	ctx *schemas.BifrostContext,
	req *schemas.HTTPRequest,
) (*schemas.HTTPResponse, error) {
	return hooks.TransportPre(ctx, req)
}

// HTTPTransportPostHook fires after the upstream provider call for
// non-streaming responses.
func HTTPTransportPostHook(
	ctx *schemas.BifrostContext,
	req *schemas.HTTPRequest,
	resp *schemas.HTTPResponse,
) error {
	return hooks.TransportPost(ctx, req, resp)
}

// HTTPTransportStreamChunkHook fires once per streamed chunk.
// PostLLMHook/PostHook do NOT fire for streaming responses, so cost
// accounting on streams hooks in here.
func HTTPTransportStreamChunkHook(
	ctx *schemas.BifrostContext,
	req *schemas.HTTPRequest,
	chunk *schemas.BifrostStreamChunk,
) (*schemas.BifrostStreamChunk, error) {
	return hooks.StreamChunk(ctx, req, chunk)
}

// PreLLMHook fires after bifrost has parsed the request into its
// internal BifrostRequest type — first place we see resolved provider
// + model.
func PreLLMHook(
	ctx *schemas.BifrostContext,
	req *schemas.BifrostRequest,
) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return hooks.LLMPre(ctx, req)
}

// PostLLMHook fires after the upstream provider call (or after a
// short-circuit).
func PostLLMHook(
	ctx *schemas.BifrostContext,
	resp *schemas.BifrostResponse,
	bifrostErr *schemas.BifrostError,
) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return hooks.LLMPost(ctx, resp, bifrostErr)
}
