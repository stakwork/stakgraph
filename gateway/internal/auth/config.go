package auth

import (
	"encoding/json"
	"sync"

	"github.com/stakwork/stakgraph/gateway/internal/env"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// Config is the auth subset of the plugin's config block in
// gateway/data/config.json. See package doc for the rollout posture.
//
// The block lives under plugins[].config in Bifrost's config.json
// alongside the existing log_level entry, so adding a field here is
// a one-line addition operators can pre-deploy:
//
//	"config": {
//	  "log_level":         "info",
//	  "enforce_macaroons": false,
//	  "agent_budgets": {
//	    "coder":      { "cap_usd": 5.00, "window": "1d" },
//	    "web-search": { "cap_usd": 1.00, "window": "1h" }
//	  }
//	}
//
// Default-zero is shadow mode (verify + log mismatches, don't reject).
// Existing swarms running an old config.json will pick up the new
// adapter in shadow mode without any operator action — that is the
// whole point of the flag. Flip to true per-swarm as rollout proceeds.
//
// The env var BIFROST_PLUGIN_ENFORCE_MACAROONS (see internal/env)
// overrides the config.json value when set, so a swarm can be flipped
// through its environment without rebuilding the image that carries
// config.json. EnforceMacaroonsSource records which one won so the
// boot log can say so.
type Config struct {
	// EnforceMacaroons gates whether verification failures actually
	// reject the request. When false (default, shadow mode), the
	// adapter still verifies + stamps claims on success, still logs
	// mismatches loudly, but lets the request continue. When true,
	// failures short-circuit with a bifrost.Error.
	EnforceMacaroons bool `json:"enforce_macaroons"`

	// EnforceMacaroonsSource is where EnforceMacaroons came from:
	// "env" (BIFROST_PLUGIN_ENFORCE_MACAROONS set and valid), "config"
	// (the key is present in the plugin config block), "default"
	// (neither — zero value, shadow mode), or "env-invalid" (the env
	// var was set to something unparseable, an ERROR was logged, and
	// the config/default value stands). Diagnostic only; never
	// persisted — grep the boot line for it.
	EnforceMacaroonsSource string `json:"-"`

	// AgentBudgets is the per-agent windowed spend cap declared
	// in plugin.yaml. Phase 6's PreLLMHook reads these to gate
	// inference; phase 8's dashboard reads them to render the
	// budget column / progress bar. Empty map ⇒ no agent has a
	// cap, every request passes the budget check.
	AgentBudgets map[string]AgentBudget `json:"agent_budgets"`

	// ModelPricing is the per-model token pricing the phase-6
	// accumulator uses to turn usage into dollars when the provider
	// response doesn't carry its own cost (Usage.Cost is populated
	// by only a handful of providers — Anthropic and OpenAI chat
	// responses don't have it in bifrost core v1.5.x). Keys are
	// resolved model names as Bifrost reports them, with or without
	// the "provider/" prefix. A model with no entry accumulates $0
	// and logs loudly — undercounting is visible in the daily
	// logs.db reconciliation rather than silently guessed at.
	ModelPricing map[string]ModelPrice `json:"model_pricing"`
}

// AgentBudget is one row in `agent_budgets`. Window uses the Bifrost
// duration vocabulary (`1d`, `1w`, `1M`, `1Y`, or sub-day `Nh`/`Nm`/`Ns`).
// See plans/phases/phase-6-plugin-enforcement.md §"Duration vocabulary"
// for the full grammar; phase 8 ships read-only — no validation needed
// beyond "the string is non-empty."
type AgentBudget struct {
	CapUSD float64 `json:"cap_usd"`
	Window string  `json:"window"`
}

// ModelPrice is one row of `model_pricing`: dollars per million
// tokens, the unit every provider publishes prices in. The phase-6
// accumulator computes prompt*input/1e6 + completion*output/1e6.
type ModelPrice struct {
	InputPerMTok  float64 `json:"input_per_mtok"`
	OutputPerMTok float64 `json:"output_per_mtok"`
}

// pluginConfigEnvelope mirrors the shape pluginlog.Init receives. We
// don't import its internal type; this is a private decoder for the
// subset of fields auth cares about. Bifrost passes the raw `config`
// JSON object verbatim, so json.Marshal+Unmarshal round-trips through
// whatever shape it actually has.
//
// EnforceMacaroons is a pointer so "key absent" (source=default) and
// "key present and false" (source=config) stay distinguishable.
type pluginConfigEnvelope struct {
	EnforceMacaroons *bool                  `json:"enforce_macaroons"`
	AgentBudgets     map[string]AgentBudget `json:"agent_budgets"`
	ModelPricing     map[string]ModelPrice  `json:"model_pricing"`
}

var (
	cfgMu sync.RWMutex
	cfg   Config
)

// Init parses the plugin's `config` block (the same `any` Bifrost
// hands to pluginlog.Init / Plugin.Init) and caches the result for
// subsequent calls to GetConfig.
//
// Idempotent: re-calling replaces the cached value. Tests use this
// to flip enforce_macaroons mid-suite without restarting the package.
//
// Safe to call with raw == nil; in that case the zero-value Config
// (shadow mode) is cached — unless the env override is set, which
// applies on top of whatever the block held, nil included.
//
// Only malformed config JSON is an error. An unparseable
// BIFROST_PLUGIN_ENFORCE_MACAROONS value is deliberately NOT fatal:
// a plugin Init error does not stop bifrost-http, it just drops this
// plugin — the wrapper then serves inference with no macaroon
// verification, no claim canonicalization, and /_plugin/* down,
// which is strictly worse than shadow mode. So the typo is logged at
// ERROR, the boot line says source=env-invalid, and the config.json
// value stands.
func Init(raw any) error {
	parsed, err := parseConfig(raw)
	if err != nil {
		return err
	}
	cfgMu.Lock()
	cfg = parsed
	cfgMu.Unlock()
	return nil
}

// GetConfig returns a snapshot of the cached config. Safe to call
// before Init — returns the zero value (shadow mode) in that case,
// which is the same posture as "the operator hasn't enabled
// enforcement yet."
func GetConfig() Config {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return cfg
}

// SetConfigForTest overrides the cached config from tests. Production
// code MUST go through Init.
func SetConfigForTest(c Config) {
	cfgMu.Lock()
	cfg = c
	cfgMu.Unlock()
}

func parseConfig(raw any) (Config, error) {
	cfg := Config{EnforceMacaroonsSource: "default"}
	if raw != nil {
		// Round-trip through JSON so we accept whatever map shape
		// Bifrost decoded the block into. No reflection on field names.
		buf, err := json.Marshal(raw)
		if err != nil {
			return Config{}, err
		}
		var envelope pluginConfigEnvelope
		if err := json.Unmarshal(buf, &envelope); err != nil {
			return Config{}, err
		}
		cfg.AgentBudgets = envelope.AgentBudgets
		cfg.ModelPricing = envelope.ModelPricing
		if envelope.EnforceMacaroons != nil {
			cfg.EnforceMacaroons = *envelope.EnforceMacaroons
			cfg.EnforceMacaroonsSource = "config"
		}
	}
	// Env override wins over the config block. A value we can't
	// parse keeps the plugin alive on the config value (see Init for
	// why fatal would be worse) but must be impossible to miss: an
	// ERROR line here and "source=env-invalid" on the boot line.
	if v, set, err := env.EnforceMacaroonsValue(); err != nil {
		pluginlog.Errf("auth: %v — ignoring the override; enforce_macaroons=%t from %s stands",
			err, cfg.EnforceMacaroons, cfg.EnforceMacaroonsSource)
		cfg.EnforceMacaroonsSource = "env-invalid"
	} else if set {
		cfg.EnforceMacaroons = v
		cfg.EnforceMacaroonsSource = "env"
	}
	return cfg, nil
}
