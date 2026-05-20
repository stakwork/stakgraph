package auth

import (
	"encoding/json"
	"sync"
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
type Config struct {
	// EnforceMacaroons gates whether verification failures actually
	// reject the request. When false (default, shadow mode), the
	// adapter still verifies + stamps claims on success, still logs
	// mismatches loudly, but lets the request continue. When true,
	// failures short-circuit with a bifrost.Error.
	EnforceMacaroons bool `json:"enforce_macaroons"`

	// AgentBudgets is the per-agent windowed spend cap declared
	// in plugin.yaml. Phase 6's PreLLMHook reads these to gate
	// inference; phase 8's dashboard reads them to render the
	// budget column / progress bar. Empty map ⇒ no agent has a
	// cap, every request passes the budget check.
	AgentBudgets map[string]AgentBudget `json:"agent_budgets"`
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

// pluginConfigEnvelope mirrors the shape pluginlog.Init receives. We
// don't import its internal type; this is a private decoder for the
// subset of fields auth cares about. Bifrost passes the raw `config`
// JSON object verbatim, so json.Marshal+Unmarshal round-trips through
// whatever shape it actually has.
type pluginConfigEnvelope struct {
	EnforceMacaroons bool                   `json:"enforce_macaroons"`
	AgentBudgets     map[string]AgentBudget `json:"agent_budgets"`
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
// (shadow mode) is cached.
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
	if raw == nil {
		return Config{}, nil
	}
	// Round-trip through JSON so we accept whatever map shape
	// Bifrost decoded the block into. No reflection on field names.
	buf, err := json.Marshal(raw)
	if err != nil {
		return Config{}, err
	}
	var env pluginConfigEnvelope
	if err := json.Unmarshal(buf, &env); err != nil {
		return Config{}, err
	}
	return Config{
		EnforceMacaroons: env.EnforceMacaroons,
		AgentBudgets:     env.AgentBudgets,
	}, nil
}
