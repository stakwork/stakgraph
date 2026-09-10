package auth

import (
	"testing"

	"github.com/stakwork/stakgraph/gateway/internal/env"
)

func TestInit_NilConfig_DefaultsToShadow(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	if err := Init(nil); err != nil {
		t.Fatalf("Init(nil): %v", err)
	}
	if GetConfig().EnforceMacaroons {
		t.Fatal("nil config should default to enforce_macaroons=false (shadow)")
	}
}

func TestInit_RawJSONConfig(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	// Mirror how Bifrost decodes the plugin's config block — JSON
	// object → map[string]any.
	raw := map[string]any{
		"log_level":         "info",
		"enforce_macaroons": true,
	}
	if err := Init(raw); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if !GetConfig().EnforceMacaroons {
		t.Fatal("enforce_macaroons=true did not stick")
	}
}

func TestInit_RawJSONConfig_FlagOff(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	raw := map[string]any{"enforce_macaroons": false}
	if err := Init(raw); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if GetConfig().EnforceMacaroons {
		t.Fatal("enforce_macaroons=false should produce shadow mode")
	}
}

func TestInit_IgnoresUnknownFields(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	raw := map[string]any{
		"enforce_macaroons":    true,
		"unrelated_future_key": "whatever",
	}
	if err := Init(raw); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if !GetConfig().EnforceMacaroons {
		t.Fatal("unknown sibling fields should not affect parsing")
	}
}

func TestGetConfig_BeforeInit_ReturnsZero(t *testing.T) {
	SetConfigForTest(Config{}) // simulate "never initialized"
	if got := GetConfig(); got.EnforceMacaroons {
		t.Fatalf("pre-Init should be shadow mode, got %+v", got)
	}
}

// --- BIFROST_PLUGIN_ENFORCE_MACAROONS override ---------------------------

func TestInit_Source_ConfigVsDefault(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "")

	if err := Init(map[string]any{"log_level": "info"}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if got := GetConfig(); got.EnforceMacaroons || got.EnforceMacaroonsSource != "default" {
		t.Fatalf("absent key: got enforce=%v source=%q, want false/default", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}

	if err := Init(map[string]any{"enforce_macaroons": false}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if got := GetConfig(); got.EnforceMacaroons || got.EnforceMacaroonsSource != "config" {
		t.Fatalf("explicit false: got enforce=%v source=%q, want false/config", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}
}

func TestInit_EnvOverride_TrueOverConfigFalse(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "true")

	if err := Init(map[string]any{"enforce_macaroons": false}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if got := GetConfig(); !got.EnforceMacaroons || got.EnforceMacaroonsSource != "env" {
		t.Fatalf("got enforce=%v source=%q, want true/env", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}
}

func TestInit_EnvOverride_FalseOverConfigTrue(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "0")

	if err := Init(map[string]any{"enforce_macaroons": true}); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if got := GetConfig(); got.EnforceMacaroons || got.EnforceMacaroonsSource != "env" {
		t.Fatalf("got enforce=%v source=%q, want false/env", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}
}

func TestInit_EnvOverride_AppliesToNilConfig(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "yes")

	if err := Init(nil); err != nil {
		t.Fatalf("Init(nil): %v", err)
	}
	if got := GetConfig(); !got.EnforceMacaroons || got.EnforceMacaroonsSource != "env" {
		t.Fatalf("nil config + env: got enforce=%v source=%q, want true/env", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}
}

func TestInit_EnvOverride_PreservesOtherFields(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "true")

	raw := map[string]any{
		"enforce_macaroons": false,
		"agent_budgets":     map[string]any{"coder": map[string]any{"cap_usd": 5, "window": "1d"}},
		"model_pricing":     map[string]any{"m": map[string]any{"input_per_mtok": 1, "output_per_mtok": 2}},
	}
	if err := Init(raw); err != nil {
		t.Fatalf("Init: %v", err)
	}
	got := GetConfig()
	if !got.EnforceMacaroons || got.AgentBudgets["coder"].CapUSD != 5 || got.ModelPricing["m"].OutputPerMTok != 2 {
		t.Fatalf("override must not drop sibling fields: %+v", got)
	}
}

// An unparseable override must NOT fail Init: a plugin that fails to
// load leaves bifrost-http serving with no verification at all, which
// is worse than either mode. It falls back to the config value and
// flags itself via the source.
func TestInit_EnvOverride_GarbageFallsBackToConfig(t *testing.T) {
	t.Cleanup(func() { SetConfigForTest(Config{}) })
	t.Setenv(env.EnforceMacaroons, "ture")

	if err := Init(map[string]any{"enforce_macaroons": true}); err != nil {
		t.Fatalf("Init must not fail on a bad override: %v", err)
	}
	if got := GetConfig(); !got.EnforceMacaroons || got.EnforceMacaroonsSource != "env-invalid" {
		t.Fatalf("config true + garbage env: got enforce=%v source=%q, want true/env-invalid", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}

	if err := Init(nil); err != nil {
		t.Fatalf("Init(nil) must not fail on a bad override: %v", err)
	}
	if got := GetConfig(); got.EnforceMacaroons || got.EnforceMacaroonsSource != "env-invalid" {
		t.Fatalf("nil config + garbage env: got enforce=%v source=%q, want false/env-invalid", got.EnforceMacaroons, got.EnforceMacaroonsSource)
	}
}
