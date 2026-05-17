package auth

import (
	"testing"
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
