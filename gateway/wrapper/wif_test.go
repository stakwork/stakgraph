package main

import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"strings"
	"testing"
)

const wifSeed = `{
  "client": { "log_retention_days": 365 },
  "providers": {
    "anthropic": {
      "keys": [
        {
          "name": "anthropic-key-1",
          "value": "env.ANTHROPIC_API_KEY",
          "models": ["*"],
          "weight": 1.0
        }
      ]
    },
    "openai": {
      "keys": [
        {
          "name": "openai-key-1",
          "value": "env.OPENAI_API_KEY",
          "models": ["*"],
          "weight": 1.0
        }
      ]
    }
  },
  "config_store": { "enabled": true }
}
`

func captureLogger() (*log.Logger, *bytes.Buffer) {
	var b bytes.Buffer
	return log.New(&b, "", 0), &b
}

var allFed = map[string]string{
	"ANTHROPIC_FEDERATION_RULE_ID": "fdrl_1",
	"ANTHROPIC_ORGANIZATION_ID":    "org-1",
	"ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_1",
}

func with(m map[string]string, kv ...string) map[string]string {
	out := map[string]string{}
	for k, v := range m {
		out[k] = v
	}
	for i := 0; i+1 < len(kv); i += 2 {
		out[kv[i]] = kv[i+1]
	}
	return out
}

func TestModeAPIKeyWins(t *testing.T) {
	l, buf := captureLogger()
	env := with(allFed, "ANTHROPIC_API_KEY", "sk-ant-x", "ANTHROPIC_WORKSPACE_ID", "wrkspc_1")
	if w := anthropicAuthFromEnv(envMap(env), l); w != nil {
		t.Fatalf("api key set must not select WIF: %+v", w)
	}
	if !strings.Contains(buf.String(), "mode: api-key") || !strings.Contains(buf.String(), "wins") {
		t.Errorf("log: %s", buf.String())
	}
	if strings.Contains(buf.String(), "sk-ant-x") {
		t.Error("key leaked into the log")
	}
	// Whitespace-only counts as empty.
	l, _ = captureLogger()
	if w := anthropicAuthFromEnv(envMap(with(allFed, "ANTHROPIC_API_KEY", "  ")), l); w == nil {
		t.Error("blank key + all federation vars should select WIF")
	}
}

func TestModeAPIKeySeedUnchanged(t *testing.T) {
	seedPath, appDir := writeSeed(t, wifSeed)
	l, _ := captureLogger()
	wif := anthropicAuthFromEnv(envMap(with(allFed, "ANTHROPIC_API_KEY", "k")), l)
	if _, err := syncSeedConfigWIF(quietLogger(), seedPath, appDir, nil, wif); err != nil {
		t.Fatal(err)
	}
	if got := readConfig(t, appDir); got != wifSeed {
		t.Errorf("config.json != seed:\n%s", got)
	}
}

func TestModeWorkloadIdentity(t *testing.T) {
	for _, ws := range []string{"", "wrkspc_1"} {
		l, buf := captureLogger()
		w := anthropicAuthFromEnv(envMap(with(allFed, "ANTHROPIC_WORKSPACE_ID", ws)), l)
		if w == nil {
			t.Fatalf("workspace=%q: want WIF", ws)
		}
		if w.HasWorkspace != (ws != "") || w.TokenFile != defaultIdentityTokenFile {
			t.Errorf("%+v", w)
		}
		if !strings.Contains(buf.String(), "mode: workload-identity") {
			t.Errorf("log: %s", buf.String())
		}
	}
	w := anthropicAuthFromEnv(envMap(with(allFed, "ANTHROPIC_IDENTITY_TOKEN_FILE", "/run/secrets/a/t")), quietLogger())
	if w.TokenFile != "/run/secrets/a/t" {
		t.Errorf("token file %q", w.TokenFile)
	}
}

func TestModeIncompleteWarns(t *testing.T) {
	for _, drop := range []string{"ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID"} {
		l, buf := captureLogger()
		env := with(allFed, drop, "", "ANTHROPIC_WORKSPACE_ID", "wrkspc_1")
		if w := anthropicAuthFromEnv(envMap(env), l); w != nil {
			t.Fatalf("missing %s must not select WIF", drop)
		}
		if !strings.Contains(buf.String(), "WARNING") || !strings.Contains(buf.String(), drop) {
			t.Errorf("missing %s: log %q", drop, buf.String())
		}
		seedPath, appDir := writeSeed(t, wifSeed)
		if _, err := syncSeedConfigWIF(quietLogger(), seedPath, appDir, nil, nil); err != nil {
			t.Fatal(err)
		}
		if readConfig(t, appDir) != wifSeed {
			t.Error("config.json != seed")
		}
	}
	// Nothing configured at all: warns too.
	l, buf := captureLogger()
	if anthropicAuthFromEnv(envMap(nil), l) != nil || !strings.Contains(buf.String(), "WARNING") {
		t.Errorf("log: %q", buf.String())
	}
}

type keyJSON struct {
	Name   string          `json:"name"`
	Value  *string         `json:"value"`
	Models []string        `json:"models"`
	Weight float64         `json:"weight"`
	AKC    json.RawMessage `json:"anthropic_key_config"`
}

func parseAnthropicKey(t *testing.T, cfg string) keyJSON {
	t.Helper()
	var doc struct {
		Providers map[string]struct {
			Keys []keyJSON `json:"keys"`
		} `json:"providers"`
	}
	if err := json.Unmarshal([]byte(cfg), &doc); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, cfg)
	}
	return doc.Providers["anthropic"].Keys[0]
}

func TestApplyWIFBlock(t *testing.T) {
	for _, ws := range []bool{false, true} {
		w := &wifConfig{TokenFile: "/var/run/secrets/anthropic.com/token", HasWorkspace: ws}
		out, err := applyWIF([]byte(wifSeed), w)
		if err != nil {
			t.Fatal(err)
		}
		k := parseAnthropicKey(t, string(out))
		if k.Value != nil {
			t.Error("value must be removed")
		}
		if k.Name != "anthropic-key-1" || len(k.Models) != 1 || k.Models[0] != "*" || k.Weight != 1.0 {
			t.Errorf("name/models/weight not kept: %+v", k)
		}
		var akc map[string]map[string]any
		if err := json.Unmarshal(k.AKC, &akc); err != nil {
			t.Fatal(err)
		}
		wi := akc["workload_identity"]
		if wi["federation_rule_id"] != "env.ANTHROPIC_FEDERATION_RULE_ID" ||
			wi["organization_id"] != "env.ANTHROPIC_ORGANIZATION_ID" ||
			wi["service_account_id"] != "env.ANTHROPIC_SERVICE_ACCOUNT_ID" {
			t.Errorf("ids: %v", wi)
		}
		_, has := wi["workspace_id"]
		if has != ws || (ws && wi["workspace_id"] != "env.ANTHROPIC_WORKSPACE_ID") {
			t.Errorf("workspace_id present=%t want %t", has, ws)
		}
		src, _ := wi["identity_source"].(map[string]any)
		if src["type"] != "file" || src["path"] != "/var/run/secrets/anthropic.com/token" || len(src) != 2 {
			t.Errorf("identity_source: %v", src)
		}
		if !strings.HasSuffix(string(out), "}\n") {
			t.Error("trailing newline lost")
		}
		// Other providers keep their bytes.
		openai := `"openai": {
      "keys": [
        {
          "name": "openai-key-1",
          "value": "env.OPENAI_API_KEY",
          "models": ["*"],
          "weight": 1.0
        }
      ]
    }`
		if !strings.Contains(string(out), openai) || !strings.Contains(string(out), `"client": { "log_retention_days": 365 }`) {
			t.Errorf("untouched parts changed:\n%s", out)
		}
		// Idempotent: a second pass is a no-op.
		again, err := applyWIF(out, w)
		if err != nil || !bytes.Equal(again, out) {
			t.Errorf("not idempotent (err=%v)", err)
		}
	}
}

func TestApplyWIFErrors(t *testing.T) {
	w := &wifConfig{TokenFile: "/x"}
	for name, seed := range map[string]string{
		"not json":     `nope`,
		"no providers": `{"client":{}}`,
		"no anthropic": `{"providers":{"openai":{}}}`,
		"no keys":      `{"providers":{"anthropic":{}}}`,
		"empty keys":   `{"providers":{"anthropic":{"keys":[]}}}`,
		"key not obj":  `{"providers":{"anthropic":{"keys":["x"]}}}`,
	} {
		if _, err := applyWIF([]byte(seed), w); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

func TestSyncSeedConfigWIFWithOffloadAndFallback(t *testing.T) {
	seedPath, appDir := writeSeed(t, wifSeed)
	o := &logsOffload{Bucket: "b", Prefix: "p"}
	applied, err := syncSeedConfigWIF(quietLogger(), seedPath, appDir, o, &wifConfig{TokenFile: "/var/run/secrets/anthropic.com/token"})
	if err != nil || applied == nil {
		t.Fatalf("applied=%v err=%v", applied, err)
	}
	got := readConfig(t, appDir)
	if !strings.Contains(got, `"logs_store"`) || !strings.Contains(got, `"workload_identity"`) || strings.Contains(got, "env.ANTHROPIC_API_KEY") {
		t.Errorf("config:\n%s", got)
	}
	parseAnthropicKey(t, got)

	// A seed that cannot take the rewrite is written as is, with an ERROR.
	seedPath, appDir = writeSeed(t, testSeed)
	l, buf := captureLogger()
	if _, err := syncSeedConfigWIF(l, seedPath, appDir, nil, &wifConfig{TokenFile: "/x"}); err != nil {
		t.Fatal(err)
	}
	if readConfig(t, appDir) != testSeed || !strings.Contains(buf.String(), "ERROR") {
		t.Errorf("fallback: %s", buf.String())
	}
}

// The real image seed must take the rewrite and stay valid JSON.
func TestRealSeedTakesWIF(t *testing.T) {
	seed, err := os.ReadFile("../data/config.json")
	if err != nil {
		t.Skip("no seed in this checkout")
	}
	out, err := applyWIF(seed, &wifConfig{TokenFile: defaultIdentityTokenFile, HasWorkspace: true})
	if err != nil {
		t.Fatal(err)
	}
	k := parseAnthropicKey(t, string(out))
	if k.Name != "anthropic-key-1" || k.Value != nil || k.AKC == nil {
		t.Errorf("%+v", k)
	}
	if !json.Valid(out) {
		t.Fatal("invalid JSON")
	}
}
