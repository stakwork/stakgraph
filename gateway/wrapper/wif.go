// Anthropic workload identity federation (WIF) from an EC2 instance
// profile.
//
// Mode selection, decided ONCE at boot from the environment:
//
//	ANTHROPIC_API_KEY non-empty                      -> api-key. Nothing
//	    changes: config.json is the seed, byte for byte, even when the
//	    federation variables are also set (the API key wins, as it does
//	    inside Bifrost for a key that carries both).
//	ANTHROPIC_API_KEY empty and ANTHROPIC_FEDERATION_RULE_ID,
//	    ANTHROPIC_ORGANIZATION_ID, ANTHROPIC_SERVICE_ACCOUNT_ID all set -> workload-identity
//	    (ANTHROPIC_WORKSPACE_ID optional, passed through when set).
//	anything else                                    -> none. config.json is
//	    the seed, a warning says what is missing.
//
// In workload-identity mode the config.json written to the app dir has
// providers.anthropic.keys[0] rewritten: `value` removed and an
// anthropic_key_config.workload_identity block added (name, models and
// weight kept). The identifiers stay env.* references, so no secret or
// id lands in the file. Bifrost does not call AWS: the identity token
// file is minted and refreshed by the id-token helper (gateway/idtoken),
// which the wrapper runs once synchronously before bifrost-http starts
// and then supervises for the process lifetime.
//
// The key keeps its NAME on purpose. Bifrost (transports/bifrost-http/
// lib/config.go mergeProviderKeys / reconcileProviderKeys) matches a
// config.json key to the one in config.db by name or hash, keeps the
// stored key ID and lets the file's version win when the file changed;
// the key hash covers anthropic_key_config, so a switch in either
// direction is a change and is written through to config.db. A new name
// would instead delete the old key row, and with it the key's links from
// virtual-key provider configs.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

const (
	// defaultIDTokenBin is where the Dockerfile puts the helper.
	defaultIDTokenBin = "/app/id-token"

	// defaultIdentityTokenFile matches the helper's default and sits in
	// a directory Bifrost may read identity tokens from.
	defaultIdentityTokenFile = "/var/run/secrets/anthropic.com/token"

	// idTokenOnceTimeout bounds the boot-time mint (the helper bounds
	// its own STS call at 30s).
	idTokenOnceTimeout = 60 * time.Second
)

// wifConfig is the workload identity block to write into config.json.
type wifConfig struct {
	TokenFile    string
	HasWorkspace bool
}

// anthropicAuthFromEnv picks the mode (see the file comment) and logs
// which one it chose. It returns nil unless the mode is workload-identity.
func anthropicAuthFromEnv(getenv func(string) string, logger *log.Logger) *wifConfig {
	set := func(k string) bool { return strings.TrimSpace(getenv(k)) != "" }
	required := []string{"ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID"}

	var missing, present []string
	for _, k := range append(required, "ANTHROPIC_WORKSPACE_ID") {
		if set(k) {
			present = append(present, k)
		} else if k != "ANTHROPIC_WORKSPACE_ID" {
			missing = append(missing, k)
		}
	}

	if set("ANTHROPIC_API_KEY") {
		if len(present) > 0 {
			logger.Printf("anthropic auth mode: api-key (ANTHROPIC_API_KEY is set, so it wins; %s ignored)",
				strings.Join(present, ", "))
		} else {
			logger.Printf("anthropic auth mode: api-key")
		}
		return nil
	}
	if len(missing) > 0 {
		if len(present) == 0 {
			logger.Printf("WARNING: anthropic auth mode: none (ANTHROPIC_API_KEY is empty and workload identity federation is not configured); anthropic requests will fail")
		} else {
			logger.Printf("WARNING: anthropic auth mode: none (ANTHROPIC_API_KEY is empty and workload identity federation is incomplete: %s not set); anthropic requests will fail",
				strings.Join(missing, ", "))
		}
		return nil
	}

	file := strings.TrimSpace(getenv("ANTHROPIC_IDENTITY_TOKEN_FILE"))
	if file == "" {
		file = defaultIdentityTokenFile
	}
	w := &wifConfig{TokenFile: file, HasWorkspace: set("ANTHROPIC_WORKSPACE_ID")}
	logger.Printf("anthropic auth mode: workload-identity (ANTHROPIC_API_KEY is empty; identity token file %s, workspace_id %t)",
		w.TokenFile, w.HasWorkspace)
	return w
}

// The block's JSON shape: core/schemas/account.go AnthropicWorkloadIdentityConfig.
type wifBlock struct {
	FederationRuleID string         `json:"federation_rule_id"`
	OrganizationID   string         `json:"organization_id"`
	ServiceAccountID string         `json:"service_account_id"`
	WorkspaceID      string         `json:"workspace_id,omitempty"`
	IdentitySource   wifIdentitySrc `json:"identity_source"`
}

type wifIdentitySrc struct {
	Type string `json:"type"`
	Path string `json:"path"`
}

func (w *wifConfig) block() wifBlock {
	b := wifBlock{
		FederationRuleID: "env.ANTHROPIC_FEDERATION_RULE_ID",
		OrganizationID:   "env.ANTHROPIC_ORGANIZATION_ID",
		ServiceAccountID: "env.ANTHROPIC_SERVICE_ACCOUNT_ID",
		IdentitySource:   wifIdentitySrc{Type: "file", Path: w.TokenFile},
	}
	if w.HasWorkspace {
		b.WorkspaceID = "env.ANTHROPIC_WORKSPACE_ID"
	}
	return b
}

// applyWIF returns config with providers.anthropic.keys[0] switched from
// a static value to workload identity. Everything else keeps its bytes;
// only the objects on the path to that key are re-laid-out in the seed's
// two-space style.
func applyWIF(config []byte, w *wifConfig) ([]byte, error) {
	top, err := parseObject(config)
	if err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}
	pi := indexOf(top, "providers")
	if pi < 0 {
		return nil, errors.New("no providers block")
	}
	providers, err := parseObject(top[pi].raw)
	if err != nil {
		return nil, fmt.Errorf("providers: %w", err)
	}
	ai := indexOf(providers, "anthropic")
	if ai < 0 {
		return nil, errors.New("no providers.anthropic block")
	}
	anthropic, err := parseObject(providers[ai].raw)
	if err != nil {
		return nil, fmt.Errorf("providers.anthropic: %w", err)
	}
	ki := indexOf(anthropic, "keys")
	if ki < 0 {
		return nil, errors.New("no providers.anthropic.keys")
	}
	var keys []json.RawMessage
	if err := json.Unmarshal(anthropic[ki].raw, &keys); err != nil || len(keys) == 0 {
		return nil, errors.New("providers.anthropic.keys is not a non-empty array")
	}
	key, err := parseObject(keys[0])
	if err != nil {
		return nil, fmt.Errorf("providers.anthropic.keys[0]: %w", err)
	}

	// Drop the static credential (and any stale block), keep the rest in order.
	kept := key[:0:0]
	for _, m := range key {
		if m.key != "value" && m.key != "anthropic_key_config" {
			kept = append(kept, m)
		}
	}
	block, err := json.MarshalIndent(map[string]any{"workload_identity": w.block()}, "          ", "  ")
	if err != nil {
		return nil, err
	}
	kept = append(kept, member{key: "anthropic_key_config", raw: block})

	// Re-assemble inside out. Array elements sit at 8 spaces.
	keys[0] = emitObject(kept, "        ")
	var arr bytes.Buffer
	arr.WriteString("[\n")
	for i, k := range keys {
		arr.WriteString("        ")
		arr.Write(k)
		if i < len(keys)-1 {
			arr.WriteByte(',')
		}
		arr.WriteByte('\n')
	}
	arr.WriteString("      ]")
	anthropic[ki].raw = arr.Bytes()
	providers[ai].raw = emitObject(anthropic, "    ")
	top[pi].raw = emitObject(providers, "  ")

	out := emitObject(top, "")
	if bytes.HasSuffix(config, []byte("\n")) {
		out = append(out, '\n')
	}
	return out, nil
}

// runIDTokenOnce runs `id-token -once`: one token on disk before
// bifrost-http starts.
func runIDTokenOnce(ctx context.Context, bin string) error {
	ctx, cancel := context.WithTimeout(ctx, idTokenOnceTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "-once")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	return cmd.Run()
}

// idTokenSupervisor keeps the id-token helper (refresh loop) running
// until its context ends, restarting it with a growing delay if it exits.
type idTokenSupervisor struct {
	logger   *log.Logger
	bin      string
	retryMin time.Duration
	retryMax time.Duration
}

func (s *idTokenSupervisor) run(ctx context.Context) {
	delay := s.retryMin
	for ctx.Err() == nil {
		started := time.Now()
		cmd := exec.CommandContext(ctx, s.bin)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		cmd.Env = os.Environ()
		// SIGTERM lets the helper finish cleanly; SIGKILL if it does not.
		cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
		cmd.WaitDelay = 5 * time.Second
		err := cmd.Run()
		if ctx.Err() != nil {
			return
		}
		if time.Since(started) > s.retryMax {
			delay = s.retryMin // it ran a good while: not a crash loop
		}
		s.logger.Printf("ERROR: id-token helper exited (%v); restarting in %s. The identity token file is no longer refreshed meanwhile", err, delay)
		select {
		case <-ctx.Done():
			return
		case <-time.After(delay):
		}
		if delay *= 2; delay > s.retryMax {
			delay = s.retryMax
		}
	}
}
