package main

import (
	"context"
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func env(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func quiet() *log.Logger { return log.New(io.Discard, "", 0) }

func TestConfigDefaults(t *testing.T) {
	c, err := configFromEnv(env(map[string]string{"AWS_REGION": "us-east-2"}))
	if err != nil {
		t.Fatal(err)
	}
	want := config{Region: "us-east-2", File: "/var/run/secrets/anthropic.com/token", Audience: "https://api.anthropic.com",
		Duration: 600 * time.Second, Refresh: 240 * time.Second, Algorithm: "RS256"}
	if c != want {
		t.Fatalf("got %+v want %+v", c, want)
	}
}

func TestConfigRegionFallbackAndOverrides(t *testing.T) {
	c, err := configFromEnv(env(map[string]string{
		"AWS_DEFAULT_REGION":                         "eu-west-1",
		"ANTHROPIC_IDENTITY_TOKEN_FILE":              "/run/secrets/x/tok",
		"ANTHROPIC_IDENTITY_TOKEN_AUDIENCE":          "https://aud",
		"ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS":  "900",
		"ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS":   "300",
		"ANTHROPIC_IDENTITY_TOKEN_SIGNING_ALGORITHM": "es384",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if c.Region != "eu-west-1" || c.File != "/run/secrets/x/tok" || c.Audience != "https://aud" ||
		c.Duration != 900*time.Second || c.Refresh != 300*time.Second || c.Algorithm != "ES384" {
		t.Fatalf("got %+v", c)
	}
	// AWS_REGION beats AWS_DEFAULT_REGION.
	c, _ = configFromEnv(env(map[string]string{"AWS_REGION": "a-1", "AWS_DEFAULT_REGION": "b-1"}))
	if c.Region != "a-1" {
		t.Fatalf("region %q", c.Region)
	}
}

func TestConfigErrors(t *testing.T) {
	base := func(k, v string) map[string]string {
		m := map[string]string{"AWS_REGION": "us-east-1"}
		if k != "" {
			m[k] = v
		}
		return m
	}
	cases := map[string]map[string]string{
		"no region":            {},
		"relative file":        base("ANTHROPIC_IDENTITY_TOKEN_FILE", "token"),
		"bad alg":              base("ANTHROPIC_IDENTITY_TOKEN_SIGNING_ALGORITHM", "HS256"),
		"duration NaN":         base("ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS", "abc"),
		"duration too short":   base("ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS", "30"),
		"duration too long":    base("ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS", "3601"),
		"refresh negative":     base("ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS", "-1"),
		"refresh >= duration":  base("ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS", "600"),
		"refresh > duration 2": {"AWS_REGION": "x-1", "ANTHROPIC_IDENTITY_TOKEN_DURATION_SECONDS": "120", "ANTHROPIC_IDENTITY_TOKEN_REFRESH_SECONDS": "240"},
	}
	for name, m := range cases {
		if _, err := configFromEnv(env(m)); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

func TestSTSEndpoint(t *testing.T) {
	if got := stsEndpoint("us-east-2"); got != "https://sts.us-east-2.amazonaws.com" {
		t.Error(got)
	}
	if got := stsEndpoint("cn-north-1"); got != "https://sts.cn-north-1.amazonaws.com.cn" {
		t.Error(got)
	}
}

func TestWriteTokenFileAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a", "b", "token")
	if err := writeTokenFile(path, "one"); err != nil {
		t.Fatal(err)
	}
	if err := writeTokenFile(path, "two"); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "two" {
		t.Fatalf("got %q", got)
	}
	fi, _ := os.Stat(path)
	if fi.Mode().Perm() != 0o600 {
		t.Errorf("mode %v", fi.Mode().Perm())
	}
	ents, _ := os.ReadDir(filepath.Dir(path))
	if len(ents) != 1 {
		t.Errorf("leftover temp files: %v", ents)
	}
}

func TestWriteTokenFileFailureKeepsOld(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "token")
	if err := writeTokenFile(path, "keep"); err != nil {
		t.Fatal(err)
	}
	// A directory where the target file should be makes the rename fail.
	bad := filepath.Join(dir, "d")
	if err := os.Mkdir(bad, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeTokenFile(bad, "x"); err == nil {
		t.Fatal("want error")
	}
	if got, _ := os.ReadFile(path); string(got) != "keep" {
		t.Fatalf("got %q", got)
	}
	ents, _ := os.ReadDir(dir)
	if len(ents) != 2 {
		t.Errorf("leftover temp files: %v", ents)
	}
}

func TestBackoff(t *testing.T) {
	want := []time.Duration{5, 10, 20, 40, 60, 60}
	for i, w := range want {
		if got := backoff(i, 240*time.Second); got != w*time.Second {
			t.Errorf("backoff(%d)=%s", i, got)
		}
	}
	if got := backoff(5, 30*time.Second); got != 30*time.Second {
		t.Errorf("capped by refresh: %s", got)
	}
}

type fakeSource struct {
	mu    sync.Mutex
	calls int
	fn    func(n int) (string, error)
}

func (f *fakeSource) Token(context.Context) (string, error) {
	f.mu.Lock()
	n := f.calls
	f.calls++
	f.mu.Unlock()
	return f.fn(n)
}

func TestMintOnce(t *testing.T) {
	cfg := config{File: filepath.Join(t.TempDir(), "d", "token"), Refresh: time.Second, Duration: time.Minute}
	src := &fakeSource{fn: func(int) (string, error) { return "jwt-1", nil }}
	if err := mintOnce(context.Background(), cfg, src, quiet()); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(cfg.File); string(got) != "jwt-1" {
		t.Fatalf("got %q", got)
	}
	src.fn = func(int) (string, error) { return "", errors.New("denied") }
	if err := mintOnce(context.Background(), cfg, src, quiet()); err == nil {
		t.Fatal("want error")
	}
	if got, _ := os.ReadFile(cfg.File); string(got) != "jwt-1" {
		t.Fatalf("previous token lost: %q", got)
	}
}

func TestLoopRefreshesAndRecovers(t *testing.T) {
	cfg := config{File: filepath.Join(t.TempDir(), "token"), Refresh: 240 * time.Second, Duration: 600 * time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	src := &fakeSource{fn: func(n int) (string, error) {
		switch n {
		case 0:
			return "tok-0", nil
		case 1, 2:
			return "", errors.New("sts unavailable")
		default:
			return "tok-" + string(rune('0'+n)), nil
		}
	}}
	var waits []time.Duration
	var seen []string
	sleep := func(_ context.Context, d time.Duration) bool {
		waits = append(waits, d)
		b, _ := os.ReadFile(cfg.File)
		seen = append(seen, string(b))
		if len(waits) == 5 {
			cancel()
			return false
		}
		return true
	}
	var buf strings.Builder
	logger := log.New(&buf, "", 0)
	if err := loop(ctx, cfg, src, logger, sleep); !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
	wantWaits := []time.Duration{240 * time.Second, 5 * time.Second, 10 * time.Second, 240 * time.Second, 240 * time.Second}
	for i := range wantWaits {
		if waits[i] != wantWaits[i] {
			t.Errorf("wait %d = %s want %s", i, waits[i], wantWaits[i])
		}
	}
	// Previous token stays through the two failures.
	wantSeen := []string{"tok-0", "tok-0", "tok-0", "tok-3", "tok-4"}
	for i := range wantSeen {
		if seen[i] != wantSeen[i] {
			t.Errorf("file after round %d = %q want %q", i, seen[i], wantSeen[i])
		}
	}
	if !strings.Contains(buf.String(), "sts unavailable") || !strings.Contains(buf.String(), "recovered") {
		t.Errorf("log: %s", buf.String())
	}
}

func TestLogsNeverContainToken(t *testing.T) {
	cfg := config{File: filepath.Join(t.TempDir(), "token"), Refresh: time.Second, Duration: time.Minute}
	var buf strings.Builder
	src := &fakeSource{fn: func(int) (string, error) { return "SECRET.JWT.VALUE", nil }}
	if err := mintOnce(context.Background(), cfg, src, log.New(&buf, "", 0)); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(buf.String(), "SECRET") {
		t.Fatalf("token leaked into log: %s", buf.String())
	}
}
