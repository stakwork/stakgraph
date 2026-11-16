package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func envMap(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestLogsOffloadFromEnv(t *testing.T) {
	cases := []struct {
		name    string
		env     map[string]string
		want    *logsOffload
		wantErr []string // substrings of the error; nil = no error
	}{
		{name: "unset", env: nil, want: nil},
		{name: "empty bucket", env: map[string]string{envLogsS3Bucket: ""}, want: nil},
		{name: "blank bucket", env: map[string]string{envLogsS3Bucket: "  "}, want: nil},
		{
			name: "other vars without a bucket stay off",
			env: map[string]string{
				envLogsS3Prefix:      "swarm38",
				envLogsS3AccessKeyID: "AKIA",
			},
			want: nil,
		},
		{
			name: "bucket only: default prefix, default credential chain",
			env:  map[string]string{envLogsS3Bucket: "fleet-logs"},
			want: &logsOffload{Bucket: "fleet-logs", Prefix: "bifrost"},
		},
		{
			name: "everything",
			env: map[string]string{
				envLogsS3Bucket:          " fleet-logs ",
				envLogsS3Prefix:          "swarm38/bifrost",
				envLogsS3Region:          "us-east-1",
				envLogsS3Endpoint:        "http://minio:9000",
				envLogsS3AccessKeyID:     "AKIA",
				envLogsS3SecretAccessKey: "s3cr3t",
			},
			want: &logsOffload{
				Bucket: "fleet-logs", Prefix: "swarm38/bifrost", Region: "us-east-1",
				Endpoint: "http://minio:9000", StaticKeys: true,
			},
		},
		{
			name: "prefix slashes trimmed",
			env:  map[string]string{envLogsS3Bucket: "b", envLogsS3Prefix: "/swarm38/"},
			want: &logsOffload{Bucket: "b", Prefix: "swarm38"},
		},
		{
			name: "prefix of only slashes falls back to the default",
			env:  map[string]string{envLogsS3Bucket: "b", envLogsS3Prefix: "/"},
			want: &logsOffload{Bucket: "b", Prefix: "bifrost"},
		},
		{
			name:    "access key id without secret",
			env:     map[string]string{envLogsS3Bucket: "b", envLogsS3AccessKeyID: "AKIA"},
			wantErr: []string{envLogsS3AccessKeyID + " is set but " + envLogsS3SecretAccessKey + " is not"},
		},
		{
			name:    "secret without access key id",
			env:     map[string]string{envLogsS3Bucket: "b", envLogsS3SecretAccessKey: "s3cr3t"},
			wantErr: []string{envLogsS3SecretAccessKey + " is set but " + envLogsS3AccessKeyID + " is not"},
		},
		{
			name: "blank secret counts as unset",
			env: map[string]string{
				envLogsS3Bucket: "b", envLogsS3AccessKeyID: "AKIA", envLogsS3SecretAccessKey: " ",
			},
			wantErr: []string{envLogsS3AccessKeyID + " is set but"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := logsOffloadFromEnv(envMap(tc.env))
			if tc.wantErr != nil {
				if err == nil {
					t.Fatalf("want error, got %+v", got)
				}
				for _, s := range tc.wantErr {
					if !strings.Contains(err.Error(), s) {
						t.Errorf("error %q does not contain %q", err, s)
					}
				}
				if strings.Contains(err.Error(), "s3cr3t") || strings.Contains(err.Error(), "AKIA") {
					t.Errorf("error leaks a key value: %q", err)
				}
				if got != nil {
					t.Errorf("want nil offload with an error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestLogsOffloadStringHasNoSecrets(t *testing.T) {
	o, err := logsOffloadFromEnv(envMap(map[string]string{
		envLogsS3Bucket:          "b",
		envLogsS3Endpoint:        "http://minio:9000",
		envLogsS3AccessKeyID:     "AKIAEXAMPLE",
		envLogsS3SecretAccessKey: "s3cr3t-value",
	}))
	if err != nil {
		t.Fatal(err)
	}
	s := o.String()
	if strings.Contains(s, "AKIAEXAMPLE") || strings.Contains(s, "s3cr3t-value") {
		t.Errorf("String() leaks a key: %q", s)
	}
	if want := "s3://b/bifrost/logs/"; !strings.Contains(s, want) {
		t.Errorf("String() = %q, want it to contain %q", s, want)
	}
	if (*logsOffload)(nil).String() != "off" {
		t.Errorf("nil String() = %q", (*logsOffload)(nil).String())
	}
}

// decodeLogsStore renders seed with o and returns the decoded config and
// its logs_store block.
func decodeLogsStore(t *testing.T, seed []byte, o *logsOffload) (map[string]json.RawMessage, map[string]any) {
	t.Helper()
	out, err := renderConfig(seed, o, "/app/data")
	if err != nil {
		t.Fatalf("renderConfig: %v", err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(out, &top); err != nil {
		t.Fatalf("rendered config is not valid JSON: %v\n%s", err, out)
	}
	var ls map[string]any
	if err := json.Unmarshal(top["logs_store"], &ls); err != nil {
		t.Fatalf("logs_store: %v", err)
	}
	return top, ls
}

const testSeed = `{
  "client": { "log_retention_days": 365 },
  "plugins": [ { "name": "stakgraph-gateway" } ],
  "config_store": { "enabled": true, "type": "sqlite", "config": { "path": "/app/data/config.db" } }
}
`

func TestRenderConfigOffOrInvalidKeepsSeed(t *testing.T) {
	out, err := renderConfig([]byte(testSeed), nil, "/app/data")
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != testSeed {
		t.Errorf("offload off must return the seed byte for byte, got:\n%s", out)
	}
}

func TestRenderConfigAddsLogsStore(t *testing.T) {
	o := &logsOffload{Bucket: "fleet-logs", Prefix: "swarm38"}
	top, ls := decodeLogsStore(t, []byte(testSeed), o)

	// Every seed key is still there, unchanged, but for the retention.
	var seedTop map[string]json.RawMessage
	if err := json.Unmarshal([]byte(testSeed), &seedTop); err != nil {
		t.Fatal(err)
	}
	for k, v := range seedTop {
		if k == "client" {
			continue
		}
		if !jsonEqual(t, top[k], v) {
			t.Errorf("seed key %q changed: %s -> %s", k, v, top[k])
		}
	}
	if !jsonEqual(t, top["client"], json.RawMessage(`{"log_retention_days": 30}`)) {
		t.Errorf("client = %s, want log_retention_days 30", top["client"])
	}
	if len(top) != len(seedTop)+1 {
		t.Errorf("want seed keys + logs_store, got %d keys", len(top))
	}

	want := map[string]any{
		"enabled": true,
		"type":    "sqlite",
		"config":  map[string]any{"path": "/app/data/logs.db"},
		"object_storage": map[string]any{
			"type":     "s3",
			"bucket":   "fleet-logs",
			"prefix":   "swarm38",
			"compress": true,
		},
	}
	if !reflect.DeepEqual(ls, want) {
		t.Errorf("logs_store =\n%v\nwant\n%v", ls, want)
	}
}

func TestRenderConfigEndpointKeysRegion(t *testing.T) {
	o := &logsOffload{
		Bucket: "b", Prefix: "p", Region: "auto",
		Endpoint: "https://acct.r2.cloudflarestorage.com", StaticKeys: true,
	}
	_, ls := decodeLogsStore(t, []byte(testSeed), o)
	got := ls["object_storage"].(map[string]any)
	want := map[string]any{
		"type":              "s3",
		"bucket":            "b",
		"prefix":            "p",
		"compress":          true,
		"region":            "auto",
		"endpoint":          "https://acct.r2.cloudflarestorage.com",
		"force_path_style":  true,
		"access_key_id":     "env.BIFROST_LOGS_S3_ACCESS_KEY_ID",
		"secret_access_key": "env.BIFROST_LOGS_S3_SECRET_ACCESS_KEY",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("object_storage =\n%v\nwant\n%v", got, want)
	}
}

// The key values themselves never reach config.json, which lives on the
// data volume: only env.* references Bifrost resolves at boot.
func TestRenderConfigNeverWritesKeyValues(t *testing.T) {
	o, err := logsOffloadFromEnv(envMap(map[string]string{
		envLogsS3Bucket:          "b",
		envLogsS3AccessKeyID:     "AKIAEXAMPLE",
		envLogsS3SecretAccessKey: "s3cr3t-value",
	}))
	if err != nil {
		t.Fatal(err)
	}
	out, err := renderConfig([]byte(testSeed), o, "/app/data")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(out, []byte("AKIAEXAMPLE")) || bytes.Contains(out, []byte("s3cr3t-value")) {
		t.Errorf("rendered config contains a key value:\n%s", out)
	}
}

func TestRenderConfigKeepsSeedBytes(t *testing.T) {
	// The real seed: the output is the seed with 365 -> 30 and the
	// logs_store block appended; every other byte as it was.
	seed, err := os.ReadFile(filepath.Join("..", "data", "config.json"))
	if err != nil {
		t.Skipf("no seed: %v", err)
	}
	out, err := renderConfig(seed, &logsOffload{Bucket: "b", Prefix: "p"}, "/app/data")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(seed, []byte(`"log_retention_days": 365,`)) {
		t.Skip("seed no longer has log_retention_days 365")
	}
	body := strings.TrimRight(string(seed), "\n")
	body = strings.TrimRight(body[:len(body)-1], "\n ")
	body = strings.Replace(body, `"log_retention_days": 365,`, `"log_retention_days": 30,`, 1)
	if !strings.HasPrefix(string(out), body+",\n  \"logs_store\": {") {
		t.Errorf("rendered config is not the seed + retention + logs_store:\n%s", out)
	}
	if !bytes.HasSuffix(out, []byte("\n}\n")) {
		t.Errorf("rendered config does not end with a closing brace line:\n%s", out)
	}
	// Deterministic: same input, same bytes (the skip-when-identical
	// check depends on it).
	again, _ := renderConfig(seed, &logsOffload{Bucket: "b", Prefix: "p"}, "/app/data")
	if !bytes.Equal(out, again) {
		t.Error("renderConfig is not deterministic")
	}
}

func TestWithRetention(t *testing.T) {
	for _, tc := range []struct {
		name, in, want string
		same           bool // returned untouched, bytes and all
	}{
		{name: "365 lowered", in: `{"a": true, "log_retention_days": 365, "b": 1}`,
			want: `{"a": true, "log_retention_days": 30, "b": 1}`},
		{name: "already 7", in: `{ "log_retention_days": 7 }`, same: true},
		{name: "exactly 30", in: `{"log_retention_days": 30}`, same: true},
		{name: "missing key", in: `{"a": 1}`, want: `{"a": 1, "log_retention_days": 30}`},
		{name: "no client", in: "", want: `{"log_retention_days": 30}`},
		{name: "zero", in: `{"log_retention_days": 0}`, want: `{"log_retention_days": 30}`},
		{name: "not a number", in: `{"log_retention_days": "365"}`, want: `{"log_retention_days": 30}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var in json.RawMessage
			if tc.in != "" {
				in = json.RawMessage(tc.in)
			}
			got, err := withRetention(in, 30)
			if err != nil {
				t.Fatal(err)
			}
			if tc.same {
				if string(got) != tc.in {
					t.Errorf("got %s, want %s untouched", got, tc.in)
				}
				return
			}
			if !jsonEqual(t, got, json.RawMessage(tc.want)) {
				t.Errorf("got %s, want %s", got, tc.want)
			}
			// Key order kept.
			if tc.name == "365 lowered" && !strings.Contains(string(got), "\"a\": true,\n    \"log_retention_days\": 30,\n    \"b\": 1") {
				t.Errorf("order or layout not kept:\n%s", got)
			}
		})
	}
	if _, err := withRetention(json.RawMessage(`[1]`), 30); err == nil {
		t.Error("client that is not an object: want an error")
	}
}

func TestRenderConfigEmptySeedObject(t *testing.T) {
	top, ls := decodeLogsStore(t, []byte("{}"), &logsOffload{Bucket: "b", Prefix: "p"})
	if len(top) != 2 || ls["type"] != "sqlite" || !jsonEqual(t, top["client"], json.RawMessage(`{"log_retention_days":30}`)) {
		t.Errorf("got %v", top)
	}
}

func TestRenderConfigUsesAppDir(t *testing.T) {
	out, err := renderConfig([]byte(testSeed), &logsOffload{Bucket: "b", Prefix: "p"}, "/srv/bifrost")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"path": "/srv/bifrost/logs.db"`)) {
		t.Errorf("logs.db path does not follow the app dir:\n%s", out)
	}
}

func TestRenderConfigRefuses(t *testing.T) {
	o := &logsOffload{Bucket: "b", Prefix: "p"}
	_, err := renderConfig([]byte(`{"logs_store": {"enabled": false}}`), o, "/app/data")
	if !errors.Is(err, errSeedHasLogsStore) {
		t.Errorf("seed with logs_store: err = %v, want errSeedHasLogsStore", err)
	}
	for _, bad := range []string{``, `nope`, `[1,2]`, `null`, `{"a":1`, `{"a":1} {}`, `{"client": 5}`} {
		if _, err := renderConfig([]byte(bad), o, "/app/data"); err == nil {
			t.Errorf("seed %q: want an error", bad)
		}
	}
}

// The image's real seed takes the block and stays valid JSON.
func TestRenderConfigRealSeed(t *testing.T) {
	seed, err := os.ReadFile(filepath.Join("..", "data", "config.json"))
	if err != nil {
		t.Skipf("no seed: %v", err)
	}
	top, ls := decodeLogsStore(t, seed, &logsOffload{Bucket: "b", Prefix: "p", StaticKeys: true})
	for _, k := range []string{"client", "auth_config", "providers", "plugins", "config_store"} {
		if _, ok := top[k]; !ok {
			t.Errorf("seed key %q missing", k)
		}
	}
	if ls["config"].(map[string]any)["path"] != "/app/data/logs.db" {
		t.Errorf("logs_store.config = %v", ls["config"])
	}
}

func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var x, y any
	if err := json.Unmarshal(a, &x); err != nil {
		return false
	}
	if err := json.Unmarshal(b, &y); err != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}

// ─── syncSeedConfig ──────────────────────────────────────────────────

func quietLogger() *log.Logger { return log.New(io.Discard, "", 0) }

func writeSeed(t *testing.T, content string) (seedPath, appDir string) {
	t.Helper()
	dir := t.TempDir()
	seedPath = filepath.Join(dir, "config.json.seed")
	if err := os.WriteFile(seedPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return seedPath, filepath.Join(dir, "data")
}

func readConfig(t *testing.T, appDir string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(appDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestSyncSeedConfigOff(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	applied, err := syncSeedConfig(quietLogger(), seedPath, appDir, nil)
	if err != nil || applied != nil {
		t.Fatalf("applied=%v err=%v", applied, err)
	}
	if got := readConfig(t, appDir); got != testSeed {
		t.Errorf("config.json != seed:\n%s", got)
	}
}

func TestSyncSeedConfigOnOffAndSkip(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	cfg := filepath.Join(appDir, "config.json")
	o := &logsOffload{Bucket: "b", Prefix: "p"}

	applied, err := syncSeedConfig(quietLogger(), seedPath, appDir, o)
	if err != nil || applied != o {
		t.Fatalf("on: applied=%v err=%v", applied, err)
	}
	if got := readConfig(t, appDir); !strings.Contains(got, `"logs_store"`) {
		t.Fatalf("on: no logs_store in config.json:\n%s", got)
	}

	// Same content again: no write (mtime untouched).
	old := time.Now().Add(-time.Hour).Truncate(time.Second)
	if err := os.Chtimes(cfg, old, old); err != nil {
		t.Fatal(err)
	}
	if _, err := syncSeedConfig(quietLogger(), seedPath, appDir, o); err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(cfg); !fi.ModTime().Equal(old) {
		t.Errorf("unchanged config was rewritten (mtime %v, want %v)", fi.ModTime(), old)
	}

	// Off again: back to the seed, byte for byte.
	if applied, err := syncSeedConfig(quietLogger(), seedPath, appDir, nil); err != nil || applied != nil {
		t.Fatalf("off: applied=%v err=%v", applied, err)
	}
	if got := readConfig(t, appDir); got != testSeed {
		t.Errorf("off: config.json != seed:\n%s", got)
	}
	if _, err := os.Stat(cfg + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("temp file left behind: %v", err)
	}
}

func TestSyncSeedConfigSeedWithLogsStoreWins(t *testing.T) {
	seed := `{"logs_store": {"enabled": true, "type": "sqlite", "config": {"path": "/x/logs.db"}}}`
	seedPath, appDir := writeSeed(t, seed)
	applied, err := syncSeedConfig(quietLogger(), seedPath, appDir, &logsOffload{Bucket: "b", Prefix: "p"})
	if err != nil || applied != nil {
		t.Fatalf("applied=%v err=%v", applied, err)
	}
	if got := readConfig(t, appDir); got != seed {
		t.Errorf("config.json != seed:\n%s", got)
	}
}

func TestSyncSeedConfigNoSeed(t *testing.T) {
	dir := t.TempDir()
	applied, err := syncSeedConfig(quietLogger(), filepath.Join(dir, "missing"), filepath.Join(dir, "data"),
		&logsOffload{Bucket: "b", Prefix: "p"})
	if err != nil || applied != nil {
		t.Fatalf("applied=%v err=%v", applied, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "data", "config.json")); !os.IsNotExist(err) {
		t.Errorf("config.json written without a seed: %v", err)
	}
}

// ─── boot fallback ───────────────────────────────────────────────────

// fakeBoot drives bootEnv.boot with real child processes (so stopChild's
// SIGTERM path runs) and a scripted readiness result per start. It
// records, for each start, whether config.json had a logs_store block.
type fakeBoot struct {
	t        *testing.T
	appDir   string
	children []string // command per start: "sleep" stays up, "exit" exits 1
	ready    []bool   // readiness per start
	starts   []bool   // config.json had logs_store at start i
	cmds     []*exec.Cmd
	dones    []chan error
}

func (f *fakeBoot) env(seedPath string) bootEnv {
	return bootEnv{
		logger:   quietLogger(),
		seedPath: seedPath,
		appDir:   f.appDir,
		start: func() (*exec.Cmd, chan error) {
			i := len(f.starts)
			f.starts = append(f.starts, strings.Contains(readConfig(f.t, f.appDir), `"logs_store"`))
			var cmd *exec.Cmd
			if f.children[i] == "exit" {
				cmd = exec.Command("sh", "-c", "exit 1")
			} else {
				cmd = exec.Command("sleep", "30")
			}
			if err := cmd.Start(); err != nil {
				f.t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			f.cmds = append(f.cmds, cmd)
			f.dones = append(f.dones, done)
			return cmd, done
		},
		wait: func(done <-chan error) error {
			i := len(f.starts) - 1
			if f.ready[i] {
				return nil
			}
			if f.children[i] == "exit" {
				// What waitForUpstream returns when the child dies.
				childErr := <-done
				return errors.Join(errChildExited, childErr)
			}
			return errors.New("bifrost-http readiness wait: context deadline exceeded")
		},
	}
}

// cleanup kills any child still running. Kill on an exited process is a
// harmless ErrProcessDone; ProcessState is not read here because the
// Wait goroutines may still be writing it.
func (f *fakeBoot) cleanup() {
	for _, c := range f.cmds {
		_ = c.Process.Kill()
	}
}

func TestBootOffloadSucceeds(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	f := &fakeBoot{t: t, appDir: appDir, children: []string{"sleep"}, ready: []bool{true}}
	defer f.cleanup()
	_, _, err := f.env(seedPath).boot(&logsOffload{Bucket: "b", Prefix: "p"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(f.starts, []bool{true}) {
		t.Errorf("starts (logs_store present) = %v, want [true]", f.starts)
	}
}

func TestBootFallsBackWhenBifrostExits(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	f := &fakeBoot{t: t, appDir: appDir, children: []string{"exit", "sleep"}, ready: []bool{false, true}}
	defer f.cleanup()
	cmd, _, err := f.env(seedPath).boot(&logsOffload{Bucket: "b", Prefix: "p"})
	if err != nil {
		t.Fatalf("boot: %v", err)
	}
	if !reflect.DeepEqual(f.starts, []bool{true, false}) {
		t.Errorf("starts (logs_store present) = %v, want [true false]", f.starts)
	}
	if cmd != f.cmds[1] {
		t.Error("boot did not return the second child")
	}
	if got := readConfig(t, appDir); got != testSeed {
		t.Errorf("after fallback config.json != seed:\n%s", got)
	}
}

func TestBootFallsBackWhenBifrostHangs(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	f := &fakeBoot{t: t, appDir: appDir, children: []string{"sleep", "sleep"}, ready: []bool{false, true}}
	defer f.cleanup()
	if _, _, err := f.env(seedPath).boot(&logsOffload{Bucket: "b", Prefix: "p"}); err != nil {
		t.Fatalf("boot: %v", err)
	}
	if !reflect.DeepEqual(f.starts, []bool{true, false}) {
		t.Errorf("starts (logs_store present) = %v, want [true false]", f.starts)
	}
	// The hung first child was stopped and reaped before the restart.
	if f.cmds[0].ProcessState == nil {
		t.Error("first bifrost-http still running after fallback")
	}
}

func TestBootNoRetryWithoutOffload(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	f := &fakeBoot{t: t, appDir: appDir, children: []string{"exit"}, ready: []bool{false}}
	defer f.cleanup()
	_, _, err := f.env(seedPath).boot(nil)
	if !errors.Is(err, errChildExited) {
		t.Fatalf("err = %v, want errChildExited", err)
	}
	if len(f.starts) != 1 {
		t.Errorf("started %d times, want 1", len(f.starts))
	}
}

func TestBootFallbackFailureIsReturned(t *testing.T) {
	seedPath, appDir := writeSeed(t, testSeed)
	f := &fakeBoot{t: t, appDir: appDir, children: []string{"exit", "exit"}, ready: []bool{false, false}}
	defer f.cleanup()
	_, _, err := f.env(seedPath).boot(&logsOffload{Bucket: "b", Prefix: "p"})
	if !errors.Is(err, errChildExited) {
		t.Fatalf("err = %v, want errChildExited", err)
	}
	if len(f.starts) != 2 {
		t.Errorf("started %d times, want 2 (no third try)", len(f.starts))
	}
}
