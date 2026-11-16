package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var moveTime = time.Date(2026, 11, 16, 4, 10, 0, 0, time.UTC)

const movedName = "20261116T041000Z"

// appDirWith makes an app dir holding the named files, each containing
// its own name (so a test can tell them apart after a move).
func appDirWith(t *testing.T, names ...string) string {
	t.Helper()
	dir := t.TempDir()
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n), []byte(n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func mustRead(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func exists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}

func readMarker(t *testing.T, appDir string) offloadMarker {
	t.Helper()
	var m offloadMarker
	if err := json.Unmarshal([]byte(mustRead(t, filepath.Join(appDir, offloadMarkerName))), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMoveLegacyLogsDBMovesTheSet(t *testing.T) {
	app := appDirWith(t, "logs.db", "logs.db-wal", "logs.db-shm", "config.db", "config.json")
	moved, err := moveLegacyLogsDB(app, moveTime)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(app, legacyDirName, movedName)
	if moved != want {
		t.Fatalf("moved to %q, want %q", moved, want)
	}
	for _, n := range []string{"logs.db", "logs.db-wal", "logs.db-shm"} {
		if exists(filepath.Join(app, n)) {
			t.Errorf("%s still in the app dir", n)
		}
		// Same names inside the dated dir: the set stays a valid database.
		if got := mustRead(t, filepath.Join(moved, n)); got != n {
			t.Errorf("%s content = %q", n, got)
		}
	}
	for _, n := range []string{"config.db", "config.json"} {
		if !exists(filepath.Join(app, n)) {
			t.Errorf("%s was moved", n)
		}
	}
	m := readMarker(t, app)
	if m.Archived != filepath.Join(legacyDirName, movedName) || m.Since != "2026-11-16T04:10:00Z" {
		t.Errorf("marker = %+v", m)
	}
	if dirs, _ := pendingLegacyDirs(app); len(dirs) != 1 || dirs[0] != want {
		t.Errorf("pending = %v", dirs)
	}
}

func TestMoveLegacyLogsDBOnce(t *testing.T) {
	app := appDirWith(t, "logs.db")
	if _, err := moveLegacyLogsDB(app, moveTime); err != nil {
		t.Fatal(err)
	}
	// Bifrost then creates a fresh logs.db; later boots leave it alone.
	if err := os.WriteFile(filepath.Join(app, "logs.db"), []byte("fresh"), 0o644); err != nil {
		t.Fatal(err)
	}
	moved, err := moveLegacyLogsDB(app, moveTime.Add(time.Hour))
	if err != nil || moved != "" {
		t.Fatalf("second boot: moved=%q err=%v", moved, err)
	}
	if got := mustRead(t, filepath.Join(app, "logs.db")); got != "fresh" {
		t.Errorf("fresh logs.db touched: %q", got)
	}
}

func TestMoveLegacyLogsDBNoDatabase(t *testing.T) {
	app := appDirWith(t, "config.json")
	moved, err := moveLegacyLogsDB(app, moveTime)
	if err != nil || moved != "" {
		t.Fatalf("moved=%q err=%v", moved, err)
	}
	if m := readMarker(t, app); m.Archived != "" {
		t.Errorf("marker = %+v", m)
	}
	if dirs, _ := pendingLegacyDirs(app); len(dirs) != 0 {
		t.Errorf("pending = %v", dirs)
	}
	if ents, _ := os.ReadDir(filepath.Join(app, legacyDirName)); len(ents) != 0 {
		t.Errorf("leftovers in %s: %v", legacyDirName, ents)
	}
}

func TestMoveLegacyLogsDBResumesAnInterruptedMove(t *testing.T) {
	// A boot died after moving the WAL: the .moving dir holds it, the
	// app dir still has logs.db and the -shm. The next boot finishes.
	app := appDirWith(t, "logs.db", "logs.db-shm")
	staging := filepath.Join(app, legacyDirName, "20261115T000000Z"+movingSuffix)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staging, "logs.db-wal"), []byte("logs.db-wal"), 0o644); err != nil {
		t.Fatal(err)
	}
	moved, err := moveLegacyLogsDB(app, moveTime)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(app, legacyDirName, "20261115T000000Z"); moved != want {
		t.Fatalf("moved = %q, want the interrupted dir %q", moved, want)
	}
	for _, n := range []string{"logs.db", "logs.db-wal", "logs.db-shm"} {
		if got := mustRead(t, filepath.Join(moved, n)); got != n {
			t.Errorf("%s = %q", n, got)
		}
	}
	if exists(staging) {
		t.Error(".moving dir left behind")
	}
}

func TestMoveLegacyLogsDBRefusesToOverwrite(t *testing.T) {
	// The interrupted dir already has a logs.db-wal and the app dir has
	// another: refuse, put back what this call moved, write no marker.
	app := appDirWith(t, "logs.db", "logs.db-wal")
	staging := filepath.Join(app, legacyDirName, "20261115T000000Z"+movingSuffix)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staging, "logs.db-wal"), []byte("older wal"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := moveLegacyLogsDB(app, moveTime); err == nil || !strings.Contains(err.Error(), "refusing to overwrite") {
		t.Fatalf("err = %v", err)
	}
	if mustRead(t, filepath.Join(app, "logs.db")) != "logs.db" || mustRead(t, filepath.Join(app, "logs.db-wal")) != "logs.db-wal" {
		t.Error("app dir files changed")
	}
	if mustRead(t, filepath.Join(staging, "logs.db-wal")) != "older wal" {
		t.Error("interrupted dir changed")
	}
	if exists(filepath.Join(app, offloadMarkerName)) {
		t.Error("marker written after a failed move")
	}
}

func TestPendingLegacyDirsSkipsMoving(t *testing.T) {
	app := t.TempDir()
	for _, d := range []string{"20261116T000000Z", "20261115T000000Z", "20261117T000000Z" + movingSuffix} {
		if err := os.MkdirAll(filepath.Join(app, legacyDirName, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	dirs, err := pendingLegacyDirs(app)
	if err != nil {
		t.Fatal(err)
	}
	if len(dirs) != 2 || filepath.Base(dirs[0]) != "20261115T000000Z" || filepath.Base(dirs[1]) != "20261116T000000Z" {
		t.Errorf("pending = %v", dirs)
	}
	if dirs, err := pendingLegacyDirs(t.TempDir()); err != nil || dirs != nil {
		t.Errorf("no legacy dir: %v %v", dirs, err)
	}
}

func TestHelperArgsNeverCarryKeys(t *testing.T) {
	o := &logsOffload{Bucket: "b", Prefix: "p", Region: "r", Endpoint: "http://e", StaticKeys: true}
	got := strings.Join(o.helperArgs(), " ")
	if got != "-bucket b -prefix p -region r -endpoint http://e" {
		t.Errorf("args = %q", got)
	}
	if got := strings.Join((&logsOffload{Bucket: "b", Prefix: "p"}).helperArgs(), " "); got != "-bucket b -prefix p" {
		t.Errorf("args = %q", got)
	}
}

// fakeHelper writes a stand-in for logs-archive: it records its argv and
// the env keys it saw, and for `upload` deletes -dir unless the fail file
// still has attempts left.
func fakeHelper(t *testing.T, failUploads int) (bin, argsLog string) {
	t.Helper()
	dir := t.TempDir()
	argsLog = filepath.Join(dir, "args.log")
	fails := filepath.Join(dir, "fails")
	if err := os.WriteFile(fails, []byte(strings.Repeat("x", failUploads)), 0o644); err != nil {
		t.Fatal(err)
	}
	bin = filepath.Join(dir, "logs-archive")
	script := `#!/bin/sh
echo "$* key=${BIFROST_LOGS_S3_ACCESS_KEY_ID:-none}" >> ` + argsLog + `
[ "$1" = preflight ] && exit ${PREFLIGHT_EXIT:-0}
dir=""
while [ $# -gt 0 ]; do [ "$1" = -dir ] && dir=$2; shift; done
left=$(cat ` + fails + `)
if [ -n "$left" ]; then printf '%s' "${left#x}" > ` + fails + `; exit 1; fi
rm -rf "$dir"
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin, argsLog
}

func TestRunPreflight(t *testing.T) {
	bin, argsLog := fakeHelper(t, 0)
	t.Setenv("BIFROST_LOGS_S3_ACCESS_KEY_ID", "AKIAEXAMPLE")
	o := &logsOffload{Bucket: "b", Prefix: "p", Endpoint: "http://minio:9000", StaticKeys: true}
	if err := runPreflight(context.Background(), bin, o); err != nil {
		t.Fatalf("ok helper: %v", err)
	}
	got := mustRead(t, argsLog)
	if !strings.Contains(got, "preflight -timeout 20s -bucket b -prefix p -endpoint http://minio:9000 key=AKIAEXAMPLE") {
		t.Errorf("helper saw %q (keys must come through the env, not argv)", got)
	}
	t.Setenv("PREFLIGHT_EXIT", "1")
	if err := runPreflight(context.Background(), bin, o); err == nil {
		t.Error("failing helper: want an error")
	}
	if err := runPreflight(context.Background(), filepath.Join(t.TempDir(), "missing"), o); err == nil {
		t.Error("missing helper: want an error")
	}
}

func legacyApp(t *testing.T, dirs ...string) string {
	t.Helper()
	app := t.TempDir()
	for _, d := range dirs {
		p := filepath.Join(app, legacyDirName, d)
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(p, "logs.db"), []byte("old"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return app
}

func TestArchiverUploadsEveryPendingDir(t *testing.T) {
	bin, argsLog := fakeHelper(t, 0)
	app := legacyApp(t, "A", "B", "C"+movingSuffix)
	a := &archiver{logger: quietLogger(), bin: bin, appDir: app, offload: &logsOffload{Bucket: "b", Prefix: "p"},
		retryMin: time.Millisecond, retryMax: time.Millisecond}
	a.run(context.Background())
	if dirs, _ := pendingLegacyDirs(app); len(dirs) != 0 {
		t.Errorf("still pending: %v", dirs)
	}
	if !exists(filepath.Join(app, legacyDirName, "C"+movingSuffix)) {
		t.Error("an unfinished move was uploaded")
	}
	if n := strings.Count(mustRead(t, argsLog), "upload -bucket b -prefix p -dir "); n != 2 {
		t.Errorf("uploads = %d, want 2", n)
	}
}

func TestArchiverRetriesFailures(t *testing.T) {
	bin, argsLog := fakeHelper(t, 2)
	app := legacyApp(t, "A")
	a := &archiver{logger: quietLogger(), bin: bin, appDir: app, offload: &logsOffload{Bucket: "b", Prefix: "p"},
		retryMin: time.Millisecond, retryMax: 2 * time.Millisecond}
	a.run(context.Background())
	if dirs, _ := pendingLegacyDirs(app); len(dirs) != 0 {
		t.Errorf("still pending: %v", dirs)
	}
	if n := strings.Count(mustRead(t, argsLog), "upload "); n != 3 {
		t.Errorf("attempts = %d, want 3 (two failures, then success)", n)
	}
}

func TestArchiverStopsOnCancelAndKeepsFiles(t *testing.T) {
	bin, _ := fakeHelper(t, 1000)
	app := legacyApp(t, "A")
	a := &archiver{logger: quietLogger(), bin: bin, appDir: app, offload: &logsOffload{Bucket: "b", Prefix: "p"},
		retryMin: time.Hour, retryMax: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { a.run(ctx); close(done) }()
	time.Sleep(200 * time.Millisecond) // first attempt fails, then it waits an hour
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("archiver did not stop on cancel")
	}
	if !exists(filepath.Join(app, legacyDirName, "A", "logs.db")) {
		t.Error("files deleted without a successful upload")
	}
}

func TestArchiverHelperThatLeavesTheDirIsAFailure(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "logs-archive")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	app := legacyApp(t, "A")
	a := &archiver{logger: quietLogger(), bin: bin, appDir: app, offload: &logsOffload{Bucket: "b", Prefix: "p"},
		retryMin: time.Hour, retryMax: time.Hour}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	a.run(ctx) // must not spin: it waits retryMin after the "success" that left the dir
	if !exists(filepath.Join(app, legacyDirName, "A", "logs.db")) {
		t.Error("files gone")
	}
}
