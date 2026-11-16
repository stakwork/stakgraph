package main

// Archiving the logs.db that predates offload.
//
// Offload only changes new rows, and SQLite never gives space back
// (Bifrost never VACUUMs), so a 133 GB logs.db stays 133 GB unless it is
// moved out. So the first boot with offload on moves the old file aside
// and Bifrost starts a fresh logs.db; the old one is uploaded to
// <prefix>/legacy/ in the background and deleted locally once the upload
// is verified. The dashboard's history starts over at that boot.
//
// Order at boot, all only when BIFROST_LOGS_S3_BUCKET is set:
//
//  1. preflight (logs-archive preflight): the bucket must take a tagged
//     put and serve it back with these credentials, or offload stays off
//     for this boot and logs.db is not touched.
//  2. move (moveLegacyLogsDB, here, before Bifrost opens the file): once
//     per app dir, recorded by the marker file logs-offload.json.
//  3. Bifrost starts (with the logs_store block, see logsoffload.go).
//  4. archive (archiver.run, in the background): every directory under
//     logs-legacy/ is handed to `logs-archive upload`, which deletes it
//     only after a verified upload. A failure keeps it and is retried
//     later in this boot and on the next one.
//
// The S3 work runs in a separate binary (gateway/logarchive) so the
// wrapper stays stdlib-only and static.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

const (
	// defaultLogsArchiveBin is where the Dockerfile puts the helper.
	defaultLogsArchiveBin = "/app/logs-archive"

	// legacyDirName holds moved-aside logs.db sets, one directory per
	// move, named by its UTC time: logs-legacy/20261116T041000Z/logs.db.
	// The files keep their names inside it, so logs.db and its -wal stay
	// a valid SQLite database together.
	legacyDirName = "logs-legacy"

	// movingSuffix marks a directory a move is still filling. A boot that
	// dies mid-move leaves it, and the next offload boot finishes it.
	movingSuffix = ".moving"

	// offloadMarkerName records that the app dir's logs.db was started
	// under offload, so the move happens once.
	offloadMarkerName = "logs-offload.json"

	// preflightTimeout bounds the bucket check at boot. An endpoint that
	// drops packets costs this, not Bifrost's much longer boot hang.
	preflightTimeout = 20 * time.Second
)

// sqliteSet is moved in this order: the -wal (and -shm) first, so a move
// cut short never leaves a logs.db next to a WAL that is not its own.
var sqliteSet = []string{"logs.db-wal", "logs.db-shm", "logs.db"}

// helperArgs are the logs-archive flags for o. Never the keys: those
// reach the helper through the environment it inherits.
func (o *logsOffload) helperArgs() []string {
	args := []string{"-bucket", o.Bucket, "-prefix", o.Prefix}
	if o.Region != "" {
		args = append(args, "-region", o.Region)
	}
	if o.Endpoint != "" {
		args = append(args, "-endpoint", o.Endpoint)
	}
	return args
}

// runPreflight runs `logs-archive preflight`; nil means the bucket works.
func runPreflight(ctx context.Context, bin string, o *logsOffload) error {
	ctx, cancel := context.WithTimeout(ctx, preflightTimeout+10*time.Second)
	defer cancel()
	args := append([]string{"preflight", "-timeout", preflightTimeout.String()}, o.helperArgs()...)
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	return cmd.Run()
}

// offloadMarker is the content of logs-offload.json.
type offloadMarker struct {
	Since    string `json:"since"`
	Archived string `json:"archived,omitempty"` // relative to the app dir
}

// moveLegacyLogsDB moves the app dir's logs.db (with its -wal and -shm)
// to logs-legacy/<UTC time>/ unless the marker says this app dir's
// logs.db was already started under offload. It returns the directory
// the files went to, or "" when nothing was moved. With no logs.db it
// only writes the marker.
//
// It must run before bifrost-http starts: nothing may have the database
// open. On error, files moved by this call are put back.
func moveLegacyLogsDB(appDir string, now time.Time) (string, error) {
	markerPath := filepath.Join(appDir, offloadMarkerName)
	if _, err := os.Stat(markerPath); err == nil {
		return "", nil
	} else if !os.IsNotExist(err) {
		return "", fmt.Errorf("stat %s: %w", markerPath, err)
	}

	root := filepath.Join(appDir, legacyDirName)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	staging, err := findMoving(root)
	if err != nil {
		return "", err
	}
	if staging == "" {
		staging = filepath.Join(root, now.UTC().Format("20060102T150405Z")+movingSuffix)
		if err := os.Mkdir(staging, 0o755); err != nil {
			return "", err
		}
	}

	var moved []string
	rollback := func(cause error) (string, error) {
		for i := len(moved) - 1; i >= 0; i-- {
			if err := os.Rename(filepath.Join(staging, moved[i]), filepath.Join(appDir, moved[i])); err != nil {
				cause = fmt.Errorf("%w; and putting %s back failed: %v", cause, moved[i], err)
			}
		}
		_ = os.Remove(staging) // only succeeds when empty
		return "", cause
	}
	for _, name := range sqliteSet {
		src, dst := filepath.Join(appDir, name), filepath.Join(staging, name)
		if _, err := os.Lstat(src); os.IsNotExist(err) {
			continue
		} else if err != nil {
			return rollback(err)
		}
		if _, err := os.Lstat(dst); err == nil {
			return rollback(fmt.Errorf("%s already holds a %s; refusing to overwrite it", staging, name))
		}
		if err := os.Rename(src, dst); err != nil {
			return rollback(err)
		}
		moved = append(moved, name)
	}

	final := ""
	ents, err := os.ReadDir(staging)
	if err != nil {
		return rollback(err)
	}
	if len(ents) == 0 {
		_ = os.Remove(staging)
	} else {
		final = strings.TrimSuffix(staging, movingSuffix)
		if err := os.Rename(staging, final); err != nil {
			return rollback(err)
		}
	}
	syncDir(root)
	syncDir(appDir)

	m := offloadMarker{Since: now.UTC().Format(time.RFC3339)}
	if final != "" {
		m.Archived = filepath.Join(legacyDirName, filepath.Base(final))
	}
	b, _ := json.MarshalIndent(m, "", "  ")
	if err := writeFileAtomic(markerPath, append(b, '\n')); err != nil {
		// The files are moved and stay moved: Bifrost must not find the
		// old WAL next to a new logs.db. Without the marker the next
		// offload boot would move whatever logs.db exists then (archived
		// too, so nothing is lost).
		return final, fmt.Errorf("moved, but writing %s failed: %w", markerPath, err)
	}
	return final, nil
}

// findMoving returns the (oldest) unfinished move under root, or "".
func findMoving(root string) (string, error) {
	ents, err := os.ReadDir(root)
	if err != nil {
		return "", err
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() && strings.HasSuffix(e.Name(), movingSuffix) {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return "", nil
	}
	sort.Strings(names)
	return filepath.Join(root, names[0]), nil
}

// pendingLegacyDirs lists the finished moves waiting for upload, oldest
// first.
func pendingLegacyDirs(appDir string) ([]string, error) {
	root := filepath.Join(appDir, legacyDirName)
	ents, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if e.IsDir() && !strings.HasSuffix(e.Name(), movingSuffix) {
			out = append(out, filepath.Join(root, e.Name()))
		}
	}
	sort.Strings(out)
	return out, nil
}

// syncDir fsyncs a directory so renames in it survive a crash. Best
// effort.
func syncDir(dir string) {
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
}

// archiver uploads pending legacy directories in the background.
type archiver struct {
	logger  *log.Logger
	bin     string
	appDir  string
	offload *logsOffload
	// retry waits between failed rounds, doubling from retryMin up to
	// retryMax. The next container start also retries.
	retryMin, retryMax time.Duration
}

// run archives every pending directory, retrying failures until they
// all succeed or ctx is done. Cancelling ctx stops a running upload with
// SIGTERM (the helper aborts its multipart upload) and keeps the files.
func (a *archiver) run(ctx context.Context) {
	wait := a.retryMin
	for {
		dirs, err := pendingLegacyDirs(a.appDir)
		if err != nil {
			a.logger.Printf("ERROR: listing %s: %v", legacyDirName, err)
			return
		}
		if len(dirs) == 0 {
			return
		}
		failed := false
		for _, dir := range dirs {
			if ctx.Err() != nil {
				return
			}
			err := a.uploadOne(ctx, dir)
			if err == nil {
				if _, serr := os.Stat(dir); serr == nil {
					err = errors.New("logs-archive exited 0 but left the directory")
				}
			}
			if err != nil {
				if ctx.Err() != nil {
					a.logger.Printf("archive of %s stopped (shutting down); the files stay and the next start resumes", dir)
					return
				}
				failed = true
				a.logger.Printf("ERROR: archive of %s failed: %v; the files stay on disk, retrying in %s", dir, err, wait)
			}
		}
		if !failed {
			continue // re-list: done when nothing is pending
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
		if wait *= 2; wait > a.retryMax {
			wait = a.retryMax
		}
	}
}

// uploadOne runs `logs-archive upload` for dir at low CPU priority.
func (a *archiver) uploadOne(ctx context.Context, dir string) error {
	args := append(append([]string{"upload"}, a.offload.helperArgs()...), "-dir", dir)
	cmd := exec.CommandContext(ctx, a.bin, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// On shutdown: SIGTERM, so the helper aborts its multipart upload,
	// then SIGKILL if it takes too long.
	cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
	cmd.WaitDelay = 20 * time.Second
	if err := cmd.Start(); err != nil {
		return err
	}
	_ = syscall.Setpriority(syscall.PRIO_PROCESS, cmd.Process.Pid, 10)
	return cmd.Wait()
}
