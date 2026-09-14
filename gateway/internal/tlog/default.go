package tlog

import (
	"errors"
	"sync"
)

// The process-wide log. main.Init opens it; the hooks append to it;
// the admin API pages from it. Kept behind accessors so tests can
// swap in a temp-dir log and so a plugin that never called Init
// (SDK mode, unit tests of other packages) gets a clean error rather
// than a nil dereference.

var (
	defMu sync.RWMutex
	def   *Log
)

// ErrNotInitialized is returned by the package-level Append when Init
// has not run.
var ErrNotInitialized = errors.New("tlog: not initialized")

// Init opens the process-wide log at path. The log is installed even
// when Open fails (disabled, carrying the reason) so the admin route
// can report why rather than "not initialized". Returns Open's error
// for the caller to log; the plugin keeps running either way — a
// gateway that drops its whole plugin over a log-file problem would
// be strictly less governed than one whose witness sees a 503.
func Init(path string) error {
	l, err := Open(path)
	defMu.Lock()
	def = l
	defMu.Unlock()
	return err
}

// Default returns the process-wide log, or nil before Init.
func Default() *Log {
	defMu.RLock()
	defer defMu.RUnlock()
	return def
}

// SetDefaultForTest replaces the process-wide log. Production code
// goes through Init.
func SetDefaultForTest(l *Log) {
	defMu.Lock()
	def = l
	defMu.Unlock()
}

// Append appends to the process-wide log.
func Append(leaf Leaf) (uint64, error) {
	l := Default()
	if l == nil {
		return 0, ErrNotInitialized
	}
	return l.Append(leaf)
}

// Close closes the process-wide log and clears it.
func Close() error {
	defMu.Lock()
	l := def
	def = nil
	defMu.Unlock()
	if l == nil {
		return nil
	}
	return l.Close()
}
