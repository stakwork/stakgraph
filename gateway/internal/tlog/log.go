package tlog

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
	"github.com/stakwork/stakgraph/gateway/internal/pluginlog"
)

// PageSize caps the number of leaves one witness poll returns. Hive
// pages with since = next until next == tree_size.
const PageSize = 5000

// ErrDisabled is wrapped by every operation on a Log that refused to
// come up or has stopped itself after an unrecoverable write failure.
// The reason is in the error chain; the boot log carries it too.
var ErrDisabled = errors.New("tlog: log disabled")

// AheadError is returned by Page when the witness asks for leaves
// from a size the tree has not reached. After a power loss the tree
// at boot can be shorter than the last head a witness signed; the
// witness must stop and involve an operator, never silently restart.
type AheadError struct {
	Since    uint64
	TreeSize uint64
}

func (e *AheadError) Error() string {
	return fmt.Sprintf("tlog: since=%d is ahead of tree_size=%d", e.Since, e.TreeSize)
}

// Page is the response body of GET /_plugin/tlog/sth?since=N.
//
// ConsistencyProof runs from since to STH.TreeSize and is empty when
// since is 0 or equals the tree size; a full-replica witness does not
// need it, light witnesses that hold only heads do. Leaves are the
// canonical JSON of indices [since, Next), byte-for-byte what was
// hashed. Both slices are always present on the wire, never null.
type Page struct {
	STH              STH               `json:"sth"`
	LogPubkey        string            `json:"log_pubkey"`
	ConsistencyProof []string          `json:"consistency_proof"`
	Leaves           []json.RawMessage `json:"leaves"`
	Next             uint64            `json:"next"`
}

// Log is one transparency log: an in-memory Merkle tree rebuilt at
// boot from a JSONL leaf file, appended to synchronously on the hot
// path, and served as signed heads to a witness.
//
// One mutex guards the tree, the file, and the cached head. Appends
// arrive from concurrent LLMPost / StreamChunk goroutines;
// MarkAccounted only dedupes within a request.
type Log struct {
	mu sync.Mutex

	path    string
	f       *os.File // append handle; nil once closed
	offsets []int64  // byte offset of each leaf's line; len == tree size
	end     int64    // file length == offset of the next line

	tree tree
	key  *logKey
	sth  *STH // cached head; valid while sth.TreeSize == tree.size()

	// lastTS is the ts of the newest leaf ("" while empty). Set on
	// Append and recovered from the last valid line at rebuild, so
	// the status card can say "last leaf 12s ago" without a signed
	// head, a page read, or a full parse of the file.
	lastTS string

	now func() time.Time
	err error // non-nil ⇒ disabled, with the reason
}

// options are the knobs tests and the fixture generator turn; the
// production path (Open) sets only the path.
type options struct {
	path string
	priv []byte           // fixed log key; nil ⇒ fresh random key
	now  func() time.Time // clock for ts / signed_at; nil ⇒ time.Now
}

// Open rebuilds the log at path (creating the file and its directory
// if needed) and generates this boot's log key.
//
// The returned *Log is never nil. On error it is disabled: every
// Append and Page returns the reason wrapped in ErrDisabled, so the
// admin route can say why. A corrupt tail line (crash mid-write) is
// not an error — it is truncated and logged. An invalid line followed
// by more data is: the file is not a crash artefact, and the safe
// move is to refuse to rebuild and leave the evidence untouched.
func Open(path string) (*Log, error) {
	return open(options{path: path})
}

func open(o options) (*Log, error) {
	l := &Log{path: o.path, now: o.now}
	if l.now == nil {
		l.now = time.Now
	}
	var err error
	if o.priv != nil {
		l.key, err = logKeyFromBytes(o.priv)
	} else {
		l.key, err = newLogKey()
	}
	if err != nil {
		return l.disabled(err), err
	}
	if err := l.rebuild(); err != nil {
		return l.disabled(err), err
	}
	return l, nil
}

func (l *Log) disabled(reason error) *Log {
	l.err = fmt.Errorf("%w: %w", ErrDisabled, reason)
	return l
}

// rebuild reads the leaf file, rehashes every line into the tree,
// truncates a corrupt tail, and opens the append handle.
func (l *Log) rebuild() error {
	if err := os.MkdirAll(filepath.Dir(l.path), 0o755); err != nil {
		return fmt.Errorf("tlog: mkdir %s: %w", filepath.Dir(l.path), err)
	}

	rf, err := os.Open(l.path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// Fresh log.
	case err != nil:
		return fmt.Errorf("tlog: open %s: %w", l.path, err)
	default:
		truncateAt, scanErr := l.scan(rf)
		_ = rf.Close()
		if scanErr != nil {
			return scanErr
		}
		if truncateAt >= 0 {
			st, statErr := os.Stat(l.path)
			if statErr != nil {
				return fmt.Errorf("tlog: stat %s: %w", l.path, statErr)
			}
			if err := os.Truncate(l.path, truncateAt); err != nil {
				return fmt.Errorf("tlog: truncate corrupt tail of %s: %w", l.path, err)
			}
			pluginlog.Warnf("tlog: dropped 1 corrupt tail line (%d bytes) from %s — crash mid-write; rebuilt at size %d",
				st.Size()-truncateAt, l.path, l.tree.size())
		}
	}

	f, err := os.OpenFile(l.path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644)
	if err != nil {
		return fmt.Errorf("tlog: open %s for append: %w", l.path, err)
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("tlog: stat %s: %w", l.path, err)
	}
	if st.Size() != l.end {
		_ = f.Close()
		return fmt.Errorf("tlog: %s changed underneath the rebuild (scanned %d bytes, file is %d)", l.path, l.end, st.Size())
	}
	l.f = f
	root := l.tree.root()
	pluginlog.Logf("tlog: opened %s tree_size=%d root=%s log_pubkey=%s",
		l.path, l.tree.size(), macaroon.BytesToHex(root[:]), l.key.pubHex)
	return nil
}

// scan walks the leaf file once, appending every valid line to the
// tree and recording its offset. It returns the offset to truncate
// at when the final line is corrupt (incomplete, or complete but not
// JSON), or -1 when the file is clean. An invalid line with anything
// after it is an error.
func (l *Log) scan(r io.Reader) (int64, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	var offset int64
	badAt := int64(-1)
	// ReadBytes hands back a fresh copy per line, so holding the
	// newest valid one costs nothing extra; only that one line is
	// parsed, after the loop.
	var lastGood []byte
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			if badAt >= 0 {
				return -1, fmt.Errorf("tlog: %s: invalid leaf line at byte %d is followed by more data — not a crash artefact; refusing to rebuild (move the file aside to start a fresh log; the witness must then reset its stored head)", l.path, badAt)
			}
			complete := line[len(line)-1] == '\n'
			body := line
			if complete {
				body = line[:len(line)-1]
			}
			if !complete || !json.Valid(body) {
				badAt = offset
			} else {
				l.offsets = append(l.offsets, offset)
				l.tree.append(LeafHash(body))
				lastGood = body
			}
			offset += int64(len(line))
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return -1, fmt.Errorf("tlog: read %s: %w", l.path, err)
		}
	}
	l.lastTS = leafTS(lastGood)
	if badAt >= 0 {
		l.end = badAt
		return badAt, nil
	}
	l.end = offset
	return -1, nil
}

// leafTS pulls the ts field out of one canonical leaf line. A line
// the tree accepted but that carries no ts (valid JSON, not a leaf
// object) yields "" — the status then reports no last leaf rather
// than refusing to rebuild over a field the tree never needed.
func leafTS(line []byte) string {
	if len(line) == 0 {
		return ""
	}
	var probe struct {
		TS string `json:"ts"`
	}
	if err := json.Unmarshal(line, &probe); err != nil {
		return ""
	}
	return probe.TS
}

// Append canonicalizes leaf, writes it as one line, and folds its
// hash into the tree. Returns the leaf's index. Synchronous and under
// the mutex; plain write, no fsync — Page syncs before a head is
// signed. Fills V, LeafID, and TS when the caller left them zero.
//
// On a failed or short write the file is rolled back to its previous
// length so a partial line can never precede a later good one; if
// even that fails the log disables itself rather than risk it.
func (l *Log) Append(leaf Leaf) (uint64, error) {
	if leaf.V == 0 {
		leaf.V = LeafVersion
	}
	if leaf.LeafID == "" {
		id, err := NewLeafID()
		if err != nil {
			return 0, err
		}
		leaf.LeafID = id
	}
	if leaf.TS == "" {
		leaf.TS = formatTime(l.now())
	}
	raw, err := leaf.Canonical()
	if err != nil {
		return 0, fmt.Errorf("tlog: canonicalize leaf: %w", err)
	}
	line := make([]byte, 0, len(raw)+1)
	line = append(line, raw...)
	line = append(line, '\n')

	l.mu.Lock()
	defer l.mu.Unlock()
	if l.err != nil {
		return 0, l.err
	}
	n, err := l.f.Write(line)
	if err != nil || n != len(line) {
		if err == nil {
			err = io.ErrShortWrite
		}
		if terr := l.f.Truncate(l.end); terr != nil {
			l.err = fmt.Errorf("%w: append failed (%v) and rollback failed (%v)", ErrDisabled, err, terr)
			pluginlog.Errf("%v — refusing further appends", l.err)
		}
		return 0, fmt.Errorf("tlog: append: %w", err)
	}
	index := l.tree.size()
	l.offsets = append(l.offsets, l.end)
	l.end += int64(n)
	l.tree.append(LeafHash(raw))
	l.lastTS = leaf.TS
	return index, nil
}

// Head returns the current signed tree head. Signed on demand and
// cached by tree size, so repeated calls at the same size return the
// same bytes and the first call after an append re-signs.
func (l *Log) Head() (STH, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.headLocked()
}

func (l *Log) headLocked() (STH, error) {
	if l.err != nil {
		return STH{}, l.err
	}
	size := l.tree.size()
	if l.sth != nil && l.sth.TreeSize == size {
		return *l.sth, nil
	}
	sth, err := signSTH(l.key, size, l.tree.root(), l.now())
	if err != nil {
		return STH{}, err
	}
	l.sth = &sth
	return sth, nil
}

// Page serves one witness poll: fsync the leaf file so the head is
// never ahead of the disk, then the signed head, the consistency
// proof from since, and up to PageSize leaves from since.
func (l *Log) Page(since uint64) (*Page, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.err != nil {
		return nil, l.err
	}
	size := l.tree.size()
	if since > size {
		return nil, &AheadError{Since: since, TreeSize: size}
	}
	if err := l.f.Sync(); err != nil {
		return nil, fmt.Errorf("tlog: sync %s: %w", l.path, err)
	}
	sth, err := l.headLocked()
	if err != nil {
		return nil, err
	}
	proof := []string{}
	if since > 0 && since < size {
		hs, err := l.tree.consistencyProof(since, size)
		if err != nil {
			return nil, err
		}
		proof = hexHashes(hs)
	}
	next := since + PageSize
	if next > size {
		next = size
	}
	leaves, err := l.readLeavesLocked(since, next)
	if err != nil {
		return nil, err
	}
	return &Page{
		STH:              sth,
		LogPubkey:        l.key.pubHex,
		ConsistencyProof: proof,
		Leaves:           leaves,
		Next:             next,
	}, nil
}

// readLeavesLocked returns the canonical bytes of leaves [from, to)
// straight from the file, split on the line breaks Append wrote.
func (l *Log) readLeavesLocked(from, to uint64) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, to-from)
	if from == to {
		return out, nil
	}
	start := l.offsets[from]
	stop := l.end
	if to < uint64(len(l.offsets)) {
		stop = l.offsets[to]
	}
	rf, err := os.Open(l.path)
	if err != nil {
		return nil, fmt.Errorf("tlog: open %s for read: %w", l.path, err)
	}
	defer rf.Close()
	buf := make([]byte, stop-start)
	if n, err := rf.ReadAt(buf, start); n != len(buf) {
		return nil, fmt.Errorf("tlog: short read of %s at %d: %d of %d bytes (%v)", l.path, start, n, len(buf), err)
	}
	for len(buf) > 0 {
		nl := -1
		for i, b := range buf {
			if b == '\n' {
				nl = i
				break
			}
		}
		if nl < 0 {
			return nil, fmt.Errorf("tlog: %s: leaf %d is not newline-terminated", l.path, from+uint64(len(out)))
		}
		out = append(out, json.RawMessage(buf[:nl]))
		buf = buf[nl+1:]
	}
	if uint64(len(out)) != to-from {
		return nil, fmt.Errorf("tlog: %s: expected %d leaves in [%d,%d), read %d", l.path, to-from, from, to, len(out))
	}
	return out, nil
}

// InclusionProof returns the proof for the leaf at index against the
// head of size. Part 1 does not serve these; Part 2 receipts do, and
// the tests and fixtures pin them now so the construction cannot
// drift underneath the verifiers already shipped in gatekey.
func (l *Log) InclusionProof(index, size uint64) ([]Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.tree.inclusionProof(index, size)
}

// ConsistencyProof returns the proof from size m to size n.
func (l *Log) ConsistencyProof(m, n uint64) ([]Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.tree.consistencyProof(m, n)
}

// RootAt returns the root the tree had at size n (n <= Size()).
func (l *Log) RootAt(n uint64) (Hash, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if n > l.tree.size() {
		return Hash{}, ErrOutOfRange
	}
	return l.tree.rootAt(n), nil
}

// Status is the snapshot behind GET /_plugin/tlog/status: every
// local fact the dashboard card shows, read under one lock. Nothing
// here is signed, synced, or read from disk — status must be free on
// a hot gateway, and a cookie session must never receive material
// the witness countersigns (leaves, the STH signature), so those
// stay on Head/Page.
type Status struct {
	Size      uint64
	Root      Hash   // RFC 9162 empty root at size 0
	PubkeyHex string // this boot's log key; "" only if key generation failed
	Path      string
	LastTS    string // ts of the newest leaf; "" while empty
	Err       error  // non-nil ⇒ disabled, with the reason
}

// Status returns the current snapshot. Safe on a disabled log: the
// numbers are whatever was rebuilt before it refused, and Err says
// why.
func (l *Log) Status() Status {
	l.mu.Lock()
	defer l.mu.Unlock()
	st := Status{
		Size:   l.tree.size(),
		Root:   l.tree.root(),
		Path:   l.path,
		LastTS: l.lastTS,
		Err:    l.err,
	}
	if l.key != nil {
		st.PubkeyHex = l.key.pubHex
	}
	return st
}

// Size is the current number of leaves.
func (l *Log) Size() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.tree.size()
}

// Root is the current root hash.
func (l *Log) Root() Hash {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.tree.root()
}

// PubkeyHex is this boot's log public key, compressed, lowercase hex.
func (l *Log) PubkeyHex() string { return l.key.pubHex }

// Path is the leaf file.
func (l *Log) Path() string { return l.path }

// Err is the reason the log is disabled, or nil.
func (l *Log) Err() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err
}

// Close syncs and closes the leaf file. The log refuses appends
// afterwards.
func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.f == nil {
		return nil
	}
	serr := l.f.Sync()
	cerr := l.f.Close()
	l.f = nil
	if l.err == nil {
		l.err = fmt.Errorf("%w: closed", ErrDisabled)
	}
	if serr != nil {
		return serr
	}
	return cerr
}

func hexHashes(hs []Hash) []string {
	out := make([]string, len(hs))
	for i, h := range hs {
		out[i] = macaroon.BytesToHex(h[:])
	}
	return out
}
