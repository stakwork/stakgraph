package tlog

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fixedClock returns a clock that starts at start and advances one
// second per call, so timestamps are deterministic but distinct.
func fixedClock(start time.Time) func() time.Time {
	var mu sync.Mutex
	n := 0
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		t := start.Add(time.Duration(n) * time.Second)
		n++
		return t
	}
}

var testStart = time.Date(2026, 9, 14, 12, 0, 0, 0, time.UTC)

func newTestLog(t testing.TB, path string) *Log {
	t.Helper()
	l, err := open(options{path: path, now: fixedClock(testStart)})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = l.Close() })
	return l
}

func tempPath(t testing.TB) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "tlog", "leaves.jsonl")
}

func strp(s string) *string { return &s }

func sha256hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func sampleLeaf(i int) Leaf {
	return Leaf{
		OrgID:            "org_acme",
		UserID:           "u_alice",
		RunID:            "r_01",
		Agent:            "coder",
		MacaroonSHA256:   sha256hex("macaroon"),
		RequestSHA256:    strp(sha256hex("request-" + string(rune('a'+i%26)))),
		Model:            "claude-sonnet-5",
		Provider:         "anthropic",
		PromptTokens:     100 + i,
		CompletionTokens: 10 + i,
		CostUSD:          0.001 * float64(i),
		Status:           StatusOK,
	}
}

func appendN(t testing.TB, l *Log, n int) [][]byte {
	t.Helper()
	var canon [][]byte
	for i := 0; i < n; i++ {
		leaf := sampleLeaf(i)
		idx, err := l.Append(leaf)
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
		if idx != uint64(i) {
			t.Fatalf("append %d returned index %d", i, idx)
		}
		raw, err := os.ReadFile(l.Path())
		if err != nil {
			t.Fatal(err)
		}
		lines := bytes.Split(bytes.TrimSuffix(raw, []byte("\n")), []byte("\n"))
		canon = append(canon, append([]byte(nil), lines[len(lines)-1]...))
	}
	return canon
}

func fileLines(t testing.TB, path string) [][]byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) == 0 {
		return nil
	}
	return bytes.Split(bytes.TrimSuffix(raw, []byte("\n")), []byte("\n"))
}

func rootOfFile(t testing.TB, path string) Hash {
	t.Helper()
	var hs []Hash
	for _, line := range fileLines(t, path) {
		hs = append(hs, LeafHash(line))
	}
	return RootFromLeafHashes(hs)
}

func TestLog_EmptyTreeHead(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	sth, err := l.Head()
	if err != nil {
		t.Fatal(err)
	}
	if sth.TreeSize != 0 || sth.RootHash != hex.EncodeToString(emptyRoot[:]) {
		t.Fatalf("empty head = %+v", sth)
	}
	if !VerifySTH(sth, l.PubkeyHex()) {
		t.Fatal("empty STH does not verify")
	}
	page, err := l.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	if page.Next != 0 || len(page.Leaves) != 0 || len(page.ConsistencyProof) != 0 || page.LogPubkey != l.PubkeyHex() {
		t.Fatalf("empty page = %+v", page)
	}
	if page.Leaves == nil || page.ConsistencyProof == nil {
		t.Fatal("empty slices must be non-nil so they serialize as [] not null")
	}
}

func TestLog_AppendFillsDefaultsAndPersistsCanonical(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	leaf := sampleLeaf(0)
	if _, err := l.Append(leaf); err != nil {
		t.Fatal(err)
	}
	lines := fileLines(t, l.Path())
	if len(lines) != 1 {
		t.Fatalf("want 1 line, got %d", len(lines))
	}
	var got Leaf
	if err := jsonUnmarshal(lines[0], &got); err != nil {
		t.Fatalf("line is not a leaf: %v", err)
	}
	if got.V != LeafVersion {
		t.Errorf("v = %d", got.V)
	}
	if len(got.LeafID) != 32 {
		t.Errorf("leaf_id = %q, want 128-bit hex", got.LeafID)
	}
	if _, err := hex.DecodeString(got.LeafID); err != nil {
		t.Errorf("leaf_id not hex: %v", err)
	}
	if got.TS != "2026-09-14T12:00:00.000Z" {
		t.Errorf("ts = %q", got.TS)
	}
	// The line is byte-identical to the leaf's own canonical form
	// (i.e. what was written is what re-canonicalizes).
	canon, err := got.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(canon, lines[0]) {
		t.Errorf("persisted line is not canonical:\n  %s\n  %s", lines[0], canon)
	}
	if got.ResponseSHA256 != nil || got.AgentRequestSig != nil {
		t.Error("reserved fields must be null")
	}
	if !strings.Contains(string(lines[0]), `"response_sha256":null`) {
		t.Errorf("nulls must be explicit on the wire: %s", lines[0])
	}
	// Caller-supplied identity/timestamp survive untouched.
	fixed := sampleLeaf(1)
	fixed.LeafID = strings.Repeat("ab", 16)
	fixed.TS = "2026-01-01T00:00:00.000Z"
	if _, err := l.Append(fixed); err != nil {
		t.Fatal(err)
	}
	if line := string(fileLines(t, l.Path())[1]); !strings.Contains(line, fixed.LeafID) || !strings.Contains(line, fixed.TS) {
		t.Errorf("explicit leaf_id/ts not preserved: %s", line)
	}
}

func TestLog_PageServesLeavesAndProofs(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	canon := appendN(t, l, 5)

	page, err := l.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	if page.STH.TreeSize != 5 || page.Next != 5 || len(page.Leaves) != 5 {
		t.Fatalf("page(0) = size %d next %d leaves %d", page.STH.TreeSize, page.Next, len(page.Leaves))
	}
	if len(page.ConsistencyProof) != 0 {
		t.Errorf("since=0 must carry no consistency proof, got %v", page.ConsistencyProof)
	}
	for i, raw := range page.Leaves {
		if !bytes.Equal(raw, canon[i]) {
			t.Errorf("leaf %d served bytes differ from persisted bytes", i)
		}
	}
	wantRoot := rootOfFile(t, l.Path())
	if page.STH.RootHash != hex.EncodeToString(wantRoot[:]) {
		t.Errorf("root_hash = %s, want %x", page.STH.RootHash, wantRoot)
	}
	if !VerifySTH(page.STH, page.LogPubkey) {
		t.Error("STH signature does not verify with log_pubkey")
	}

	// Mid-tree since: proof from 2 to 5 verifies against the roots.
	page, err = l.Page(2)
	if err != nil {
		t.Fatal(err)
	}
	if page.Next != 5 || len(page.Leaves) != 3 || !bytes.Equal(page.Leaves[0], canon[2]) {
		t.Fatalf("page(2) = next %d leaves %d", page.Next, len(page.Leaves))
	}
	old, _ := l.RootAt(2)
	proof := make([]Hash, len(page.ConsistencyProof))
	for i, s := range page.ConsistencyProof {
		proof[i] = mustHash(t, s)
	}
	if !VerifyConsistency(2, 5, proof, old, wantRoot) {
		t.Error("served consistency proof does not verify")
	}

	// since == tree_size: nothing new, no proof.
	page, err = l.Page(5)
	if err != nil {
		t.Fatal(err)
	}
	if page.Next != 5 || len(page.Leaves) != 0 || len(page.ConsistencyProof) != 0 {
		t.Fatalf("page(5) = %+v", page)
	}

	// since > tree_size: the witness is ahead of us.
	_, err = l.Page(6)
	var ahead *AheadError
	if !errors.As(err, &ahead) || ahead.TreeSize != 5 || ahead.Since != 6 {
		t.Fatalf("page(6) err = %v", err)
	}
}

func TestLog_PageCapsAtPageSize(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	const extra = 3
	for i := 0; i < PageSize+extra; i++ {
		if _, err := l.Append(sampleLeaf(i)); err != nil {
			t.Fatal(err)
		}
	}
	page, err := l.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	if page.Next != PageSize || len(page.Leaves) != PageSize || page.STH.TreeSize != PageSize+extra {
		t.Fatalf("page(0): next %d leaves %d size %d", page.Next, len(page.Leaves), page.STH.TreeSize)
	}
	page, err = l.Page(page.Next)
	if err != nil {
		t.Fatal(err)
	}
	if page.Next != PageSize+extra || len(page.Leaves) != extra {
		t.Fatalf("page(%d): next %d leaves %d", PageSize, page.Next, len(page.Leaves))
	}
}

func TestLog_HeadCachedUntilAppend(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	appendN(t, l, 2)
	a, _ := l.Head()
	b, _ := l.Head() // clock advanced, size did not
	if a != b {
		t.Fatalf("head re-signed at the same size:\n%+v\n%+v", a, b)
	}
	p, _ := l.Page(0)
	if p.STH != a {
		t.Fatal("Page must serve the cached head")
	}
	if _, err := l.Append(sampleLeaf(2)); err != nil {
		t.Fatal(err)
	}
	c, _ := l.Head()
	if c.TreeSize != 3 || c.SignedAt == a.SignedAt || c.Sig == a.Sig || c.RootHash == a.RootHash {
		t.Fatalf("head not re-signed after append: %+v vs %+v", a, c)
	}
	if !VerifySTH(c, l.PubkeyHex()) {
		t.Fatal("re-signed head does not verify")
	}
	tampered := c
	tampered.TreeSize++
	if VerifySTH(tampered, l.PubkeyHex()) {
		t.Fatal("tampered head verified")
	}
}

func TestLog_RebuildFromFileMatchesMemory(t *testing.T) {
	path := tempPath(t)
	l := newTestLog(t, path)
	canon := appendN(t, l, 37)
	wantRoot := l.Root()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Append(sampleLeaf(0)); !errors.Is(err, ErrDisabled) {
		t.Fatalf("append after close err = %v", err)
	}

	re := newTestLog(t, path)
	if re.Size() != 37 || re.Root() != wantRoot {
		t.Fatalf("rebuilt size %d root %x, want 37 %x", re.Size(), re.Root(), wantRoot)
	}
	page, err := re.Page(0)
	if err != nil {
		t.Fatal(err)
	}
	for i, raw := range page.Leaves {
		if !bytes.Equal(raw, canon[i]) {
			t.Fatalf("rebuilt leaf %d differs", i)
		}
	}
	// Proofs across the boundary still verify.
	old, _ := re.RootAt(20)
	proof, _ := re.ConsistencyProof(20, 37)
	if !VerifyConsistency(20, 37, proof, old, wantRoot) {
		t.Fatal("consistency proof after rebuild does not verify")
	}
	// Appending continues the sequence.
	idx, err := re.Append(sampleLeaf(99))
	if err != nil || idx != 37 {
		t.Fatalf("append after rebuild: idx %d err %v", idx, err)
	}
	if re.Root() != rootOfFile(t, path) {
		t.Fatal("root after rebuild+append disagrees with the file")
	}
}

func TestLog_CorruptTailIsTruncated(t *testing.T) {
	for name, garbage := range map[string]string{
		"partial line":     `{"v":1,"leaf_id":"abc`,
		"complete garbage": "garbage\n",
		"nul run":          string(make([]byte, 64)),
		"blank line":       "\n",
	} {
		t.Run(name, func(t *testing.T) {
			path := tempPath(t)
			l := newTestLog(t, path)
			appendN(t, l, 3)
			wantRoot := l.Root()
			_ = l.Close()
			clean, _ := os.ReadFile(path)
			f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
			_, _ = f.WriteString(garbage)
			_ = f.Close()

			re := newTestLog(t, path)
			if re.Size() != 3 || re.Root() != wantRoot {
				t.Fatalf("size %d root %x after truncation, want 3 %x", re.Size(), re.Root(), wantRoot)
			}
			now, _ := os.ReadFile(path)
			if !bytes.Equal(now, clean) {
				t.Fatalf("file not truncated back to the clean prefix (%d vs %d bytes)", len(now), len(clean))
			}
			if idx, err := re.Append(sampleLeaf(3)); err != nil || idx != 3 {
				t.Fatalf("append after truncation: idx %d err %v", idx, err)
			}
			if len(fileLines(t, path)) != 4 {
				t.Fatal("append after truncation did not land as line 4")
			}
		})
	}
}

func TestLog_MidFileCorruptionDisablesWithoutTouchingFile(t *testing.T) {
	path := tempPath(t)
	l := newTestLog(t, path)
	appendN(t, l, 3)
	_ = l.Close()
	lines := fileLines(t, path)
	lines[1] = []byte("not json")
	corrupt := append(bytes.Join(lines, []byte("\n")), '\n')
	if err := os.WriteFile(path, corrupt, 0o644); err != nil {
		t.Fatal(err)
	}

	re, err := open(options{path: path})
	if err == nil {
		t.Fatal("open must fail on mid-file corruption")
	}
	if re == nil || !errors.Is(re.Err(), ErrDisabled) {
		t.Fatalf("log must come back disabled, got %v", re)
	}
	if _, err := re.Append(sampleLeaf(0)); !errors.Is(err, ErrDisabled) {
		t.Errorf("append on disabled log err = %v", err)
	}
	if _, err := re.Page(0); !errors.Is(err, ErrDisabled) {
		t.Errorf("page on disabled log err = %v", err)
	}
	if _, err := re.Head(); !errors.Is(err, ErrDisabled) {
		t.Errorf("head on disabled log err = %v", err)
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(after, corrupt) {
		t.Fatal("a disabled rebuild must leave the file exactly as it found it")
	}
}

func TestLog_ConcurrentAppends(t *testing.T) {
	path := tempPath(t)
	l := newTestLog(t, path)
	const workers, each = 8, 50
	var wg sync.WaitGroup
	seen := make([]bool, workers*each)
	var mu sync.Mutex
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				leaf := sampleLeaf(w*each + i)
				idx, err := l.Append(leaf)
				if err != nil {
					t.Errorf("append: %v", err)
					return
				}
				mu.Lock()
				if seen[idx] {
					t.Errorf("index %d handed out twice", idx)
				}
				seen[idx] = true
				mu.Unlock()
			}
		}(w)
	}
	wg.Wait()
	if l.Size() != workers*each {
		t.Fatalf("size = %d", l.Size())
	}
	if l.Root() != rootOfFile(t, path) {
		t.Fatal("in-memory root disagrees with the file after concurrent appends")
	}
	if lines := fileLines(t, path); len(lines) != workers*each {
		t.Fatalf("file has %d lines", len(lines))
	}
}

// Closing the descriptor out from under the log is the closest a unit
// test gets to a dead disk: the write fails, the rollback fails, and
// the log must stop rather than risk a torn file.
func TestLog_WriteFailureDisables(t *testing.T) {
	l := newTestLog(t, tempPath(t))
	appendN(t, l, 2)
	_ = l.f.Close()
	if _, err := l.Append(sampleLeaf(2)); err == nil {
		t.Fatal("append on a closed descriptor must fail")
	}
	if l.Size() != 2 {
		t.Fatalf("failed append changed the tree: size %d", l.Size())
	}
	if !errors.Is(l.Err(), ErrDisabled) {
		t.Fatalf("log must disable itself when rollback fails, err = %v", l.Err())
	}
	l.f = nil // Cleanup's Close must not double-close
}

func TestDefault_NotInitialized(t *testing.T) {
	prev := Default()
	SetDefaultForTest(nil)
	t.Cleanup(func() { SetDefaultForTest(prev) })
	if _, err := Append(sampleLeaf(0)); !errors.Is(err, ErrNotInitialized) {
		t.Fatalf("err = %v", err)
	}
	if err := Close(); err != nil {
		t.Fatalf("close of nil default: %v", err)
	}
}

func TestInit_InstallsDisabledLogOnFailure(t *testing.T) {
	prev := Default()
	t.Cleanup(func() { SetDefaultForTest(prev) })
	// A path under a regular file cannot be created.
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Init(filepath.Join(blocker, "leaves.jsonl"))
	if err == nil {
		t.Fatal("Init must report the failure")
	}
	if l := Default(); l == nil || !errors.Is(l.Err(), ErrDisabled) {
		t.Fatal("Init must still install a (disabled) log so the route can say why")
	}
	if _, err := Append(sampleLeaf(0)); !errors.Is(err, ErrDisabled) {
		t.Fatalf("append err = %v", err)
	}
	_ = Close()
}

// The newest leaf's ts is tracked on Append and recovered at rebuild
// from the last valid line alone, so the status card can say "last
// leaf 12s ago" after a restart without a signed head or a page.
func TestLog_LastTSSurvivesRebuild(t *testing.T) {
	path := tempPath(t)
	l := newTestLog(t, path)
	if st := l.Status(); st.LastTS != "" || st.Size != 0 || st.Root != emptyRoot || st.Err != nil {
		t.Fatalf("empty status = %+v", st)
	}
	if st := l.Status(); st.PubkeyHex != l.PubkeyHex() || st.Path != path {
		t.Fatalf("status identity = %+v", st)
	}

	appendN(t, l, 3) // fixedClock: 12:00:00, :01, :02
	st := l.Status()
	if st.LastTS != "2026-09-14T12:00:02.000Z" || st.Size != 3 || st.Root != l.Root() {
		t.Fatalf("status after appends = %+v", st)
	}
	// A caller-supplied ts is what gets recorded, not the clock.
	fixed := sampleLeaf(3)
	fixed.TS = "2026-01-01T00:00:00.000Z"
	if _, err := l.Append(fixed); err != nil {
		t.Fatal(err)
	}
	if got := l.Status().LastTS; got != fixed.TS {
		t.Fatalf("lastTS after explicit ts = %q", got)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if st := l.Status(); st.Err == nil || !errors.Is(st.Err, ErrDisabled) {
		t.Fatalf("closed log must report disabled, got %+v", st)
	}

	re := newTestLog(t, path)
	st = re.Status()
	if st.LastTS != fixed.TS || st.Size != 4 || st.Err != nil {
		t.Fatalf("rebuilt status = %+v", st)
	}

	// Rebuild after a corrupt tail: the last *valid* line's ts.
	_ = re.Close()
	f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	_, _ = f.WriteString(`{"v":1,"ts":"2099-01-01T00:00:00.000Z","leaf_id":"trunc`)
	_ = f.Close()
	re2 := newTestLog(t, path)
	if got := re2.Status().LastTS; got != fixed.TS {
		t.Fatalf("lastTS after corrupt-tail rebuild = %q, want %q", got, fixed.TS)
	}
	// And a fresh append moves it again.
	if _, err := re2.Append(sampleLeaf(4)); err != nil {
		t.Fatal(err)
	}
	if got := re2.Status().LastTS; got == fixed.TS || got == "" {
		t.Fatalf("lastTS after append on rebuilt log = %q", got)
	}
}

// A line the tree accepts (valid JSON) but that is not a leaf object
// yields no ts rather than a failed rebuild.
func TestLog_LastTSIgnoresNonLeafLine(t *testing.T) {
	path := tempPath(t)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("[1,2,3]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	l := newTestLog(t, path)
	if st := l.Status(); st.Size != 1 || st.LastTS != "" || st.Err != nil {
		t.Fatalf("status = %+v", st)
	}
}
