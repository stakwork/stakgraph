package tlog

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// Cross-language fixtures for the transparency log, mirror image of
// the macaroon fixtures: here Go is the producer, so Go generates and
// gatekey (gateway/auth/ts/test/tlog.test.ts) checks byte-for-byte.
//
//	go test ./internal/tlog -update     # regenerate gateway/auth/fixtures/tlog-*.json
//	go test ./internal/tlog             # assert this implementation reproduces them
//
// Two files:
//
//	tlog-00-rfc9162-vectors.json  raw-byte leaves from the CT reference
//	                              suite; the test also pins the expected
//	                              values to the reference constants in
//	                              merkle_test.go, so a generator bug
//	                              cannot launder itself into the fixture.
//	tlog-01-leaves.json           eight JSON leaves through the real Log,
//	                              with a fixed log key and clock: canonical
//	                              bytes, hashes, roots, every proof, and
//	                              signed heads for the empty and full tree.

var update = flag.Bool("update", false, "regenerate gateway/auth/fixtures/tlog-*.json from this implementation")

const (
	fixturesDir       = "../../auth/fixtures"
	vectorFixtureName = "tlog-00-rfc9162-vectors.json"
	leavesFixtureName = "tlog-01-leaves.json"

	// Deterministic log key for the fixtures. Published in this repo;
	// never use it for anything real.
	fixtureLogPrivHex = "0000000000000000000000000000000000000000000000000000000000000042"
	fixtureSignedAt   = "2026-09-14T12:00:00.000Z"
)

type inclusionVector struct {
	LeafIndex uint64   `json:"leaf_index"`
	TreeSize  uint64   `json:"tree_size"`
	Proof     []string `json:"proof"`
}

type consistencyVector struct {
	OldSize uint64   `json:"old_size"`
	NewSize uint64   `json:"new_size"`
	Proof   []string `json:"proof"`
}

type sthVector struct {
	STH             STH    `json:"sth"`
	SigningInput    string `json:"signing_input"`
	SigningBytesHex string `json:"signing_bytes_hex"`
}

type vectorFixture struct {
	Description string `json:"description"`
	Inputs      struct {
		LeavesHex []string `json:"leaves_hex"`
	} `json:"inputs"`
	Expected struct {
		LeafHashes        []string            `json:"leaf_hashes"`
		Roots             []string            `json:"roots"`
		InclusionProofs   []inclusionVector   `json:"inclusion_proofs"`
		ConsistencyProofs []consistencyVector `json:"consistency_proofs"`
	} `json:"expected"`
}

type leavesFixture struct {
	Description string `json:"description"`
	Inputs      struct {
		LogPrivHex string `json:"log_priv_hex"`
		SignedAt   string `json:"signed_at"`
		Leaves     []Leaf `json:"leaves"`
	} `json:"inputs"`
	Expected struct {
		LogPubkey         string              `json:"log_pubkey"`
		LeafCanonicalJSON []string            `json:"leaf_canonical_json"`
		LeafHashes        []string            `json:"leaf_hashes"`
		Roots             []string            `json:"roots"`
		InclusionProofs   []inclusionVector   `json:"inclusion_proofs"`
		ConsistencyProofs []consistencyVector `json:"consistency_proofs"`
		STHs              []sthVector         `json:"sths"`
	} `json:"expected"`
}

// fixtureLeaves are chosen to exercise the JCS corners a witness has
// to agree on: explicit nulls, floats in every ES6 rendering (plain,
// integer-valued, exponent), large integers, non-ASCII, and the
// characters JSON must escape.
func fixtureLeaves() []Leaf {
	ts := func(i int) string {
		return time.Date(2026, 9, 14, 11, 59, 50+i, 0, time.UTC).Format("2006-01-02T15:04:05.000Z07:00")
	}
	id := func(i int) string { return hex.EncodeToString(bytes.Repeat([]byte{byte(i)}, 16)) }
	mac := func(i int) string { return sha256hex("macaroon-" + string(rune('0'+i))) }
	req := func(i int) *string { return strp(sha256hex("request-" + string(rune('0'+i)))) }
	return []Leaf{
		{V: 1, LeafID: id(1), TS: ts(0), OrgID: "org_acme", UserID: "u_alice", RunID: "r_01HZXK4T4M0000000000000A", Agent: "coder",
			MacaroonSHA256: mac(1), RequestSHA256: req(1), Model: "claude-sonnet-5", Provider: "anthropic",
			PromptTokens: 1234, CompletionTokens: 56, CostUSD: 0.0123, Status: StatusOK},
		{V: 1, LeafID: id(2), TS: ts(1), OrgID: "org_acme", UserID: "u_alice", RunID: "r_01HZXK4T4M0000000000000A", Agent: "coder",
			MacaroonSHA256: mac(1), RequestSHA256: req(2), Model: "claude-sonnet-5", Provider: "anthropic",
			PromptTokens: 0, CompletionTokens: 0, CostUSD: 0, Status: StatusError},
		{V: 1, LeafID: id(3), TS: ts(2), OrgID: "org_acme", UserID: "u_alice", RunID: "r_01HZXK4T4M0000000000000A", Agent: "coder",
			MacaroonSHA256: mac(1), RequestSHA256: nil, Model: "claude-sonnet-5", Provider: "anthropic",
			PromptTokens: 91000, CompletionTokens: 12, CostUSD: 0.0042, Status: StatusOK},
		{V: 1, LeafID: id(4), TS: ts(3), OrgID: "org_acme", UserID: "u_bob", RunID: "r_01HZXK4T4M0000000000000B", Agent: "übersetzer",
			MacaroonSHA256: mac(2), RequestSHA256: req(4), Model: "gpt-5", Provider: "openai",
			PromptTokens: 100000, CompletionTokens: 2048, CostUSD: 12.5, Status: StatusOK},
		{V: 1, LeafID: id(5), TS: ts(4), OrgID: "org_acme", UserID: "u_bob", RunID: "r_01HZXK4T4M0000000000000B", Agent: "web-search",
			MacaroonSHA256: mac(2), RequestSHA256: req(5), Model: "claude-haiku-4-5-20251001", Provider: "anthropic",
			PromptTokens: 3, CompletionTokens: 1, CostUSD: 0.00000123, Status: StatusOK},
		{V: 1, LeafID: id(6), TS: ts(5), OrgID: "org_acme", UserID: "u_bob", RunID: "r_01HZXK4T4M0000000000000B::C-001", Agent: `code"r\<&>`,
			MacaroonSHA256: mac(3), RequestSHA256: req(6), Model: "grok-4", Provider: "xai",
			PromptTokens: 512, CompletionTokens: 128, CostUSD: 0.5, Status: StatusOK},
		{V: 1, LeafID: id(7), TS: ts(6), OrgID: "org_acme", UserID: "u_bob", RunID: "r_01HZXK4T4M0000000000000B::C-001", Agent: "coder",
			MacaroonSHA256: mac(3), RequestSHA256: req(7), Model: "gemini-2.5-pro", Provider: "gemini",
			PromptTokens: 1, CompletionTokens: 1, CostUSD: 1, Status: StatusOK},
		{V: 1, LeafID: id(8), TS: ts(7), OrgID: "org_other", UserID: "u_carol", RunID: "r_01HZXK4T4M0000000000000C", Agent: "pr-monitor",
			MacaroonSHA256: mac(4), RequestSHA256: req(8), Model: "claude-opus-5", Provider: "anthropic",
			PromptTokens: 200000, CompletionTokens: 32000, CostUSD: 1e21, Status: StatusOK},
	}
}

func hexList(hs []Hash) []string { return hexHashes(hs) }

func allProofs(t *testing.T, l *Log, size uint64) ([]inclusionVector, []consistencyVector) {
	t.Helper()
	var inc []inclusionVector
	for n := uint64(1); n <= size; n++ {
		for i := uint64(0); i < n; i++ {
			p, err := l.InclusionProof(i, n)
			if err != nil {
				t.Fatal(err)
			}
			inc = append(inc, inclusionVector{LeafIndex: i, TreeSize: n, Proof: hexList(p)})
		}
	}
	var con []consistencyVector
	for m := uint64(0); m <= size; m++ {
		for n := m; n <= size; n++ {
			p, err := l.ConsistencyProof(m, n)
			if err != nil {
				t.Fatal(err)
			}
			con = append(con, consistencyVector{OldSize: m, NewSize: n, Proof: hexList(p)})
		}
	}
	return inc, con
}

func rootsThrough(t *testing.T, l *Log, size uint64) []string {
	t.Helper()
	out := make([]string, 0, size+1)
	for n := uint64(0); n <= size; n++ {
		r, err := l.RootAt(n)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, hex.EncodeToString(r[:]))
	}
	return out
}

func buildVectorFixture(t *testing.T) vectorFixture {
	t.Helper()
	l := newTestLog(t, tempPath(t))
	var fx vectorFixture
	fx.Description = "RFC 9162 §2.1 (RFC 6962) Merkle tree vectors over the eight classic Certificate Transparency test leaves: leaf hashes, roots at every size 0..8, every inclusion proof, every consistency proof (old_size 0 included: the empty tree is extended by every tree with an empty proof). Values are pinned to the transparency-dev/merkle reference implementation by gateway/internal/tlog/merkle_test.go."
	for _, leaf := range referenceLeaves {
		fx.Inputs.LeavesHex = append(fx.Inputs.LeavesHex, hex.EncodeToString(leaf))
		// Feed raw bytes through the tree directly: these leaves are
		// not JSON, so Append (which canonicalizes) is bypassed.
		l.mu.Lock()
		l.tree.append(LeafHash(leaf))
		l.mu.Unlock()
		h := LeafHash(leaf)
		fx.Expected.LeafHashes = append(fx.Expected.LeafHashes, hex.EncodeToString(h[:]))
	}
	size := uint64(len(referenceLeaves))
	fx.Expected.Roots = rootsThrough(t, l, size)
	fx.Expected.InclusionProofs, fx.Expected.ConsistencyProofs = allProofs(t, l, size)
	return fx
}

func buildLeavesFixture(t *testing.T) leavesFixture {
	t.Helper()
	priv, err := hex.DecodeString(fixtureLogPrivHex)
	if err != nil {
		t.Fatal(err)
	}
	signedAt, err := time.Parse(time.RFC3339Nano, fixtureSignedAt)
	if err != nil {
		t.Fatal(err)
	}
	l, err := open(options{path: tempPath(t), priv: priv, now: func() time.Time { return signedAt }})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = l.Close() })

	var fx leavesFixture
	fx.Description = "Eight transparency-log leaves appended through the gateway's Log with a fixed log key and clock: canonical (JCS) bytes and leaf hash per leaf, roots at every size, every inclusion and consistency proof, and signed tree heads for the empty tree and the full tree with their exact signing input. Leaves cover explicit nulls, integer-valued and exponent-form floats, non-ASCII, and characters JSON must escape."
	fx.Inputs.LogPrivHex = fixtureLogPrivHex
	fx.Inputs.SignedAt = fixtureSignedAt
	fx.Inputs.Leaves = fixtureLeaves()
	fx.Expected.LogPubkey = l.PubkeyHex()

	// The empty tree's head, signed before anything is appended.
	emptySTH, err := l.Head()
	if err != nil {
		t.Fatal(err)
	}
	for i, leaf := range fx.Inputs.Leaves {
		idx, err := l.Append(leaf)
		if err != nil {
			t.Fatalf("append %d: %v", i, err)
		}
		if idx != uint64(i) {
			t.Fatalf("append %d got index %d", i, idx)
		}
		canon, err := leaf.Canonical()
		if err != nil {
			t.Fatal(err)
		}
		fx.Expected.LeafCanonicalJSON = append(fx.Expected.LeafCanonicalJSON, string(canon))
		h := LeafHash(canon)
		fx.Expected.LeafHashes = append(fx.Expected.LeafHashes, hex.EncodeToString(h[:]))
	}
	size := uint64(len(fx.Inputs.Leaves))
	fx.Expected.Roots = rootsThrough(t, l, size)
	fx.Expected.InclusionProofs, fx.Expected.ConsistencyProofs = allProofs(t, l, size)
	fullSTH, err := l.Head()
	if err != nil {
		t.Fatal(err)
	}
	for _, sth := range []STH{emptySTH, fullSTH} {
		in, err := SigningBytes(sth)
		if err != nil {
			t.Fatal(err)
		}
		fx.Expected.STHs = append(fx.Expected.STHs, sthVector{
			STH: sth, SigningInput: string(in), SigningBytesHex: hex.EncodeToString(in),
		})
	}
	return fx
}

// writeFixture emits indented JSON without HTML escaping, so the
// canonical strings inside stay readable and byte-identical to what
// was hashed.
func writeFixture(t *testing.T, name string, v any) {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fixturesDir, name), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", filepath.Join(fixturesDir, name))
}

func readFixture(t *testing.T, name string, into any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(fixturesDir, name))
	if err != nil {
		t.Fatalf("read %s: %v (run `go test ./internal/tlog -update` to generate)", name, err)
	}
	if err := json.Unmarshal(raw, into); err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
}

func TestFixtures_Vectors(t *testing.T) {
	built := buildVectorFixture(t)

	// The generated values must equal the reference constants — this
	// is what stops a generator bug from becoming the fixture.
	ref := parseReference(t)
	for n, want := range ref.roots {
		if built.Expected.Roots[n] != hex.EncodeToString(want[:]) {
			t.Fatalf("generated root %d disagrees with the reference", n)
		}
	}
	for _, iv := range built.Expected.InclusionProofs {
		if !reflect.DeepEqual(iv.Proof, hexList(ref.inclusion[[2]uint64{iv.LeafIndex, iv.TreeSize}])) {
			t.Fatalf("generated inclusion (%d,%d) disagrees with the reference", iv.LeafIndex, iv.TreeSize)
		}
	}
	for _, cv := range built.Expected.ConsistencyProofs {
		if cv.OldSize == 0 {
			if len(cv.Proof) != 0 {
				t.Fatalf("consistency (0,%d) must be empty", cv.NewSize)
			}
			continue
		}
		if !reflect.DeepEqual(cv.Proof, hexList(ref.consistency[[2]uint64{cv.OldSize, cv.NewSize}])) {
			t.Fatalf("generated consistency (%d,%d) disagrees with the reference", cv.OldSize, cv.NewSize)
		}
	}

	if *update {
		writeFixture(t, vectorFixtureName, built)
	}
	var onDisk vectorFixture
	readFixture(t, vectorFixtureName, &onDisk)
	if !reflect.DeepEqual(onDisk, built) {
		t.Fatalf("%s is stale: regenerate with `go test ./internal/tlog -update`", vectorFixtureName)
	}
}

func TestFixtures_Leaves(t *testing.T) {
	built := buildLeavesFixture(t)

	// Self-check before trusting the output: every head verifies with
	// the fixture key, and the signature is what the shared helper
	// produces over the recorded signing input.
	for _, sv := range built.Expected.STHs {
		if !VerifySTH(sv.STH, built.Expected.LogPubkey) {
			t.Fatalf("generated STH at size %d does not verify", sv.STH.TreeSize)
		}
		priv, _ := hex.DecodeString(fixtureLogPrivHex)
		sig, err := macaroon.EcdsaSecp256k1Sign(priv, []byte(sv.SigningInput))
		if err != nil {
			t.Fatal(err)
		}
		if hex.EncodeToString(sig) != sv.STH.Sig {
			t.Fatalf("STH signature is not deterministic over the signing input")
		}
	}
	if built.Expected.STHs[1].STH.RootHash != built.Expected.Roots[len(built.Expected.Roots)-1] {
		t.Fatal("full-tree STH root does not match roots[size]")
	}
	if built.Expected.STHs[0].STH.RootHash != hex.EncodeToString(emptyRoot[:]) {
		t.Fatal("empty-tree STH root is not the empty root")
	}
	// Every canonical leaf round-trips: parse → JCS → same bytes.
	for i, canon := range built.Expected.LeafCanonicalJSON {
		var leaf Leaf
		if err := json.Unmarshal([]byte(canon), &leaf); err != nil {
			t.Fatalf("leaf %d canonical is not parseable: %v", i, err)
		}
		again, _ := leaf.Canonical()
		if string(again) != canon {
			t.Fatalf("leaf %d canonical does not round-trip:\n%s\n%s", i, canon, again)
		}
	}

	if *update {
		writeFixture(t, leavesFixtureName, built)
	}
	var onDisk leavesFixture
	readFixture(t, leavesFixtureName, &onDisk)
	if !reflect.DeepEqual(onDisk, built) {
		t.Fatalf("%s is stale: regenerate with `go test ./internal/tlog -update`", leavesFixtureName)
	}
}
