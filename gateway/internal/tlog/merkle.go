package tlog

import (
	"crypto/sha256"
	"errors"
	"math/bits"
)

// Merkle tree per RFC 9162 §2.1 (Certificate Transparency v2, which
// obsoletes RFC 6962 without changing the hashing):
//
//	leaf_hash(d)    = SHA256(0x00 || d)
//	node_hash(l, r) = SHA256(0x01 || l || r)
//	MTH({})         = SHA256("")
//	MTH(D[n])       = node_hash(MTH(D[0:k]), MTH(D[k:n]))   k = largest power of two < n
//
// Inclusion proofs follow §2.1.3.1, consistency proofs §2.1.4.1, and
// the two verifiers below are transcriptions of §2.1.3.2 / §2.1.4.2.
// The vectors in merkle_test.go come from the reference
// implementation (transparency-dev/merkle) so the construction here
// is pinned to what every CT log speaks, not to itself.

// Hash is one SHA-256 digest.
type Hash [sha256.Size]byte

// ErrOutOfRange is returned for a proof request outside the tree.
var ErrOutOfRange = errors.New("tlog: index or size out of range")

var emptyRoot = sha256.Sum256(nil)

// EmptyRoot is MTH({}): the root of the empty tree, SHA-256 of the
// empty string.
func EmptyRoot() Hash { return emptyRoot }

// LeafHash is SHA256(0x00 || data).
func LeafHash(data []byte) Hash {
	h := sha256.New()
	h.Write([]byte{0x00})
	h.Write(data)
	var out Hash
	copy(out[:], h.Sum(nil))
	return out
}

// NodeHash is SHA256(0x01 || left || right).
func NodeHash(left, right Hash) Hash {
	h := sha256.New()
	h.Write([]byte{0x01})
	h.Write(left[:])
	h.Write(right[:])
	var out Hash
	copy(out[:], h.Sum(nil))
	return out
}

// RootFromLeafHashes computes MTH over a slice of leaf hashes with the
// recursive definition. It shares no code with the incremental tree
// below, which is the point: the tests compare the two at every size,
// and a full-replica witness (Hive) computes the root exactly this
// way from the leaves it holds.
func RootFromLeafHashes(hashes []Hash) Hash {
	switch len(hashes) {
	case 0:
		return emptyRoot
	case 1:
		return hashes[0]
	}
	k := largestPowerOfTwoBelow(uint64(len(hashes)))
	return NodeHash(RootFromLeafHashes(hashes[:k]), RootFromLeafHashes(hashes[k:]))
}

// largestPowerOfTwoBelow returns the largest power of two strictly
// less than n. n must be >= 2.
func largestPowerOfTwoBelow(n uint64) uint64 {
	return 1 << (bits.Len64(n-1) - 1)
}

func isPowerOfTwo(n uint64) bool { return n != 0 && n&(n-1) == 0 }

// tree is the in-memory Merkle tree. Every level's hashes are kept:
// levels[0] holds the leaf hashes; levels[i][j] is the hash of the
// perfect subtree covering leaves [j<<i, (j+1)<<i). Append pushes one
// leaf hash and folds every newly complete pair upward, so it is
// O(log n) and the tree never rehashes what it already has.
//
// Not safe for concurrent use; Log holds the mutex.
type tree struct {
	levels [][]Hash
}

func (t *tree) size() uint64 {
	if len(t.levels) == 0 {
		return 0
	}
	return uint64(len(t.levels[0]))
}

func (t *tree) append(h Hash) {
	if len(t.levels) == 0 {
		t.levels = append(t.levels, nil)
	}
	t.levels[0] = append(t.levels[0], h)
	for i := 0; len(t.levels[i])%2 == 0; i++ {
		lvl := t.levels[i]
		parent := NodeHash(lvl[len(lvl)-2], lvl[len(lvl)-1])
		if len(t.levels) == i+1 {
			t.levels = append(t.levels, nil)
		}
		t.levels[i+1] = append(t.levels[i+1], parent)
	}
}

// rangeHash is MTH(D[lo:hi]) for 0 <= lo < hi <= size. Perfect,
// aligned subtrees come straight out of the level cache; anything
// else splits at k the way the RFC definition does.
func (t *tree) rangeHash(lo, hi uint64) Hash {
	n := hi - lo
	if n == 1 {
		return t.levels[0][lo]
	}
	if isPowerOfTwo(n) && lo%n == 0 {
		level := bits.TrailingZeros64(n)
		return t.levels[level][lo>>level]
	}
	k := largestPowerOfTwoBelow(n)
	return NodeHash(t.rangeHash(lo, lo+k), t.rangeHash(lo+k, hi))
}

// rootAt is MTH(D[0:n]) for any n <= size — the tree's root as it was
// when it had n leaves. Every prefix's perfect subtrees are in the
// cache, so this is O(log n).
func (t *tree) rootAt(n uint64) Hash {
	if n == 0 {
		return emptyRoot
	}
	return t.rangeHash(0, n)
}

func (t *tree) root() Hash { return t.rootAt(t.size()) }

// inclusionProof is PATH(index, D[0:size]) per §2.1.3.1: the sibling
// hashes from the leaf up to the root, nearest the leaf first.
func (t *tree) inclusionProof(index, size uint64) ([]Hash, error) {
	if size > t.size() || index >= size {
		return nil, ErrOutOfRange
	}
	var path []Hash
	lo, hi := uint64(0), size
	for hi-lo > 1 {
		k := largestPowerOfTwoBelow(hi - lo)
		if index < lo+k {
			path = append(path, t.rangeHash(lo+k, hi))
			hi = lo + k
		} else {
			path = append(path, t.rangeHash(lo, lo+k))
			lo += k
		}
	}
	reverseHashes(path)
	if path == nil {
		path = []Hash{}
	}
	return path, nil
}

// consistencyProof is PROOF(m, D[0:n]) per §2.1.4.1: what a verifier
// holding the root at size m needs to check that the root at size n
// extends it. Empty when m == 0 or m == n.
func (t *tree) consistencyProof(m, n uint64) ([]Hash, error) {
	if n > t.size() || m > n {
		return nil, ErrOutOfRange
	}
	if m == 0 || m == n {
		return []Hash{}, nil
	}
	var path []Hash
	lo, hi := uint64(0), n
	rel := m // m relative to [lo, hi)
	complete := true
	for {
		if rel == hi-lo {
			if !complete {
				path = append(path, t.rangeHash(lo, hi))
			}
			break
		}
		k := largestPowerOfTwoBelow(hi - lo)
		if rel <= k {
			path = append(path, t.rangeHash(lo+k, hi))
			hi = lo + k
		} else {
			path = append(path, t.rangeHash(lo, lo+k))
			lo += k
			rel -= k
			complete = false
		}
	}
	reverseHashes(path)
	return path, nil
}

func reverseHashes(hs []Hash) {
	for i, j := 0, len(hs)-1; i < j; i, j = i+1, j-1 {
		hs[i], hs[j] = hs[j], hs[i]
	}
}

// VerifyInclusion checks an inclusion proof for leafHash at index in a
// tree of size whose root is root. Transcription of RFC 9162
// §2.1.3.2.
func VerifyInclusion(leafHash Hash, index, size uint64, proof []Hash, root Hash) bool {
	if index >= size {
		return false
	}
	fn, sn := index, size-1
	r := leafHash
	for _, p := range proof {
		if sn == 0 {
			return false
		}
		if fn&1 == 1 || fn == sn {
			r = NodeHash(p, r)
			if fn&1 == 0 {
				for fn&1 == 0 && fn != 0 {
					fn >>= 1
					sn >>= 1
				}
			}
		} else {
			r = NodeHash(r, p)
		}
		fn >>= 1
		sn >>= 1
	}
	return sn == 0 && r == root
}

// VerifyConsistency checks that a tree of newSize with root newRoot
// extends a tree of oldSize with root oldRoot, given the consistency
// proof between them. Transcription of RFC 9162 §2.1.4.2, plus the
// two degenerate cases the algorithm excludes: equal sizes need an
// empty proof and equal roots, and every tree extends the empty one.
func VerifyConsistency(oldSize, newSize uint64, proof []Hash, oldRoot, newRoot Hash) bool {
	switch {
	case oldSize > newSize:
		return false
	case oldSize == newSize:
		return len(proof) == 0 && oldRoot == newRoot
	case oldSize == 0:
		return len(proof) == 0 && oldRoot == emptyRoot
	case len(proof) == 0:
		return false
	}
	path := proof
	if isPowerOfTwo(oldSize) {
		path = append([]Hash{oldRoot}, proof...)
	}
	fn, sn := oldSize-1, newSize-1
	for fn&1 == 1 {
		fn >>= 1
		sn >>= 1
	}
	fr, sr := path[0], path[0]
	for _, c := range path[1:] {
		if sn == 0 {
			return false
		}
		if fn&1 == 1 || fn == sn {
			fr = NodeHash(c, fr)
			sr = NodeHash(c, sr)
			if fn&1 == 0 {
				for fn&1 == 0 && fn != 0 {
					fn >>= 1
					sn >>= 1
				}
			}
		} else {
			sr = NodeHash(sr, c)
		}
		fn >>= 1
		sn >>= 1
	}
	return fr == oldRoot && sr == newRoot && sn == 0
}
