package tlog

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"
)

// referenceLeaves are the eight classic CT test leaves.
var referenceLeaves = [][]byte{
	{},
	{0x00},
	{0x10},
	{0x20, 0x21},
	{0x30, 0x31},
	{0x40, 0x41, 0x42, 0x43},
	{0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57},
	{0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f},
}

// referenceVectors is the verbatim output of the RFC 6962 / RFC 9162
// reference implementation (github.com/transparency-dev/merkle:
// rfc6962.DefaultHasher, proof.Inclusion, proof.Consistency) over
// referenceLeaves. Lines: "leaf_hash i hex", "root n hex",
// "inclusion i n hex…", "consistency m n hex…". Pinning to an
// external producer is what makes this a conformance test rather
// than a tautology.
const referenceVectors = `
leaf_hash 0 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d
leaf_hash 1 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7
leaf_hash 2 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7
leaf_hash 3 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7
leaf_hash 4 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
leaf_hash 5 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658
leaf_hash 6 b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f
leaf_hash 7 46f6ffadd3d06a09ff3c5860d2755c8b9819db7df44251788c7d8e3180de8eb1
root 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
root 1 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d
root 2 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
root 3 aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77
root 4 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
root 5 4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4
root 6 76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef
root 7 ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c
root 8 5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328
inclusion 0 1
inclusion 0 2 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7
inclusion 1 2 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d
inclusion 0 3 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7
inclusion 1 3 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7
inclusion 2 3 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
inclusion 0 4 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e
inclusion 1 4 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e
inclusion 2 4 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
inclusion 3 4 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
inclusion 0 5 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
inclusion 1 5 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
inclusion 2 5 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
inclusion 3 5 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
inclusion 4 5 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 0 6 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
inclusion 1 6 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
inclusion 2 6 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
inclusion 3 6 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
inclusion 4 6 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 5 6 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 0 7 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
inclusion 1 7 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
inclusion 2 7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
inclusion 3 7 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
inclusion 4 7 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 5 7 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 6 7 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 0 8 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
inclusion 1 8 6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
inclusion 2 8 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
inclusion 3 8 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
inclusion 4 8 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 5 8 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 6 8 46f6ffadd3d06a09ff3c5860d2755c8b9819db7df44251788c7d8e3180de8eb1 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
inclusion 7 8 b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 1 1
consistency 1 2 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7
consistency 1 3 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7
consistency 1 4 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e
consistency 1 5 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
consistency 1 6 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
consistency 1 7 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
consistency 1 8 96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
consistency 2 2
consistency 2 3 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7
consistency 2 4 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e
consistency 2 5 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
consistency 2 6 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
consistency 2 7 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
consistency 2 8 5f083f0a1a33ca076a95279832580db3e0ef4584bdff1f54c8a360f50de3031e 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
consistency 3 3
consistency 3 4 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125
consistency 3 5 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
consistency 3 6 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
consistency 3 7 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
consistency 3 8 0298d122906dcfc10892cb53a73992fc5b9f493ea4c9badb27b791b4127a7fe7 07506a85fd9dd2f120eb694f86011e5bb4662e5c415a62917033d4a9624487e7 fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
consistency 4 4
consistency 4 5 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b
consistency 4 6 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a
consistency 4 7 837dbb152e9b079010717e84e865da4ebc0fa198a806d59d31bf15accef22d0e
consistency 4 8 6b47aaf29ee3c2af9af889bc1fb9254dabd31177f16232dd6aab035ca39bf6e4
consistency 5 5
consistency 5 6 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 5 7 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 5 8 bc1a0643b12e4d2d7c77918f44e0f4f79a838b6cf9ec5b5c283e1f4d88599e6b 4271a26be0d8a84f0bd54c8c302e7cb3a3b5d1fa6780a40bcce2873477dab658 ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 6 6
consistency 6 7 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 6 8 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a ca854ea128ed050b41b35ffc1b87b8eb2bde461e9e3b5596ece6b9d5975a0ae0 d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 7 7
consistency 7 8 b08693ec2e721597130641e8211e7eedccb4c26413963eee6c1e2ed16ffb1a5f 46f6ffadd3d06a09ff3c5860d2755c8b9819db7df44251788c7d8e3180de8eb1 0ebc5d3437fbe2db158b9f126a1d118e308181031d0a949f8dededebc558ef6a d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7
consistency 8 8
`

type reference struct {
	leafHashes  []Hash
	roots       []Hash               // index = size
	inclusion   map[[2]uint64][]Hash // (index, size)
	consistency map[[2]uint64][]Hash // (m, n)
}

func mustHash(t testing.TB, s string) Hash {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != len(Hash{}) {
		t.Fatalf("bad hash literal %q: %v", s, err)
	}
	var h Hash
	copy(h[:], b)
	return h
}

func parseReference(t testing.TB) reference {
	t.Helper()
	ref := reference{
		leafHashes:  make([]Hash, 8),
		roots:       make([]Hash, 9),
		inclusion:   map[[2]uint64][]Hash{},
		consistency: map[[2]uint64][]Hash{},
	}
	sc := bufio.NewScanner(strings.NewReader(referenceVectors))
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) == 0 {
			continue
		}
		num := func(i int) uint64 {
			n, err := strconv.ParseUint(f[i], 10, 64)
			if err != nil {
				t.Fatalf("bad vector line %q", sc.Text())
			}
			return n
		}
		hashes := func(from int) []Hash {
			out := []Hash{}
			for _, s := range f[from:] {
				out = append(out, mustHash(t, s))
			}
			return out
		}
		switch f[0] {
		case "leaf_hash":
			ref.leafHashes[num(1)] = mustHash(t, f[2])
		case "root":
			ref.roots[num(1)] = mustHash(t, f[2])
		case "inclusion":
			ref.inclusion[[2]uint64{num(1), num(2)}] = hashes(3)
		case "consistency":
			ref.consistency[[2]uint64{num(1), num(2)}] = hashes(3)
		default:
			t.Fatalf("unknown vector line %q", sc.Text())
		}
	}
	return ref
}

func referenceTree() *tree {
	t := &tree{}
	for _, l := range referenceLeaves {
		t.append(LeafHash(l))
	}
	return t
}

func equalHashes(a, b []Hash) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestReference_LeafHashesAndRoots(t *testing.T) {
	ref := parseReference(t)
	for i, l := range referenceLeaves {
		if got := LeafHash(l); got != ref.leafHashes[i] {
			t.Errorf("leaf %d hash = %x, want %x", i, got, ref.leafHashes[i])
		}
	}
	if EmptyRoot() != ref.roots[0] {
		t.Errorf("empty root = %x, want %x", EmptyRoot(), ref.roots[0])
	}
	tr := &tree{}
	for n := 0; n <= 8; n++ {
		if n > 0 {
			tr.append(ref.leafHashes[n-1])
		}
		if got := tr.root(); got != ref.roots[n] {
			t.Errorf("incremental root at size %d = %x, want %x", n, got, ref.roots[n])
		}
		if got := RootFromLeafHashes(ref.leafHashes[:n]); got != ref.roots[n] {
			t.Errorf("recursive root at size %d = %x, want %x", n, got, ref.roots[n])
		}
	}
	// Roots of earlier sizes stay reachable from the full tree.
	for n := 0; n <= 8; n++ {
		if got := tr.rootAt(uint64(n)); got != ref.roots[n] {
			t.Errorf("rootAt(%d) = %x, want %x", n, got, ref.roots[n])
		}
	}
}

func TestReference_InclusionProofs(t *testing.T) {
	ref := parseReference(t)
	tr := referenceTree()
	for key, want := range ref.inclusion {
		index, size := key[0], key[1]
		got, err := tr.inclusionProof(index, size)
		if err != nil {
			t.Fatalf("inclusionProof(%d, %d): %v", index, size, err)
		}
		if !equalHashes(got, want) {
			t.Errorf("inclusionProof(%d, %d) = %x, want %x", index, size, got, want)
		}
		if !VerifyInclusion(ref.leafHashes[index], index, size, want, ref.roots[size]) {
			t.Errorf("VerifyInclusion(%d, %d) rejected the reference proof", index, size)
		}
	}
}

func TestReference_ConsistencyProofs(t *testing.T) {
	ref := parseReference(t)
	tr := referenceTree()
	for key, want := range ref.consistency {
		m, n := key[0], key[1]
		got, err := tr.consistencyProof(m, n)
		if err != nil {
			t.Fatalf("consistencyProof(%d, %d): %v", m, n, err)
		}
		if !equalHashes(got, want) {
			t.Errorf("consistencyProof(%d, %d) = %x, want %x", m, n, got, want)
		}
		if !VerifyConsistency(m, n, want, ref.roots[m], ref.roots[n]) {
			t.Errorf("VerifyConsistency(%d, %d) rejected the reference proof", m, n)
		}
	}
}

func randomLeafHashes(t testing.TB, n int) []Hash {
	t.Helper()
	out := make([]Hash, n)
	for i := range out {
		var buf [40]byte
		if _, err := rand.Read(buf[:]); err != nil {
			t.Fatal(err)
		}
		out[i] = LeafHash(buf[:])
	}
	return out
}

// Every size up to 130 (past 128 so the tree grows an eighth level):
// the incremental tree agrees with the recursive definition, every
// inclusion proof verifies, every consistency proof verifies, and the
// obvious tampering is caught.
func TestTree_AllSizesRoundTrip(t *testing.T) {
	const max = 130
	hashes := randomLeafHashes(t, max)
	tr := &tree{}
	for n := uint64(0); n <= max; n++ {
		if n > 0 {
			tr.append(hashes[n-1])
		}
		want := RootFromLeafHashes(hashes[:n])
		if got := tr.root(); got != want {
			t.Fatalf("size %d: incremental root %x != recursive %x", n, got, want)
		}
		for m := uint64(0); m <= n; m++ {
			if got := tr.rootAt(m); got != RootFromLeafHashes(hashes[:m]) {
				t.Fatalf("rootAt(%d) at size %d wrong", m, n)
			}
			proof, err := tr.consistencyProof(m, n)
			if err != nil {
				t.Fatalf("consistencyProof(%d, %d): %v", m, n, err)
			}
			if !VerifyConsistency(m, n, proof, tr.rootAt(m), want) {
				t.Fatalf("consistency (%d, %d) did not verify", m, n)
			}
			if m > 0 && m < n {
				bad := RootFromLeafHashes(hashes[1 : m+1])
				if VerifyConsistency(m, n, proof, bad, want) {
					t.Fatalf("consistency (%d, %d) accepted a wrong old root", m, n)
				}
			}
		}
		if n%7 != 0 && n != max {
			continue // inclusion is O(n log n) per size; sample sizes
		}
		for i := uint64(0); i < n; i++ {
			proof, err := tr.inclusionProof(i, n)
			if err != nil {
				t.Fatalf("inclusionProof(%d, %d): %v", i, n, err)
			}
			if !VerifyInclusion(hashes[i], i, n, proof, want) {
				t.Fatalf("inclusion (%d, %d) did not verify", i, n)
			}
			if n > 1 {
				other := (i + 1) % n
				if VerifyInclusion(hashes[other], i, n, proof, want) {
					t.Fatalf("inclusion (%d, %d) accepted another leaf", i, n)
				}
				if VerifyInclusion(hashes[i], other, n, proof, want) {
					t.Fatalf("inclusion (%d, %d) accepted a wrong index", i, n)
				}
				if VerifyInclusion(hashes[i], i, n, proof[:len(proof)-1], want) {
					t.Fatalf("inclusion (%d, %d) accepted a truncated proof", i, n)
				}
			}
		}
	}
}

func TestVerify_DegenerateCases(t *testing.T) {
	hashes := randomLeafHashes(t, 5)
	root5 := RootFromLeafHashes(hashes)
	root3 := RootFromLeafHashes(hashes[:3])

	// Any tree extends the empty tree, with an empty proof and the
	// empty root — and nothing else.
	if !VerifyConsistency(0, 5, nil, EmptyRoot(), root5) {
		t.Error("empty→5 with empty proof must verify")
	}
	if VerifyConsistency(0, 5, []Hash{hashes[0]}, EmptyRoot(), root5) {
		t.Error("empty→5 with a non-empty proof must fail")
	}
	if VerifyConsistency(0, 5, nil, root3, root5) {
		t.Error("empty→5 with a non-empty old root must fail")
	}
	// Same size: empty proof, equal roots.
	if !VerifyConsistency(5, 5, nil, root5, root5) {
		t.Error("5→5 must verify")
	}
	if VerifyConsistency(5, 5, nil, root3, root5) {
		t.Error("5→5 with different roots must fail")
	}
	if VerifyConsistency(5, 5, []Hash{hashes[0]}, root5, root5) {
		t.Error("5→5 with a proof must fail")
	}
	// Shrinking is never consistent; a missing proof never is.
	if VerifyConsistency(5, 3, nil, root5, root3) {
		t.Error("5→3 must fail")
	}
	if VerifyConsistency(3, 5, nil, root3, root5) {
		t.Error("3→5 with an empty proof must fail")
	}
	// Inclusion: index must be inside the tree; a single-leaf tree
	// has an empty proof; a bigger tree does not.
	if VerifyInclusion(hashes[0], 5, 5, nil, root5) {
		t.Error("index == size must fail")
	}
	if !VerifyInclusion(hashes[0], 0, 1, nil, hashes[0]) {
		t.Error("single-leaf inclusion with empty proof must verify")
	}
	if VerifyInclusion(hashes[0], 0, 5, nil, root5) {
		t.Error("empty proof against a 5-leaf root must fail")
	}
	// Out-of-range proof requests are refused, not computed.
	tr := &tree{}
	for _, h := range hashes {
		tr.append(h)
	}
	if _, err := tr.inclusionProof(5, 5); err != ErrOutOfRange {
		t.Errorf("inclusionProof(5,5) err = %v", err)
	}
	if _, err := tr.inclusionProof(0, 6); err != ErrOutOfRange {
		t.Errorf("inclusionProof(0,6) err = %v", err)
	}
	if _, err := tr.consistencyProof(4, 6); err != ErrOutOfRange {
		t.Errorf("consistencyProof(4,6) err = %v", err)
	}
	if _, err := tr.consistencyProof(4, 3); err != ErrOutOfRange {
		t.Errorf("consistencyProof(4,3) err = %v", err)
	}
}
