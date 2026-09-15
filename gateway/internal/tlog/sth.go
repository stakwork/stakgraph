package tlog

import (
	"fmt"
	"time"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// STH is a signed tree head. Sig is the log key's ECDSA signature over
// SHA256(JCS(sth \ sig)), the same construction as the macaroon
// layers, so gatekey verifies it with the primitives it already has.
//
// The empty tree is a valid, signed head: tree_size 0 and the RFC 9162
// empty root.
type STH struct {
	TreeSize uint64 `json:"tree_size"`
	RootHash string `json:"root_hash"`
	SignedAt string `json:"signed_at"`
	Sig      string `json:"sig"`
}

// SigningBytes returns the exact bytes the log key signs for sth:
// JCS of the head with the sig field removed.
func SigningBytes(sth STH) ([]byte, error) {
	return macaroon.JCSStripField(sth, "sig")
}

// VerifySTH checks sth.Sig against a compressed-hex log pubkey. This
// is the Go twin of gatekey's verifySth, for tests and any Go witness.
func VerifySTH(sth STH, logPubkeyHex string) bool {
	msg, err := SigningBytes(sth)
	if err != nil {
		return false
	}
	pub, err := macaroon.HexToBytes(logPubkeyHex)
	if err != nil {
		return false
	}
	sig, err := macaroon.HexToBytes(sth.Sig)
	if err != nil {
		return false
	}
	return macaroon.EcdsaSecp256k1Verify(pub, msg, sig)
}

func signSTH(key *logKey, size uint64, root Hash, at time.Time) (STH, error) {
	sth := STH{
		TreeSize: size,
		RootHash: macaroon.BytesToHex(root[:]),
		SignedAt: formatTime(at),
	}
	msg, err := SigningBytes(sth)
	if err != nil {
		return STH{}, fmt.Errorf("tlog: sth signing bytes: %w", err)
	}
	sig, err := key.sign(msg)
	if err != nil {
		return STH{}, fmt.Errorf("tlog: sign sth: %w", err)
	}
	sth.Sig = macaroon.BytesToHex(sig)
	return sth, nil
}

// formatTime renders an RFC 3339 UTC timestamp with millisecond
// precision, the format used for leaf.ts and sth.signed_at.
func formatTime(t time.Time) string {
	return t.UTC().Format("2006-01-02T15:04:05.000Z07:00")
}
