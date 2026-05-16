package macaroon

import (
	"crypto/ed25519"
	"crypto/sha256"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
	"github.com/decred/dcrd/dcrec/secp256k1/v4/ecdsa"
)

// ─── Ed25519 ──────────────────────────────────────────────────────────

// Ed25519Verify checks a 64-byte Ed25519 signature over msg against a
// 32-byte raw pubkey. Ed25519 signs the raw message (no prehash) per
// RFC 8032.
func Ed25519Verify(pubKey, msg, sig []byte) bool {
	if len(pubKey) != ed25519.PublicKeySize || len(sig) != ed25519.SignatureSize {
		return false
	}
	return ed25519.Verify(ed25519.PublicKey(pubKey), msg, sig)
}

// ─── ECDSA-secp256k1-SHA256 ───────────────────────────────────────────

// EcdsaSecp256k1Verify checks a 64-byte compact (r||s) ECDSA signature
// over secp256k1 with SHA-256 prehash. The signature MUST be low-s
// (BIP 62) — high-s signatures are rejected to keep "same logical
// signature → same bytes" enforceable.
//
// pubKey is the 33-byte compressed secp256k1 public key.
func EcdsaSecp256k1Verify(pubKey, msg, sig []byte) bool {
	if len(sig) != 64 {
		return false
	}
	pk, err := secp256k1.ParsePubKey(pubKey)
	if err != nil {
		return false
	}
	r := new(secp256k1.ModNScalar)
	if overflow := r.SetByteSlice(sig[:32]); overflow {
		return false
	}
	s := new(secp256k1.ModNScalar)
	if overflow := s.SetByteSlice(sig[32:]); overflow {
		return false
	}
	if r.IsZero() || s.IsZero() {
		return false
	}
	// Enforce low-s: reject signatures where s > n/2.
	if s.IsOverHalfOrder() {
		return false
	}
	digest := sha256.Sum256(msg)
	parsed := ecdsa.NewSignature(r, s)
	return parsed.Verify(digest[:], pk)
}
