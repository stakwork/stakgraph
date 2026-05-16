package macaroon

import (
	"crypto/ed25519"
	"crypto/sha256"
	"fmt"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"
	"github.com/decred/dcrd/dcrec/secp256k1/v4/ecdsa"
)

// ─── Ed25519 ──────────────────────────────────────────────────────────

// Ed25519Sign signs msg with a 32-byte Ed25519 seed (the standard
// "private key" representation used in the TS sibling and in
// fixtures). Returns the 64-byte signature. Ed25519 is deterministic
// — same seed + same msg → same signature, byte-for-byte, across
// implementations.
func Ed25519Sign(seed, msg []byte) ([]byte, error) {
	if len(seed) != ed25519.SeedSize {
		return nil, fmt.Errorf("ed25519: seed must be %d bytes, got %d",
			ed25519.SeedSize, len(seed))
	}
	priv := ed25519.NewKeyFromSeed(seed)
	return ed25519.Sign(priv, msg), nil
}

// Ed25519Verify checks a 64-byte Ed25519 signature over msg against a
// 32-byte raw pubkey. Ed25519 signs the raw message (no prehash) per
// RFC 8032.
func Ed25519Verify(pubKey, msg, sig []byte) bool {
	if len(pubKey) != ed25519.PublicKeySize || len(sig) != ed25519.SignatureSize {
		return false
	}
	return ed25519.Verify(ed25519.PublicKey(pubKey), msg, sig)
}

// Ed25519PublicKey derives the 32-byte raw public key from a 32-byte
// seed. Useful for issuer key bootstrapping and tests.
func Ed25519PublicKey(seed []byte) ([]byte, error) {
	if len(seed) != ed25519.SeedSize {
		return nil, fmt.Errorf("ed25519: seed must be %d bytes, got %d",
			ed25519.SeedSize, len(seed))
	}
	priv := ed25519.NewKeyFromSeed(seed)
	pub := priv.Public().(ed25519.PublicKey)
	out := make([]byte, len(pub))
	copy(out, pub)
	return out, nil
}

// ─── ECDSA-secp256k1-SHA256 ───────────────────────────────────────────

// EcdsaSecp256k1Sign produces a 64-byte compact (r||s) ECDSA signature
// over secp256k1, SHA-256 prehashed, RFC 6979 deterministic, BIP 62
// low-s normalized. Cross-language byte-equivalence with TS sibling
// (which uses @noble/curves with the same defaults) depends on all
// three properties — don't substitute a randomized signer here.
//
// privKey is the 32-byte secp256k1 private scalar.
func EcdsaSecp256k1Sign(privKey, msg []byte) ([]byte, error) {
	if len(privKey) != 32 {
		return nil, fmt.Errorf("secp256k1: private key must be 32 bytes, got %d", len(privKey))
	}
	priv := secp256k1.PrivKeyFromBytes(privKey)
	digest := sha256.Sum256(msg)
	sig := ecdsa.Sign(priv, digest[:]) // RFC 6979 + low-s by construction
	r := sig.R()
	s := sig.S()
	out := make([]byte, 64)
	var rBuf, sBuf [32]byte
	r.PutBytes(&rBuf)
	s.PutBytes(&sBuf)
	copy(out[:32], rBuf[:])
	copy(out[32:], sBuf[:])
	return out, nil
}

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

// EcdsaSecp256k1PublicKey derives the 33-byte compressed secp256k1
// public key from a 32-byte private scalar. Useful for issuer key
// bootstrapping and tests.
func EcdsaSecp256k1PublicKey(privKey []byte) ([]byte, error) {
	if len(privKey) != 32 {
		return nil, fmt.Errorf("secp256k1: private key must be 32 bytes, got %d", len(privKey))
	}
	priv := secp256k1.PrivKeyFromBytes(privKey)
	return priv.PubKey().SerializeCompressed(), nil
}
