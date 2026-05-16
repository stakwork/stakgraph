// Package macaroon implements both signer and verifier for the
// three-layer macaroon defined in gateway/plans/phases/phase-4-macaroon-shape.md.
//
// Pure means: no I/O, no Bifrost types, no ambient state. Two public
// entry-point surfaces:
//
//   - Verify (verify.go) takes the macaroon (base64url), the trusting
//     org's policy, and the current time, and returns either verified
//     claims or an error with a machine-readable Code.
//   - SignUserAuthorizationSingle / SignUserAuthorizationMultisig /
//     SignInvocation (sign.go), Attenuate (attenuate.go), and
//     EncodeMacaroon (encode.go) build a wire-format macaroon from
//     unsigned layer objects and private keys. These are what a Go
//     issuer (e.g. a standalone reference issuer, or a future Go
//     migration of Hive's /macaroons/issue handler) imports.
//
// Adapters (e.g. gateway/internal/auth/) wrap the verifier with header
// extraction, trust-registry lookup, Redis-backed revocation, and
// bifrost.Error shaping. Adapters hold no cryptographic knowledge of
// their own.
//
// Cross-language byte-equivalence with the TypeScript sibling
// (gateway/auth/ts/) is enforced by gateway/auth/fixtures/. Both
// sides load the same fixtures and assert every intermediate value
// reproduces byte-for-byte — including Go-side re-signing, which is
// well-defined because Ed25519 is deterministic per RFC 8032 and
// ECDSA-secp256k1 is deterministic per RFC 6979 with BIP 62 low-s
// normalization in both implementations.
package macaroon
