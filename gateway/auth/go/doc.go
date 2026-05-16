// Package macaroon implements the pure verifier for the three-layer
// macaroon defined in gateway/plans/phases/phase-4-macaroon-shape.md.
//
// Pure means: no I/O, no Bifrost types, no ambient state. The public
// entry point is [Verify], which takes the macaroon (base64url), the
// trusting org's policy, and the current time, and returns either
// verified claims or an error with a machine-readable Code.
//
// Adapters (e.g. gateway/internal/auth/) wrap this package with
// header extraction, trust-registry lookup, Redis-backed revocation,
// and bifrost.Error shaping. Adapters hold no cryptographic knowledge.
//
// Cross-language byte-equivalence with the TypeScript sibling
// (gateway/auth/ts/) is enforced by gateway/auth/fixtures/. Both
// sides load the same fixtures and assert every intermediate value
// reproduces byte-for-byte.
package macaroon
