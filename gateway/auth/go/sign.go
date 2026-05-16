package macaroon

// Layer-level signers — Go mirror of gateway/auth/ts/src/sign.ts.
//
// Phase 1 custodial mode: the issuer (Hive today, or a standalone
// reference issuer in Go) holds both org and user private keys and
// produces both layer signatures. Phase 2+ moves the user sign step
// to the user's device; the API stays the same — only which key the
// caller hands in changes.
//
// Cross-language byte-equivalence with TS: both Ed25519 (RFC 8032)
// and ECDSA-secp256k1-SHA256 (RFC 6979 + BIP 62 low-s) are
// deterministic in the implementations we use here, so a fixture
// signed in TS and re-signed in Go produces identical bytes. The
// fixture suite asserts this.

import (
	"fmt"
)

// MultisigSigner is one participating signer in a multisig org_sig.
// KeyIndex must match the position of the signer's pubkey in the
// trust-registry policy's Keys array; the verifier rejects out-of-
// range or mismatched-alg entries.
type MultisigSigner struct {
	KeyIndex int
	PrivKey  []byte // 32-byte secp256k1 private scalar
}

// SignUserAuthorizationSingle signs a user_authorization envelope with
// a single secp256k1 key. The caller passes a UserAuthorization with
// OrgSig at its zero value; this function returns a new copy with
// OrgSig populated. Custodial phase 1 uses this; phase 3+ uses
// SignUserAuthorizationMultisig once the org adopts an m-of-n policy.
func SignUserAuthorizationSingle(unsigned UserAuthorization, orgPrivKey []byte) (UserAuthorization, error) {
	msg, err := JCSStripField(unsigned, "org_sig")
	if err != nil {
		return UserAuthorization{}, fmt.Errorf("sign ua: jcs strip: %w", err)
	}
	sigBytes, err := EcdsaSecp256k1Sign(orgPrivKey, msg)
	if err != nil {
		return UserAuthorization{}, fmt.Errorf("sign ua: ecdsa: %w", err)
	}
	out := unsigned
	out.OrgSig = Sig{
		Alg: AlgEcdsaSecp256k1Sha256,
		Sig: BytesToHex(sigBytes),
	}
	return out, nil
}

// SignUserAuthorizationMultisig signs a user_authorization envelope
// under an m-of-n multisig policy. Each participating signer is
// identified by its KeyIndex into the policy's Keys array. Signers
// are sorted by KeyIndex ascending so the assembled multisig-v1
// envelope is deterministic.
//
// Threshold enforcement happens at verify time — this function will
// happily produce an under-threshold envelope, which the verifier
// will then reject. That's by design: the signer's job is to produce
// bytes, not to gate on policy.
func SignUserAuthorizationMultisig(unsigned UserAuthorization, signers []MultisigSigner) (UserAuthorization, error) {
	msg, err := JCSStripField(unsigned, "org_sig")
	if err != nil {
		return UserAuthorization{}, fmt.Errorf("sign ua multisig: jcs strip: %w", err)
	}
	// Sort by KeyIndex ascending without mutating caller's slice.
	sorted := make([]MultisigSigner, len(signers))
	copy(sorted, signers)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1].KeyIndex > sorted[j].KeyIndex; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}
	subSigs := make([]SubSig, 0, len(sorted))
	for _, s := range sorted {
		sigBytes, err := EcdsaSecp256k1Sign(s.PrivKey, msg)
		if err != nil {
			return UserAuthorization{}, fmt.Errorf("sign ua multisig: signer %d: %w", s.KeyIndex, err)
		}
		subSigs = append(subSigs, SubSig{
			KeyIndex: s.KeyIndex,
			Alg:      AlgEcdsaSecp256k1Sha256,
			Sig:      BytesToHex(sigBytes),
		})
	}
	out := unsigned
	out.OrgSig = Sig{
		Alg:  AlgMultisigV1,
		Sigs: subSigs,
	}
	return out, nil
}

// SignInvocation signs an invocation envelope with the user's Ed25519
// private key (passed as a 32-byte seed). Returns a new Invocation
// with UserSig populated.
func SignInvocation(unsigned Invocation, userPrivKey []byte) (Invocation, error) {
	msg, err := JCSStripField(unsigned, "user_sig")
	if err != nil {
		return Invocation{}, fmt.Errorf("sign invocation: jcs strip: %w", err)
	}
	sigBytes, err := Ed25519Sign(userPrivKey, msg)
	if err != nil {
		return Invocation{}, fmt.Errorf("sign invocation: ed25519: %w", err)
	}
	out := unsigned
	out.UserSig = Sig{
		Alg: AlgEd25519,
		Sig: BytesToHex(sigBytes),
	}
	return out, nil
}

// InvocationSigBytes returns the raw bytes of an invocation's
// user_sig.sig, for use as the HMAC key of the first attenuation in
// the chain.
func InvocationSigBytes(inv Invocation) ([]byte, error) {
	return HexToBytes(inv.UserSig.Sig)
}
