package macaroon

// Sub-agent attenuation — Go mirror of gateway/auth/ts/src/attenuate.ts.
//
// Appends one HMAC link to the chain. The HMAC primitive itself lives
// in hmac.go (already shared with the verifier); this file wraps it
// with the "produce a complete Attenuation struct" helper and the
// two prev-sig-bytes accessors that issuers and parent agents need
// when building chains.
//
// Polyglot note: any language with JCS + HMAC-SHA256 + hex can
// reproduce these functions in ~20 lines. They're the only crypto
// a sub-agent needs.

// Attenuate builds one chain link: computes the HMAC over the
// canonicalized caveats keyed by the previous link's signature bytes,
// and returns the assembled Attenuation struct.
//
// prevSigBytes is:
//   - For the first attenuation: the raw bytes of
//     invocation.user_sig.sig (use InvocationSigBytes).
//   - For later attenuations: the raw bytes of the previous
//     attenuation's hmac (use AttenuationSigBytes).
func Attenuate(prevSigBytes []byte, caveats AttenuationCaveats) (Attenuation, error) {
	hmacBytes, err := ComputeAttenuationHMAC(prevSigBytes, caveats)
	if err != nil {
		return Attenuation{}, err
	}
	return Attenuation{
		Caveats: caveats,
		HMAC:    BytesToHex(hmacBytes),
	}, nil
}

// AttenuationSigBytes returns the raw bytes of an attenuation's HMAC,
// for use as the prev-sig-bytes of the next attenuation in the chain.
func AttenuationSigBytes(att Attenuation) ([]byte, error) {
	return HexToBytes(att.HMAC)
}
