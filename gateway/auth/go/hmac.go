package macaroon

import (
	"crypto/hmac"
	"crypto/sha256"
)

// ComputeAttenuationHMAC implements the phase-4 HMAC chain step:
//
//	hmac_i = HMAC-SHA256(key = prev_sig_bytes, msg = utf8(JCS(caveats)))
//
// For the first attenuation, prevSigBytes is the raw bytes of the
// invocation's user_sig.sig. For later attenuations, it is the raw
// bytes of the previous attenuation's hmac.
func ComputeAttenuationHMAC(prevSigBytes []byte, caveats AttenuationCaveats) ([]byte, error) {
	msg, err := JCS(caveats)
	if err != nil {
		return nil, err
	}
	mac := hmac.New(sha256.New, prevSigBytes)
	mac.Write(msg)
	return mac.Sum(nil), nil
}
