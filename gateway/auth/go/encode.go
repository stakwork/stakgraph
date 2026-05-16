package macaroon

import "fmt"

// EncodeMacaroon serializes an assembled macaroon to the on-wire form
// carried by the x-macaroon HTTP header: base64url(JCS(macaroon)),
// unpadded.
//
// The macaroon must already be fully signed (UserAuthorization.OrgSig,
// Invocation.UserSig, and every attenuation's HMAC populated).
// EncodeMacaroon does no signing of its own; it just canonicalizes
// and encodes.
//
// Companion to Verify, which takes the same b64url string and
// reverses this transform.
func EncodeMacaroon(m Macaroon) (string, error) {
	canonical, err := JCS(m)
	if err != nil {
		return "", fmt.Errorf("encode macaroon: jcs: %w", err)
	}
	return BytesToBase64url(canonical), nil
}
