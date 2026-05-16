package macaroon

import (
	"encoding/json"
	"fmt"

	"github.com/gowebpki/jcs"
)

// JCS canonicalizes a Go value to RFC 8785 canonical JSON.
//
// The value is first marshalled with encoding/json (standard library)
// then transformed by github.com/gowebpki/jcs to RFC 8785 form. The
// result is the canonical UTF-8 JSON byte sequence — feed it directly
// to signers / HMAC primitives, no further encoding.
func JCS(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("json.Marshal: %w", err)
	}
	out, err := jcs.Transform(raw)
	if err != nil {
		return nil, fmt.Errorf("jcs.Transform: %w", err)
	}
	return out, nil
}

// JCSStripField marshals v to JSON, removes one top-level field
// (used to strip OrgSig / UserSig before signing), and returns the
// JCS-canonical form of the rest.
//
// This is the "signing input" recipe from phase 4:
//
//	sign_bytes(layer, sig_field_name) = utf8(JCS(layer without sig_field_name))
func JCSStripField(v any, fieldName string) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("json.Marshal: %w", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("json.Unmarshal to map: %w", err)
	}
	delete(m, fieldName)
	stripped, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("json.Marshal stripped: %w", err)
	}
	out, err := jcs.Transform(stripped)
	if err != nil {
		return nil, fmt.Errorf("jcs.Transform: %w", err)
	}
	return out, nil
}
