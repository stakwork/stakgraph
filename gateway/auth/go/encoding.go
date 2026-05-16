package macaroon

import (
	"encoding/base64"
	"encoding/hex"
	"fmt"
)

// Phase 4 wire conventions: lowercase hex WITHOUT 0x prefix, base64url
// WITHOUT padding (RFC 4648 §5). All helpers in this file enforce that.

// BytesToHex encodes bytes as lowercase hex with no prefix.
func BytesToHex(b []byte) string {
	return hex.EncodeToString(b)
}

// HexToBytes decodes a lowercase-or-uppercase hex string with no prefix.
func HexToBytes(s string) ([]byte, error) {
	out, err := hex.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("hex decode: %w", err)
	}
	return out, nil
}

// BytesToBase64url encodes bytes as base64url with no padding.
func BytesToBase64url(b []byte) string {
	return base64.RawURLEncoding.EncodeToString(b)
}

// Base64urlToBytes decodes base64url; tolerates padding because some
// callers (curl, custom clients) emit it even though we don't.
func Base64urlToBytes(s string) ([]byte, error) {
	// Try unpadded first (the canonical form).
	if out, err := base64.RawURLEncoding.DecodeString(s); err == nil {
		return out, nil
	}
	// Tolerate padded base64url for input lenience.
	out, err := base64.URLEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("base64url decode: %w", err)
	}
	return out, nil
}
