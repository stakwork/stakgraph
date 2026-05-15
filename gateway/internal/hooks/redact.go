package hooks

import "fmt"

// redact returns a short fingerprint of a secret-ish string. We use it
// to log header values like x-macaroon without dumping the whole
// credential to stderr.
//
// Examples:
//
//	redact("")                → "<unset>"
//	redact("hi")              → "set(hi…)"
//	redact("abcdef1234567890") → "set(abcd…7890,len=16)"
//
// The format is deliberately greppable — "<unset>" vs "set(" — so log
// scrapers can match either form without a regex over potentially
// secret bytes.
func redact(s string) string {
	if s == "" {
		return "<unset>"
	}
	if len(s) <= 12 {
		n := len(s)
		if n > 4 {
			n = 4
		}
		return "set(" + s[:n] + "…)"
	}
	return fmt.Sprintf("set(%s…%s,len=%d)", s[:4], s[len(s)-4:], len(s))
}
