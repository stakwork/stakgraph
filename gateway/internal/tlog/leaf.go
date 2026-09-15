package tlog

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// LeafVersion is the current leaf schema version (the "v" field).
const LeafVersion = 1

// Leaf status values.
const (
	StatusOK    = "ok"
	StatusError = "error"
)

// Leaf is one accounted LLM call. Every field is always present on
// the wire and nulls are explicit — no omitempty — so the same object
// canonicalizes to the same bytes in Go and TypeScript.
//
// Identity fields (OrgID, UserID, RunID, Agent) come from the verified
// macaroon claims, never from caller-supplied dimension headers.
// MacaroonSHA256 is over the raw x-macaroon header string and binds
// the leaf to the exact authorization chain. RequestSHA256 is over the
// raw request body bytes as received at the transport layer and is
// what makes the log a transcript commitment rather than a cost
// ledger; it is nil only when bifrost skipped the body copy.
// ResponseSHA256 and AgentRequestSig are reserved nil in v1.
type Leaf struct {
	V                int     `json:"v"`
	LeafID           string  `json:"leaf_id"`
	TS               string  `json:"ts"`
	OrgID            string  `json:"org_id"`
	UserID           string  `json:"user_id"`
	RunID            string  `json:"run_id"`
	Agent            string  `json:"agent"`
	MacaroonSHA256   string  `json:"macaroon_sha256"`
	RequestSHA256    *string `json:"request_sha256"`
	ResponseSHA256   *string `json:"response_sha256"`
	Model            string  `json:"model"`
	Provider         string  `json:"provider"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	CostUSD          float64 `json:"cost_usd"`
	Status           string  `json:"status"`
	AgentRequestSig  *string `json:"agent_request_sig"`
}

// Canonical returns the RFC 8785 (JCS) bytes of the leaf. These are
// the bytes that are hashed into the tree and written to disk, and
// the bytes a witness re-hashes.
func (l Leaf) Canonical() ([]byte, error) {
	return macaroon.JCS(l)
}

// Hash is the leaf's Merkle leaf hash: SHA256(0x00 || Canonical()).
func (l Leaf) Hash() (Hash, error) {
	raw, err := l.Canonical()
	if err != nil {
		return Hash{}, err
	}
	return LeafHash(raw), nil
}

// NewLeafID returns a fresh 128-bit hex leaf identifier. It is the
// handle receipts and the dashboard refer to and carries no meaning
// beyond identity.
func NewLeafID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("tlog: leaf id: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}
