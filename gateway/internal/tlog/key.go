package tlog

import (
	"fmt"

	"github.com/decred/dcrd/dcrec/secp256k1/v4"

	macaroon "github.com/stakwork/stakgraph/gateway/auth/go"
)

// logKey is the per-boot signing key. secp256k1, generated with
// crypto/rand at Init and held in memory only. It is generated
// outside the plugin config block on purpose: pluginlog.Init dumps
// the whole config to stderr at boot, and a key in there would land
// in `docker logs`. Nothing ever logs the private scalar; wire shapes
// carry only the compressed-hex public key as log_pubkey.
//
// It is not an identity key. Compromising it forges nothing that the
// witness's countersignature does not already have to corroborate,
// and a restart (fresh key) is invisible to the chain because the
// witness verifies heads by recomputing the root from leaves it holds.
type logKey struct {
	priv   []byte // 32-byte scalar
	pubHex string // 33-byte compressed pubkey, lowercase hex
}

func newLogKey() (*logKey, error) {
	k, err := secp256k1.GeneratePrivateKey()
	if err != nil {
		return nil, fmt.Errorf("tlog: generate log key: %w", err)
	}
	return logKeyFromBytes(k.Serialize())
}

// logKeyFromBytes wraps a caller-supplied 32-byte scalar. Used by the
// fixture generator and tests so signatures are reproducible; the
// production path always goes through newLogKey.
func logKeyFromBytes(priv []byte) (*logKey, error) {
	pub, err := macaroon.EcdsaSecp256k1PublicKey(priv)
	if err != nil {
		return nil, fmt.Errorf("tlog: log key: %w", err)
	}
	return &logKey{priv: append([]byte(nil), priv...), pubHex: macaroon.BytesToHex(pub)}, nil
}

// sign produces the 64-byte compact ECDSA signature over SHA256(msg),
// RFC 6979 deterministic and low-s — byte-identical to what gatekey
// produces and verifies, via the shared auth/go helper.
func (k *logKey) sign(msg []byte) ([]byte, error) {
	return macaroon.EcdsaSecp256k1Sign(k.priv, msg)
}
