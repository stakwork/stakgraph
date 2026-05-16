module github.com/stakwork/stakgraph/gateway

go 1.26.2

// Version is overridden in Docker builds via `go mod edit -replace` to
// point at the local bifrost checkout — this guarantees the plugin and
// bifrost-http compile against byte-identical core sources, which Go's
// plugin loader requires.
require github.com/maximhq/bifrost/core v1.5.10

require (
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.1 // indirect
	github.com/gowebpki/jcs v1.0.1 // indirect
)
