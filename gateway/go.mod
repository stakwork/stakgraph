module github.com/stakwork/stakgraph/gateway

go 1.26.2

// Version is overridden in Docker builds via `go mod edit -replace` to
// point at the local bifrost checkout — this guarantees the plugin and
// bifrost-http compile against byte-identical core sources, which Go's
// plugin loader requires.
require github.com/maximhq/bifrost/core v1.5.10

require (
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.1
	github.com/gowebpki/jcs v1.0.1
	// go-redis MUST match the version bifrost-http itself depends
	// on transitively (see transports/go.mod in the bifrost release
	// matching BIFROST_VERSION). Go's plugin loader rejects with
	// "plugin was built with a different version of package …" if
	// any imported package differs from the host binary's build.
	// Bump in lockstep with BIFROST_VERSION in the Dockerfile.
	github.com/redis/go-redis/v9 v9.17.2
)

require (
	github.com/andybalholm/brotli v1.2.0 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/bytedance/gopkg v0.1.3 // indirect
	github.com/bytedance/sonic v1.15.0 // indirect
	github.com/bytedance/sonic/loader v0.5.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/cloudwego/base64x v0.1.6 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/invopop/jsonschema v0.13.0 // indirect
	github.com/klauspost/compress v1.18.2 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/mailru/easyjson v0.9.1 // indirect
	github.com/mark3labs/mcp-go v0.43.2 // indirect
	github.com/spf13/cast v1.10.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.0 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/twitchyliquid64/golang-asm v0.15.1 // indirect
	github.com/valyala/bytebufferpool v1.0.0 // indirect
	github.com/valyala/fasthttp v1.68.0 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/arch v0.23.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
