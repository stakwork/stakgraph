// Wrapper module — kept separate from the plugin (../) on purpose.
// The plugin needs github.com/maximhq/bifrost/core at a very specific
// pinned version (Go's plugin loader requires byte-identical deps with
// bifrost-http). The wrapper has zero such constraints and stays on
// stdlib only.
//
// Bumping Go here is safe and independent of the plugin's Go version
// requirement.
module github.com/stakwork/stakgraph/gateway/wrapper

go 1.24
