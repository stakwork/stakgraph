// Package hooks contains the bodies of every Bifrost plugin entry
// point. main.go in the parent package is just thin re-exports that
// call into here.
//
// Why bodies live in a non-main package
// -------------------------------------
// Bifrost's plugin loader (framework/plugins/soloader.go) only looks
// up exported symbols on the main package of the .so. Those exports
// MUST live in main, but they can be (and are) one-liners that
// delegate into here.
//
// Splitting this out means:
//   - main.go is auditable at a glance — you can see every exported
//     symbol the plugin contributes to bifrost-http in ~30 lines.
//   - Each hook's body has its own file and can grow independently
//     without dragging unrelated bodies into the diff.
//   - The hook bodies can import other internal/ packages (auth,
//     ratelimit, …) without main.go's import block becoming a wall.
//
// Each hook is a single exported function named after the entry
// point but without the leading "HTTPTransport" / "LLM" mouthful:
//
//	HTTPTransportPreHook       -> hooks.TransportPre
//	HTTPTransportPostHook      -> hooks.TransportPost
//	HTTPTransportStreamChunkHook -> hooks.StreamChunk
//	PreLLMHook                 -> hooks.LLMPre
//	PostLLMHook                -> hooks.LLMPost
package hooks
