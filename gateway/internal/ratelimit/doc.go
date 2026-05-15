// Package ratelimit will hold the gateway's per-request limit logic.
//
// Status
// ------
// Stub. Reserved import path so the first rate-limit PR lands here
// (and the design conversation in plans/llm-governance-v2.md doesn't
// have to also bikeshed the package layout).
//
// Planned scope
// -------------
//   - Lookup keys: (workspace, user), (workspace, agent-name),
//     (run-id), (session-id, agent-name). Each can have an independent
//     limit; the request is rejected if ANY exceeds.
//   - Sliding-window counters in Redis (with an in-process LRU
//     fallback for dev/standalone).
//   - Pre-check in hooks.TransportPre (before provider call), commit
//     in hooks.LLMPost (after we know token counts).
//   - Read-through cache for the limit config — Hive owns
//     authoritative state in its DB and pushes via
//     /_plugin/limits/refresh.
//
// Until the first feature lands this package exports nothing.
package ratelimit
