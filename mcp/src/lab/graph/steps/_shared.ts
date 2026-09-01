/**
 * NOT a seeded step — the canonical copy of the per-step preamble for the
 * graph/* steps, the vein-native twins of the jarvis/* steps.
 *
 * Same input/output shapes as the jarvis/* steps, but backed by vein's own
 * Neo4j-over-bolt graph layer (`vein/src/graph/*`) instead of the Jarvis
 * HTTP API, so workflows swap by step type (`jarvis/graph-search` ↔
 * `graph/graph-search`). What jarvis's API does beyond what these steps
 * touch is deliberately out of scope (plans/jarvis-graph-compat.md §7).
 *
 * Seeded steps must be SELF-CONTAINED (value-imports from "vein" only), so
 * each step file inlines its own copy of the small `graphCtx` helper below.
 * If you change the contract (secret names, backend options), update every
 * step in this directory — they are deliberately duplicated, not shared.
 *
 * Contract (all via `ctx.services.secrets`, secret store → env fallback):
 *   - `NEO4J_URI`             — bolt:// URI (required).
 *   - `NEO4J_USER` / `NEO4J_PASSWORD` — credentials (default neo4j / "").
 *   - `NEO4J_DATABASE`        — optional database name.
 *   - `VEIN_GRAPH_NAMESPACE`  — default jarvis namespace (default "default").
 *   - `VEIN_GRAPH_EMBEDDINGS` — "off" disables the local MiniLM embedder
 *     (writes leave vectors NULL; search is fulltext-only).
 *
 * The backend is opened once per config and cached process-wide; the first
 * open runs the boot obligations (domain seeding + embedding backfill).
 */
export {};
