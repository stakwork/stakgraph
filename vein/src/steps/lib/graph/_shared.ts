import type { StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import type { GraphBackend } from "../../../graph/backend.js";

export type { GraphBackend };

/**
 * The `graph/*` steps are thin plumbing over vein's own Neo4j graph backend
 * (`src/graph/*`, plans/jarvis-graph-compat.md) — the vein-native twins of
 * the lab's `jarvis/*` steps: same step names, input schemas, and output
 * shapes, so a workflow swaps backends by step type
 * (`jarvis/graph-search` ↔ `graph/graph-search`). No jarvis in the loop;
 * everything written follows jarvis's conventions so a jarvis mounted on
 * the same database later treats the `Vein` domain as native.
 *
 * Config (all via `ctx.services.secrets`, secret store → env fallback):
 *   - `NEO4J_URI`             — bolt:// URI (required).
 *   - `NEO4J_USER` / `NEO4J_PASSWORD` — credentials (default neo4j / "").
 *   - `NEO4J_DATABASE`        — optional database name.
 *   - `VEIN_GRAPH_NAMESPACE`  — default jarvis namespace (default "default").
 *   - `VEIN_GRAPH_EMBEDDINGS` — "off" disables the local MiniLM embedder
 *     (writes leave vectors NULL; search is fulltext-only).
 *
 * The backend is opened once per config and cached process-wide; the first
 * open runs the boot obligations (domain seeding + embedding backfill).
 * Per the lib dependency convention the backend module (and with it
 * neo4j-driver) is imported lazily here, inside `run()`, never at module
 * top level.
 */
export async function graphCtx(ctx?: StepContext<VeinCapabilities>): Promise<GraphBackend> {
  const secrets = ctx?.services?.secrets;
  const uri = await secrets?.get("NEO4J_URI");
  if (!uri) throw new Error("graph: NEO4J_URI not configured (set it in the env or the vein secret store)");
  const emb = ((await secrets?.get("VEIN_GRAPH_EMBEDDINGS")) ?? "").toLowerCase();
  const { openGraphBackend } = await import("../../../graph/backend.js");
  return openGraphBackend(
    {
      uri,
      user: (await secrets?.get("NEO4J_USER")) ?? "neo4j",
      password: (await secrets?.get("NEO4J_PASSWORD")) ?? "",
      namespace: (await secrets?.get("VEIN_GRAPH_NAMESPACE")) || "default",
      database: (await secrets?.get("NEO4J_DATABASE")) || undefined,
    },
    { embeddings: !["off", "0", "false"].includes(emb) },
  );
}

/** The `code` of a `GraphValidationError` / `GraphReadError`, else undefined.
 *  Duck-typed on `name` so this module never imports the graph classes. */
export function graphErrorCode(e: unknown): string | undefined {
  const n = (e as { name?: unknown } | null)?.name;
  if (n === "GraphValidationError" || n === "GraphReadError") return String((e as { code: unknown }).code);
  return undefined;
}

/** Render a graph error as a plain string the agent can read (the same
 *  convention the jarvis/* steps use for HTTP failures). */
export function errText(step: string, e: unknown): string {
  const code = graphErrorCode(e);
  const msg = e instanceof Error ? e.message : String(e);
  return code ? `${step} failed — ${code}: ${msg}` : `${step} failed: ${msg}`;
}
