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
 * Config (all via `ctx.services.secrets`, secret store → env fallback; the
 * same names + defaults as the mcp host's own Neo4j client, so a local
 * Neo4j needs nothing and a deployment's existing vars just work):
 *   - `NEO4J_URI`             — bolt:// URI; else `bolt://<NEO4J_HOST>`
 *     (`NEO4J_HOST` default `localhost:7687`).
 *   - `NEO4J_USER` / `NEO4J_PASSWORD` — credentials (default neo4j / testtest).
 *   - `NEO4J_DATABASE`        — optional database name.
 *   - `VEIN_GRAPH_NAMESPACE`  — default jarvis namespace (default "default").
 *   - `VEIN_GRAPH_EMBEDDINGS` — "off" disables the local MiniLM embedder
 *     (writes leave vectors NULL; search is fulltext-only).
 *   - `VEIN_GRAPH_SEED_ONTOLOGY` — "1" also seeds the bundled jarvis
 *     ontology on first open (add-only), so a STANDALONE Neo4j can host
 *     jarvis-typed data (Document, EvalSet, …) with no jarvis process.
 *
 * The backend is opened once per config and cached process-wide; the first
 * open runs the boot obligations (domain seeding + embedding backfill).
 * Per the lib dependency convention the backend module (and with it
 * neo4j-driver) is imported lazily here, inside `run()`, never at module
 * top level.
 */
export async function graphCtx(ctx?: StepContext<VeinCapabilities>): Promise<GraphBackend> {
  const secrets = ctx?.services?.secrets;
  // Same resolution as the mcp host's own Neo4j client: `NEO4J_URI`, else
  // `bolt://<NEO4J_HOST>` (default localhost:7687), user/password defaulting
  // to neo4j/testtest. Nothing needs configuring for a local Neo4j, and a
  // deployment that already carries NEO4J_HOST/USER/PASSWORD is picked up
  // as-is (secret store → env, per the secrets capability).
  const uri = (await secrets?.get("NEO4J_URI")) || `bolt://${(await secrets?.get("NEO4J_HOST")) || "localhost:7687"}`;
  const emb = ((await secrets?.get("VEIN_GRAPH_EMBEDDINGS")) ?? "").toLowerCase();
  const ont = ((await secrets?.get("VEIN_GRAPH_SEED_ONTOLOGY")) ?? "").toLowerCase();
  const { openGraphBackend } = await import("../../../graph/backend.js");
  return openGraphBackend(
    {
      uri,
      user: (await secrets?.get("NEO4J_USER")) || "neo4j",
      password: (await secrets?.get("NEO4J_PASSWORD")) || "testtest",
      namespace: (await secrets?.get("VEIN_GRAPH_NAMESPACE")) || "default",
      database: (await secrets?.get("NEO4J_DATABASE")) || undefined,
    },
    { embeddings: !["off", "0", "false"].includes(emb), seedOntology: ["1", "true", "on"].includes(ont) },
  );
}

export interface EdgeWriteArgs {
  edge_type: string;
  source_ref_id: string;
  target_ref_id: string;
  edge_data?: Record<string, unknown>;
  weight?: number;
  /** jarvis `create_schema_if_missing`: when the (source type, edge, target
   *  type) triple has no edge schema, register one between the endpoint
   *  types and retry once. */
  create_schema_if_missing?: boolean;
}

/** Write one edge through the backend, honouring `create_schema_if_missing`
 *  the way jarvis's edge endpoint does. Throws the writer's error otherwise. */
export async function writeEdge(b: GraphBackend, a: EdgeWriteArgs) {
  const edge = a.edge_type.toUpperCase().replace(/ /g, "_");
  const input = {
    edge,
    source_ref_id: a.source_ref_id,
    target_ref_id: a.target_ref_id,
    ...(a.edge_data ? { properties: a.edge_data } : {}),
    ...(a.weight !== undefined ? { weight: a.weight } : {}),
  };
  try {
    return await b.edges.write(input);
  } catch (e) {
    if (!a.create_schema_if_missing || graphErrorCode(e) !== "WRONG_TYPE") throw e;
    const [s, t] = await Promise.all([b.reader.getNode(a.source_ref_id), b.reader.getNode(a.target_ref_id)]);
    if (!s?.node_type || !t?.node_type) throw e;
    await b.schemas.createEdgeSchema(s.node_type, edge, t.node_type);
    return await b.edges.write(input);
  }
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
