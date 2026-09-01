import { z, defineStep, openGraphBackend, GraphValidationError, GraphReadError, type StepContext, type VeinCapabilities, type GraphBackend } from "vein";

/** Resolve the Neo4j connection via the secrets capability (secret store →
 *  env fallback) and open — or reuse — the shared vein graph backend.
 *  Duplicated in every graph/* step — see _shared.ts. */
async function graphCtx(ctx?: StepContext<VeinCapabilities>): Promise<GraphBackend> {
  const secrets = ctx?.services?.secrets;
  const uri = await secrets?.get("NEO4J_URI");
  if (!uri) throw new Error("graph: NEO4J_URI not configured (set it in the env or the vein secret store)");
  const emb = ((await secrets?.get("VEIN_GRAPH_EMBEDDINGS")) ?? "").toLowerCase();
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

/** Render a graph error the way the jarvis/* steps render HTTP failures. */
function errText(step: string, e: unknown): string {
  if (e instanceof GraphValidationError || e instanceof GraphReadError) return `${step} failed — ${e.code}: ${e.message}`;
  return `${step} failed: ${e instanceof Error ? e.message : String(e)}`;
}

export default defineStep({
  type: "graph/create-node",
  description:
    "Create (or merge) a SINGLE node in the vein knowledge graph as DATA, with no edge " +
    "(to assert a relationship at the same time, use graph_create_triplet instead). Writes live to the graph. " +
    "REUSE existing nodes: graph_graph_search first, and only create when the entity genuinely doesn't exist yet — " +
    "duplicate nodes fragment the graph. The node type must be one of the Vein types in the ontology " +
    "(check with graph_get_ontology; graph_get_ontology_type shows its attributes and which are required). " +
    "Every attribute is validated against the type's schema — unknown attributes are rejected. " +
    "Create-or-merge semantics: if a node with the same identity key already exists, its ref_id is " +
    "returned (reported as a Warning) instead of creating a duplicate — so re-running is safe.",
  input: z.object({
    node_type: z
      .string()
      .describe("Node type (a Vein type from graph_get_ontology, e.g. 'VeinWorkflow'). Never pass the wildcard sentinel \"*\"."),
    node_data: z
      .record(z.string(), z.any())
      .describe('Properties for the node, e.g. {"name": "harvey-deliver"}. Must satisfy the type\'s schema, including its node_key attribute(s).'),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) to create the node in — must be registered. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const namespace = await b.reader.resolveNamespace(cfg.namespace);
      const r = await b.nodes.write({ type: cfg.node_type, data: cfg.node_data }, "create", { namespace });
      const existed = r.outcome === "existing";
      return {
        // "Warning" here means the node already existed (idempotent merge).
        status: existed ? "Warning" : "Success",
        ref_id: r.ref_id,
        node_type: cfg.node_type,
        ...(existed ? { messages: [`Node already exists in the graph with node_key: ${r.node_key}`] } : {}),
        ...(r.outcome === "restored" ? { messages: ["Node restored"] } : {}),
      };
    } catch (e) {
      return errText("graph/create-node", e);
    }
  },
});
