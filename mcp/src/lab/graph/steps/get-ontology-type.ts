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
  type: "graph/get-ontology-type",
  description:
    "Fetch the attribute schema for a SINGLE ontology node type. Returns exactly " +
    "one field — `attributes` — and nothing else; for a type's domain, parent or " +
    "description, use graph_get_ontology. " +
    "Each attribute value is a type string (e.g. 'string', 'int'); a `?` prefix " +
    "(e.g. '?string') means the attribute is OPTIONAL, no prefix means REQUIRED. " +
    "`attributes` is complete: it already includes everything inherited from parent types. " +
    "Lookup is case-insensitive. NODE types only — edge type names (e.g. 'IN_RUN') are not " +
    "schema nodes. Call graph_get_ontology first if you don't already know the exact type name.",
  input: z.object({
    type: z.string().describe("The node type name, e.g. 'VeinRun' (case-insensitive)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const schema = await b.reader.getSchema(cfg.type);
      if (!schema) return `graph/get-ontology-type: unknown type ${cfg.type}`;
      return { attributes: schema.attributes };
    } catch (e) {
      return errText("graph/get-ontology-type", e);
    }
  },
});
