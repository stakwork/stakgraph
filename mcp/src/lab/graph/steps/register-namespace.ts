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
  type: "graph/register-namespace",
  description:
    "Register (create) a graph NAMESPACE — a named data partition that scopes node/edge writes " +
    "(pass the same name as the `namespace` config of the graph write steps). Idempotent: registering " +
    "a namespace that already exists is a success. Names are lowercased. Namespaces are not an access-control boundary.",
  input: z.object({
    namespace: z
      .string()
      .min(1)
      .describe("Namespace name to register, e.g. a task slug. Reuse the exact same string in later graph write steps."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const r = await b.reader.registerNamespace(cfg.namespace);
      return { namespace: r.namespace, registered: true, ...(r.created ? {} : { alreadyExisted: true }) };
    } catch (e) {
      return errText("graph/register-namespace", e);
    }
  },
});
