import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/** Resolve the Jarvis base URL + auth via the secrets capability (secret
 *  store → env fallback). Duplicated in every jarvis/* step — see _shared.ts. */
async function jarvisCtx(ctx?: StepContext<VeinCapabilities>) {
  const http = ctx?.services?.http;
  if (!http) throw new Error("jarvis: ctx.services.http unavailable — run with a services bag");
  const secrets = ctx?.services?.secrets;
  const base = (await secrets?.get("JARVIS_URL"))?.replace(/\/+$/, "");
  if (!base) throw new Error("jarvis: JARVIS_URL not configured (set it in the mcp env or the vein secret store)");
  const token = (await secrets?.get("API_TOKEN")) ?? "";
  const rawTimeout = Number(await secrets?.get("JARVIS_HTTP_TIMEOUT_MS"));
  const timeout = Number.isFinite(rawTimeout) && rawTimeout > 0 ? rawTimeout : 180_000;
  return { base, http, timeout, headers: { "X-Api-Token": token } };
}

export default defineStep({
  type: "jarvis/graph-search",
  description:
    "Search the Jarvis knowledge graph for ontology nodes — people, topics, concepts, organizations, workflows, and more. " +
    "Provide at least one of `q`, `input_q`, `output_q` — they can be combined, each acting as its own " +
    "retriever fused into one ranked result set. " +
    "Each result includes an `edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
    "which relationship types you can traverse next with jarvis_graph_neighbors. " +
    "Call jarvis_get_ontology first to discover valid values for the `type` parameter. " +
    "An EMPTY `edges` map is NOT proof the node has no relationships (edge-count computation can be unavailable on some deployments) — confirm with jarvis_graph_neighbors before concluding a node is unconnected. ",
  input: z.object({
    q: z
      .string()
      .optional()
      .describe("General hybrid (keyword + semantic) search query over node names, descriptions, bodies, and schemas."),
    input_q: z
      .string()
      .optional()
      .describe(
        "Semantic search scoped to node INPUT schemas — find nodes by what they take as input, " +
        "e.g. 'a video file url'. Applies to node types with input embeddings (Workflow, Skill).",
      ),
    output_q: z
      .string()
      .optional()
      .describe(
        "Semantic search scoped to node OUTPUT schemas — find nodes by what they produce, " +
        "e.g. 'transcript with word-level timestamps'. Applies to node types with output embeddings (Workflow, Skill).",
      ),
    type: z
      .string()
      .optional()
      .describe("Comma-separated node type filter, e.g. 'Concept' or 'Person,Topic'. Call jarvis_get_ontology to see all valid values."),
    limit: z.number().optional().default(10).describe("Maximum number of results to return"),
    domains: z
      .string()
      .optional()
      .describe("Comma-separated domain filter, e.g. 'entity' or 'content,entity'. Not required. Call jarvis_get_ontology to see valid domains."),
    namespace: z
      .string()
      .optional()
      .describe("Scope the search to a Jarvis namespace (data partition). Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    if (!cfg.q && !cfg.input_q && !cfg.output_q) {
      return "jarvis/graph-search requires at least one of: q, input_q, output_q";
    }
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const query: Record<string, string | number | boolean> = {
      limit: cfg.limit ?? 10,
      // Per-node {EDGE_TYPE: count} map inline, so connectivity + hop targets
      // come back in one call.
      include_edge_counts: true,
    };
    if (cfg.q) query.q = cfg.q;
    // Field-scoped vector search: Jarvis embeds these against the per-field
    // input/output schema embeddings and fuses them with `q` via RRF.
    if (cfg.input_q) query.input_q = cfg.input_q;
    if (cfg.output_q) query.output_q = cfg.output_q;
    if (cfg.type) query.type = cfg.type;
    if (cfg.domains) query.domains = cfg.domains;
    if (cfg.namespace) query.namespace = cfg.namespace;

    const res = await http(`${base}/v2/nodes`, { headers, query, timeout });
    if (!res.ok) return `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    const data = res.body as any;
    const nodes: any[] = Array.isArray(data) ? data : (data?.nodes ?? []);
    return nodes.map((n: any) => ({
      ref_id: n.ref_id ?? n.properties?.ref_id,
      name:
        n.properties?.name ??
        n.properties?.workflow_name ??
        n.properties?.episode_title ??
        n.properties?.entity,
      node_type: n.node_type,
      description: n.properties?.description ?? n.properties?.summary ?? n.properties?.text ?? "",
      // {EDGE_TYPE: count} map of this node's relationships.
      edges: (n.edges ?? {}) as Record<string, number>,
      ...(n.properties?.workflow_id !== undefined ? { workflow_id: n.properties.workflow_id } : {}),
      ...(n.properties?.skill_id !== undefined ? { skill_id: n.properties.skill_id } : {}),
    }));
  },
});
