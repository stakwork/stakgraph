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

/** Created/merged node ref_id from a Jarvis POST /v2/nodes response body.
 *  Both plain success and the "already exists" warning carry data.ref_id. */
function extractNodeRefId(body: any): string | undefined {
  const refId = body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

export default defineStep({
  type: "jarvis/create-node",
  description:
    "Create (or merge) a SINGLE node in the Jarvis knowledge graph as DATA, with no edge " +
    "(to assert a relationship at the same time, use jarvis_create_triplet instead). Writes live to the graph. " +
    "REUSE existing nodes: jarvis_graph_search first, and only create when the entity genuinely doesn't exist yet — " +
    "duplicate nodes fragment the graph. The node type must already exist in the ontology " +
    "(check with jarvis_get_ontology; jarvis_get_ontology_type shows its attributes and which are required). " +
    "Create-or-merge semantics: if a node with the same identity key already exists, its ref_id is " +
    "returned (reported as a Warning) instead of creating a duplicate — so re-running is safe.",
  input: z.object({
    node_type: z
      .string()
      .describe('Node type (must exist in the ontology — see jarvis_get_ontology). Never pass the wildcard sentinel "*".'),
    node_data: z
      .record(z.string(), z.any())
      .describe('Properties for the node, e.g. {"name": "Alice"}. Must satisfy the type\'s schema, including its node_key attribute.'),
    namespace: z
      .string()
      .optional()
      .describe("Jarvis namespace (data partition) to create the node in. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const query: Record<string, string> = {};
    if (cfg.namespace) query.namespace = cfg.namespace;
    const res = await http(`${base}/v2/nodes`, {
      method: "POST",
      headers,
      query,
      timeout,
      body: { node_type: cfg.node_type, node_data: cfg.node_data },
    });
    const body = res.body as any;
    const refId = extractNodeRefId(body);
    if (!refId) {
      return `jarvis/create-node failed — HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    }
    return {
      // "Warning" here means the node already existed (idempotent merge).
      status: body?.status ?? "Success",
      ref_id: refId,
      node_type: cfg.node_type,
      ...(Array.isArray(body?.status_messages) && body.status_messages.length > 0
        ? { messages: body.status_messages }
        : {}),
    };
  },
});
