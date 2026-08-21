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
  type: "jarvis/edit-node",
  description:
    "Update an EXISTING node in the Jarvis knowledge graph by ref_id (writes live to the graph). " +
    "PARTIAL update: properties in node_data are merged over the node's current properties " +
    "(validated against the type's schema) — properties you omit are left untouched. " +
    "Use properties_to_be_deleted to remove properties entirely. " +
    "Get the ref_id from jarvis_graph_search, and inspect the node with jarvis_graph_get first so you know its " +
    "current state before changing it. " +
    "Pass node_type ONLY to change the node's type (with type_to_be_deleted listing the old type " +
    "label(s) to remove); omit both for normal property edits. " +
    "If the update would change the node's identity key to collide with another node, the write " +
    "fails with 'Node already exists in the graph'.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to update (from jarvis_graph_search/jarvis_graph_get)."),
    node_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties to set/overwrite, e.g. {"description": "..."}. Merged over the node\'s existing properties.'),
    properties_to_be_deleted: z
      .array(z.string())
      .optional()
      .describe("Property names to REMOVE from the node."),
    node_type: z
      .string()
      .optional()
      .describe("New node type — pass ONLY when changing the node's type (must exist in the ontology)."),
    type_to_be_deleted: z
      .array(z.string())
      .optional()
      .describe("When changing the node's type: the old type label(s) to remove from the node."),
    namespace: z
      .string()
      .optional()
      .describe("Jarvis namespace (data partition) the node lives in. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const hasSet = cfg.node_data && Object.keys(cfg.node_data).length > 0;
    const hasDelete = cfg.properties_to_be_deleted && cfg.properties_to_be_deleted.length > 0;
    if (!hasSet && !hasDelete && !cfg.node_type) {
      return "jarvis/edit-node invalid input — pass at least one change: node_data (properties to set), properties_to_be_deleted, or node_type";
    }
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const query: Record<string, string> = {};
    if (cfg.namespace) query.namespace = cfg.namespace;
    // Always send node_data (even empty) — its presence selects Jarvis's
    // modern schema-validated update path over the legacy flat-property one.
    const res = await http(`${base}/v2/nodes/${encodeURIComponent(cfg.ref_id)}`, {
      method: "POST",
      headers,
      query,
      timeout,
      body: {
        node_data: cfg.node_data ?? {},
        ...(hasDelete ? { properties_to_be_deleted: cfg.properties_to_be_deleted } : {}),
        ...(cfg.node_type ? { node_type: cfg.node_type } : {}),
        ...(cfg.type_to_be_deleted && cfg.type_to_be_deleted.length > 0
          ? { type_to_be_deleted: cfg.type_to_be_deleted }
          : {}),
      },
    });
    const body = res.body as any;
    // Jarvis returns HTTP 200 with {status: "fail", message} on some write
    // failures (e.g. node_key collision), so res.ok alone is not enough.
    const succeeded = res.ok && body?.status === "success";
    if (!succeeded) {
      const detail = body?.message ?? body?.errorCode ?? (typeof res.body === "string" ? res.body : JSON.stringify(res.body));
      return `jarvis/edit-node failed — HTTP ${res.status}: ${detail}`;
    }
    // Compact confirmation — deliberately NOT the full updated node, whose
    // properties include bulky derived fields. jarvis_graph_get to verify.
    return {
      status: "Success",
      ref_id: cfg.ref_id,
      ...(hasSet ? { updated: Object.keys(cfg.node_data!) } : {}),
      ...(hasDelete ? { deleted: cfg.properties_to_be_deleted } : {}),
      ...(cfg.node_type ? { node_type: cfg.node_type } : {}),
    };
  },
});
