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

/** Validate one side of a triplet: either ref_id XOR (type + data). */
function validateTripletSide(
  side: "source" | "target",
  refId?: string,
  nodeType?: string,
  nodeData?: Record<string, any>,
): string | null {
  const hasRef = typeof refId === "string" && refId.length > 0;
  const hasInline = Boolean(nodeType) || Boolean(nodeData);
  if (hasRef && hasInline) return `${side}: pass either ${side}_ref_id OR ${side}_type + ${side}_data, not both`;
  if (hasRef) return null;
  if (nodeType && nodeData) return null;
  return `${side}: pass ${side}_ref_id (an existing node), or both ${side}_type and ${side}_data (create/merge inline)`;
}

/** Created/merged node ref_id from a Jarvis POST /v2/nodes response body. */
function extractNodeRefId(body: any): string | undefined {
  const refId = body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

/** Edge ref_id from a Jarvis POST /v2/edges response body: a fresh edge lands
 *  in edges[0].ref_id; the "already exists" warning carries data.ref_id. */
function extractEdgeRefId(body: any): string | undefined {
  const refId = body?.edges?.[0]?.ref_id ?? body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

export default defineStep({
  type: "jarvis/create-triplet",
  description:
    "Assert a fact into the Jarvis knowledge graph as DATA: a triplet of source node -[edge]-> target node " +
    "(instances, not schema). Writes live to the graph. " +
    "For each side pass EITHER the ref_id of an existing node (preferred — find it with jarvis_graph_search) " +
    "OR a node type + data object to create/merge the node inline. " +
    "REUSE existing nodes wherever possible: search first, and only create inline when the entity " +
    "genuinely doesn't exist yet — duplicate nodes fragment the graph. " +
    "Node types and the edge type must already exist in the ontology (check with jarvis_get_ontology); " +
    "create_schema_if_missing auto-creates a missing edge schema as a last resort. " +
    "WILDCARD EDGE MATCHING: when checking source_type/target_type against jarvis_get_ontology's edges for validity, " +
    'an edge entry with "*" on either side matches any concrete type on that side. ' +
    '"*" is NEVER a valid value to SUPPLY as source_type or target_type — it is a backend sentinel, not a real node type.',
  input: z.object({
    source_ref_id: z
      .string()
      .optional()
      .describe("ref_id of an EXISTING source node (from jarvis_graph_search/jarvis_graph_get). Preferred over inline creation."),
    source_type: z
      .string()
      .optional()
      .describe("Node type for an INLINE source node (must exist in the ontology). Requires source_data; omit when source_ref_id is set."),
    source_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties for an INLINE source node, e.g. {"name": "Alice"}. Must satisfy the type\'s schema, including its node_key attribute.'),
    target_ref_id: z
      .string()
      .optional()
      .describe("ref_id of an EXISTING target node. Preferred over inline creation."),
    target_type: z
      .string()
      .optional()
      .describe("Node type for an INLINE target node (must exist in the ontology). Requires target_data; omit when target_ref_id is set."),
    target_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties for an INLINE target node, e.g. {"name": "Acme Corp"}.'),
    edge_type: z
      .string()
      .describe("The relationship type, e.g. 'WORKS_AT'. Uppercased by Jarvis. Must exist in the ontology between the two node types unless create_schema_if_missing is set."),
    edge_data: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional properties to set on the edge."),
    weight: z.number().optional().describe("Optional edge weight (defaults to 1)."),
    create_schema_if_missing: z
      .boolean()
      .optional()
      .default(false)
      .describe(
        "Auto-create the edge schema when the (source_type, edge_type, target_type) relationship is not yet in the ontology. " +
        'Last resort — check jarvis_get_ontology\'s edges for an existing wildcard ("*") rule covering the same edge_type first.',
      ),
    namespace: z
      .string()
      .optional()
      .describe("Jarvis namespace (data partition) for inline node creation. Not an access-control boundary."),
    allow_scratchpad: z
      .boolean()
      .optional()
      .describe(
        "Last-resort capture: when the write would otherwise be rejected (unregistered edge type/pair, or " +
          "data that fails schema validation) the payload is preserved as a ScratchpadEntry node instead of " +
          "being dropped. No effect on writes that validate cleanly.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    for (const err of [
      validateTripletSide("source", cfg.source_ref_id, cfg.source_type, cfg.source_data),
      validateTripletSide("target", cfg.target_ref_id, cfg.target_type, cfg.target_data),
    ]) {
      if (err) return `jarvis/create-triplet invalid input — ${err}`;
    }

    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);

    // Resolve one side to a concrete ref_id, creating/merging an inline node
    // via POST /v2/nodes when no ref_id was given. Inline nodes are
    // pre-created (rather than sent straight to /v2/edges) both for per-node
    // error attribution and because Jarvis's edge endpoint mis-orders its
    // ref_id list when only the source is inline (which would reverse the
    // edge direction). Namespace applies to node creation only — the edge
    // endpoint matches by globally-unique ref_id.
    const resolveSide = async (
      side: "source" | "target",
      refId?: string,
      nodeType?: string,
      nodeData?: Record<string, any>,
    ): Promise<string> => {
      if (refId) return refId;
      const query: Record<string, string> = {};
      if (cfg.namespace) query.namespace = cfg.namespace;
      const res = await http(`${base}/v2/nodes`, {
        method: "POST",
        headers,
        query,
        timeout,
        body: {
          node_type: nodeType,
          node_data: nodeData,
          ...(cfg.allow_scratchpad ? { allow_scratchpad: true } : {}),
        },
      });
      const created = extractNodeRefId(res.body);
      if (!created) {
        throw new Error(`could not create/merge ${side} node (HTTP ${res.status}): ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`);
      }
      return created;
    };

    try {
      // Sequential so a failure names the side that broke.
      const sourceRef = await resolveSide("source", cfg.source_ref_id, cfg.source_type, cfg.source_data);
      const targetRef = await resolveSide("target", cfg.target_ref_id, cfg.target_type, cfg.target_data);

      const res = await http(`${base}/v2/edges`, {
        method: "POST",
        headers,
        timeout,
        body: {
          edge: {
            edge_type: cfg.edge_type,
            ...(cfg.weight !== undefined ? { weight: cfg.weight } : {}),
            ...(cfg.edge_data ? { edge_data: cfg.edge_data } : {}),
          },
          source: { ref_id: sourceRef },
          target: { ref_id: targetRef },
          create_schema_if_missing: cfg.create_schema_if_missing ?? false,
          ...(cfg.allow_scratchpad ? { allow_scratchpad: true } : {}),
        },
      });
      const body = res.body as any;
      const edgeRef = extractEdgeRefId(body);
      if (!res.ok || !edgeRef) {
        return (
          `jarvis/create-triplet: nodes resolved (source=${sourceRef}, target=${targetRef}) ` +
          `but the edge write failed — HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`
        );
      }
      return {
        // "Warning" here means the edge already existed (idempotent merge).
        status: body?.status ?? "Success",
        source_ref_id: sourceRef,
        target_ref_id: targetRef,
        edge_ref_id: edgeRef,
        edge_type: cfg.edge_type,
        ...(Array.isArray(body?.status_messages) && body.status_messages.length > 0
          ? { messages: body.status_messages }
          : {}),
      };
    } catch (err: any) {
      return `jarvis/create-triplet failed: ${err?.message ?? String(err)}`;
    }
  },
});
