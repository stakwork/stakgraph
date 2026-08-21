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

const LABEL_MAX = 160;

/** Jarvis nodes keep their human label under wildly different keys depending
 *  on node type — try a generous ordered candidate list. */
function deriveNodeName(node: any, p: Record<string, any>): string {
  const candidates = [
    node?.name, p.name, p.title, p.label, p.display_name, p.displayName,
    p.identifier, p.file_name, p.fileName, p.file, p.path, p.symbol,
    p.function_name, p.class_name, p.method_name, p.operation_id, p.endpoint,
    p.route, p.url, p.entity, p.key, p.slug, p.episode_title, p.show_title,
    p.username, p.email, p.summary, p.description, p.text, p.content, p.body, p.docs,
  ];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim().length > 0) {
      const t = c.trim();
      return t.length > LABEL_MAX ? t.slice(0, LABEL_MAX) : t;
    }
  }
  return "";
}

/** Collapse /connection-counts rows into a compact {EDGE_TYPE: total} map. */
function collapseConnectionCounts(
  counts: Array<{ edge_type: string; target_type?: string; count: number }>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of counts ?? []) {
    if (!c?.edge_type) continue;
    out[c.edge_type] = (out[c.edge_type] ?? 0) + Number(c.count ?? 0);
  }
  return out;
}

export default defineStep({
  type: "jarvis/graph-get",
  description:
    "Resolve a single node in the Jarvis knowledge graph to its full content by ref_id. " +
    "Use the ref_id from jarvis_graph_search or jarvis_graph_neighbors results. " +
    "Returns the node's ref_id, node_type, derived name, properties, and an " +
    "`edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
    "which relationship types you can traverse next with jarvis_graph_neighbors. " +
    "To resolve several ref_ids at once, use jarvis_graph_get_batched instead.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to resolve."),
    namespace: z
      .string()
      .optional()
      .describe("Scope edge-count computation to a Jarvis namespace (data partition). Only affects the `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    // limit=1 keeps Jarvis from materializing the node's whole neighborhood
    // (which can OOM Neo4j for hub nodes) — we only read the node itself.
    const res = await http(`${base}/v2/nodes/${encodeURIComponent(cfg.ref_id)}`, {
      headers,
      query: { limit: 1 },
      timeout,
    });
    if (!res.ok) return `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    const data = res.body as any;
    // Deployed Jarvis wraps the node in `{ nodes, edges, status }`; some
    // builds return the node directly. Handle both shapes.
    const raw = Array.isArray(data?.nodes)
      ? data.nodes.find((n: any) => n.ref_id === cfg.ref_id) ?? data.nodes[0]
      : data;
    if (!raw || !raw.ref_id) return `node not found: ${cfg.ref_id}`;
    const properties = (raw.properties ?? {}) as Record<string, any>;

    // Edge-type connectivity from the cheap counts endpoint. Best effort —
    // never fail the whole call if this lookup errors.
    let edges: Record<string, number> = {};
    try {
      const ccQuery: Record<string, string> = {};
      if (cfg.namespace) ccQuery.namespace = cfg.namespace;
      const cc = await http(`${base}/v2/nodes/${encodeURIComponent(cfg.ref_id)}/connection-counts`, {
        headers,
        query: ccQuery,
        timeout,
      });
      if (cc.ok) edges = collapseConnectionCounts((cc.body as any)?.counts ?? []);
    } catch {
      // best effort — edges stays {}
    }

    return {
      ref_id: raw.ref_id,
      node_type: raw.node_type,
      name: deriveNodeName(raw, properties),
      properties: raw.properties,
      edges,
    };
  },
});
