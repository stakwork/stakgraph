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
const BATCH_MAX = 50;
const CONCURRENCY = 8;

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
  type: "jarvis/graph-get-batched",
  description:
    `Resolve up to ${BATCH_MAX} nodes in one call by ref_id — the batched form of jarvis_graph_get. ` +
    "ALWAYS prefer this over calling jarvis_graph_get in a loop: it fetches them concurrently in a single call. " +
    "Returns `{ requested, returned, truncated, omitted_ref_ids, nodes }`, where each entry in " +
    "`nodes` is either the full node (ref_id, node_type, name, properties, edges) or " +
    "`{ ref_id, error }` if that one could not be resolved — one bad ref_id never fails the rest. " +
    `If you pass more than ${BATCH_MAX} ref_ids, the excess comes back in ` +
    "`omitted_ref_ids` and `truncated` is true; call again with those to finish the job. " +
    "An EMPTY `edges` map is NOT proof the node has no relationships (edge-count computation can be unavailable on some deployments) — confirm with jarvis_graph_neighbors before concluding a node is unconnected. ",
  input: z.object({
    ref_ids: z
      .array(z.string())
      .min(1)
      .describe(`The ref_ids to resolve, in the order you want them back. Up to ${BATCH_MAX} per call.`),
    namespace: z
      .string()
      .optional()
      .describe("Scope edge-count computation to a Jarvis namespace (data partition). Only affects each node's `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);

    async function fetchNode(
      ref_id: string,
    ): Promise<Record<string, any>> {
      try {
        const res = await http(`${base}/v2/nodes/${encodeURIComponent(ref_id)}`, {
          headers,
          query: { limit: 1 },
          timeout,
        });
        if (!res.ok) {
          return { ref_id, error: `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}` };
        }
        const data = res.body as any;
        const raw = Array.isArray(data?.nodes)
          ? data.nodes.find((n: any) => n.ref_id === ref_id) ?? data.nodes[0]
          : data;
        if (!raw || !raw.ref_id) return { ref_id, error: `node not found: ${ref_id}` };
        const properties = (raw.properties ?? {}) as Record<string, any>;
        let edges: Record<string, number> = {};
        try {
          const ccQuery: Record<string, string> = {};
          if (cfg.namespace) ccQuery.namespace = cfg.namespace;
          const cc = await http(`${base}/v2/nodes/${encodeURIComponent(ref_id)}/connection-counts`, {
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
      } catch (err: any) {
        return { ref_id, error: `graph-get failed: ${err?.message ?? String(err)}` };
      }
    }

    // Dedupe while preserving the caller's ordering.
    const unique = Array.from(new Set(cfg.ref_ids.filter((r) => r && r.trim())));
    if (unique.length === 0) {
      return { requested: cfg.ref_ids.length, returned: 0, truncated: false, omitted_ref_ids: [], nodes: [], note: "no usable ref_ids supplied" };
    }
    const selected = unique.slice(0, BATCH_MAX);
    const omitted = unique.slice(BATCH_MAX);

    // Bounded concurrency without a queue dep: chunked waves.
    const nodes: Record<string, any>[] = [];
    for (let i = 0; i < selected.length; i += CONCURRENCY) {
      const wave = selected.slice(i, i + CONCURRENCY);
      nodes.push(...(await Promise.all(wave.map(fetchNode))));
    }

    return {
      requested: cfg.ref_ids.length,
      returned: nodes.length,
      truncated: omitted.length > 0,
      omitted_ref_ids: omitted,
      nodes,
    };
  },
});
