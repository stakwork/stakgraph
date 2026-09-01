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

const LABEL_MAX = 160;
const BATCH_MAX = 50;
const CONCURRENCY = 8;

/** Nodes keep their human label under different keys depending on node
 *  type — try a generous ordered candidate list (same as jarvis/graph-get). */
function deriveNodeName(node: any, p: Record<string, any>): string {
  const candidates = [
    node?.name, p.name, p.title, p.label, p.display_name, p.displayName,
    p.identifier, p.file_name, p.fileName, p.file, p.path, p.symbol,
    p.function_name, p.class_name, p.method_name, p.operation_id, p.endpoint,
    p.route, p.url, p.entity, p.key, p.slug, p.episode_title, p.show_title,
    p.username, p.email, p.summary, p.description, p.text, p.content, p.body, p.docs,
    p.workflow_name, p.step_type, p.run_id, p.tool_name, p.chat_id,
  ];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim().length > 0) {
      const t = c.trim();
      return t.length > LABEL_MAX ? t.slice(0, LABEL_MAX) : t;
    }
  }
  return "";
}

/** Collapse connection-count rows into a compact {EDGE_TYPE: total} map. */
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
  type: "graph/graph-get-batched",
  description:
    `Resolve up to ${BATCH_MAX} nodes in one call by ref_id — the batched form of graph_graph_get. ` +
    "ALWAYS prefer this over calling graph_graph_get in a loop: it fetches them concurrently in a single call. " +
    "Returns `{ requested, returned, truncated, omitted_ref_ids, nodes }`, where each entry in " +
    "`nodes` is either the full node (ref_id, node_type, name, properties, edges) or " +
    "`{ ref_id, error }` if that one could not be resolved — one bad ref_id never fails the rest. " +
    `If you pass more than ${BATCH_MAX} ref_ids, the excess comes back in ` +
    "`omitted_ref_ids` and `truncated` is true; call again with those to finish the job.",
  input: z.object({
    ref_ids: z
      .array(z.string())
      .min(1)
      .describe(`The ref_ids to resolve, in the order you want them back. Up to ${BATCH_MAX} per call.`),
    namespace: z
      .string()
      .optional()
      .describe("Scope edge-count computation to a namespace (data partition). Only affects each node's `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    let b: GraphBackend;
    try {
      b = await graphCtx(ctx as StepContext<VeinCapabilities>);
    } catch (e) {
      return errText("graph/graph-get-batched", e);
    }

    async function fetchNode(ref_id: string): Promise<Record<string, any>> {
      try {
        const raw = await b.reader.getNode(ref_id);
        if (!raw) return { ref_id, error: `node not found: ${ref_id}` };
        const properties = (raw.properties ?? {}) as Record<string, any>;
        let edges: Record<string, number> = {};
        try {
          edges = collapseConnectionCounts(await b.reader.connectionCounts(ref_id, cfg.namespace));
        } catch {
          // best effort — edges stays {}
        }
        return { ref_id: raw.ref_id, node_type: raw.node_type, name: deriveNodeName(raw, properties), properties: raw.properties, edges };
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
