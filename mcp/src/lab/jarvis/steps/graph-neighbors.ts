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
const NEIGHBOR_CAP = 50;
const EXCLUDED_NODE_TYPES = ["Hint", "Memory", "Clip", "Turn"];

/** Encode an array as a Python list literal, e.g. `["MODIFIES","CITES"]`
 *  (the format the Jarvis endpoint parses for list params). */
function toPythonListLiteral(arr: string[]): string {
  return `[${arr.map((s) => `"${s}"`).join(",")}]`;
}

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

export default defineStep({
  type: "jarvis/graph-neighbors",
  description:
    "Return all nodes adjacent (one hop) to a node in the Jarvis knowledge graph, " +
    "with edge_type and direction. Use the ref_id from jarvis_graph_search or jarvis_graph_get. " +
    "Each neighbor also includes an `edges` map ({EDGE_TYPE: count}) showing how " +
    "connected that neighbor is and which relationship types you can hop along next. " +
    "Optionally filter by edge_type and/or node_type. " +
    "Use this to traverse relationships between people, topics, concepts, code, etc.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to expand."),
    edge_type: z
      .array(z.string())
      .optional()
      .describe('Filter edges by type, e.g. ["MODIFIES", "CITES"].'),
    node_type: z
      .array(z.string())
      .optional()
      .describe('Filter neighbor nodes by type, e.g. ["File", "Concept"].'),
    namespace: z
      .string()
      .optional()
      .describe("Scope neighbor edge-count computation to a Jarvis namespace (data partition). Only affects each neighbor's `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    // `limit` bounds the Cypher traversal so a hub node doesn't OOM Neo4j.
    // `sort_by=importance` orders edges before LIMIT so the cap keeps the most
    // important neighbors. `canonicalize=false` matches the real Neo4j label.
    const query: Record<string, string | number | boolean> = {
      expand: "edges",
      limit: NEIGHBOR_CAP,
      sort_by: "importance",
      canonicalize: "false",
      exclude_node_type: toPythonListLiteral(EXCLUDED_NODE_TYPES),
      include_edge_counts: true,
    };
    if (cfg.edge_type && cfg.edge_type.length > 0) query.edge_type = toPythonListLiteral(cfg.edge_type);
    if (cfg.node_type && cfg.node_type.length > 0) query.node_type = toPythonListLiteral(cfg.node_type);
    if (cfg.namespace) query.namespace = cfg.namespace;

    const res = await http(`${base}/v2/nodes/${encodeURIComponent(cfg.ref_id)}`, { headers, query, timeout });
    if (!res.ok) return `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    const data = res.body as any;

    // Node details by ref_id (excluding the queried node) so each neighbor
    // carries a label and its own connectivity map alongside its ref_id.
    const nodeMap = new Map<string, { node_type: string; name: string; edges: Record<string, number> }>();
    for (const node of data.nodes ?? []) {
      if (node.ref_id !== cfg.ref_id) {
        nodeMap.set(node.ref_id, {
          node_type: node.node_type,
          name: deriveNodeName(node, (node.properties ?? {}) as Record<string, any>),
          edges: (node.edges ?? {}) as Record<string, number>,
        });
      }
    }

    const neighbors: any[] = [];
    const seen = new Set<string>();
    for (const edge of data.edges ?? []) {
      const direction = edge.source === cfg.ref_id ? "forward" : "reverse";
      const neighborRefId = direction === "forward" ? edge.target : edge.source;
      if (neighborRefId === cfg.ref_id) continue; // self-loop guard
      if (seen.has(neighborRefId)) continue; // parallel-edge dedup: keep the first
      seen.add(neighborRefId);

      const detail = nodeMap.get(neighborRefId);
      const importance = edge.properties?.importance as number | undefined;
      neighbors.push({
        ref_id: neighborRefId,
        node_type: detail?.node_type ?? "unknown",
        name: detail?.name ?? "",
        edge_type: edge.edge_type,
        direction,
        edges: detail?.edges ?? {},
        ...(importance !== undefined ? { importance } : {}),
      });
      if (neighbors.length >= NEIGHBOR_CAP) break;
    }

    return neighbors;
  },
});
