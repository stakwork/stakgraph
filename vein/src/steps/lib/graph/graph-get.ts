import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
const LABEL_MAX = 160;

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
  type: "graph/graph-get",
  description:
    "Resolve a single node in the vein knowledge graph to its full content by ref_id. " +
    "Use the ref_id from graph_graph_search or graph_graph_neighbors results. " +
    "Returns the node's ref_id, node_type, derived name, properties, and an " +
    "`edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
    "which relationship types you can traverse next with graph_graph_neighbors. " +
    "To resolve several ref_ids at once, use graph_graph_get_batched instead.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to resolve."),
    namespace: z
      .string()
      .optional()
      .describe("Scope edge-count computation to a namespace (data partition). Only affects the `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const raw = await b.reader.getNode(cfg.ref_id);
      if (!raw) return `node not found: ${cfg.ref_id}`;
      const properties = (raw.properties ?? {}) as Record<string, any>;
      // Edge-type connectivity. Best effort — never fail the whole call.
      let edges: Record<string, number> = {};
      try {
        edges = collapseConnectionCounts(await b.reader.connectionCounts(cfg.ref_id, cfg.namespace));
      } catch {
        // edges stays {}
      }
      return {
        ref_id: raw.ref_id,
        node_type: raw.node_type,
        name: deriveNodeName(raw, properties),
        properties: raw.properties,
        edges,
      };
    } catch (e) {
      return errText("graph/graph-get", e);
    }
  },
});
