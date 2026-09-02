import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
const LABEL_MAX = 160;
const NEIGHBOR_CAP = 50;
const EXCLUDED_NODE_TYPES = ["Hint", "Memory", "Clip", "Turn"];

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

export default defineStep({
  type: "graph/graph-neighbors",
  description:
    "Return all nodes adjacent (one hop) to a node in the vein knowledge graph, " +
    "with edge_type and direction. Use the ref_id from graph_graph_search or graph_graph_get. " +
    "Each neighbor also includes an `edges` map ({EDGE_TYPE: count}) showing how " +
    "connected that neighbor is and which relationship types you can hop along next. " +
    "Optionally filter by edge_type and/or node_type. " +
    "Use this to traverse relationships between workflows, versions, steps, runs, sessions, tool calls, chats, and concepts.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to expand."),
    edge_type: z
      .array(z.string())
      .optional()
      .describe('Filter edges by type, e.g. ["VERSION_OF", "USES_STEP"].'),
    node_type: z
      .array(z.string())
      .optional()
      .describe('Filter neighbor nodes by type, e.g. ["VeinStep", "Concept"].'),
    namespace: z
      .string()
      .optional()
      .describe("Scope neighbor edge-count computation to a namespace (data partition). Only affects each neighbor's `edges` map."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      // `limit` bounds the traversal so a hub node doesn't OOM Neo4j;
      // edges are importance-sorted before the cap keeps the top ones.
      const data = await b.reader.neighbors(cfg.ref_id, {
        edge_types: cfg.edge_type && cfg.edge_type.length > 0 ? cfg.edge_type : undefined,
        node_types: cfg.node_type && cfg.node_type.length > 0 ? cfg.node_type : undefined,
        exclude_node_types: EXCLUDED_NODE_TYPES,
        limit: NEIGHBOR_CAP,
        namespace: cfg.namespace,
        include_edge_counts: true,
      });

      // Node details by ref_id (excluding the queried node) so each neighbor
      // carries a label and its own connectivity map alongside its ref_id.
      const nodeMap = new Map<string, { node_type: string; name: string; edges: Record<string, number> }>();
      for (const node of data.nodes ?? []) {
        if (node.ref_id !== cfg.ref_id) {
          nodeMap.set(node.ref_id, {
            node_type: node.node_type ?? "unknown",
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
        const importance = (edge.properties as Record<string, unknown>)?.["importance"] as number | undefined;
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
    } catch (e) {
      return errText("graph/graph-neighbors", e);
    }
  },
});
