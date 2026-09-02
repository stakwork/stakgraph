import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { getVeinSchema } from "../../../graph/vein-schemas.js";
import { graphCtx, errText } from "./_shared.js";
const splitList = (s?: string) => s?.split(",").map((x) => x.trim()).filter(Boolean);

/** A node's human label: the Vein schema's title_key when it is a Vein type,
 *  else the same candidate chain the jarvis/* step uses. */
function nameOf(nodeType: string | undefined, p: Record<string, any>): string | undefined {
  const vein = nodeType ? getVeinSchema(nodeType) : undefined;
  if (vein && typeof p[vein.title_key] === "string") return p[vein.title_key];
  return p.name ?? p.workflow_name ?? p.episode_title ?? p.entity;
}

function descriptionOf(nodeType: string | undefined, p: Record<string, any>): string {
  const vein = nodeType ? getVeinSchema(nodeType) : undefined;
  if (vein && typeof p[vein.description_key] === "string" && vein.description_key !== vein.title_key) return p[vein.description_key];
  return p.description ?? p.summary ?? p.text ?? "";
}

export default defineStep({
  type: "graph/graph-search",
  description:
    "Search the vein knowledge graph for nodes — workflows, workflow versions, steps, runs, agent sessions, tool calls, chats, turns, and any jarvis-owned types sharing the database. " +
    "Provide at least one of `q`, `input_q`, `output_q` — they can be combined, each acting as its own " +
    "retriever fused into one ranked result set. " +
    "Each result includes an `edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
    "which relationship types you can traverse next with graph_graph_neighbors. " +
    "Call graph_get_ontology first to discover valid values for the `type` parameter.",
  input: z.object({
    q: z
      .string()
      .optional()
      .describe("General hybrid (keyword + semantic) search query over node names, descriptions, summaries, and schemas."),
    input_q: z
      .string()
      .optional()
      .describe(
        "Semantic search scoped to node INPUT schemas — find nodes by what they take as input, " +
        "e.g. 'a video file url'. Applies to node types with input embeddings (VeinWorkflowVersion, VeinStep).",
      ),
    output_q: z
      .string()
      .optional()
      .describe(
        "Semantic search scoped to node OUTPUT schemas — find nodes by what they produce, " +
        "e.g. 'transcript with word-level timestamps'. Applies to node types with output embeddings (VeinWorkflowVersion, VeinStep).",
      ),
    type: z
      .string()
      .optional()
      .describe("Comma-separated node type filter, e.g. 'VeinWorkflow' or 'VeinRun,VeinChat'. Call graph_get_ontology to see all valid values."),
    limit: z.number().optional().default(10).describe("Maximum number of results to return"),
    domains: z
      .string()
      .optional()
      .describe("Comma-separated domain filter, e.g. 'vein' or 'vein,entity'. Not required. Call graph_get_ontology to see valid domains."),
    namespace: z
      .string()
      .optional()
      .describe("Scope the search to a namespace (data partition). Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    if (!cfg.q && !cfg.input_q && !cfg.output_q) {
      return "graph/graph-search requires at least one of: q, input_q, output_q";
    }
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const res = await b.reader.search({
        q: cfg.q,
        input_q: cfg.input_q,
        output_q: cfg.output_q,
        types: splitList(cfg.type),
        domains: splitList(cfg.domains),
        namespace: cfg.namespace,
        limit: cfg.limit ?? 10,
        // Per-node {EDGE_TYPE: count} map inline, so connectivity + hop
        // targets come back in one call.
        include_edge_counts: true,
      });
      return res.nodes.map((n) => {
        const p = (n.properties ?? {}) as Record<string, any>;
        return {
          ref_id: n.ref_id,
          name: nameOf(n.node_type, p),
          node_type: n.node_type,
          description: descriptionOf(n.node_type, p),
          // {EDGE_TYPE: count} map of this node's relationships.
          edges: (n.edges ?? {}) as Record<string, number>,
          ...(p.workflow_id !== undefined ? { workflow_id: p.workflow_id } : {}),
          ...(p.skill_id !== undefined ? { skill_id: p.skill_id } : {}),
        };
      });
    } catch (e) {
      return errText("graph/graph-search", e);
    }
  },
});
