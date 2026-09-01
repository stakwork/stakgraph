import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
export default defineStep({
  type: "graph/create-node",
  description:
    "Create (or merge) a SINGLE node in the vein knowledge graph as DATA, with no edge " +
    "(to assert a relationship at the same time, use graph_create_triplet instead). Writes live to the graph. " +
    "REUSE existing nodes: graph_graph_search first, and only create when the entity genuinely doesn't exist yet — " +
    "duplicate nodes fragment the graph. The node type must already exist in the ontology " +
    "(check with graph_get_ontology; graph_get_ontology_type shows its attributes and which are required). " +
    "Every attribute is validated against the type's schema — unknown attributes are rejected. " +
    "Create-or-merge semantics: if a node with the same identity key already exists, its ref_id is " +
    "returned (reported as a Warning) instead of creating a duplicate — so re-running is safe.",
  input: z.object({
    node_type: z
      .string()
      .describe('Node type (must exist in the ontology — see graph_get_ontology). Never pass the wildcard sentinel "*".'),
    node_data: z
      .record(z.string(), z.any())
      .describe('Properties for the node, e.g. {"name": "harvey-deliver"}. Must satisfy the type\'s schema, including its node_key attribute(s).'),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) to create the node in — must be registered. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const namespace = await b.reader.resolveNamespace(cfg.namespace);
      const r = await b.nodes.write({ type: cfg.node_type, data: cfg.node_data }, "create", { namespace });
      const existed = r.outcome === "existing";
      return {
        // "Warning" here means the node already existed (idempotent merge).
        status: existed ? "Warning" : "Success",
        ref_id: r.ref_id,
        node_type: cfg.node_type,
        ...(existed ? { messages: [`Node already exists in the graph with node_key: ${r.node_key}`] } : {}),
        ...(r.outcome === "restored" ? { messages: ["Node restored"] } : {}),
      };
    } catch (e) {
      return errText("graph/create-node", e);
    }
  },
});
