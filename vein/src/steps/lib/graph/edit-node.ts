import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";
export default defineStep({
  type: "graph/edit-node",
  description:
    "Update an EXISTING node in the vein knowledge graph by ref_id (writes live to the graph). " +
    "PARTIAL update: properties in node_data are merged over the node's current properties " +
    "(the merged result is validated against the type's schema) — properties you omit are left untouched. " +
    "Use properties_to_be_deleted to remove optional properties entirely (required ones cannot be removed). " +
    "Get the ref_id from graph_graph_search, and inspect the node with graph_graph_get first so you know its " +
    "current state before changing it. " +
    "Changing a node's TYPE is not supported (the Vein type set is closed) — create a new node instead. " +
    "If the update would change the node's identity key to collide with another node, the write " +
    "fails with 'Node already exists in the graph'.",
  input: z.object({
    ref_id: z.string().describe("The ref_id of the node to update (from graph_graph_search/graph_graph_get)."),
    node_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties to set/overwrite, e.g. {"description": "..."}. Merged over the node\'s existing properties.'),
    properties_to_be_deleted: z
      .array(z.string())
      .optional()
      .describe("Property names to REMOVE from the node (optional attributes only)."),
    node_type: z
      .string()
      .optional()
      .describe("Not supported by the vein graph backend — passing it returns an error. Kept for input parity with jarvis/edit-node."),
    type_to_be_deleted: z
      .array(z.string())
      .optional()
      .describe("Not supported by the vein graph backend. Kept for input parity with jarvis/edit-node."),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) the node lives in. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const hasSet = cfg.node_data && Object.keys(cfg.node_data).length > 0;
    const hasDelete = cfg.properties_to_be_deleted && cfg.properties_to_be_deleted.length > 0;
    if (cfg.node_type || (cfg.type_to_be_deleted && cfg.type_to_be_deleted.length > 0)) {
      return "graph/edit-node: changing a node's type is not supported by the vein graph backend (closed Vein type set) — create a new node of the right type instead";
    }
    if (!hasSet && !hasDelete) {
      return "graph/edit-node invalid input — pass at least one change: node_data (properties to set) or properties_to_be_deleted";
    }
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      await b.nodes.update(cfg.ref_id, { set: cfg.node_data ?? {}, remove: cfg.properties_to_be_deleted ?? [] });
      // Compact confirmation — deliberately NOT the full updated node.
      // graph_graph_get to verify.
      return {
        status: "Success",
        ref_id: cfg.ref_id,
        ...(hasSet ? { updated: Object.keys(cfg.node_data!) } : {}),
        ...(hasDelete ? { deleted: cfg.properties_to_be_deleted } : {}),
      };
    } catch (e) {
      return errText("graph/edit-node", e);
    }
  },
});
