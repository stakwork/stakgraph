import { z } from "zod";
import { defineStep, type StepContext, withAccessedNodes } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText } from "./_shared.js";

const EXAMPLE = `- id: set_strength
  type: graph/edit-edge
  config:
    source_ref_id: "{{ claim.ref_id }}"
    edge_type: EVIDENCED_BY
    target_ref_id: "{{ evidence.ref_id }}"
    edge_data:
      strength: "{{ grounded.strength }}"`;

export default defineStep({
  type: "graph/edit-edge",
  description:
    "Update the properties of an EXISTING edge in the vein knowledge graph (writes live to the graph). " +
    "Edges are create-only in graph/create-triplet: writing the same source-[EDGE]->target again returns the " +
    "existing edge unchanged ('Warning'), so this is the one way to change an edge's data afterwards. " +
    "Locate the edge by `edge_ref_id` (from create-triplet's output) OR by the triple `source_ref_id` + `edge_type` + " +
    "`target_ref_id`. PARTIAL update: `edge_data` properties are set/overwritten over the edge's current properties, " +
    "`properties_to_be_deleted` removes properties; everything else is left untouched. `weight` may be set. " +
    "The identity stamps (ref_id, edge_key, date_added_to_graph, unique_source_id) cannot be changed, and an edge " +
    "cannot be re-pointed at other nodes — create a new edge instead.\n\n" +
    EXAMPLE,
  input: z.object({
    edge_ref_id: z.string().optional().describe("ref_id of the edge (create-triplet returns it as edge_ref_id). Preferred when you have it."),
    source_ref_id: z.string().optional().describe("With edge_type + target_ref_id: locate the edge by its endpoints instead of edge_ref_id."),
    edge_type: z.string().optional().describe("The relationship type, e.g. 'EVIDENCED_BY' (uppercased). Required when locating by endpoints."),
    target_ref_id: z.string().optional().describe("Target node ref_id, when locating by endpoints."),
    edge_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties to set/overwrite on the edge, e.g. {"strength": -0.75}. Merged over the existing properties.'),
    properties_to_be_deleted: z.array(z.string()).optional().describe("Property names to REMOVE from the edge."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const byRef = Boolean(cfg.edge_ref_id);
    const byTriple = Boolean(cfg.source_ref_id || cfg.edge_type || cfg.target_ref_id);
    if (!byRef && !byTriple) return "graph/edit-edge invalid input — pass edge_ref_id, or source_ref_id + edge_type + target_ref_id";
    if (byRef && byTriple) return "graph/edit-edge invalid input — pass EITHER edge_ref_id OR the source_ref_id/edge_type/target_ref_id triple, not both";
    if (byTriple && !(cfg.source_ref_id && cfg.edge_type && cfg.target_ref_id)) {
      return "graph/edit-edge invalid input — locating by endpoints needs all of source_ref_id, edge_type and target_ref_id";
    }
    const hasSet = cfg.edge_data && Object.keys(cfg.edge_data).length > 0;
    const hasDelete = cfg.properties_to_be_deleted && cfg.properties_to_be_deleted.length > 0;
    if (!hasSet && !hasDelete) return "graph/edit-edge invalid input — pass at least one change: edge_data (properties to set) or properties_to_be_deleted";
    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const r = await b.edges.update(
        byRef ? { ref_id: cfg.edge_ref_id! } : { edge: cfg.edge_type!, source_ref_id: cfg.source_ref_id!, target_ref_id: cfg.target_ref_id! },
        { set: cfg.edge_data ?? {}, remove: cfg.properties_to_be_deleted ?? [] },
      );
      return withAccessedNodes(
        {
          status: "Success",
          edge_ref_id: r.ref_id,
          edge_type: r.edge,
          source_ref_id: r.source_ref_id,
          target_ref_id: r.target_ref_id,
          ...(hasSet ? { updated: r.updated } : {}),
          ...(hasDelete ? { deleted: r.removed } : {}),
        },
        // Provenance: both endpoints (the edge itself is not a node).
        [{ ref_id: r.source_ref_id }, { ref_id: r.target_ref_id }],
      );
    } catch (e) {
      return errText("graph/edit-edge", e);
    }
  },
});
