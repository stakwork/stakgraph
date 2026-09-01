import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText, graphErrorCode } from "./_shared.js";
/** Validate one side of a triplet: either ref_id XOR (type + data). */
function validateTripletSide(
  side: "source" | "target",
  refId?: string,
  nodeType?: string,
  nodeData?: Record<string, any>,
): string | null {
  const hasRef = typeof refId === "string" && refId.length > 0;
  const hasInline = Boolean(nodeType) || Boolean(nodeData);
  if (hasRef && hasInline) return `${side}: pass either ${side}_ref_id OR ${side}_type + ${side}_data, not both`;
  if (hasRef) return null;
  if (nodeType && nodeData) return null;
  return `${side}: pass ${side}_ref_id (an existing node), or both ${side}_type and ${side}_data (create/merge inline)`;
}

export default defineStep({
  type: "graph/create-triplet",
  description:
    "Assert a fact into the vein knowledge graph as DATA: a triplet of source node -[edge]-> target node " +
    "(instances, not schema). Writes live to the graph. " +
    "For each side pass EITHER the ref_id of an existing node (preferred — find it with graph_graph_search) " +
    "OR a node type + data object to create/merge the node inline. " +
    "REUSE existing nodes wherever possible: search first, and only create inline when the entity " +
    "genuinely doesn't exist yet — duplicate nodes fragment the graph. " +
    "Node types and the (source_type, edge_type, target_type) triple must be in the ontology " +
    "(check graph_get_ontology with include_edges) — the edge registry is closed; " +
    "create_schema_if_missing and allow_scratchpad are accepted for input parity but have no effect. " +
    "The source must be a Vein node; ACCESSED may point at ANY node (including jarvis-owned types like Concept).",
  input: z.object({
    source_ref_id: z
      .string()
      .optional()
      .describe("ref_id of an EXISTING source node (from graph_graph_search/graph_graph_get). Preferred over inline creation."),
    source_type: z
      .string()
      .optional()
      .describe("Node type for an INLINE source node (a Vein type). Requires source_data; omit when source_ref_id is set."),
    source_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties for an INLINE source node, e.g. {"name": "harvey-deliver"}. Must satisfy the type\'s schema, including its node_key attribute(s).'),
    target_ref_id: z
      .string()
      .optional()
      .describe("ref_id of an EXISTING target node. Preferred over inline creation."),
    target_type: z
      .string()
      .optional()
      .describe("Node type for an INLINE target node (a Vein type). Requires target_data; omit when target_ref_id is set."),
    target_data: z
      .record(z.string(), z.any())
      .optional()
      .describe('Properties for an INLINE target node.'),
    edge_type: z
      .string()
      .describe("The relationship type, e.g. 'VERSION_OF'. Uppercased. Must be a registered (source_type, edge_type, target_type) triple."),
    edge_data: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional properties to set on the edge."),
    weight: z.number().optional().describe("Optional edge weight (defaults to 1)."),
    create_schema_if_missing: z
      .boolean()
      .optional()
      .default(false)
      .describe("Accepted for input parity with jarvis/create-triplet; the vein edge registry is closed, so this has no effect."),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) for inline node creation — must be registered. Not an access-control boundary."),
    allow_scratchpad: z
      .boolean()
      .optional()
      .describe("Accepted for input parity with jarvis/create-triplet; the vein graph has no scratchpad, so this has no effect."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    for (const err of [
      validateTripletSide("source", cfg.source_ref_id, cfg.source_type, cfg.source_data),
      validateTripletSide("target", cfg.target_ref_id, cfg.target_type, cfg.target_data),
    ]) {
      if (err) return `graph/create-triplet invalid input — ${err}`;
    }

    try {
      const b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      const namespace = await b.reader.resolveNamespace(cfg.namespace);

      // Resolve one side to a concrete ref_id, creating/merging an inline
      // node when no ref_id was given. Sequential so a failure names the
      // side that broke.
      const resolveSide = async (side: "source" | "target", refId?: string, nodeType?: string, nodeData?: Record<string, any>): Promise<string> => {
        if (refId) return refId;
        try {
          return (await b.nodes.write({ type: nodeType!, data: nodeData! }, "create", { namespace })).ref_id;
        } catch (e) {
          throw new Error(`could not create/merge ${side} node: ${e instanceof Error ? e.message : String(e)}`);
        }
      };
      const sourceRef = await resolveSide("source", cfg.source_ref_id, cfg.source_type, cfg.source_data);
      const targetRef = await resolveSide("target", cfg.target_ref_id, cfg.target_type, cfg.target_data);

      let edge;
      try {
        edge = await b.edges.write({
          edge: cfg.edge_type.toUpperCase().replace(/ /g, "_"),
          source_ref_id: sourceRef,
          target_ref_id: targetRef,
          ...(cfg.edge_data ? { properties: cfg.edge_data } : {}),
          ...(cfg.weight !== undefined ? { weight: cfg.weight } : {}),
        });
      } catch (e) {
        return (
          `graph/create-triplet: nodes resolved (source=${sourceRef}, target=${targetRef}) ` +
          `but the edge write failed — ${graphErrorCode(e) ? `${graphErrorCode(e)}: ` : ""}${e instanceof Error ? e.message : String(e)}`
        );
      }
      return {
        // "Warning" here means the edge already existed (idempotent merge).
        status: edge.created ? "Success" : "Warning",
        source_ref_id: edge.source_ref_id,
        target_ref_id: edge.target_ref_id,
        edge_ref_id: edge.ref_id,
        edge_type: cfg.edge_type.toUpperCase().replace(/ /g, "_"),
        ...(edge.created ? {} : { messages: ["Edge already exists in the graph"] }),
      };
    } catch (e) {
      return errText("graph/create-triplet", e);
    }
  },
});
