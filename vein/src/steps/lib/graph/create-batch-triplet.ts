import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { graphCtx, errText, graphErrorCode, type GraphBackend } from "./_shared.js";
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

/** Stable dedup key for an inline node side (type + canonical sorted JSON),
 *  so identical inline sides across the batch resolve once. */
function nodeDedupKey(nodeType: string, nodeData: Record<string, any>): string {
  function sortedJson(v: any): any {
    if (v === null || typeof v !== "object" || Array.isArray(v)) return v;
    const s: Record<string, any> = {};
    for (const k of Object.keys(v).sort()) s[k] = sortedJson(v[k]);
    return s;
  }
  return `${nodeType}::${JSON.stringify(sortedJson(nodeData))}`;
}

const TripletSchema = z.object({
  source_ref_id: z.string().optional(),
  source_type: z.string().optional(),
  source_data: z.record(z.string(), z.any()).optional(),
  target_ref_id: z.string().optional(),
  target_type: z.string().optional(),
  target_data: z.record(z.string(), z.any()).optional(),
  edge_type: z.string(),
  edge_data: z.record(z.string(), z.any()).optional(),
  weight: z.number().optional(),
  create_schema_if_missing: z.boolean().optional().default(false),
});

export default defineStep({
  type: "graph/create-batch-triplet",
  description:
    "Assert MANY facts into the vein knowledge graph in a single call. " +
    "Each item in `triplets` has the same shape as graph_create_triplet (source/target as ref_id or inline " +
    "node_type+node_data, plus edge_type and optional edge_data/weight). " +
    "A single top-level `namespace` applies to all inline node creation. " +
    "REUSE existing nodes wherever possible: supply ref_ids from graph_graph_search when entities already exist — " +
    "inline creation is a last resort. Identical inline sides across the batch are resolved once (deduped). " +
    "Triplets are written sequentially in input order; the result is a per-triplet array in the same order, " +
    "each entry either the created edge info or `{ error }` — one failed triplet never fails the rest.",
  input: z.object({
    triplets: z.array(TripletSchema).min(1).describe("The facts to assert, in order."),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) for all inline node creation — must be registered. Not an access-control boundary."),
    allow_scratchpad: z
      .boolean()
      .optional()
      .describe("Accepted for input parity with jarvis/create-batch-triplet; the vein graph has no scratchpad, so this has no effect."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    let b: GraphBackend;
    let namespace: string;
    try {
      b = await graphCtx(ctx as StepContext<VeinCapabilities>);
      namespace = await b.reader.resolveNamespace(cfg.namespace);
    } catch (e) {
      return errText("graph/create-batch-triplet", e);
    }

    // Inline-node resolution cache: identical sides resolve once per batch.
    const refCache = new Map<string, string>();
    const resolveSide = async (side: "source" | "target", refId?: string, nodeType?: string, nodeData?: Record<string, any>): Promise<string> => {
      if (refId) return refId;
      const key = nodeDedupKey(nodeType!, nodeData!);
      const cached = refCache.get(key);
      if (cached) return cached;
      try {
        const created = (await b.nodes.write({ type: nodeType!, data: nodeData! }, "create", { namespace })).ref_id;
        refCache.set(key, created);
        return created;
      } catch (e) {
        throw new Error(`could not create/merge ${side} node: ${e instanceof Error ? e.message : String(e)}`);
      }
    };

    const results: any[] = [];
    for (const t of cfg.triplets) {
      const invalid =
        validateTripletSide("source", t.source_ref_id, t.source_type, t.source_data) ??
        validateTripletSide("target", t.target_ref_id, t.target_type, t.target_data);
      if (invalid) {
        results.push({ error: `invalid input — ${invalid}`, edge_type: t.edge_type });
        continue;
      }
      const edgeType = t.edge_type.toUpperCase().replace(/ /g, "_");
      try {
        const sourceRef = await resolveSide("source", t.source_ref_id, t.source_type, t.source_data);
        const targetRef = await resolveSide("target", t.target_ref_id, t.target_type, t.target_data);
        let edge;
        try {
          edge = await b.edges.write({
            edge: edgeType,
            source_ref_id: sourceRef,
            target_ref_id: targetRef,
            ...(t.edge_data ? { properties: t.edge_data } : {}),
            ...(t.weight !== undefined ? { weight: t.weight } : {}),
          });
        } catch (e) {
          results.push({
            error: `edge write failed — ${graphErrorCode(e) ? `${graphErrorCode(e)}: ` : ""}${e instanceof Error ? e.message : String(e)}`,
            source_ref_id: sourceRef,
            target_ref_id: targetRef,
            edge_type: edgeType,
          });
          continue;
        }
        results.push({
          status: edge.created ? "Success" : "Warning",
          source_ref_id: edge.source_ref_id,
          target_ref_id: edge.target_ref_id,
          edge_ref_id: edge.ref_id,
          edge_type: edgeType,
        });
      } catch (err: any) {
        results.push({ error: err?.message ?? String(err), edge_type: edgeType });
      }
    }

    const failed = results.filter((r) => r.error).length;
    return { requested: cfg.triplets.length, succeeded: results.length - failed, failed, results };
  },
});
