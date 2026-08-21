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

/** Created/merged node ref_id from a Jarvis POST /v2/nodes response body. */
function extractNodeRefId(body: any): string | undefined {
  const refId = body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

/** Edge ref_id from a Jarvis POST /v2/edges response body. */
function extractEdgeRefId(body: any): string | undefined {
  const refId = body?.edges?.[0]?.ref_id ?? body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
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
  type: "jarvis/create-batch-triplet",
  description:
    "Assert MANY facts into the Jarvis knowledge graph in a single call. " +
    "Each item in `triplets` has the same shape as jarvis_create_triplet (source/target as ref_id or inline " +
    "node_type+node_data, plus edge_type and optional edge_data/weight/create_schema_if_missing). " +
    "A single top-level `namespace` applies to all inline node creation. " +
    "REUSE existing nodes wherever possible: supply ref_ids from jarvis_graph_search when entities already exist — " +
    "inline creation is a last resort. Identical inline sides across the batch are resolved once (deduped). " +
    "Triplets are written sequentially in input order; the result is a per-triplet array in the same order, " +
    "each entry either the created edge info or `{ error }` — one failed triplet never fails the rest.",
  input: z.object({
    triplets: z.array(TripletSchema).min(1).describe("The facts to assert, in order."),
    namespace: z
      .string()
      .optional()
      .describe("Jarvis namespace (data partition) for all inline node creation. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);

    // Inline-node resolution cache: identical sides resolve once per batch.
    const refCache = new Map<string, string>();

    const resolveSide = async (
      side: "source" | "target",
      refId?: string,
      nodeType?: string,
      nodeData?: Record<string, any>,
    ): Promise<string> => {
      if (refId) return refId;
      const key = nodeDedupKey(nodeType!, nodeData!);
      const cached = refCache.get(key);
      if (cached) return cached;
      const query: Record<string, string> = {};
      if (cfg.namespace) query.namespace = cfg.namespace;
      const res = await http(`${base}/v2/nodes`, {
        method: "POST",
        headers,
        query,
        timeout,
        body: { node_type: nodeType, node_data: nodeData },
      });
      const created = extractNodeRefId(res.body);
      if (!created) {
        throw new Error(`could not create/merge ${side} node (HTTP ${res.status}): ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`);
      }
      refCache.set(key, created);
      return created;
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
      try {
        const sourceRef = await resolveSide("source", t.source_ref_id, t.source_type, t.source_data);
        const targetRef = await resolveSide("target", t.target_ref_id, t.target_type, t.target_data);
        const res = await http(`${base}/v2/edges`, {
          method: "POST",
          headers,
          timeout,
          body: {
            edge: {
              edge_type: t.edge_type,
              ...(t.weight !== undefined ? { weight: t.weight } : {}),
              ...(t.edge_data ? { edge_data: t.edge_data } : {}),
            },
            source: { ref_id: sourceRef },
            target: { ref_id: targetRef },
            create_schema_if_missing: t.create_schema_if_missing ?? false,
          },
        });
        const body = res.body as any;
        const edgeRef = extractEdgeRefId(body);
        if (!res.ok || !edgeRef) {
          results.push({
            error: `edge write failed — HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`,
            source_ref_id: sourceRef,
            target_ref_id: targetRef,
            edge_type: t.edge_type,
          });
          continue;
        }
        results.push({
          status: body?.status ?? "Success",
          source_ref_id: sourceRef,
          target_ref_id: targetRef,
          edge_ref_id: edgeRef,
          edge_type: t.edge_type,
        });
      } catch (err: any) {
        results.push({ error: err?.message ?? String(err), edge_type: t.edge_type });
      }
    }

    const failed = results.filter((r) => r.error).length;
    return { requested: cfg.triplets.length, succeeded: results.length - failed, failed, results };
  },
});
