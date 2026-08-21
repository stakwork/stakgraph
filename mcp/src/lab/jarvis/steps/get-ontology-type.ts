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

export default defineStep({
  type: "jarvis/get-ontology-type",
  description:
    "Fetch the attribute schema for a SINGLE ontology node type. Returns exactly " +
    "one field — `attributes` — and nothing else; for a type's domain, parent or " +
    "description, use jarvis_get_ontology. " +
    "Each attribute value is a type string (e.g. 'string', 'int'); a `?` prefix " +
    "(e.g. '?string') means the attribute is OPTIONAL, no prefix means REQUIRED. " +
    "`attributes` is complete: it already includes everything inherited from parent types. " +
    "Lookup is case-insensitive for every type EXCEPT the root type 'Thing' (exact casing). " +
    "A schema ref_id is also accepted instead of a type name. NODE types only — edge type " +
    "names (e.g. 'KNOWS') are not schema nodes. Call jarvis_get_ontology first if you " +
    "don't already know the exact type name.",
  input: z.object({
    type: z.string().describe(
      "The node type name, e.g. 'Person' (case-insensitive, except the literal root type 'Thing'). A schema ref_id is also accepted.",
    ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const res = await http(`${base}/v2/schema/${encodeURIComponent(cfg.type)}`, { headers, timeout });
    if (!res.ok) return `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    const data = res.body as any;
    // Trim to the attribute schema (own + inherited are already merged into
    // `attributes` on this endpoint). If the shape is unexpected, return it
    // untouched so it stays debuggable.
    if (!data || typeof data !== "object" || Array.isArray(data)) return data;
    return data.attributes !== undefined ? { attributes: data.attributes } : data;
  },
});
