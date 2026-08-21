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

/** Build the enriched ontology payload from the raw /v2/schema response:
 *  filter "*"/deleted schemas, group node types by (lowercased) domain
 *  ("ungrouped" for null), derive the sorted domains list, and optionally
 *  append deduped compact edge triples sorted by edge_type. */
function buildOntologyPayload(schemaData: any, includeEdges: boolean, includeAttributes: boolean) {
  const schemas: any[] = schemaData?.schemas ?? [];
  const rawEdges: any[] = schemaData?.edges ?? [];

  const nodeTypes = schemas
    .filter((s: any) => s.type && s.type !== "*" && !s.is_deleted)
    .map((s: any) => {
      const td = (s.type_description as string) ?? "";
      const desc = (s.description as string) ?? "";
      return {
        type: s.type as string,
        _domain: s.domain ? (s.domain as string).toLowerCase() : null,
        description: td.trim() !== "" ? td : desc,
        ...(includeAttributes && {
          attributes: (s.attributes ?? {}) as Record<string, string>,
          inherited_attributes: (s.inherited_attributes ?? {}) as Record<string, string>,
        }),
      };
    });

  const domains = Array.from(new Set(nodeTypes.map((n) => n._domain).filter((d): d is string => d !== null))).sort();

  const grouped: Record<string, any[]> = {};
  for (const { _domain, ...entry } of nodeTypes) {
    const key = _domain ?? "ungrouped";
    (grouped[key] ??= []).push(entry);
  }

  if (!includeEdges) return { domains, node_types: grouped };

  const edgeSeen = new Set<string>();
  const edges: Array<{ edge_type: string; source_type: string; target_type: string }> = [];
  for (const e of rawEdges) {
    const triple = { edge_type: e.edge_type as string, source_type: e.source_type as string, target_type: e.target_type as string };
    const key = `${triple.edge_type}|${triple.source_type}|${triple.target_type}`;
    if (!edgeSeen.has(key)) {
      edgeSeen.add(key);
      edges.push(triple);
    }
  }
  edges.sort((a, b) => a.edge_type.localeCompare(b.edge_type));
  return { domains, node_types: grouped, edges };
}

export default defineStep({
  type: "jarvis/get-ontology",
  description:
    "Fetch the ontology of the Jarvis knowledge graph: node types grouped by domain " +
    "and the canonical list of valid `domains`. " +
    "Call this once before jarvis_graph_search to discover valid values for both the `type` and `domains` parameters. " +
    "Node types are grouped by domain key in `node_types[<domain>]`; types with no domain land in the `ungrouped` bucket. " +
    "Pass `domains` to filter results (comma-separated, e.g. 'Legal,Entity'); omit to receive all domains. " +
    "Relationship edges are omitted by default — jarvis_graph_neighbors returns edge types live as you traverse. " +
    "Set `include_edges` to also get the full relationship map (source_type -> target_type triples). " +
    "Set `include_attributes` to also get each node type's attribute schema (field names, types, required/optional status). " +
    "WILDCARD EDGES: when include_edges is true, an edge entry whose source_type and/or target_type is \"*\" " +
    "means that edge type applies to ANY node type on that side. \"*\" is intentionally absent from node_types — " +
    "it is a backend sentinel, not a real type.",
  input: z.object({
    domains: z
      .string()
      .optional()
      .describe(
        "Comma-separated list of domains to filter results to (e.g. 'Legal,Entity'). " +
        "Omit to receive node types from all domains. Matched case-insensitively.",
      ),
    include_edges: z
      .boolean()
      .optional()
      .default(false)
      .describe(
        "Include the full list of relationship edges (source_type/edge_type/target_type triples). " +
        "Off by default — the edge list is large and jarvis_graph_neighbors surfaces edge types live.",
      ),
    include_attributes: z
      .boolean()
      .optional()
      .default(false)
      .describe(
        "Include each node type's attribute schema maps (`attributes` and `inherited_attributes`). " +
        "Off by default to keep the payload lean. A `?` prefix on a value type (e.g. '?string') means optional.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { base, http, timeout, headers } = await jarvisCtx(ctx as StepContext<VeinCapabilities>);
    const query: Record<string, string | boolean> = {
      include_edges: cfg.include_edges ?? false,
      include_attributes: cfg.include_attributes ?? false,
    };
    if (cfg.domains && cfg.domains.trim() !== "") query.domains = cfg.domains.trim();
    const res = await http(`${base}/v2/schema`, { headers, query, timeout });
    if (!res.ok) return `HTTP ${res.status}: ${typeof res.body === "string" ? res.body : JSON.stringify(res.body)}`;
    return buildOntologyPayload(res.body, cfg.include_edges ?? false, cfg.include_attributes ?? false);
  },
});
