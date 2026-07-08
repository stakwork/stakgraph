import { tool, Tool } from "ai";
import { z } from "zod";
import axios from "axios";

async function jarvisFetch(url: string, headers: Record<string, string>) {
  const resp = await axios.get(url, { headers, validateStatus: () => true, responseType: "text" });
  const text: string = typeof resp.data === "string" ? resp.data : JSON.stringify(resp.data);
  return {
    ok: resp.status >= 200 && resp.status < 300,
    status: resp.status,
    text: async () => text,
    json: async () => JSON.parse(text) as unknown,
  };
}

/** Max neighbors returned in a single hop — keeps tool output within budget. */
const KG_NEIGHBOR_CAP = 50;

/** Max length of a derived label so a single row doesn't flood the context. */
const LABEL_MAX = 160;

/**
 * Node types that must never surface in neighbor expansion — internal /
 * low-signal types (hint nodes, agent memory, media clips, transcript turns).
 * Excluded server-side by Jarvis via `exclude_node_type` before LIMIT.
 */
const EXCLUDED_NODE_TYPES = ["Hint", "Memory", "Clip", "Turn"];

/** Encode an array as a Python list literal, e.g. `["MODIFIES","CITES"]`. */
function toPythonListLiteral(arr: string[]): string {
  return `[${arr.map((s) => `"${s}"`).join(",")}]`;
}

/**
 * Jarvis nodes keep their human label under wildly different keys depending on
 * node type. Try a generous ordered list of candidates — short identifier-like
 * fields first, long descriptive fields as a truncated last resort. Returns ""
 * only when nothing usable exists.
 */
function deriveNodeName(node: any, properties: Record<string, any>): string {
  const candidates = [
    node?.name,
    properties.name,
    properties.title,
    properties.label,
    properties.display_name,
    properties.displayName,
    properties.identifier,
    properties.file_name,
    properties.fileName,
    properties.file,
    properties.path,
    properties.symbol,
    properties.function_name,
    properties.class_name,
    properties.method_name,
    properties.operation_id,
    properties.endpoint,
    properties.route,
    properties.url,
    properties.entity,
    properties.key,
    properties.slug,
    properties.episode_title,
    properties.show_title,
    properties.username,
    properties.email,
    properties.summary,
    properties.description,
    properties.text,
    properties.content,
    properties.body,
    properties.docs,
  ];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim().length > 0) {
      const trimmed = c.trim();
      return trimmed.length > LABEL_MAX ? trimmed.slice(0, LABEL_MAX) : trimmed;
    }
  }
  return "";
}

export interface OntologyNodeType {
  type: string;
  domain: string | null;
  description: string;
}

export interface OntologyEdge {
  edge_type: string;
  source_type: string;
  target_type: string;
}

export interface OntologyPayload {
  domains: string[];
  node_types: Record<string, OntologyNodeType[]>;
  edges: OntologyEdge[];
}

/**
 * Pure transform: given the raw `/v2/schema` response (full mode), build the
 * enriched ontology payload that `get_ontology` returns.
 *
 * - Filters out `type === "*"` and `is_deleted` schema entries.
 * - Lowercases the per-entry `domain` so it matches `graph_search`'s `domains` param.
 * - Groups node types by domain; null-domain types land in the `"ungrouped"` bucket.
 * - `domains` list is the distinct, non-null, lowercased, sorted set.
 * - Edges are deduped compact triples sorted by `edge_type`.
 */
export function buildOntologyPayload(schemaData: any): OntologyPayload {
  const schemas: any[] = schemaData?.schemas ?? [];
  const rawEdges: any[] = schemaData?.edges ?? [];

  // Build node type list
  const nodeTypes: OntologyNodeType[] = schemas
    .filter((s: any) => s.type && s.type !== "*" && !s.is_deleted)
    .map((s: any) => ({
      type: s.type as string,
      domain: s.domain ? (s.domain as string).toLowerCase() : null,
      description: (s.description as string) ?? "",
    }));

  // Derive canonical domains list (distinct, non-null, sorted)
  const domainsSet = new Set<string>();
  for (const nt of nodeTypes) {
    if (nt.domain !== null) domainsSet.add(nt.domain);
  }
  const domains = Array.from(domainsSet).sort();

  // Group node types by domain (null → "ungrouped")
  const grouped: Record<string, OntologyNodeType[]> = {};
  for (const nt of nodeTypes) {
    const key = nt.domain ?? "ungrouped";
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(nt);
  }

  // Build deduped compact edge triples sorted by edge_type
  const edgeSeen = new Set<string>();
  const edges: OntologyEdge[] = [];
  for (const e of rawEdges) {
    const triple: OntologyEdge = {
      edge_type: e.edge_type as string,
      source_type: e.source_type as string,
      target_type: e.target_type as string,
    };
    const key = `${triple.edge_type}|${triple.source_type}|${triple.target_type}`;
    if (!edgeSeen.has(key)) {
      edgeSeen.add(key);
      edges.push(triple);
    }
  }
  edges.sort((a, b) => a.edge_type.localeCompare(b.edge_type));

  return { domains, node_types: grouped, edges };
}

/**
 * Registers Jarvis knowledge-graph tools into the given `allTools` map whenever
 * `JARVIS_URL` is set in the environment. Registers four tools:
 *   - `get_ontology`    — list all available node types in the ontology
 *   - `graph_search`    — keyword search across ontology nodes
 *   - `graph_get`       — resolve a single ref_id to its full node content
 *   - `graph_neighbors` — return all adjacent nodes reachable in one hop
 */
export function registerJarvisTools(
  allTools: Record<string, Tool<any, any>>,
): void {
  const jarvisUrl = process.env.JARVIS_URL;
  if (!jarvisUrl) {
    console.error(
      "[repo agent] JARVIS_URL is not set — skipping Jarvis knowledge-graph tools",
    );
    return;
  }

  const jarvisHeaders = {
    "Content-Type": "application/json",
    "X-Api-Token": process.env.API_TOKEN ?? "",
  };

  allTools.get_ontology = tool({
    description:
      "Fetch the full ontology of the Jarvis knowledge graph: node types (with their domain), " +
      "relationship edges, and the canonical list of valid `domains`. " +
      "Call this once before graph_search to discover valid values for both the `type` and `domains` parameters. " +
      "Node types are grouped by domain; types in the `ungrouped` bucket have no domain and cannot be scoped with `domains`.",
    inputSchema: z.object({}),
    execute: async () => {
      const url = `${jarvisUrl}/v2/schema`;
      console.log(`[get_ontology] fetching ${url}`);
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        return JSON.stringify(buildOntologyPayload(data));
      } catch (err: any) {
        return `get_ontology failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.graph_search = tool({
    description:
      "Search the Jarvis knowledge graph for ontology nodes — people, topics, episodes, clips, organizations, workflows, and more. " +
      "Unlike stakgraph_search (code nodes only), this queries the full Jarvis ontology. " +
      "Call get_ontology first to discover valid values for the `type` parameter.",
    inputSchema: z.object({
      q: z.string().describe("The search query"),
      type: z
        .string()
        .optional()
        .describe(
          "Comma-separated node type filter, e.g. 'Episode' or 'Person,Topic'. " +
          "Call get_ontology to see all valid values."
        ),
      limit: z
        .number()
        .optional()
        .default(10)
        .describe("Maximum number of results to return"),
      domains: z
        .string()
        .optional()
        .describe(
          "Comma-separated domain filter, e.g. 'entity' or 'content,entity'. " +
          "Not required — the search works without it. " +
          "Call `get_ontology` to see valid domains."
        ),
    }),
    execute: async ({
      q,
      type,
      limit = 10,
      domains,
    }: {
      q: string;
      type?: string;
      limit?: number;
      domains?: string;
    }) => {
      const params = new URLSearchParams({ q, limit: String(limit) });
      if (type) params.set("type", type);
      if (domains) params.set("domains", domains);
      const url = `${jarvisUrl}/v2/nodes?${params.toString()}`;
      console.log(
        `[graph_search] q=${q} type=${type ?? "*"} domains=${domains ?? "*"} limit=${limit}`,
      );
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        const nodes: any[] = Array.isArray(data) ? data : (data.nodes ?? []);
        return JSON.stringify(
          nodes.map((n: any) => ({
            ref_id: n.ref_id ?? n.properties?.ref_id,
            name:
              n.properties?.name ??
              n.properties?.episode_title ??
              n.properties?.entity,
            node_type: n.node_type,
            description:
              n.properties?.description ??
              n.properties?.summary ??
              n.properties?.text ??
              "",
          }))
        );
      } catch (err: any) {
        return `graph_search failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.graph_get = tool({
    description:
      "Resolve a single node in the Jarvis knowledge graph to its full content by ref_id. " +
      "Use the ref_id from graph_search or graph_neighbors results. " +
      "Returns the node's ref_id, node_type, derived name, and properties.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the node to resolve."),
    }),
    execute: async ({ ref_id }: { ref_id: string }) => {
      // limit=1 keeps Jarvis from materializing the node's whole neighborhood
      // (which can OOM Neo4j for hub nodes) — we only read the node itself.
      const url = `${jarvisUrl}/v2/nodes/${encodeURIComponent(ref_id)}?limit=1`;
      console.log(`[graph_get] fetching ${url}`);
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        // Deployed Jarvis wraps the node in `{ nodes, edges, status }`; some
        // builds return the node directly. Handle both shapes.
        const raw = Array.isArray(data?.nodes)
          ? data.nodes.find((n: any) => n.ref_id === ref_id) ?? data.nodes[0]
          : data;
        if (!raw || !raw.ref_id) return `node not found: ${ref_id}`;
        const properties = (raw.properties ?? {}) as Record<string, any>;
        return JSON.stringify({
          ref_id: raw.ref_id,
          node_type: raw.node_type,
          name: deriveNodeName(raw, properties),
          properties: raw.properties,
        });
      } catch (err: any) {
        return `graph_get failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.graph_neighbors = tool({
    description:
      "Return all nodes adjacent (one hop) to a node in the Jarvis knowledge graph, " +
      "with edge_type and direction. Use the ref_id from graph_search or graph_get. " +
      "Optionally filter by edge_type and/or node_type. " +
      "Use this to traverse relationships between people, topics, episodes, code, etc.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the node to expand."),
      edge_type: z
        .array(z.string())
        .optional()
        .describe('Filter edges by type, e.g. ["MODIFIES", "CITES"].'),
      node_type: z
        .array(z.string())
        .optional()
        .describe('Filter neighbor nodes by type, e.g. ["File", "Function"].'),
    }),
    execute: async ({
      ref_id,
      edge_type,
      node_type,
    }: {
      ref_id: string;
      edge_type?: string[];
      node_type?: string[];
    }) => {
      // `limit` bounds the Cypher traversal so a hub node doesn't OOM Neo4j.
      // `sort_by=importance` orders edges before LIMIT so the cap keeps the most
      // important neighbors. `canonicalize=false` matches the real Neo4j label.
      const params = new URLSearchParams({
        expand: "edges",
        limit: String(KG_NEIGHBOR_CAP),
        sort_by: "importance",
        canonicalize: "false",
        exclude_node_type: toPythonListLiteral(EXCLUDED_NODE_TYPES),
      });
      if (edge_type && edge_type.length > 0) {
        params.set("edge_type", toPythonListLiteral(edge_type));
      }
      if (node_type && node_type.length > 0) {
        params.set("node_type", toPythonListLiteral(node_type));
      }
      const url = `${jarvisUrl}/v2/nodes/${encodeURIComponent(ref_id)}?${params.toString()}`;
      console.log(
        `[graph_neighbors] ref_id=${ref_id} edge_type=${edge_type?.join(",") ?? "*"} node_type=${node_type?.join(",") ?? "*"}`,
      );
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;

        // Look up node details by ref_id, excluding the queried node itself, so
        // each neighbor carries a human-readable label alongside its ref_id.
        const nodeMap = new Map<string, { node_type: string; name: string }>();
        for (const node of data.nodes ?? []) {
          if (node.ref_id !== ref_id) {
            nodeMap.set(node.ref_id, {
              node_type: node.node_type,
              name: deriveNodeName(node, (node.properties ?? {}) as Record<string, any>),
            });
          }
        }

        const neighbors: any[] = [];
        const seen = new Set<string>();
        for (const edge of data.edges ?? []) {
          const direction = edge.source === ref_id ? "forward" : "reverse";
          const neighborRefId = direction === "forward" ? edge.target : edge.source;
          // Self-loop guard / source dedup.
          if (neighborRefId === ref_id) continue;
          // A node can be reached via multiple parallel edges — keep the first.
          if (seen.has(neighborRefId)) continue;
          seen.add(neighborRefId);

          const detail = nodeMap.get(neighborRefId);
          const importance = edge.properties?.importance as number | undefined;
          neighbors.push({
            ref_id: neighborRefId,
            node_type: detail?.node_type ?? "unknown",
            name: detail?.name ?? "",
            edge_type: edge.edge_type,
            direction,
            ...(importance !== undefined ? { importance } : {}),
          });
          if (neighbors.length >= KG_NEIGHBOR_CAP) break;
        }

        return JSON.stringify(neighbors);
      } catch (err: any) {
        return `graph_neighbors failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  console.log(
    "===> registered graph_search + get_ontology + graph_get + graph_neighbors tools",
  );
}
