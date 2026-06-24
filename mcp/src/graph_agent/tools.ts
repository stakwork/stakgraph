import { tool } from "ai";
import { z } from "zod";
import axios from "axios";

const JARVIS_URL = process.env.JARVIS_URL ?? "";

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

/** Build Authorization header from the forwarded L402 token (if present). */
function authHeaders(authToken?: string): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (authToken) {
    headers["Authorization"] = authToken;
  }
  return headers;
}

export type GraphToolsConfig = {
  authToken?: string; // forwarded L402 / Authorization header
};

/**
 * Create the three graph agent tools with the caller's auth token baked in.
 */
export function get_graph_tools(config: GraphToolsConfig = {}) {
  if (!JARVIS_URL) {
    console.error(
      "[graph_agent] JARVIS_URL is not set — graph_search / graph_node / graph_neighbors will fail"
    );
  }
  const { authToken } = config;

  // ── graph_search ────────────────────────────────────────────────────────
  const graph_search = tool({
    description:
      "Search the knowledge graph by keyword, semantic meaning, or hybrid. " +
      "Returns a list of matching nodes with ref_id, name, type, and description. " +
      "Always call graph_node with a ref_id from these results to get full details.",
    inputSchema: z.object({
      q: z.string().describe("The search query"),
      search_method: z
        .enum(["keyword", "semantic", "hybrid"])
        .optional()
        .default("hybrid")
        .describe("Search method to use"),
      type: z
        .string()
        .optional()
        .describe("Node type filter, e.g. 'Episode', 'Topic', 'Clip'"),
      limit: z
        .number()
        .optional()
        .default(10)
        .describe("Maximum number of results to return"),
    }),
    execute: async ({ q, search_method = "hybrid", type, limit = 10 }: { q: string; search_method?: string; type?: string; limit?: number }) => {
      const start = Date.now();
      const params = new URLSearchParams({ q, search_method: search_method ?? "hybrid", limit: String(limit) });
      if (type) params.set("type", type);
      const url = `${JARVIS_URL}/v2/nodes?${params.toString()}`;
      console.log(`[graph_search] q=${q} method=${search_method} type=${type ?? "*"} limit=${limit} url=${url}`);
      try {
        const resp = await jarvisFetch(url, authHeaders(authToken) as Record<string, string>);
        if (!resp.ok) {
          const text = await resp.text();
          console.error(`[graph_search] HTTP ${resp.status}: ${text}`);
          return { error: `HTTP ${resp.status}: ${text}`, nodes: [] };
        }
        const data = await resp.json() as any;
        const nodes: any[] = Array.isArray(data) ? data : (data.nodes ?? data.results ?? []);
        const ms = Date.now() - start;
        console.log(`[graph_search] returned ${nodes.length} nodes in ${ms}ms`);
        return { nodes, count: nodes.length };
      } catch (err: any) {
        const ms = Date.now() - start;
        console.error(`[graph_search] Error after ${ms}ms:`, err?.message, err?.cause);
        return { error: err?.message ?? String(err), nodes: [] };
      }
    },
  });

  // ── graph_node ───────────────────────────────────────────────────────────
  const graph_node = tool({
    description:
      "Fetch the full details of a single graph node by its ref_id. " +
      "Use this after graph_search returns results — pass the ref_id to get complete node data. " +
      "Always use nodes returned here as citations in your final answer.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the node to fetch"),
    }),
    execute: async ({ ref_id }: { ref_id: string }) => {
      const start = Date.now();
      const url = `${JARVIS_URL}/v2/nodes/${encodeURIComponent(ref_id)}`;
      console.log(`[graph_node] ref_id=${ref_id} url=${url}`);
      try {
        const resp = await jarvisFetch(url, authHeaders(authToken) as Record<string, string>);
        if (!resp.ok) {
          const text = await resp.text();
          console.error(`[graph_node] HTTP ${resp.status}: ${text}`);
          return { error: `HTTP ${resp.status}: ${text}`, node: null };
        }
        const node = await resp.json();
        const ms = Date.now() - start;
        console.log(`[graph_node] fetched ref_id=${ref_id} in ${ms}ms`);
        return { node };
      } catch (err: any) {
        const ms = Date.now() - start;
        console.error(`[graph_node] Error after ${ms}ms:`, err?.message, err?.cause);
        return { error: err?.message ?? String(err), node: null };
      }
    },
  });

  // ── graph_neighbors ──────────────────────────────────────────────────────
  const graph_neighbors = tool({
    description:
      "Explore the 1-hop neighborhood (related nodes) of a graph node. " +
      "Use this to discover connected topics, episodes, clips, or other related content. " +
      "Provide the ref_id from graph_search or graph_node results.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the node whose neighbors to explore"),
    }),
    execute: async ({ ref_id }: { ref_id: string }) => {
      const start = Date.now();
      const url = `${JARVIS_URL}/v2/nodes/${encodeURIComponent(ref_id)}?expand=edges`;
      console.log(`[graph_neighbors] ref_id=${ref_id} url=${url}`);
      try {
        const resp = await jarvisFetch(url, authHeaders(authToken) as Record<string, string>);

        if (!resp.ok) {
          const text = await resp.text();
          console.error(`[graph_neighbors] HTTP ${resp.status}: ${text}`);
          return { error: `HTTP ${resp.status}: ${text}`, neighbors: [] };
        }

        const data = await resp.json() as any;
        const neighbors: any[] = Array.isArray(data)
          ? data
          : (data.neighbors ?? data.nodes ?? data.edges ?? []);
        const ms = Date.now() - start;
        console.log(`[graph_neighbors] ref_id=${ref_id} returned ${neighbors.length} neighbors in ${ms}ms`);
        return { neighbors, count: neighbors.length };
      } catch (err: any) {
        const ms = Date.now() - start;
        console.error(`[graph_neighbors] Error after ${ms}ms:`, err?.message, err?.cause);
        return { error: err?.message ?? String(err), neighbors: [] };
      }
    },
  });

  return { graph_search, graph_node, graph_neighbors };
}
