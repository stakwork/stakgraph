import { tool, Tool } from "ai";
import { z } from "zod";

/**
 * Registers Jarvis ontology tools (`get_ontology` and `graph_search`) into the
 * given `allTools` map when both `JARVIS_URL` is set in the environment and
 * the caller has enabled them via `toolsConfig.jarvis`.
 *
 * Note: `jarvis` is a virtual toggle in `ToolsConfig` — it is not itself a
 * registered tool. It only gates the registration of the two real tools below.
 */
export function registerJarvisTools(
  allTools: Record<string, Tool<any, any>>,
  jarvisEnabled: boolean | undefined,
): void {
  const jarvisUrl = process.env.JARVIS_URL;
  if (!jarvisUrl || !jarvisEnabled) {
    if (!jarvisUrl) {
      console.log(
        "===> no JARVIS_URL set, skipping graph_search + get_ontology tools",
      );
    } else {
      console.log(
        "===> jarvis not enabled in toolsConfig, skipping graph_search + get_ontology tools",
      );
    }
    return;
  }

  const jarvisHeaders = {
    "Content-Type": "application/json",
    "X-Api-Token": process.env.API_TOKEN ?? "",
  };

  allTools.get_ontology = tool({
    description:
      "Fetch the list of all available node types in the Jarvis knowledge graph ontology. " +
      "Call this before graph_search to discover valid values for the `type` parameter.",
    inputSchema: z.object({}),
    execute: async () => {
      const url = `${jarvisUrl}/v2/schema?concise=true`;
      console.log(`[get_ontology] fetching ${url}`);
      try {
        const resp = await fetch(url, { headers: jarvisHeaders });
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        const schemas: any[] = data.schemas ?? [];
        return JSON.stringify(
          schemas
            .filter((s: any) => s.type && !s.is_deleted)
            .map((s: any) => s.type)
            .sort()
        );
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
          "Valid values: entity, content, knowledgeartifact, workflow, codeartifact."
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
        const resp = await fetch(url, { headers: jarvisHeaders });
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

  console.log("===> registered graph_search + get_ontology tools");
}
