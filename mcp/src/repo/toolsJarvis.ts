import { tool, Tool, ToolLoopAgent, stepCountIs } from "ai";
import { z } from "zod";
import axios from "axios";
import {
  getModelDetails,
  getProviderOptions,
  ModelName,
} from "../aieo/src/index.js";
import {
  extractFinalAnswer,
  createHasEndMarkerCondition,
} from "./utils.js";

function appendNamespace(params: URLSearchParams, namespace?: string): void {
  if (namespace && namespace.length > 0) {
    params.set("namespace", namespace);
  }
}

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

/**
 * Perform a write (POST/PUT/DELETE) against Jarvis. Mirrors `jarvisFetch` but
 * for mutations. Never throws on non-2xx (validateStatus) so the tool can
 * surface Jarvis's `errorCode`/`message` body back to the agent verbatim.
 */
async function jarvisMutate(
  method: "post" | "put" | "delete",
  url: string,
  headers: Record<string, string>,
  body?: unknown,
) {
  const resp = await axios.request({
    method,
    url,
    headers,
    data: body,
    validateStatus: () => true,
    responseType: "text",
  });
  const text: string = typeof resp.data === "string" ? resp.data : JSON.stringify(resp.data);
  return {
    ok: resp.status >= 200 && resp.status < 300,
    status: resp.status,
    text,
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
 * Collapse Jarvis `/connection-counts` rows ([{edge_type, target_type, count}])
 * into a compact `{EDGE_TYPE: totalCount}` map, summing across target types.
 * This mirrors the inline `edges` map returned by graph_search so both tools
 * present connectivity the same way.
 */
export function collapseConnectionCounts(
  counts: Array<{ edge_type: string; target_type?: string; count: number }>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of counts ?? []) {
    if (!c?.edge_type) continue;
    out[c.edge_type] = (out[c.edge_type] ?? 0) + Number(c.count ?? 0);
  }
  return out;
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
  edges?: OntologyEdge[];
}

/**
 * Pure transform: given the raw `/v2/schema` response (full mode), build the
 * enriched ontology payload that `get_ontology` returns.
 *
 * - Filters out `type === "*"` and `is_deleted` schema entries.
 * - Lowercases the per-entry `domain` so it matches `graph_search`'s `domains` param.
 * - Groups node types by domain; null-domain types land in the `"ungrouped"` bucket.
 * - `domains` list is the distinct, non-null, lowercased, sorted set.
 * - Edges are omitted by default (they dominate the payload); pass
 *   `includeEdges` to append deduped compact triples sorted by `edge_type`.
 */
export function buildOntologyPayload(
  schemaData: any,
  includeEdges = false,
): OntologyPayload {
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

  if (!includeEdges) {
    return { domains, node_types: grouped };
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

/** Default recursion depth for nested `graph_sub_agent` spawning. */
const DEFAULT_SUBAGENT_MAX_DEPTH = 2;

/** Default tool-loop step cap for a single sub-agent run. */
const DEFAULT_SUBAGENT_MAX_STEPS = 20;

/**
 * Config for the recursive `graph_sub_agent` tool. When present (see
 * `JarvisToolsOptions.subAgent`) a child agent tool is registered that spawns an
 * in-process ToolLoopAgent with its own copy of the graph tools.
 */
export interface JarvisSubAgentConfig {
  /** Override the tool description shown to the parent LLM. */
  description?: string;
  /**
   * Max nesting depth for sub-agents. depth 0 = the top-level agent, so a
   * maxDepth of 2 means the top agent can spawn children, those children can
   * spawn grandchildren, but grandchildren get no `graph_sub_agent` tool.
   * Defaults to `DEFAULT_SUBAGENT_MAX_DEPTH`.
   */
  maxDepth?: number;
  /** Max tool-loop steps a single sub-agent run may take. */
  maxSteps?: number;
  /** LLM selection forwarded to child agents (reuses the parent's provider). */
  modelName?: ModelName;
  apiKey?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
  /**
   * Current recursion depth. Internal — callers should leave this unset (0);
   * it is incremented automatically as sub-agents spawn sub-agents.
   */
  depth?: number;
}

export interface JarvisToolsOptions {
  /**
   * When provided, registers the recursive `graph_sub_agent` tool alongside the
   * read tools. Omit to expose only the read tools (get_ontology, graph_search,
   * graph_get, graph_neighbors).
   */
  subAgent?: JarvisSubAgentConfig;
  /**
   * When true, registers the ontology WRITE tools (ontology_create_type,
   * ontology_update_type, ontology_delete_type, ontology_create_edge,
   * ontology_update_edge, ontology_delete_edge, ontology_rename_attribute)
   * that POST/PUT/DELETE directly against Jarvis `/v2/schema`. Opt-in and
   * off by default — the default posture stays read-only.
   */
  ontologyEdit?: boolean;
}

const DEFAULT_SUBAGENT_DESCRIPTION =
  "Spawn a focused child agent to explore the Jarvis knowledge graph and report back. " +
  "The child has its own copy of the graph tools (get_ontology, graph_search, graph_get, graph_neighbors) " +
  "and runs an independent exploration loop, returning a synthesized text summary of its findings. " +
  "Use this to parallelize or delegate: after you locate a few key nodes, fan out one sub-agent per " +
  "node/subtopic with a specific, self-contained prompt (include the relevant ref_ids and exactly what " +
  "to find), then collate their answers. Each prompt must stand alone — the child does not see this " +
  "conversation. Prefer a handful of targeted sub-agents over one broad one.";

/** System prompt for a spawned graph exploration sub-agent. */
const GRAPH_SUBAGENT_SYSTEM = `You are a focused knowledge-graph exploration sub-agent. A parent agent has delegated a specific exploration task to you. Answer ONLY the task you were given — do not expand scope.

You traverse a knowledge graph of interconnected entities (people, topics, episodes, organizations, workflows, code, and their relationships) using these tools:
- \`get_ontology\` — list node types (grouped by domain) and valid \`domains\`. Call FIRST if you don't already know the relevant types.
- \`graph_search\` — keyword search. Returns compact results (ref_id, name, node_type, description, edges). Scope with \`type\`/\`domains\`.
- \`graph_neighbors\` — nodes one hop away, with \`edge_type\` and \`direction\`. This is how you follow relationships.
- \`graph_get\` — resolve a single ref_id to its full content.
- \`graph_sub_agent\` (only if available) — delegate an even more focused subtask to a further child agent.

Workflow:
1. If the parent gave you ref_ids, start with \`graph_get\`/\`graph_neighbors\` on them. Otherwise start with \`graph_search\`.
2. Walk outward hop-by-hop, filtering by \`node_type\`/\`edge_type\`, following the \`name\` labels to decide where to go.
3. Stop calling tools as soon as you have enough to answer — extra calls rarely improve a complete answer.

Be efficient and concrete. Cite the node names/ref_ids you relied on so the parent can verify or dig deeper.

CRITICAL: When ready, output your complete findings followed by [END_OF_ANSWER] on a new line. Always finish with this marker.`;

/**
 * Register the recursive `graph_sub_agent` tool. Each invocation builds a fresh
 * child tool set (via `registerJarvisTools` with an incremented depth) and runs
 * an in-process ToolLoopAgent, returning the child's synthesized text answer.
 * Depth capping happens at the call site (only registered while depth < maxDepth),
 * so leaf children never receive another `graph_sub_agent` tool.
 */
function registerGraphSubAgentTool(
  allTools: Record<string, Tool<any, any>>,
  sub: JarvisSubAgentConfig,
  depth: number,
): void {
  allTools.graph_sub_agent = tool({
    description: sub.description ?? DEFAULT_SUBAGENT_DESCRIPTION,
    inputSchema: z.object({
      prompt: z
        .string()
        .describe(
          "A focused, self-contained exploration task for the child agent. " +
          "Include any relevant ref_ids and state exactly what to find and report back. " +
          "The child cannot see this conversation.",
        ),
    }),
    execute: async ({ prompt }: { prompt: string }) => {
      const childTools: Record<string, Tool<any, any>> = {};
      // Recurse with depth+1 so nested sub-agents stop at maxDepth.
      registerJarvisTools(childTools, {
        subAgent: { ...sub, depth: depth + 1 },
      });

      try {
        const { model, provider, modelId } = getModelDetails(
          sub.modelName,
          sub.apiKey,
          sub.baseUrl,
          sub.headers,
        );
        const maxSteps = sub.maxSteps ?? DEFAULT_SUBAGENT_MAX_STEPS;
        const hasEndMarker = createHasEndMarkerCondition<typeof childTools>();
        const agent = new ToolLoopAgent({
          model,
          instructions: GRAPH_SUBAGENT_SYSTEM,
          tools: childTools,
          providerOptions: getProviderOptions(provider, undefined, modelId) as any,
          stopWhen: maxSteps > 0 ? [hasEndMarker, stepCountIs(maxSteps)] : hasEndMarker,
          stopSequences: ["[END_OF_ANSWER]"],
        });
        console.log(
          `[graph_sub_agent] depth=${depth + 1} spawning child: ${prompt.slice(0, 200)}`,
        );
        const result = await agent.generate({ prompt });
        const final = extractFinalAnswer(result.steps);
        return final.answer || "Sub-agent returned no findings.";
      } catch (err: any) {
        return `graph_sub_agent failed: ${err?.message ?? String(err)}`;
      }
    },
  });
  console.log(`===> registered graph_sub_agent tool (depth ${depth})`);
}

/** Human-readable summary of a Jarvis mutation response for the agent. */
function formatMutationResult(
  label: string,
  res: { ok: boolean; status: number; text: string },
): string {
  if (res.ok) {
    return `${label} succeeded (HTTP ${res.status}): ${res.text}`;
  }
  return `${label} failed — HTTP ${res.status}: ${res.text}`;
}

/**
 * Valid attribute type descriptors for schema node/edge attributes. Prefix any
 * value with `?` to mark it optional (e.g. `?string`). `delete` removes an
 * existing attribute on an update.
 */
const ATTRIBUTE_TYPES_DOC =
  'Map of attribute name → type descriptor. Valid types: "string", "boolean", ' +
  '"int", "float", "datetime", "list", "complex". Prefix with "?" to make it ' +
  'optional (e.g. "?string"). Use "delete" as the value to remove an attribute ' +
  "on an update. Attribute names cannot contain '-' or be reserved system " +
  "properties (status, is_deleted, boost, algo_*).";

/**
 * Register the ontology WRITE tools. Each tool POST/PUT/DELETEs directly
 * against the Jarvis `/v2/schema` endpoints (which already apply changes live
 * to Neo4j and invalidate caches). Gated by `JarvisToolsOptions.ontologyEdit`.
 */
function registerOntologyWriteTools(
  allTools: Record<string, Tool<any, any>>,
  jarvisUrl: string,
  jarvisHeaders: Record<string, string>,
): void {
  const schemaUrl = `${jarvisUrl}/v2/schema`;

  allTools.ontology_create_type = tool({
    description:
      "Create a new NODE TYPE in the Jarvis ontology (writes live to the graph). " +
      "Call get_ontology first to confirm the type does not already exist and to pick a valid `parent`. " +
      "Node types form an inheritance tree rooted at `Thing`; children inherit parent attributes.",
    inputSchema: z.object({
      type: z.string().describe("The new node type name, e.g. 'Statute' (PascalCase)."),
      parent: z
        .string()
        .describe("The parent node type it inherits from (e.g. 'Thing'). Required."),
      attributes: z
        .record(z.string(), z.string())
        .describe(ATTRIBUTE_TYPES_DOC),
      node_key: z
        .string()
        .describe(
          "A unique key for the type — usually one of the attribute names used to identify a node. " +
          "Jarvis prefixes it with the lowercased type automatically.",
        ),
      domain: z
        .string()
        .optional()
        .describe("Domain grouping (defaults to 'entity'). Call get_ontology to see existing domains."),
      description: z.string().optional().describe("Human-readable description of the type."),
    }),
    execute: async (input: {
      type: string;
      parent: string;
      attributes: Record<string, string>;
      node_key: string;
      domain?: string;
      description?: string;
    }) => {
      console.log(`[ontology_create_type] type=${input.type} parent=${input.parent}`);
      try {
        const res = await jarvisMutate("post", schemaUrl, jarvisHeaders, input);
        return formatMutationResult(`create node type '${input.type}'`, res);
      } catch (err: any) {
        return `ontology_create_type failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_update_type = tool({
    description:
      "Update an existing NODE TYPE in the Jarvis ontology by ref_id (writes live to the graph). " +
      "Use to add/change attributes, description, or domain. To remove an attribute, set its value to 'delete'. " +
      "Get the ref_id from get_ontology (include_edges) or graph_get.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the schema node type to update."),
      type: z
        .string()
        .optional()
        .describe("New type name — only pass this when RENAMING the type."),
      attributes: z
        .record(z.string(), z.string())
        .optional()
        .describe(ATTRIBUTE_TYPES_DOC),
      domain: z.string().optional().describe("New domain grouping."),
      description: z.string().optional().describe("New description."),
    }),
    execute: async (input: {
      ref_id: string;
      type?: string;
      attributes?: Record<string, string>;
      domain?: string;
      description?: string;
    }) => {
      const { ref_id, ...body } = input;
      console.log(`[ontology_update_type] ref_id=${ref_id}`);
      try {
        const url = `${schemaUrl}/${encodeURIComponent(ref_id)}`;
        const res = await jarvisMutate("put", url, jarvisHeaders, body);
        return formatMutationResult(`update node type ${ref_id}`, res);
      } catch (err: any) {
        return `ontology_update_type failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_delete_type = tool({
    description:
      "Soft-delete a NODE TYPE from the Jarvis ontology (sets is_deleted=true; writes live to the graph). " +
      "DESTRUCTIVE — only call after the user has explicitly confirmed. " +
      "Accepts either the ref_id or the type name.",
    inputSchema: z.object({
      ref_id_or_type: z
        .string()
        .describe("The ref_id or the type name of the node type to soft-delete."),
    }),
    execute: async ({ ref_id_or_type }: { ref_id_or_type: string }) => {
      console.log(`[ontology_delete_type] ${ref_id_or_type}`);
      try {
        const url = `${schemaUrl}/${encodeURIComponent(ref_id_or_type)}`;
        const res = await jarvisMutate("delete", url, jarvisHeaders);
        return formatMutationResult(`delete node type '${ref_id_or_type}'`, res);
      } catch (err: any) {
        return `ontology_delete_type failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_create_edge = tool({
    description:
      "Create a new EDGE TYPE (relationship) between two node types in the Jarvis ontology (writes live to the graph). " +
      "Call get_ontology first to confirm both source and target types exist. " +
      "Use '*' for source or target to define a wildcard relationship rule.",
    inputSchema: z.object({
      source: z.string().describe("Source node type (or '*' wildcard)."),
      target: z.string().describe("Target node type (or '*' wildcard)."),
      edge_type: z
        .string()
        .describe("The relationship name, e.g. 'CITES'. Uppercased with spaces→underscores by Jarvis."),
      attributes: z
        .record(z.string(), z.string())
        .optional()
        .describe(ATTRIBUTE_TYPES_DOC),
      display_name: z.string().optional().describe("Human-readable label for the edge."),
      temporal: z
        .boolean()
        .optional()
        .describe(
          "When true, Jarvis auto-adds bitemporal attributes (valid_at, invalid_at, expired_at, etc.).",
        ),
    }),
    execute: async (input: {
      source: string;
      target: string;
      edge_type: string;
      attributes?: Record<string, string>;
      display_name?: string;
      temporal?: boolean;
    }) => {
      console.log(
        `[ontology_create_edge] ${input.source} -[${input.edge_type}]-> ${input.target}`,
      );
      try {
        const url = `${schemaUrl}/edge`;
        const res = await jarvisMutate("post", url, jarvisHeaders, input);
        return formatMutationResult(
          `create edge '${input.source}-[${input.edge_type}]->${input.target}'`,
          res,
        );
      } catch (err: any) {
        return `ontology_create_edge failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_update_edge = tool({
    description:
      "Update an existing EDGE TYPE in the Jarvis ontology by ref_id (writes live to the graph). " +
      "Use to rename the edge_type, change its display_name, or add/change attributes. " +
      "Get the ref_id from get_ontology (include_edges). CHILD_OF edges cannot be modified.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the edge schema to update."),
      edge_type: z
        .string()
        .optional()
        .describe("New relationship name — only pass this when RENAMING the edge."),
      display_name: z.string().optional().describe("New human-readable label."),
      attributes: z
        .record(z.string(), z.string())
        .optional()
        .describe(ATTRIBUTE_TYPES_DOC),
    }),
    execute: async (input: {
      ref_id: string;
      edge_type?: string;
      display_name?: string;
      attributes?: Record<string, string>;
    }) => {
      const { ref_id, ...body } = input;
      console.log(`[ontology_update_edge] ref_id=${ref_id}`);
      try {
        const url = `${schemaUrl}/edge/${encodeURIComponent(ref_id)}`;
        const res = await jarvisMutate("put", url, jarvisHeaders, body);
        return formatMutationResult(`update edge ${ref_id}`, res);
      } catch (err: any) {
        return `ontology_update_edge failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_delete_edge = tool({
    description:
      "Soft-delete an EDGE TYPE from the Jarvis ontology by ref_id (sets is_deleted=true; writes live to the graph). " +
      "DESTRUCTIVE — only call after the user has explicitly confirmed. CHILD_OF edges cannot be deleted.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the edge schema to soft-delete."),
    }),
    execute: async ({ ref_id }: { ref_id: string }) => {
      console.log(`[ontology_delete_edge] ref_id=${ref_id}`);
      try {
        const url = `${schemaUrl}/edge/${encodeURIComponent(ref_id)}`;
        const res = await jarvisMutate("delete", url, jarvisHeaders);
        return formatMutationResult(`delete edge ${ref_id}`, res);
      } catch (err: any) {
        return `ontology_delete_edge failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.ontology_rename_attribute = tool({
    description:
      "Rename an attribute on a NODE TYPE and migrate all existing node data to the new name (writes live to the graph). " +
      "DESTRUCTIVE data migration — only call after the user has explicitly confirmed. " +
      "Get the ref_id from get_ontology (include_edges) or graph_get.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the schema node type."),
      current_attribute: z.string().describe("The existing attribute name to rename."),
      new_attribute: z.string().describe("The new attribute name."),
    }),
    execute: async (input: {
      ref_id: string;
      current_attribute: string;
      new_attribute: string;
    }) => {
      const { ref_id, ...body } = input;
      console.log(
        `[ontology_rename_attribute] ref_id=${ref_id} ${input.current_attribute}→${input.new_attribute}`,
      );
      try {
        const url = `${schemaUrl}/${encodeURIComponent(ref_id)}/attribute`;
        const res = await jarvisMutate("put", url, jarvisHeaders, body);
        return formatMutationResult(
          `rename attribute '${input.current_attribute}'→'${input.new_attribute}' on ${ref_id}`,
          res,
        );
      } catch (err: any) {
        return `ontology_rename_attribute failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  console.log(
    "===> registered ontology write tools: ontology_create_type, ontology_update_type, " +
    "ontology_delete_type, ontology_create_edge, ontology_update_edge, ontology_delete_edge, " +
    "ontology_rename_attribute",
  );
}

/**
 * Registers Jarvis knowledge-graph tools into the given `allTools` map whenever
 * `JARVIS_URL` is set in the environment. Registers four read tools:
 *   - `get_ontology`    — list all available node types in the ontology
 *   - `graph_search`    — keyword search across ontology nodes
 *   - `graph_get`       — resolve a single ref_id to its full node content
 *   - `graph_neighbors` — return all adjacent nodes reachable in one hop
 *
 * When `options.subAgent` is provided (and the recursion depth is below
 * `maxDepth`) it additionally registers `graph_sub_agent`, which spawns an
 * in-process child agent with its own copy of these tools.
 */
export function registerJarvisTools(
  allTools: Record<string, Tool<any, any>>,
  options: JarvisToolsOptions = {},
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
      "Fetch the ontology of the Jarvis knowledge graph: node types (with their domain) " +
      "and the canonical list of valid `domains`. " +
      "Call this once before graph_search to discover valid values for both the `type` and `domains` parameters. " +
      "Node types are grouped by domain; types in the `ungrouped` bucket have no domain and cannot be scoped with `domains`. " +
      "Relationship edges are omitted by default — graph_neighbors returns edge types live as you traverse. " +
      "Set `include_edges` to also get the full relationship map (source_type -> target_type triples).",
    inputSchema: z.object({
      include_edges: z
        .boolean()
        .optional()
        .default(false)
        .describe(
          "Include the full list of relationship edges (source_type/edge_type/target_type triples). " +
          "Off by default — the edge list is large and graph_neighbors surfaces edge types live. " +
          "Only enable when you need the complete relationship map up front."
        ),
    }),
    execute: async ({ include_edges = false }: { include_edges?: boolean }) => {
      const url = `${jarvisUrl}/v2/schema`;
      console.log(`[get_ontology] fetching ${url} include_edges=${include_edges}`);
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        return JSON.stringify(buildOntologyPayload(data, include_edges));
      } catch (err: any) {
        return `get_ontology failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.graph_search = tool({
    description:
      "Search the Jarvis knowledge graph for ontology nodes — people, topics, episodes, clips, organizations, workflows, and more. " +
      "Unlike stakgraph_search (code nodes only), this queries the full Jarvis ontology. " +
      "Each result includes an `edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
      "which relationship types you can traverse next with graph_neighbors. " +
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
      namespace: z
        .string()
        .optional()
        .describe(
          "Scope the search to a Jarvis namespace (data partition). Not an access-control boundary."
        ),
    }),
    execute: async ({
      q,
      type,
      limit = 10,
      domains,
      namespace,
    }: {
      q: string;
      type?: string;
      limit?: number;
      domains?: string;
      namespace?: string;
    }) => {
      const params = new URLSearchParams({ q, limit: String(limit) });
      if (type) params.set("type", type);
      if (domains) params.set("domains", domains);
      // Ask Jarvis to attach a per-node {EDGE_TYPE: count} map inline so the
      // agent can gauge connectivity and see hop targets in one call.
      params.set("include_edge_counts", "true");
      appendNamespace(params, namespace);
      const url = `${jarvisUrl}/v2/nodes?${params.toString()}`;
      console.log(
        `[graph_search] q=${q} type=${type ?? "*"} domains=${domains ?? "*"} limit=${limit} namespace=${namespace ?? "*"}`,
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
            // {EDGE_TYPE: count} map of this node's relationships — shows how
            // connected it is and which edge types graph_neighbors can follow.
            edges: (n.edges ?? {}) as Record<string, number>,
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
      "Returns the node's ref_id, node_type, derived name, properties, and an " +
      "`edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
      "which relationship types you can traverse next with graph_neighbors.",
    inputSchema: z.object({
      ref_id: z.string().describe("The ref_id of the node to resolve."),
      namespace: z
        .string()
        .optional()
        .describe(
          "Scope edge-count computation to a Jarvis namespace (data partition). " +
          "Only affects the `edges` map. Not an access-control boundary."
        ),
    }),
    execute: async ({
      ref_id,
      namespace,
    }: {
      ref_id: string;
      namespace?: string;
    }) => {
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

        // Fetch edge-type connectivity from the dedicated aggregation endpoint
        // (cheap: counts only, no neighbor materialization). Collapse the
        // (edge_type, target_type) breakdown into a {EDGE_TYPE: count} map so
        // graph_get and graph_search present connectivity identically. Best
        // effort — never fail the whole call if this lookup errors.
        let edges: Record<string, number> = {};
        try {
          const ccParams = new URLSearchParams();
          appendNamespace(ccParams, namespace);
          const ccQuery = ccParams.toString();
          const ccUrl = `${jarvisUrl}/v2/nodes/${encodeURIComponent(ref_id)}/connection-counts${ccQuery ? `?${ccQuery}` : ""}`;
          const ccResp = await jarvisFetch(ccUrl, jarvisHeaders);
          if (ccResp.ok) {
            const ccData = (await ccResp.json()) as any;
            edges = collapseConnectionCounts(ccData?.counts ?? []);
          }
        } catch {
          // ignore — edges stays {}
        }

        return JSON.stringify({
          ref_id: raw.ref_id,
          node_type: raw.node_type,
          name: deriveNodeName(raw, properties),
          properties: raw.properties,
          edges,
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
      "Each neighbor also includes an `edges` map ({EDGE_TYPE: count}) showing how " +
      "connected that neighbor is and which relationship types you can hop along next. " +
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
      namespace: z
        .string()
        .optional()
        .describe(
          "Scope neighbor edge-count computation to a Jarvis namespace (data partition). " +
          "Only affects each neighbor's `edges` map. Not an access-control boundary."
        ),
    }),
    execute: async ({
      ref_id,
      edge_type,
      node_type,
      namespace,
    }: {
      ref_id: string;
      edge_type?: string[];
      node_type?: string[];
      namespace?: string;
    }) => {
      // `limit` bounds the Cypher traversal so a hub node doesn't OOM Neo4j.
      // `sort_by=importance` orders edges before LIMIT so the cap keeps the most
      // important neighbors. `canonicalize=false` matches the real Neo4j label.
      // `include_edge_counts` attaches each neighbor's {EDGE_TYPE: count} map.
      const params = new URLSearchParams({
        expand: "edges",
        limit: String(KG_NEIGHBOR_CAP),
        sort_by: "importance",
        canonicalize: "false",
        exclude_node_type: toPythonListLiteral(EXCLUDED_NODE_TYPES),
        include_edge_counts: "true",
      });
      if (edge_type && edge_type.length > 0) {
        params.set("edge_type", toPythonListLiteral(edge_type));
      }
      if (node_type && node_type.length > 0) {
        params.set("node_type", toPythonListLiteral(node_type));
      }
      appendNamespace(params, namespace);
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
        // each neighbor carries a human-readable label (and its own connectivity
        // map) alongside its ref_id.
        const nodeMap = new Map<
          string,
          { node_type: string; name: string; edges: Record<string, number> }
        >();
        for (const node of data.nodes ?? []) {
          if (node.ref_id !== ref_id) {
            nodeMap.set(node.ref_id, {
              node_type: node.node_type,
              name: deriveNodeName(node, (node.properties ?? {}) as Record<string, any>),
              edges: (node.edges ?? {}) as Record<string, number>,
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
            // {EDGE_TYPE: count} map of this neighbor's own relationships —
            // shows how connected it is and which edges to follow next.
            edges: detail?.edges ?? {},
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

  // Recursive sub-agent tool, gated by config + depth so children can't spawn
  // forever. Registered only while the current depth is below maxDepth, meaning
  // leaf children never receive a `graph_sub_agent` tool.
  const sub = options.subAgent;
  if (sub) {
    const depth = sub.depth ?? 0;
    const maxDepth = sub.maxDepth ?? DEFAULT_SUBAGENT_MAX_DEPTH;
    if (depth < maxDepth) {
      registerGraphSubAgentTool(allTools, sub, depth);
    }
  }

  // Ontology write tools — opt-in via toolsConfig.ontology_edit. Off by default
  // so the standard posture stays read-only.
  if (options.ontologyEdit) {
    registerOntologyWriteTools(allTools, jarvisUrl, jarvisHeaders);
  }
}


/*
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/mikeoss",
    "prompt": "please spin up graph sub agents (graph_sub_agent tool) so i can see if that actually works. explore graph nodes. MAKE SURE YOU USE graph_sub_agent tool!!!",
    "mode": "graph",
    "toolsConfig": { "graph_sub_agent": true }
  }' \
  "http://localhost:3355/repo/agent"

curl "http://localhost:3355/progress?request_id=77591d86-994f-4758-8b66-26a7b13a7bf8"

*/