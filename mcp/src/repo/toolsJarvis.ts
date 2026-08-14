import { tool, Tool, ToolLoopAgent, stepCountIs } from "ai";
import { z } from "zod";
import axios from "axios";
import { randomUUID } from "crypto";
import PQueueModule from "p-queue";
const PQueue = (PQueueModule as any).default ?? PQueueModule;
import {
  getModelDetails,
  getProviderOptions,
  ModelName,
} from "../aieo/src/index.js";
import {
  extractFinalAnswer,
  createHasEndMarkerCondition,
  extractMessagesFromSteps,
  logStep,
} from "./utils.js";
import {
  createSession,
  appendMessages,
  appendStepMeta,
  appendSessionEnd,
  mergeReflection,
  type StepMeta,
} from "./session.js";
import {
  withConceptCollection,
  normalizeConceptReads,
  type ConceptCollector,
} from "./concepts.js";
import {
  addUsage,
  normalizeUsage,
  withProviderCacheUsage,
} from "../aieo/src/usage.js";

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

/**
 * Max ref_ids honoured by one `graph_get_batched` call. Anything beyond this is
 * reported back as `omitted_ref_ids` rather than dropped silently, so the agent
 * can issue a follow-up call for the remainder instead of inferring the gap.
 */
const KG_BATCH_GET_MAX = 50;

/**
 * In-flight node fetches per batched call. Each ref_id costs two Jarvis
 * requests (the node + its connection-counts), so this is ~2x this many
 * sockets against Jarvis at once.
 */
const KG_BATCH_GET_CONCURRENCY = 8;

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

// NOTE on attribute semantics: the bulk list endpoint backing `get_ontology`
// (`/v2/schema`) splits attributes into non-overlapping "own-only" `attributes`
// and `inherited_attributes` buckets. The single-schema endpoint
// (`format_single_schema` in jarvis-backend) keeps all attributes (own +
// inherited) in `attributes` for backward compatibility and derives
// `inherited_attributes` as an overlapping read-only view. This implementation
// assumes the non-overlapping bulk-endpoint shape — if the `get_ontology` fetch
// is ever repointed at the single-schema endpoint, the spread logic below needs
// re-verification.
export interface OntologyNodeType {
  type: string;
  /** Domain is conveyed by the grouping key in `node_types[<domain>]`; null-domain types land in the "ungrouped" bucket. */
  description: string;
  attributes?: Record<string, string>;
  inherited_attributes?: Record<string, string>;
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
 * - Lowercases each schema entry's domain locally for grouping and `domains` derivation
 *   (not emitted per entry — domain is conveyed by the `node_types[<domain>]` key).
 * - Groups node types by domain; null-domain types land in the `"ungrouped"` bucket.
 * - `domains` list is the distinct, non-null, lowercased, sorted set.
 * - Edges are omitted by default (they dominate the payload); pass
 *   `includeEdges` to append deduped compact triples sorted by `edge_type`.
 */
export function buildOntologyPayload(
  schemaData: any,
  includeEdges = false,
  includeAttributes = false,
): OntologyPayload {
  const schemas: any[] = schemaData?.schemas ?? [];
  const rawEdges: any[] = schemaData?.edges ?? [];

  // Build node type list; compute lowercased domain locally for grouping only (not emitted per entry)
  type SchemaWithDomain = OntologyNodeType & { _domain: string | null };
  const nodeTypes: SchemaWithDomain[] = schemas
    .filter((s: any) => s.type && s.type !== "*" && !s.is_deleted)
    .map((s: any) => {
      const td = (s.type_description as string) ?? "";
      const desc = (s.description as string) ?? "";
      const description = td.trim() !== "" ? td : desc;
      const _domain = s.domain ? (s.domain as string).toLowerCase() : null;
      return {
        type: s.type as string,
        _domain,
        description,
        ...(includeAttributes && {
          attributes: (s.attributes ?? {}) as Record<string, string>,
          inherited_attributes: (s.inherited_attributes ?? {}) as Record<string, string>,
        }),
      };
    });

  // Derive canonical domains list (distinct, non-null, sorted)
  const domainsSet = new Set<string>();
  for (const nt of nodeTypes) {
    if (nt._domain !== null) domainsSet.add(nt._domain);
  }
  const domains = Array.from(domainsSet).sort();

  // Group node types by domain (null → "ungrouped"); strip the internal _domain field before emitting
  const grouped: Record<string, OntologyNodeType[]> = {};
  for (const { _domain, ...entry } of nodeTypes) {
    const key = _domain ?? "ungrouped";
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(entry as OntologyNodeType);
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

/**
 * Fields of a single `/v2/schema/<type>` response that `get_ontology_type`
 * forwards to the model. Everything else the endpoint returns is either UI
 * chrome (icon, shape, primary_color, secondary_color), a key the agent never
 * addresses a node by (ref_id, node_key, title_key, description_key, index),
 * or already available from `get_ontology` (type, domain, parent, description,
 * type_description) — together ~65% of the response.
 *
 * `inherited_attributes` is dropped too: on THIS endpoint `attributes` is the
 * complete own+inherited set and `inherited_attributes` is an overlapping
 * read-only view of the inherited subset, so it is pure duplication (verified
 * as a strict subset across every observed response).
 *
 * WARNING: the bulk `/v2/schema` endpoint behind `get_ontology` does NOT share
 * this shape — there the two buckets are non-overlapping (`attributes` is
 * own-only), so applying the same trim in buildOntologyPayload would silently
 * drop every inherited field. See the note above OntologyNodeType.
 */
export const ONTOLOGY_TYPE_FIELDS = ["attributes"] as const;

/**
 * Pure transform: trim a raw `/v2/schema/<type>` response down to the attribute
 * schema the model actually reasons about.
 *
 * If NEITHER field is present the raw object is returned untouched, so an
 * unexpected response shape stays debuggable rather than being flattened into
 * `{}` — which the model would read as "this type has no attributes" and act on.
 */
export function buildOntologyTypePayload(data: any): any {
  if (!data || typeof data !== "object" || Array.isArray(data)) return data;
  const kept: Record<string, any> = {};
  for (const f of ONTOLOGY_TYPE_FIELDS) {
    if (data[f] !== undefined) kept[f] = data[f];
  }
  return Object.keys(kept).length > 0 ? kept : data;
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
   * Session id of the agent that owns this tool. When set, each sub-agent run
   * is persisted as a first-class session (`<parentSessionId>-sub-<rand>`)
   * with a `parent_session_id` link, so it shows up in /sessions like any
   * other run. When unset (parent runs without session persistence),
   * sub-agent runs are not persisted either.
   */
  parentSessionId?: string;
  /** Repo label stamped on persisted sub-agent sessions. */
  repo?: string;
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
  /**
   * When true, registers the `create_triplet` graph DATA write tool, which
   * asserts source -[edge]-> target instance facts (creating/merging nodes as
   * needed) via Jarvis `/v2/nodes` + `/v2/edges`. Distinct from `ontologyEdit`
   * (schema writes). Opt-in and off by default.
   */
  graphWrite?: boolean;
  /**
   * Comma-separated ontology domains to scope `get_ontology` to when the model
   * omits the `domains` argument (e.g. "Legal,Entity,Content"). Set from the
   * `ontologyDomains` request field so a legal-only session never pays for the
   * CodeArtifact and Workflow halves of the ontology.
   *
   * Omit to send no `domains` to Jarvis at all, leaving every domain available.
   * A model-supplied `domains` always wins over this default. Jarvis
   * `/v2/schema` applies the filter to BOTH `schemas` and `edges`, so this
   * trims the edge list too (edges are ~80% of the payload).
   */
  defaultDomains?: string;
}

const DEFAULT_SUBAGENT_DESCRIPTION =
  "Spawn a focused child agent to explore the Jarvis knowledge graph and report back. " +
  "The child has its own copy of the graph tools (get_ontology, get_ontology_type, graph_search, graph_get, graph_get_batched, graph_neighbors) " +
  "and runs an independent exploration loop, returning a synthesized text summary of its findings. " +
  "Use this to parallelize or delegate: after you locate a few key nodes, fan out one sub-agent per " +
  "node/subtopic with a specific, self-contained prompt (include the relevant ref_ids and exactly what " +
  "to find), then collate their answers. Each prompt must stand alone — the child does not see this " +
  "conversation. Prefer a handful of targeted sub-agents over one broad one. " +
  "Do NOT spawn sub-agents merely to fetch a list of ref_ids — that is what graph_get_batched is for. " +
  "Delegate reasoning and open-ended search, not bulk retrieval.";

/** System prompt for a spawned graph exploration sub-agent. */
const GRAPH_SUBAGENT_SYSTEM = `You are a focused knowledge-graph exploration sub-agent. A parent agent has delegated a specific exploration task to you. Answer ONLY the task you were given — do not expand scope.

You traverse a knowledge graph of interconnected entities (people, topics, episodes, organizations, workflows, code, and their relationships) using these tools:
- \`get_ontology\` — list node types (grouped by domain) and valid \`domains\`. Call FIRST if you don't already know the relevant types.
- \`get_ontology_type\` — fetch the full schema for a single node type (attributes + required/optional). Use when you need field-level detail for one type instead of the whole ontology.
- \`graph_search\` — keyword search. Returns compact results (ref_id, name, node_type, description, edges). Scope with \`type\`/\`domains\`, and \`namespace\` (data partition) when one applies.
- \`graph_get_batched\` — resolve up to ${KG_BATCH_GET_MAX} ref_ids in ONE call. Always use this instead of calling \`graph_get\` repeatedly.
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
  defaultDomains?: string,
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
      // Persist the child run as its own session (linked to the parent) only
      // when the parent itself is session-backed. Grandchildren link to the
      // child, giving a full chain back to the top-level session.
      const childSessionId = sub.parentSessionId
        ? `${sub.parentSessionId}-sub-${randomUUID().slice(0, 8)}`
        : undefined;
      // Recurse with depth+1 so nested sub-agents stop at maxDepth.
      registerJarvisTools(childTools, {
        subAgent: { ...sub, depth: depth + 1, parentSessionId: childSessionId },
        // Children inherit the parent's ontology scope; otherwise a sub-agent's
        // get_ontology would pull the full unfiltered payload back in.
        defaultDomains,
      });

      // Record every Concept whose body a child tool hands back, exactly like
      // the top-level run in agent.ts. Reads persist deterministically in
      // endSession; sub-agents never get a reflect/ranking pass.
      const conceptCollector: ConceptCollector = { reads: [] };
      const runTools = withConceptCollection(childTools, conceptCollector);

      if (childSessionId) {
        createSession(
          childSessionId,
          GRAPH_SUBAGENT_SYSTEM,
          "graph_sub_agent",
          sub.repo,
          sub.parentSessionId,
        );
      }

      const startTime = Date.now();
      let lastStepTime = startTime;
      const stepMetas: StepMeta[] = [];
      let cumInput = 0;
      let cumOutput = 0;
      let modelId = "";
      let provider: string | undefined;

      const endSession = async (
        status: "success" | "error",
        errorMessage?: string,
      ) => {
        if (!childSessionId) return;
        // Best-effort, like reflectOnConcepts' read-only path: a failure here
        // must never eat the child's answer or its session-end record.
        if (conceptCollector.reads.length > 0) {
          try {
            const concepts = await normalizeConceptReads(
              conceptCollector.reads,
              sub.repo,
            );
            if (concepts.length > 0) {
              mergeReflection(childSessionId, {
                concepts: concepts.map((c) => ({
                  id: c.id,
                  ref_id: c.ref_id,
                  repo: c.repo,
                  name: c.name,
                  rank: null,
                })),
              });
            }
          } catch (e) {
            console.error("[concepts] could not record sub-agent concept reads:", e);
          }
        }
        appendStepMeta(childSessionId, stepMetas);
        await appendSessionEnd(childSessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status,
          error_message: errorMessage,
          token_usage:
            stepMetas.length > 0
              ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
              : undefined,
        });
      };

      try {
        const details = getModelDetails(
          sub.modelName,
          sub.apiKey,
          sub.baseUrl,
          sub.headers,
        );
        const model = details.model;
        provider = details.provider;
        modelId = details.modelId;
        const maxSteps = sub.maxSteps ?? DEFAULT_SUBAGENT_MAX_STEPS;
        const hasEndMarker = createHasEndMarkerCondition<typeof runTools>();
        const agent = new ToolLoopAgent({
          model,
          instructions: GRAPH_SUBAGENT_SYSTEM,
          tools: runTools,
          providerOptions: getProviderOptions(details.provider, undefined, modelId) as any,
          stopWhen: maxSteps > 0 ? [hasEndMarker, stepCountIs(maxSteps)] : hasEndMarker,
          stopSequences: ["[END_OF_ANSWER]"],
          onStepFinish: (sf) => {
            if (!childSessionId) return;
            const now = Date.now();
            const elapsedMs = now - lastStepTime;
            lastStepTime = now;
            logStep(sf.content, childSessionId, elapsedMs);
            const u = withProviderCacheUsage(
              normalizeUsage(sf.usage),
              sf.providerMetadata as Record<string, any> | undefined,
            );
            cumInput += u.inputTokens ?? 0;
            cumOutput += u.outputTokens ?? 0;
            stepMetas.push({
              step: stepMetas.length,
              turn: 1,
              finishReason: sf.finishReason,
              rawFinishReason: sf.rawFinishReason,
              usage: u,
              cumulativeInput: cumInput,
              cumulativeOutput: cumOutput,
              toolCalls: (sf.toolCalls ?? []).map(
                (tc: { toolName: string }) => tc.toolName,
              ),
              timestamp: new Date().toISOString(),
              sessionId: childSessionId,
              elapsedMs,
            });
          },
        });
        console.log(
          `[graph_sub_agent] depth=${depth + 1}${childSessionId ? ` session=${childSessionId}` : ""} spawning child: ${prompt.slice(0, 200)}`,
        );
        const result = await agent.generate({ prompt });
        if (childSessionId) {
          appendMessages(
            childSessionId,
            extractMessagesFromSteps({ role: "user", content: prompt }, result.steps),
          );
        }
        await endSession("success");
        const final = extractFinalAnswer(result.steps);
        return final.answer || "Sub-agent returned no findings.";
      } catch (err: any) {
        await endSession("error", err?.message ?? String(err));
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
      "LAST RESORT — only use when NO existing type can represent the concept. " +
      "You MUST call get_ontology first and confirm no existing type already covers this concept " +
      "(match on meaning, not just spelling — check synonyms, plurals, and broader/narrower terms, " +
      "e.g. reuse `Episode` instead of creating `Podcast`, `Person` instead of `Individual`). " +
      "Prefer extending an existing type (ontology_update_type to add an attribute, or ontology_create_edge) " +
      "over inventing a new node type. " +
      "When you do create, inherit from the closest existing `parent` (not `Thing` if a more specific ancestor exists) " +
      "and keep naming consistent with existing types. Children inherit parent attributes.",
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
      "You MUST call get_ontology (include_edges: true) first to confirm both source and target types exist " +
      "and that no existing edge type already expresses this relationship (reuse an existing edge type when one fits, " +
      "matching on meaning not just spelling). Do not create near-duplicate relationships. " +
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
 * Validate one side (source/target) of a triplet. A side must be EITHER an
 * existing node (`<side>_ref_id`) OR an inline node (`<side>_type` +
 * `<side>_data`) — never both, never neither. Returns an error message for the
 * agent, or null when valid.
 */
export function validateTripletSide(
  side: "source" | "target",
  refId?: string,
  nodeType?: string,
  nodeData?: Record<string, any>,
): string | null {
  const hasRef = typeof refId === "string" && refId.length > 0;
  const hasInline = Boolean(nodeType) || Boolean(nodeData);
  if (hasRef && hasInline) {
    return `${side}: pass either ${side}_ref_id OR ${side}_type + ${side}_data, not both`;
  }
  if (hasRef) return null;
  if (nodeType && nodeData) return null;
  return (
    `${side}: pass ${side}_ref_id (an existing node), ` +
    `or both ${side}_type and ${side}_data (create/merge inline)`
  );
}

/**
 * Build a stable dedup key for an inline node side so identical sides across a
 * batch collapse to a single resolution call. Key = node_type + canonical JSON
 * of node_data with object keys sorted recursively — key order in node_data is
 * ignored, matching Jarvis's own identity semantics.
 */
export function buildNodeDedupKey(
  nodeType: string,
  nodeData: Record<string, any>,
): string {
  function sortedJson(v: any): any {
    if (v === null || typeof v !== "object" || Array.isArray(v)) return v;
    const sorted: Record<string, any> = {};
    for (const k of Object.keys(v).sort()) {
      sorted[k] = sortedJson(v[k]);
    }
    return sorted;
  }
  return `${nodeType}::${JSON.stringify(sortedJson(nodeData))}`;
}

/**
 * Resolved triplet for the edge pass — all three fields are guaranteed
 * non-empty when the triplet reaches this stage.
 */
export interface ResolvedTriplet {
  /** original input index */
  index: number;
  source_ref_id: string;
  target_ref_id: string;
  edge_type: string;
  edge_data?: Record<string, any>;
  weight?: number;
  create_schema_if_missing: boolean;
}

/**
 * Match returned edges from the bulk `POST /v2/edges` response back to the
 * originating triplets.  Matching key = (source_ref_id, target_ref_id,
 * edge_type).  Each returned edge is consumed **at most once** in input order
 * so duplicate keys (same triple but different edge_data/weight) are
 * disambiguated deterministically.
 *
 * Returns `{ matched: Map<index, edgeRefId>, unmatched: ResolvedTriplet[] }`.
 */
export function matchEdgeResults(
  triplets: ResolvedTriplet[],
  returnedEdges: Array<{ ref_id: string; source?: string; target?: string; edge_type?: string }>,
): { matched: Map<number, string>; unmatched: ResolvedTriplet[] } {
  // Build a pool of returned edges grouped by (src, tgt, edge_type) key.
  // Each pool entry is a queue; we shift() one per matched triplet so each
  // returned edge is consumed at most once.
  const pool = new Map<string, string[]>();
  for (const e of returnedEdges) {
    if (!e?.ref_id) continue;
    const key = `${e.source ?? ""}|${e.target ?? ""}|${e.edge_type ?? ""}`;
    if (!pool.has(key)) pool.set(key, []);
    pool.get(key)!.push(e.ref_id);
  }

  const matched = new Map<number, string>();
  const unmatched: ResolvedTriplet[] = [];

  for (const t of triplets) {
    const key = `${t.source_ref_id}|${t.target_ref_id}|${t.edge_type}`;
    const queue = pool.get(key);
    if (queue && queue.length > 0) {
      matched.set(t.index, queue.shift()!);
    } else {
      unmatched.push(t);
    }
  }

  return { matched, unmatched };
}

/**
 * Pull the created/merged node ref_id out of a Jarvis `POST /v2/nodes`
 * response body. Both plain success and the "Node already exists in the graph"
 * warning carry `data.ref_id` (merge semantics); anything else is a failure.
 */
export function extractNodeRefId(body: any): string | undefined {
  const refId = body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

/**
 * Pull the edge ref_id out of a Jarvis `POST /v2/edges` response body. A fresh
 * edge lands in `edges[0].ref_id`; the "Edge already exists in the graph"
 * warning carries `data.ref_id` instead.
 */
export function extractEdgeRefId(body: any): string | undefined {
  const refId = body?.edges?.[0]?.ref_id ?? body?.data?.ref_id;
  return typeof refId === "string" && refId.length > 0 ? refId : undefined;
}

/**
 * Register the graph DATA write tool (`create_triplet`): assert a fact as
 * source -[edge]-> target instance data (not schema). Gated by
 * `JarvisToolsOptions.graphWrite`.
 *
 * Inline nodes are pre-created via `POST /v2/nodes` and the edge is then
 * created with two concrete ref_ids in a second call — rather than sending
 * inline nodes straight to `POST /v2/edges` — both for per-node error
 * attribution and because Jarvis's edge endpoint mis-orders its ref_id list
 * when only the source is inline (which would reverse the edge direction).
 */
function registerGraphWriteTools(
  allTools: Record<string, Tool<any, any>>,
  jarvisUrl: string,
  jarvisHeaders: Record<string, string>,
): void {
  allTools.create_triplet = tool({
    description:
      "Assert a fact into the Jarvis knowledge graph as DATA: a triplet of source node -[edge]-> target node " +
      "(instances, not schema — for schema changes use the ontology_* tools). Writes live to the graph. " +
      "For each side pass EITHER the ref_id of an existing node (preferred — find it with graph_search) " +
      "OR a node type + data object to create/merge the node inline. " +
      "REUSE existing nodes wherever possible: search first, and only create inline when the entity " +
      "genuinely doesn't exist yet — duplicate nodes fragment the graph. " +
      "Node types and the edge type must already exist in the ontology (check with get_ontology); " +
      "create_schema_if_missing auto-creates a missing edge schema as a last resort. " +
      "WILDCARD EDGE MATCHING: when checking source_type/target_type against get_ontology's edges for validity, " +
      "an edge entry with \"*\" on either side matches any concrete type on that side — do not require an exact string match. " +
      "IMPORTANT: \"*\" is valid to SEE in get_ontology's edge output, but is NEVER a valid value to SUPPLY as " +
      "source_type or target_type when calling create_triplet — supplying \"*\" would create a node of type \"*\", " +
      "which is an unintended backend sentinel, not a real node type.",
    inputSchema: z.object({
      source_ref_id: z
        .string()
        .optional()
        .describe(
          "ref_id of an EXISTING source node (from graph_search/graph_get). Preferred over inline creation.",
        ),
      source_type: z
        .string()
        .optional()
        .describe(
          "Node type for an INLINE source node (must exist in the ontology — see get_ontology). " +
          "Requires source_data; omit when source_ref_id is set.",
        ),
      source_data: z
        .record(z.string(), z.any())
        .optional()
        .describe(
          'Properties for an INLINE source node, e.g. {"name": "Alice"}. ' +
          "Must satisfy the type's schema, including its node_key attribute.",
        ),
      target_ref_id: z
        .string()
        .optional()
        .describe(
          "ref_id of an EXISTING target node (from graph_search/graph_get). Preferred over inline creation.",
        ),
      target_type: z
        .string()
        .optional()
        .describe(
          "Node type for an INLINE target node (must exist in the ontology — see get_ontology). " +
          "Requires target_data; omit when target_ref_id is set.",
        ),
      target_data: z
        .record(z.string(), z.any())
        .optional()
        .describe(
          'Properties for an INLINE target node, e.g. {"name": "Acme Corp"}. ' +
          "Must satisfy the type's schema, including its node_key attribute.",
        ),
      edge_type: z
        .string()
        .describe(
          "The relationship type, e.g. 'WORKS_AT'. Uppercased by Jarvis. Must exist in the ontology " +
          "between the two node types unless create_schema_if_missing is set.",
        ),
      edge_data: z
        .record(z.string(), z.any())
        .optional()
        .describe("Optional properties to set on the edge."),
      weight: z.number().optional().describe("Optional edge weight (defaults to 1)."),
      create_schema_if_missing: z
        .boolean()
        .optional()
        .default(false)
        .describe(
          "Auto-create the edge schema when the (source_type, edge_type, target_type) relationship " +
          "is not yet in the ontology. Last resort — prefer defining it deliberately with ontology_create_edge. " +
          "Before enabling, check get_ontology's edges for an existing wildcard (\"*\") rule covering the same " +
          "edge_type — a wildcard schema already matches any concrete type pair, so creating a new concrete-type " +
          "schema on top of it would produce a redundant, overlapping rule.",
        ),
      namespace: z
        .string()
        .optional()
        .describe(
          "Jarvis namespace (data partition) for inline node creation. Not an access-control boundary.",
        ),
    }),
    execute: async (input: {
      source_ref_id?: string;
      source_type?: string;
      source_data?: Record<string, any>;
      target_ref_id?: string;
      target_type?: string;
      target_data?: Record<string, any>;
      edge_type: string;
      edge_data?: Record<string, any>;
      weight?: number;
      create_schema_if_missing?: boolean;
      namespace?: string;
    }) => {
      const {
        source_ref_id,
        source_type,
        source_data,
        target_ref_id,
        target_type,
        target_data,
        edge_type,
        edge_data,
        weight,
        create_schema_if_missing = false,
        namespace,
      } = input;

      for (const err of [
        validateTripletSide("source", source_ref_id, source_type, source_data),
        validateTripletSide("target", target_ref_id, target_type, target_data),
      ]) {
        if (err) return `create_triplet invalid input — ${err}`;
      }

      console.log(
        `[create_triplet] ${source_ref_id ?? source_type} -[${edge_type}]-> ${target_ref_id ?? target_type} namespace=${namespace ?? "-"}`,
      );

      // Resolve one side to a concrete ref_id, creating/merging an inline node
      // via /v2/nodes when no ref_id was given. Namespace applies to node
      // creation only — the edge endpoint matches by globally-unique ref_id.
      const resolveSide = async (
        side: "source" | "target",
        refId?: string,
        nodeType?: string,
        nodeData?: Record<string, any>,
      ): Promise<string> => {
        if (refId) return refId;
        const params = new URLSearchParams();
        appendNamespace(params, namespace);
        const qs = params.toString();
        const url = `${jarvisUrl}/v2/nodes${qs ? `?${qs}` : ""}`;
        const res = await jarvisMutate("post", url, jarvisHeaders, {
          node_type: nodeType,
          node_data: nodeData,
        });
        let body: any;
        try {
          body = JSON.parse(res.text);
        } catch {
          // non-JSON body — fall through to the error below
        }
        const created = extractNodeRefId(body);
        if (!created) {
          throw new Error(
            `could not create/merge ${side} node (HTTP ${res.status}): ${res.text}`,
          );
        }
        return created;
      };

      try {
        // Sequential so a failure names the side that broke.
        const sourceRef = await resolveSide("source", source_ref_id, source_type, source_data);
        const targetRef = await resolveSide("target", target_ref_id, target_type, target_data);

        const res = await jarvisMutate("post", `${jarvisUrl}/v2/edges`, jarvisHeaders, {
          edge: {
            edge_type,
            ...(weight !== undefined ? { weight } : {}),
            ...(edge_data ? { edge_data } : {}),
          },
          source: { ref_id: sourceRef },
          target: { ref_id: targetRef },
          create_schema_if_missing,
        });
        let body: any;
        try {
          body = JSON.parse(res.text);
        } catch {
          // non-JSON body — fall through to the error below
        }
        const edgeRef = extractEdgeRefId(body);
        if (!res.ok || !edgeRef) {
          return (
            `create_triplet: nodes resolved (source=${sourceRef}, target=${targetRef}) ` +
            `but the edge write failed — HTTP ${res.status}: ${res.text}`
          );
        }
        return JSON.stringify({
          // "Warning" here means the edge already existed (idempotent merge).
          status: body?.status ?? "Success",
          source_ref_id: sourceRef,
          target_ref_id: targetRef,
          edge_ref_id: edgeRef,
          edge_type,
          ...(Array.isArray(body?.status_messages) && body.status_messages.length > 0
            ? { messages: body.status_messages }
            : {}),
        });
      } catch (err: any) {
        return `create_triplet failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  console.log("===> registered graph write tool: create_triplet");

  // ── create_batch_triplet ──────────────────────────────────────────────────
  // Accepts an array of triplet specs and asserts all of them in one call,
  // returning a per-triplet result array in input order.
  allTools.create_batch_triplet = tool({
    description:
      "Assert MANY facts into the Jarvis knowledge graph in a single call. " +
      "Each item in `triplets` has the same shape as `create_triplet` (source/target as ref_id or inline " +
      "node_type+node_data, plus edge_type and optional edge_data/weight/create_schema_if_missing). " +
      "A single top-level `namespace` applies to all inline node creation. " +
      "REUSE existing nodes wherever possible: supply ref_ids from graph_search when entities already exist — " +
      "inline creation is a last resort. Identical inline sides across the batch are resolved once (deduped). " +
      "Returns one result entry per input triplet in input order; a failed item does not abort the rest. " +
      "Use `create_triplet` for a single fact; use this tool when asserting multiple related facts at once " +
      "to avoid redundant round-trips.",
    inputSchema: z.object({
      triplets: z
        .array(
          z.object({
            source_ref_id: z
              .string()
              .optional()
              .describe(
                "ref_id of an EXISTING source node. Preferred over inline creation.",
              ),
            source_type: z
              .string()
              .optional()
              .describe(
                "Node type for an INLINE source node. Requires source_data; omit when source_ref_id is set.",
              ),
            source_data: z
              .record(z.string(), z.any())
              .optional()
              .describe(
                "Properties for an INLINE source node. Must satisfy the type's schema.",
              ),
            target_ref_id: z
              .string()
              .optional()
              .describe(
                "ref_id of an EXISTING target node. Preferred over inline creation.",
              ),
            target_type: z
              .string()
              .optional()
              .describe(
                "Node type for an INLINE target node. Requires target_data; omit when target_ref_id is set.",
              ),
            target_data: z
              .record(z.string(), z.any())
              .optional()
              .describe(
                "Properties for an INLINE target node. Must satisfy the type's schema.",
              ),
            edge_type: z
              .string()
              .describe(
                "The relationship type, e.g. 'WORKS_AT'. Must exist in the ontology unless " +
                "create_schema_if_missing is set.",
              ),
            edge_data: z
              .record(z.string(), z.any())
              .optional()
              .describe("Optional properties to set on the edge."),
            weight: z.number().optional().describe("Optional edge weight."),
            create_schema_if_missing: z
              .boolean()
              .optional()
              .default(false)
              .describe(
                "Auto-create the edge schema when the relationship type is not yet in the ontology. " +
                "Last resort — prefer defining it deliberately with ontology_create_edge.",
              ),
          }),
        )
        .describe("Array of triplet specs to assert."),
      namespace: z
        .string()
        .optional()
        .describe(
          "Jarvis namespace (data partition) for inline node creation. Applies to all items. " +
          "Not an access-control boundary.",
        ),
      return_edge_ids: z
        .boolean()
        .optional()
        .default(false)
        .describe(
          "Include `edge_ref_id` on each successful result. OFF by default and rarely needed — " +
          "the ref_ids of the NODES you created are always returned, and those are what you use " +
          "to build further triplets. Enable ONLY if you will address specific edges by ref_id " +
          "later in this session (e.g. to update or delete an individual edge). Leaving it off " +
          "keeps results small, which matters because every tool result stays in context.",
        ),
    }),
    execute: async (input: {
      triplets: Array<{
        source_ref_id?: string;
        source_type?: string;
        source_data?: Record<string, any>;
        target_ref_id?: string;
        target_type?: string;
        target_data?: Record<string, any>;
        edge_type: string;
        edge_data?: Record<string, any>;
        weight?: number;
        create_schema_if_missing?: boolean;
      }>;
      namespace?: string;
      return_edge_ids?: boolean;
    }) => {
      const { triplets, namespace, return_edge_ids = false } = input;
      console.log(
        `[create_batch_triplet] count=${triplets.length} namespace=${namespace ?? "-"}`,
      );

      // ── Phase 0: per-item validation ────────────────────────────────────
      // Keep a parallel array tracking failures so a bad item doesn't abort
      // the rest. null = still alive.
      const failures: Array<string | null> = triplets.map(() => null);

      for (let i = 0; i < triplets.length; i++) {
        const t = triplets[i];
        const srcErr = validateTripletSide(
          "source",
          t.source_ref_id,
          t.source_type,
          t.source_data,
        );
        const tgtErr = validateTripletSide(
          "target",
          t.target_ref_id,
          t.target_type,
          t.target_data,
        );
        const err = srcErr ?? tgtErr;
        if (err) failures[i] = `invalid input — ${err}`;
      }

      // ── Phase 1: dedup + resolve inline node sides ───────────────────────
      // Collect all unique (node_type, node_data) pairs across the batch and
      // resolve each once via single-object POST /v2/nodes.
      //
      // Map: dedupKey → resolved ref_id (or Error if resolution failed).
      const resolvedNodes = new Map<string, string | Error>();
      // Which dedupKeys still need to be fetched.
      const toResolve = new Map<string, { nodeType: string; nodeData: Record<string, any> }>();

      for (let i = 0; i < triplets.length; i++) {
        if (failures[i]) continue;
        const t = triplets[i];
        for (const [refId, nodeType, nodeData] of [
          [t.source_ref_id, t.source_type, t.source_data],
          [t.target_ref_id, t.target_type, t.target_data],
        ] as [string | undefined, string | undefined, Record<string, any> | undefined][]) {
          const hasRef = typeof refId === "string" && refId.length > 0;
          if (hasRef || !nodeType || !nodeData) continue; // ref_id side — no resolution needed
          const key = buildNodeDedupKey(nodeType, nodeData);
          if (!resolvedNodes.has(key) && !toResolve.has(key)) {
            toResolve.set(key, { nodeType, nodeData });
          }
        }
      }

      // Resolve each unique inline side sequentially.
      let uniqueNodesResolved = 0;
      for (const [key, { nodeType, nodeData }] of toResolve) {
        try {
          const params = new URLSearchParams();
          appendNamespace(params, namespace);
          const qs = params.toString();
          const url = `${jarvisUrl}/v2/nodes${qs ? `?${qs}` : ""}`;
          const res = await jarvisMutate("post", url, jarvisHeaders, {
            node_type: nodeType,
            node_data: nodeData,
          });
          let body: any;
          try {
            body = JSON.parse(res.text);
          } catch {
            // non-JSON — will fail below
          }
          const ref = extractNodeRefId(body);
          if (!ref) {
            resolvedNodes.set(
              key,
              new Error(
                `could not create/merge node type=${nodeType} (HTTP ${res.status}): ${res.text}`,
              ),
            );
          } else {
            resolvedNodes.set(key, ref);
            uniqueNodesResolved++;
          }
        } catch (err: any) {
          resolvedNodes.set(
            key,
            new Error(`could not create/merge node type=${nodeType}: ${err?.message ?? String(err)}`),
          );
        }
      }

      // Propagate node-resolution failures to every dependent triplet and
      // build the concrete ref_id pair for surviving triplets.
      const sourceRefs: Array<string | null> = triplets.map(() => null);
      const targetRefs: Array<string | null> = triplets.map(() => null);

      for (let i = 0; i < triplets.length; i++) {
        if (failures[i]) continue;
        const t = triplets[i];

        // Source side
        const srcHasRef = typeof t.source_ref_id === "string" && t.source_ref_id.length > 0;
        if (srcHasRef) {
          sourceRefs[i] = t.source_ref_id!;
        } else {
          const key = buildNodeDedupKey(t.source_type!, t.source_data!);
          const r = resolvedNodes.get(key);
          if (r instanceof Error) {
            failures[i] = `source node resolution failed: ${r.message}`;
          } else {
            sourceRefs[i] = r ?? null;
          }
        }

        if (failures[i]) continue;

        // Target side
        const tgtHasRef = typeof t.target_ref_id === "string" && t.target_ref_id.length > 0;
        if (tgtHasRef) {
          targetRefs[i] = t.target_ref_id!;
        } else {
          const key = buildNodeDedupKey(t.target_type!, t.target_data!);
          const r = resolvedNodes.get(key);
          if (r instanceof Error) {
            failures[i] = `target node resolution failed: ${r.message}`;
          } else {
            targetRefs[i] = r ?? null;
          }
        }
      }

      // ── Phase 2: bulk edge write ─────────────────────────────────────────
      // Build the edge list for all triplets that survived validation +
      // node resolution.
      const resolvedTriplets: ResolvedTriplet[] = [];
      for (let i = 0; i < triplets.length; i++) {
        if (failures[i] || !sourceRefs[i] || !targetRefs[i]) continue;
        const t = triplets[i];
        resolvedTriplets.push({
          index: i,
          source_ref_id: sourceRefs[i]!,
          target_ref_id: targetRefs[i]!,
          edge_type: t.edge_type,
          edge_data: t.edge_data,
          weight: t.weight,
          create_schema_if_missing: t.create_schema_if_missing ?? false,
        });
      }

      // Per-triplet edge ref_id accumulator (index → edge_ref_id).
      const edgeResults = new Map<number, string>();
      const bulkStatusMessages: string[] = [];

      if (resolvedTriplets.length > 0) {
        // Build list-body for sequential bulk endpoint.
        const edgeList = resolvedTriplets.map((rt) => ({
          edge: {
            edge_type: rt.edge_type,
            ...(rt.weight !== undefined ? { weight: rt.weight } : {}),
            ...(rt.edge_data ? { edge_data: rt.edge_data } : {}),
          },
          source: { ref_id: rt.source_ref_id },
          target: { ref_id: rt.target_ref_id },
          create_schema_if_missing: rt.create_schema_if_missing,
        }));

        const edgeParams = new URLSearchParams();
        appendNamespace(edgeParams, namespace);
        const edgeQs = edgeParams.toString();
        const edgeUrl = `${jarvisUrl}/v2/edges${edgeQs ? `?${edgeQs}` : ""}`;

        let bulkBody: any;
        try {
          const bulkRes = await jarvisMutate("post", edgeUrl, jarvisHeaders, edgeList);
          try {
            bulkBody = JSON.parse(bulkRes.text);
          } catch {
            // non-JSON — treat as total failure; every triplet will fall back
          }
        } catch (err: any) {
          // Network / timeout — every edge will fall through to the fallback.
        }

        const returnedEdges: Array<{
          ref_id: string;
          source?: string;
          target?: string;
          edge_type?: string;
        }> = Array.isArray(bulkBody?.edges) ? bulkBody.edges : [];

        if (Array.isArray(bulkBody?.status_messages)) {
          bulkStatusMessages.push(...bulkBody.status_messages);
        }

        const { matched, unmatched } = matchEdgeResults(resolvedTriplets, returnedEdges);

        // Record matched edges.
        for (const [idx, refId] of matched) {
          edgeResults.set(idx, refId);
        }

        // Fallback: re-issue each unmatched triplet as a single-object POST.
        for (const rt of unmatched) {
          try {
            const res = await jarvisMutate("post", edgeUrl, jarvisHeaders, {
              edge: {
                edge_type: rt.edge_type,
                ...(rt.weight !== undefined ? { weight: rt.weight } : {}),
                ...(rt.edge_data ? { edge_data: rt.edge_data } : {}),
              },
              source: { ref_id: rt.source_ref_id },
              target: { ref_id: rt.target_ref_id },
              create_schema_if_missing: rt.create_schema_if_missing,
            });
            let body: any;
            try {
              body = JSON.parse(res.text);
            } catch {
              // non-JSON — edgeRef will be undefined
            }
            const edgeRef = extractEdgeRefId(body);
            if (edgeRef) {
              edgeResults.set(rt.index, edgeRef);
            } else {
              failures[rt.index] =
                `edge write failed (HTTP ${res.status}): ${res.text}`;
            }
          } catch (err: any) {
            failures[rt.index] =
              `edge write failed: ${err?.message ?? String(err)}`;
          }
        }
      }

      // ── Phase 3: assemble results in input order ─────────────────────────
      const results = triplets.map((t, i) => {
        if (failures[i]) {
          return {
            status: "Error",
            index: i,
            edge_type: t.edge_type,
            error: failures[i],
          };
        }
        const edgeRef = edgeResults.get(i);
        if (!edgeRef) {
          return {
            status: "Error",
            index: i,
            edge_type: t.edge_type,
            error: "edge ref_id could not be recovered",
          };
        }
        // Successful entries return only what the caller does NOT already have.
        // `edge_ref_id` was measured across four production traces: 1,426 returned,
        // 0 ever referenced in a later call — so it is opt-in via `return_edge_ids`.
        // `edge_type` is echoed straight back from the request and is always dropped.
        // Both are kept on the Error branches, where the context is worth the tokens
        // and the volume is negligible.
        return {
          status: "Success",
          source_ref_id: sourceRefs[i]!,
          target_ref_id: targetRefs[i]!,
          ...(return_edge_ids ? { edge_ref_id: edgeRef } : {}),
        };
      });

      const failedCount = results.filter((r) => r.status === "Error").length;
      const bulkMatched = resolvedTriplets.length - (
        resolvedTriplets.filter((rt) => !edgeResults.has(rt.index) || failures[rt.index]).length
      );
      const fallbackRecovered = results.filter(
        (r) => r.status === "Success",
      ).length - Math.max(0, bulkMatched);

      console.log(
        `[create_batch_triplet] resolvedNodes=${uniqueNodesResolved} edgesWritten=${bulkMatched} edgesMerged=${fallbackRecovered} failed=${failedCount}`,
      );

      return JSON.stringify({
        results,
        ...(bulkStatusMessages.length > 0 ? { status_messages: bulkStatusMessages } : {}),
      });
    },
  });

  console.log("===> registered graph write tool: create_triplet + create_batch_triplet");
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

  const { defaultDomains } = options;

  const jarvisHeaders = {
    "Content-Type": "application/json",
    "X-Api-Token": process.env.API_TOKEN ?? "",
  };

  // Wildcard sentinel: jarvis-backend uses a real Schema node with type="*" to
  // mean "this edge type applies to any node type on that side." This sentinel
  // is created by `_ensure_wildcard_sentinel`, honoured by `create_edge_schema`
  // (which skips existence checks when source/target is "*"), and used as a
  // fallback in `get_schema_edge_by_edge_type` — all in
  // jarvis-backend/api/helper/schema_crud.py. The descriptions below explain
  // this convention to agents so they can interpret wildcard edges correctly.
  allTools.get_ontology = tool({
    description:
      "Fetch the ontology of the Jarvis knowledge graph: node types grouped by domain " +
      "and the canonical list of valid `domains`. " +
      "Call this once before graph_search to discover valid values for both the `type` and `domains` parameters. " +
      "Node types are grouped by domain key in `node_types[<domain>]`; types with no domain land in the `ungrouped` bucket and cannot be scoped with `domains`. " +
      "Pass `domains` to filter results to one or more specific domains (comma-separated, e.g. 'Legal,Entity'); omit to receive all domains. " +
      "Relationship edges are omitted by default — graph_neighbors returns edge types live as you traverse. " +
      "Set `include_edges` to also get the full relationship map (source_type -> target_type triples). " +
      "Set `include_attributes` to also get each node type's attribute schema (field names, types, required/optional status). " +
      "WILDCARD EDGES: when include_edges is true, an edge entry whose source_type and/or target_type is \"*\" " +
      "means that edge type applies to ANY node type on that side (use \"*\" for source or target to define a wildcard relationship rule). " +
      "\"*\" is intentionally absent from node_types — it is a backend sentinel, not a real type. " +
      "Wildcard edges are guaranteed to appear only via the default fetch path; if the backend route applies " +
      "visibility filtering (get_all_schemas in jarvis-backend filters edges to those whose source/target are " +
      "in visible_types, silently dropping wildcards), wildcard edges are omitted rather than passed through.",
    inputSchema: z.object({
      domains: z
        .string()
        .optional()
        .describe(
          "Comma-separated list of domains to filter results to (e.g. 'Legal,Entity'). " +
          "Omit to receive node types from all domains. " +
          "Values are matched case-insensitively against the domain grouping keys returned by this tool."
        ),
      include_edges: z
        .boolean()
        .optional()
        .default(false)
        .describe(
          "Include the full list of relationship edges (source_type/edge_type/target_type triples). " +
          "Off by default — the edge list is large and graph_neighbors surfaces edge types live. " +
          "Only enable when you need the complete relationship map up front. " +
          "Edges may include \"*\" as a wildcard source_type/target_type — see the tool description above for what this means."
        ),
      include_attributes: z
        .boolean()
        .optional()
        .default(false)
        .describe(
          "Include each node type's attribute schema maps (`attributes` and `inherited_attributes`). " +
          "Off by default to keep the payload lean. Enable when you need to inspect field names, " +
          "required/optional status (`?` prefix = optional), and value types per node type. " +
          ATTRIBUTE_TYPES_DOC
        ),
    }),
    execute: async ({
      domains,
      include_edges = false,
      include_attributes = false,
    }: {
      domains?: string;
      include_edges?: boolean;
      include_attributes?: boolean;
    }) => {
      // Forward include_edges and include_attributes to jarvis unconditionally (always-set,
      // mirroring the graph_search pattern). NOTE: jarvis-backend currently only parses
      // `include_deleted`, `concise`, and `visible_only` on GET /v2/schema — a separate
      // in-flight jarvis task will add support for these two params. The param names
      // (`include_edges`, `include_attributes`) are a cross-repo contract with that task
      // and must stay in sync with it. Client-side trimming in buildOntologyPayload
      // remains the source of truth until jarvis honors these params.
      //
      // `domains` IS honoured by jarvis today, and it filters both `schemas` and
      // `edges`. An explicit model-supplied value always wins; otherwise we fall
      // back to the caller's request-level scope (options.defaultDomains). With
      // neither set, no `domains` param is sent and every domain comes back.
      const effectiveDomains =
        domains && domains.trim() !== ""
          ? domains.trim()
          : (defaultDomains ?? "").trim();
      const params = new URLSearchParams();
      params.set("include_edges", String(include_edges));
      params.set("include_attributes", String(include_attributes));
      if (effectiveDomains !== "") {
        params.set("domains", effectiveDomains);
      }
      const url = `${jarvisUrl}/v2/schema?${params.toString()}`;
      console.log(
        `[get_ontology] fetching ${url} domains=${effectiveDomains}${domains ? "" : " (default)"} include_edges=${include_edges} include_attributes=${include_attributes}`
      );
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        return JSON.stringify(buildOntologyPayload(data, include_edges, include_attributes));
      } catch (err: any) {
        return `get_ontology failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.get_ontology_type = tool({
    description:
      "Fetch the attribute schema for a SINGLE ontology node type. Returns exactly " +
      "one field — `attributes` — and nothing else; for a type's domain, parent or " +
      "description, use get_ontology. " +
      "Each attribute value is a type string (e.g. 'string', 'int'); a `?` prefix " +
      "(e.g. '?string') means the attribute is OPTIONAL, no prefix means REQUIRED. " +
      "`attributes` is complete: it already includes both the type's own attributes " +
      "AND everything inherited from parent types, so it is the only field you need. " +
      "Lookup is case-insensitive for every type EXCEPT the root type 'Thing', which " +
      "must be passed with exact casing. You may also pass a schema ref_id instead " +
      "of a type name — the lookup falls back to ref_id resolution automatically. " +
      "This is for NODE types only; edge type names (e.g. 'KNOWS') are not schema " +
      "nodes and will return the not-found error below. " +
      "Call get_ontology first if you don't already know the exact type name.",
    inputSchema: z.object({
      type: z.string().describe(
        "The node type name, e.g. 'Person' (case-insensitive, except the literal " +
        "root type 'Thing' which is case-sensitive). A schema ref_id is also accepted."
      ),
    }),
    execute: async ({ type }: { type: string }) => {
      const url = `${jarvisUrl}/v2/schema/${encodeURIComponent(type)}`;
      console.log(`[get_ontology_type] fetching ${url}`);
      try {
        const resp = await jarvisFetch(url, jarvisHeaders);
        if (!resp.ok) {
          const text = await resp.text();
          return `HTTP ${resp.status}: ${text}`;
        }
        const data = (await resp.json()) as any;
        return JSON.stringify(buildOntologyTypePayload(data));
      } catch (err: any) {
        return `get_ontology_type failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  allTools.graph_search = tool({
    description:
      "Search the Jarvis knowledge graph for ontology nodes — people, topics, episodes, clips, organizations, workflows, and more. " +
      "Unlike stakgraph_search (code nodes only), this queries the full Jarvis ontology. " +
      "Provide at least one of `q`, `input_q`, `output_q` — they can be combined, each acting as its own " +
      "retriever fused into one ranked result set. " +
      "Each result includes an `edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
      "which relationship types you can traverse next with graph_neighbors. " +
      "Call get_ontology first to discover valid values for the `type` parameter.",
    inputSchema: z.object({
      q: z
        .string()
        .optional()
        .describe(
          "General hybrid (keyword + semantic) search query over node names, descriptions, bodies, and schemas."
        ),
      input_q: z
        .string()
        .optional()
        .describe(
          "Semantic search scoped to node INPUT schemas — find nodes by what they take as input, " +
          "e.g. 'a video file url'. Applies to node types with input embeddings (Workflow, Skill)."
        ),
      output_q: z
        .string()
        .optional()
        .describe(
          "Semantic search scoped to node OUTPUT schemas — find nodes by what they produce, " +
          "e.g. 'transcript with word-level timestamps'. Applies to node types with output embeddings (Workflow, Skill)."
        ),
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
      input_q,
      output_q,
      type,
      limit = 10,
      domains,
      namespace,
    }: {
      q?: string;
      input_q?: string;
      output_q?: string;
      type?: string;
      limit?: number;
      domains?: string;
      namespace?: string;
    }) => {
      if (!q && !input_q && !output_q) {
        return "graph_search requires at least one of: q, input_q, output_q";
      }
      const params = new URLSearchParams({ limit: String(limit) });
      if (q) params.set("q", q);
      // Field-scoped vector search: Jarvis embeds these against the per-field
      // input/output schema embeddings and fuses them with `q` via RRF.
      if (input_q) params.set("input_q", input_q);
      if (output_q) params.set("output_q", output_q);
      if (type) params.set("type", type);
      if (domains) params.set("domains", domains);
      // Ask Jarvis to attach a per-node {EDGE_TYPE: count} map inline so the
      // agent can gauge connectivity and see hop targets in one call.
      params.set("include_edge_counts", "true");
      appendNamespace(params, namespace);
      const url = `${jarvisUrl}/v2/nodes?${params.toString()}`;
      console.log(
        `[graph_search] q=${q ?? "-"} input_q=${input_q ?? "-"} output_q=${output_q ?? "-"} type=${type ?? "*"} domains=${domains ?? "*"} limit=${limit} namespace=${namespace ?? "*"}`,
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
              n.properties?.workflow_name ??
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
            // Human-facing Stakwork ids, present on Workflow/Skill nodes —
            // used to cite components to users (and build Stakwork UI links)
            // without resolving the full node.
            ...(n.properties?.workflow_id !== undefined
              ? { workflow_id: n.properties.workflow_id }
              : {}),
            ...(n.properties?.skill_id !== undefined
              ? { skill_id: n.properties.skill_id }
              : {}),
          }))
        );
      } catch (err: any) {
        return `graph_search failed: ${err?.message ?? String(err)}`;
      }
    },
  });

  /**
   * Resolve one ref_id to the node shape both `graph_get` and
   * `graph_get_batched` return. Yields a discriminated result rather than
   * throwing or returning a string, so the batched caller can report a
   * per-node failure without sinking the whole call.
   */
  async function fetchGraphNode(
    ref_id: string,
    namespace?: string,
  ): Promise<
    | { ok: true; node: Record<string, any> }
    | { ok: false; error: string }
  > {
    // limit=1 keeps Jarvis from materializing the node's whole neighborhood
    // (which can OOM Neo4j for hub nodes) — we only read the node itself.
    const url = `${jarvisUrl}/v2/nodes/${encodeURIComponent(ref_id)}?limit=1`;
    try {
      const resp = await jarvisFetch(url, jarvisHeaders);
      if (!resp.ok) {
        const text = await resp.text();
        return { ok: false, error: `HTTP ${resp.status}: ${text}` };
      }
      const data = (await resp.json()) as any;
      // Deployed Jarvis wraps the node in `{ nodes, edges, status }`; some
      // builds return the node directly. Handle both shapes.
      const raw = Array.isArray(data?.nodes)
        ? data.nodes.find((n: any) => n.ref_id === ref_id) ?? data.nodes[0]
        : data;
      if (!raw || !raw.ref_id) return { ok: false, error: `node not found: ${ref_id}` };
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

      return {
        ok: true,
        node: {
          ref_id: raw.ref_id,
          node_type: raw.node_type,
          name: deriveNodeName(raw, properties),
          properties: raw.properties,
          edges,
        },
      };
    } catch (err: any) {
      return { ok: false, error: `graph_get failed: ${err?.message ?? String(err)}` };
    }
  }

  allTools.graph_get = tool({
    description:
      "Resolve a single node in the Jarvis knowledge graph to its full content by ref_id. " +
      "Use the ref_id from graph_search or graph_neighbors results. " +
      "Returns the node's ref_id, node_type, derived name, properties, and an " +
      "`edges` map ({EDGE_TYPE: count}) showing how connected the node is and " +
      "which relationship types you can traverse next with graph_neighbors. " +
      "To resolve several ref_ids at once, use graph_get_batched instead.",
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
      console.log(`[graph_get] fetching ${ref_id}`);
      const res = await fetchGraphNode(ref_id, namespace);
      return res.ok ? JSON.stringify(res.node) : res.error;
    },
  });

  allTools.graph_get_batched = tool({
    description:
      `Resolve up to ${KG_BATCH_GET_MAX} nodes in one call by ref_id — the batched form of graph_get. ` +
      "ALWAYS prefer this over calling graph_get in a loop, and over delegating a list of " +
      "ref_ids to sub-agents: it fetches them concurrently in a single tool call. " +
      "Returns `{ requested, returned, truncated, omitted_ref_ids, nodes }`, where each entry in " +
      "`nodes` is either the full node (ref_id, node_type, name, properties, edges) or " +
      "`{ ref_id, error }` if that one could not be resolved — one bad ref_id never fails the rest. " +
      `If you pass more than ${KG_BATCH_GET_MAX} ref_ids, the excess comes back in ` +
      "`omitted_ref_ids` and `truncated` is true; call again with those to finish the job.",
    inputSchema: z.object({
      ref_ids: z
        .array(z.string())
        .min(1)
        .describe(
          `The ref_ids to resolve, in the order you want them back. Up to ${KG_BATCH_GET_MAX} per call.`,
        ),
      namespace: z
        .string()
        .optional()
        .describe(
          "Scope edge-count computation to a Jarvis namespace (data partition). " +
          "Only affects each node's `edges` map. Not an access-control boundary."
        ),
    }),
    execute: async ({
      ref_ids,
      namespace,
    }: {
      ref_ids: string[];
      namespace?: string;
    }) => {
      // Dedupe while preserving the caller's ordering — a repeated ref_id is a
      // wasted round trip, not a second entry.
      const unique = Array.from(new Set(ref_ids.filter((r) => r && r.trim())));
      if (unique.length === 0) {
        return JSON.stringify({
          requested: ref_ids.length,
          returned: 0,
          truncated: false,
          omitted_ref_ids: [],
          nodes: [],
          note: "no usable ref_ids supplied",
        });
      }

      const selected = unique.slice(0, KG_BATCH_GET_MAX);
      const omitted = unique.slice(KG_BATCH_GET_MAX);
      console.log(
        `[graph_get_batched] resolving ${selected.length} ref_ids (requested ${ref_ids.length}, omitted ${omitted.length}) namespace=${namespace ?? "*"}`,
      );

      const queue = new PQueue({ concurrency: KG_BATCH_GET_CONCURRENCY });
      const nodes = await Promise.all(
        selected.map((ref_id) =>
          queue.add(async () => {
            const res = await fetchGraphNode(ref_id, namespace);
            return res.ok ? res.node : { ref_id, error: res.error };
          }),
        ),
      );

      return JSON.stringify({
        requested: ref_ids.length,
        returned: nodes.length,
        truncated: omitted.length > 0,
        omitted_ref_ids: omitted,
        nodes,
      });
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
    "===> registered graph_search + get_ontology + get_ontology_type + graph_get + graph_get_batched + graph_neighbors tools",
  );

  // Recursive sub-agent tool, gated by config + depth so children can't spawn
  // forever. Registered only while the current depth is below maxDepth, meaning
  // leaf children never receive a `graph_sub_agent` tool.
  const sub = options.subAgent;
  if (sub) {
    const depth = sub.depth ?? 0;
    const maxDepth = sub.maxDepth ?? DEFAULT_SUBAGENT_MAX_DEPTH;
    if (depth < maxDepth) {
      registerGraphSubAgentTool(allTools, sub, depth, defaultDomains);
    }
  }

  // Ontology write tools — opt-in via toolsConfig.ontology_edit. Off by default
  // so the standard posture stays read-only.
  if (options.ontologyEdit) {
    registerOntologyWriteTools(allTools, jarvisUrl, jarvisHeaders);
  }

  // Graph data-write tool — opt-in via toolsConfig.create_triplet. Off by
  // default so the standard posture stays read-only.
  if (options.graphWrite) {
    registerGraphWriteTools(allTools, jarvisUrl, jarvisHeaders);
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