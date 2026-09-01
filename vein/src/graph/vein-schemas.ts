/**
 * The Vein schema library — a TypeScript mirror of one jarvis schema library
 * (`schema_library.py` in jarvis-backend). Every node type vein writes to the
 * graph is declared here, and ONLY here: the node writer rejects any type not
 * in `VEIN_SCHEMAS`, any attribute not declared on its schema, and any edge
 * not in `VEIN_EDGES` (plan §6).
 *
 * Vocabulary (labels + edges) comes from the label registry in
 * `plans/generic-storage.md`; key/index choices from `jarvis-graph-compat.md`
 * §5. Do not invent labels — every one here was verified absent from jarvis's
 * own library.
 *
 * Attribute grammar (jarvis `_assert_is_valid_schema`):
 *   string | boolean | int | float | datetime | list, optionally `?`-prefixed
 *   for optional. `datetime` values are normalized to epoch SECONDS (int)
 *   before write. jarvis also knows `complex`; vein never uses it.
 */

export type AttrBase = "string" | "boolean" | "int" | "float" | "datetime" | "list";
export type AttrType = AttrBase | `?${AttrBase}`;

export interface VeinSchema {
  type: string;
  parent: "Thing";
  domain: "Vein";
  /** `-`-joined spec: token 0 = type.lower(), then attribute names. */
  node_key: string;
  /** Fields that build `Data_Bank` (search text + `text_embeddings`), in
   *  declared order. Keep large payloads OUT of here. */
  index: string[];
  /** Fields that get their own `{stem}_embeddings` vector + per-label vector
   *  index (jarvis `vector_index`). Only the `input_q`/`output_q` search
   *  types declare this. */
  vector_index?: string[];
  icon: string;
  shape: string;
  primary_color: string;
  secondary_color: string;
  title_key: string;
  description_key: string;
  type_description: string;
  attributes: Record<string, AttrType>;
}

/** One row of the edge registry: (source label, edge type, target label). */
export interface VeinEdgeDef {
  edge: string;
  source: string;
  target: string;
  /** Jarvis schema types outside the Vein domain that this edge points at
   *  (e.g. `Person`, `Thing`). Seeding skips the edge-schema row when the
   *  target Schema node is absent (standalone mode without that type). */
  note?: string;
}

// ── Constants shared with jarvis ────────────────────────────────────────────

export const VEIN_DOMAIN = "Vein";
export const VEIN_DOMAIN_LABEL = "Domain_vein";
export const THING_TYPE = "Thing";

/** Colour pair for every Vein type (one of jarvis's `get_color_pairs()`). */
const COLORS = { primary_color: "#1D3140", secondary_color: "#4FA7D9" };

/**
 * jarvis's root schema, verbatim (`default_schemas.py:11-33`). Seeded only in
 * standalone mode; a jarvis-seeded `Thing` is never touched. `name` sits at
 * the top level in jarvis's dict, so it flattens onto the node like an
 * attribute — every type inherits it.
 */
export const THING_SCHEMA = {
  type: "Thing",
  name: "string",
  node_key: "thing-name",
  index: ["name", "description"],
  icon: "NodesIcon",
  shape: "sphere",
  primary_color: "#36292D",
  secondary_color: "#A96755",
  type_description:
    "The highest-level node in the ontology hierarchy, representing an abstract concept with no direct individual instances",
  title_key: "name",
  description_key: "description",
  attributes: {
    description: "?string",
    weight: "?float",
    is_muted: "?boolean",
    unique_source_id: "?string",
    image_url: "?string",
  } as Record<string, AttrType>,
} as const;

/** Attributes every type inherits from `Thing` (jarvis `get_schema` walks
 *  CHILD_OF and unions the parent's flattened keys). */
export const THING_INHERITED_ATTRIBUTES: Record<string, AttrType> = {
  name: "?string",
  ...THING_SCHEMA.attributes,
};

/** jarvis `USAGE_ATTRIBUTES` — never written by jarvis, read by its
 *  `?sort=usage` and search tiebreak. Vein owns updating them. */
export const USAGE_ATTRIBUTES: Record<string, AttrType> = {
  usage_count: "?int",
  usage_count_30d: "?int",
};

/** Properties jarvis stamps on every node, never declared as attributes and
 *  never accepted from a caller (`GENERIC_NODE_PROPERTIES` + the write-time
 *  system props). */
export const GENERIC_NODE_PROPERTIES = new Set([
  "Data_Bank",
  "namespace",
  "spelling_verification",
  "topic_lower",
  "ref_id",
  "node_key",
  "relevancy_score",
  "date_added_to_graph",
  "updated_at",
  "text_embeddings",
  "input_embeddings",
  "output_embeddings",
  "embeddings",
  "algo_page_rank",
  "algo_score",
  "algo_community_id",
  "algo_embedding",
  "_search_fields_used",
  "is_deleted",
]);

/** Attribute names a schema may not declare (jarvis reserved keys). */
export const RESERVED_ATTRIBUTE_NAMES = new Set(["type", "parent", "node_key", "index"]);

/** Preview fields are capped so search/embedding text stays light; full
 *  payloads stay in the run/chat log behind `log_ref`. */
export const PREVIEW_MAX_CHARS = 500;

// ── The nine Vein node types ────────────────────────────────────────────────

const base = {
  parent: "Thing",
  domain: "Vein",
  icon: "NodesIcon",
  shape: "sphere",
  ...COLORS,
} as const;

export const VEIN_SCHEMAS: readonly VeinSchema[] = [
  {
    ...base,
    type: "VeinWorkflow",
    node_key: "veinworkflow-name",
    index: ["name", "description"],
    title_key: "name",
    description_key: "description",
    type_description: "A vein workflow by name — the stable identity across versions",
    attributes: {
      name: "string",
      description: "?string",
      category: "?string",
      publisher: "?string",
      /** Content hash of the active version (mirrors `ACTIVE_VERSION`). */
      active_version: "?string",
      ...USAGE_ATTRIBUTES,
    },
  },
  {
    ...base,
    type: "VeinWorkflowVersion",
    node_key: "veinworkflowversion-name-content_hash",
    index: ["name", "description"],
    vector_index: ["input_schema", "output_schema"],
    title_key: "name",
    description_key: "description",
    type_description: "One content-hashed version of a vein workflow",
    attributes: {
      name: "string",
      content_hash: "string",
      version_label: "?string",
      description: "?string",
      created_at: "datetime",
      /** Full workflow source — deliberately NOT indexed. */
      source: "?string",
      input_schema: "?string",
      output_schema: "?string",
      params_json: "?string",
      publisher: "?string",
    },
  },
  {
    ...base,
    type: "VeinStep",
    node_key: "veinstep-step_type",
    index: ["step_type", "description"],
    vector_index: ["input_schema", "output_schema"],
    title_key: "step_type",
    description_key: "description",
    type_description: "A published vein step type (custom tier)",
    attributes: {
      step_type: "string",
      description: "?string",
      publisher: "?string",
      active_version: "?string",
      input_schema: "?string",
      output_schema: "?string",
      ...USAGE_ATTRIBUTES,
    },
  },
  {
    ...base,
    type: "VeinStepVersion",
    node_key: "veinstepversion-step_type-content_hash",
    index: ["step_type", "description"],
    title_key: "step_type",
    description_key: "description",
    type_description: "One version of a vein step's source",
    attributes: {
      step_type: "string",
      content_hash: "string",
      version_label: "?string",
      description: "?string",
      created_at: "datetime",
      /** Full step source — deliberately NOT indexed. */
      source: "?string",
      publisher: "?string",
    },
  },
  {
    ...base,
    type: "VeinRun",
    node_key: "veinrun-run_id",
    index: ["workflow_name", "status", "summary"],
    title_key: "workflow_name",
    description_key: "summary",
    type_description: "One vein workflow run — status, timings, params, and a pointer to its log",
    attributes: {
      run_id: "string",
      workflow_name: "string",
      status: "string",
      summary: "?string",
      started_at: "datetime",
      finished_at: "?datetime",
      duration_ms: "?int",
      workflow_hash: "?string",
      params_json: "?string",
      input_preview: "?string",
      output_preview: "?string",
      error_message: "?string",
      /** Pointer into the raw run log (store-specific locator). */
      log_ref: "?string",
    },
  },
  {
    ...base,
    type: "VeinAgentSession",
    node_key: "veinagentsession-run_id-path",
    index: ["prompt_preview", "result_preview"],
    title_key: "path",
    description_key: "prompt_preview",
    type_description: "One agent-step execution inside a vein run",
    attributes: {
      run_id: "string",
      path: "string",
      step_type: "?string",
      model: "?string",
      iteration: "?int",
      prompt_preview: "?string",
      result_preview: "?string",
      started_at: "?datetime",
      duration_ms: "?int",
      error_message: "?string",
      log_ref: "?string",
    },
  },
  {
    ...base,
    type: "VeinToolCall",
    node_key: "veintoolcall-run_id-path-seq",
    index: ["tool_name", "input_preview"],
    title_key: "tool_name",
    description_key: "input_preview",
    type_description: "One tool call inside a vein agent session",
    attributes: {
      run_id: "string",
      path: "string",
      seq: "int",
      tool_name: "string",
      input_preview: "?string",
      output_preview: "?string",
      started_at: "?datetime",
      duration_ms: "?int",
      error_message: "?string",
      log_ref: "?string",
    },
  },
  {
    ...base,
    type: "VeinChat",
    node_key: "veinchat-chat_id",
    index: ["title", "summary"],
    title_key: "title",
    description_key: "summary",
    type_description: "A long-lived vein chat",
    attributes: {
      chat_id: "string",
      title: "?string",
      summary: "?string",
      status: "?string",
      model: "?string",
      created_at: "datetime",
      last_active_at: "?datetime",
      turn_count: "?int",
      log_ref: "?string",
    },
  },
  {
    ...base,
    type: "VeinTurn",
    node_key: "veinturn-chat_id-turn",
    index: ["user_text_preview"],
    title_key: "user_text_preview",
    description_key: "assistant_text_preview",
    type_description: "One turn of a vein chat",
    attributes: {
      chat_id: "string",
      turn: "int",
      user_text_preview: "?string",
      assistant_text_preview: "?string",
      started_at: "?datetime",
      log_ref: "?string",
    },
  },
];

// ── Edge registry ───────────────────────────────────────────────────────────

/** Every (source, edge, target) vein may write. `ACCESSED` is declared
 *  against `Thing` — provenance may point at ANY node — and the writer
 *  accepts any target label for it (plan §6 item 6). */
export const VEIN_EDGES: readonly VeinEdgeDef[] = [
  { edge: "VERSION_OF", source: "VeinWorkflowVersion", target: "VeinWorkflow" },
  { edge: "VERSION_OF", source: "VeinStepVersion", target: "VeinStep" },
  { edge: "ACTIVE_VERSION", source: "VeinWorkflow", target: "VeinWorkflowVersion" },
  { edge: "ACTIVE_VERSION", source: "VeinStep", target: "VeinStepVersion" },
  { edge: "USES_STEP", source: "VeinWorkflowVersion", target: "VeinStep" },
  { edge: "DEPENDS_ON", source: "VeinWorkflowVersion", target: "VeinWorkflow" },
  { edge: "PUBLISHED_BY", source: "VeinStepVersion", target: "Person", note: "jarvis type; seeded only when Person exists" },
  { edge: "EXECUTED", source: "VeinRun", target: "VeinWorkflowVersion" },
  { edge: "PROMOTED_FROM", source: "VeinWorkflowVersion", target: "VeinRun" },
  { edge: "IN_RUN", source: "VeinAgentSession", target: "VeinRun" },
  { edge: "IN_SESSION", source: "VeinToolCall", target: "VeinAgentSession" },
  { edge: "SPAWNED", source: "VeinChat", target: "VeinRun" },
  { edge: "IN_CHAT", source: "VeinTurn", target: "VeinChat" },
  { edge: "ACCESSED", source: "VeinToolCall", target: "Thing", note: "any node" },
];

/** Edge types whose declared target is a wildcard (any node label). */
export const WILDCARD_TARGET_EDGES = new Set(["ACCESSED"]);

// ── Lookups ─────────────────────────────────────────────────────────────────

const BY_TYPE = new Map(VEIN_SCHEMAS.map((s) => [s.type, s]));

/** Exact-match lookup (jarvis resolves case-insensitively; we don't). */
export function getVeinSchema(type: string): VeinSchema | undefined {
  return BY_TYPE.get(type);
}

export function isVeinType(type: string): boolean {
  return BY_TYPE.has(type);
}

/** A schema's full attribute map: its own attributes + `Thing`'s. This is
 *  what jarvis's `get_schema` returns (minus the core keys), so it is what
 *  the validator checks payloads against. */
export function effectiveAttributes(schema: VeinSchema): Record<string, AttrType> {
  return { ...THING_INHERITED_ATTRIBUTES, ...schema.attributes };
}

export function isOptional(t: AttrType): boolean {
  return t.startsWith("?");
}

export function baseType(t: AttrType): AttrBase {
  return (isOptional(t) ? t.slice(1) : t) as AttrBase;
}

/** `input_schema` → `input`; anything else unchanged (jarvis `stem`). */
export function vectorStem(prop: string): string {
  return prop.endsWith("_schema") ? prop.slice(0, -"_schema".length) : prop;
}

/** `{stem}_embeddings` (jarvis `embedding_column`). */
export function embeddingColumn(prop: string): string {
  return `${vectorStem(prop)}_embeddings`;
}

/** `{label.lower()}_{stem}_vector_index` (jarvis `vector_index_name`). */
export function vectorIndexName(type: string, prop: string): string {
  return `${type.toLowerCase()}_${vectorStem(prop)}_vector_index`;
}

/** Node-key tokens after the type token. */
export function nodeKeyFields(schema: VeinSchema): string[] {
  return schema.node_key.split("-").slice(1);
}

/**
 * Searchable attributes across the Vein library, per jarvis's tier-1 rule
 * for schemas with an explicit `index` (`get_searchable_attributes_from_schema`):
 * index fields + title_key + description_key. Used to build the domain
 * fulltext index (sorted, plus `node_key` appended by the seeder).
 */
export function searchableAttributes(schemas: readonly VeinSchema[] = VEIN_SCHEMAS): string[] {
  const out = new Set<string>();
  for (const s of schemas) {
    for (const f of s.index) out.add(f);
    out.add(s.title_key);
    out.add(s.description_key);
  }
  return [...out].sort();
}

/** (type, property) pairs that declare a per-property vector index. */
export function vectorIndexedPairs(
  schemas: readonly VeinSchema[] = VEIN_SCHEMAS,
): Array<{ type: string; prop: string }> {
  const out: Array<{ type: string; prop: string }> = [];
  for (const s of schemas) for (const p of s.vector_index ?? []) out.push({ type: s.type, prop: p });
  return out;
}

// ── Author-time validation ──────────────────────────────────────────────────

const IDENT = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const EDGE_TYPE = /^[A-Z][A-Z0-9_]*$/;
const ATTR_TYPES = new Set<string>(["string", "boolean", "int", "float", "datetime", "list"]);

/**
 * Structural checks on the library itself (mirrors jarvis `valid_node_key`
 * `schema_validation.py:311-338` + `_assert_is_valid_schema`, tightened):
 * bare-identifier attribute names, no reserved names, every node_key token
 * a REQUIRED attribute, index/vector_index/title/description fields
 * declared, edge types uppercase, edge endpoints known. Throws on the first
 * violation. Runs once at module load so a bad edit fails fast.
 */
export function assertLibraryWellFormed(
  schemas: readonly VeinSchema[] = VEIN_SCHEMAS,
  edges: readonly VeinEdgeDef[] = VEIN_EDGES,
): void {
  const types = new Set<string>();
  for (const s of schemas) {
    if (!IDENT.test(s.type)) throw new Error(`schema ${s.type}: type is not a bare identifier`);
    if (types.has(s.type)) throw new Error(`schema ${s.type}: duplicate type`);
    types.add(s.type);
    if (s.parent !== THING_TYPE) throw new Error(`schema ${s.type}: parent must be Thing`);
    if (s.domain !== VEIN_DOMAIN) throw new Error(`schema ${s.type}: domain must be Vein`);

    for (const [name, t] of Object.entries(s.attributes)) {
      if (!IDENT.test(name)) throw new Error(`schema ${s.type}: attribute "${name}" is not a bare identifier`);
      if (RESERVED_ATTRIBUTE_NAMES.has(name)) throw new Error(`schema ${s.type}: attribute "${name}" is reserved`);
      if (GENERIC_NODE_PROPERTIES.has(name)) throw new Error(`schema ${s.type}: attribute "${name}" is a generic node property`);
      if (!ATTR_TYPES.has(baseType(t))) throw new Error(`schema ${s.type}: attribute "${name}" has unknown type "${t}"`);
    }

    const tokens = s.node_key.split("-");
    if (s.node_key.startsWith("-") || s.node_key.endsWith("-") || tokens.length < 2) {
      throw new Error(`schema ${s.type}: malformed node_key "${s.node_key}"`);
    }
    if (tokens[0] !== s.type.toLowerCase()) {
      throw new Error(`schema ${s.type}: node_key must start with "${s.type.toLowerCase()}"`);
    }
    for (const tok of tokens.slice(1)) {
      const t = s.attributes[tok];
      if (!t) throw new Error(`schema ${s.type}: node_key token "${tok}" is not a declared attribute`);
      if (isOptional(t)) throw new Error(`schema ${s.type}: node_key token "${tok}" must be required`);
    }

    const declared = effectiveAttributes(s);
    if (s.index.length === 0) throw new Error(`schema ${s.type}: index must be non-empty`);
    for (const f of [...s.index, ...(s.vector_index ?? []), s.title_key, s.description_key]) {
      if (!declared[f]) throw new Error(`schema ${s.type}: field "${f}" is not a declared attribute`);
    }
    for (const f of s.vector_index ?? []) {
      if (baseType(declared[f]!) !== "string") throw new Error(`schema ${s.type}: vector_index field "${f}" must be a string`);
    }
  }

  for (const e of edges) {
    if (!EDGE_TYPE.test(e.edge)) throw new Error(`edge ${e.edge}: type must match ^[A-Z][A-Z0-9_]*$`);
    if (!types.has(e.source)) throw new Error(`edge ${e.edge}: source "${e.source}" is not a Vein type`);
    if (!types.has(e.target) && !e.note) {
      throw new Error(`edge ${e.edge}: target "${e.target}" is not a Vein type (add a note if it is a jarvis type)`);
    }
  }
}

assertLibraryWellFormed();
