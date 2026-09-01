/**
 * jarvis-dialect node writes (`plans/jarvis-graph-compat.md` §1, §2, §6).
 *
 * Every node vein puts in the graph goes through `NodeWriter`. The writer
 * validates BEFORE building any Cypher (§6 — nothing invalid or unexpected
 * reaches the graph through vein), composes `node_key` exactly like jarvis
 * (`schema_validation.py:350-399`), builds the `Data_Bank` search text
 * (`schema_node_helper.py:141-234`, including jarvis's kitchen-sink fallback
 * for schemas without a usable explicit index), and MERGEs with jarvis's
 * label set and generic stamps (`schema_node_helper.py:238-268`).
 *
 * Types: Vein's own come from the in-code registry; any other type
 * (Document, EvalSet, Concept, …) is resolved from the live `:Schema`
 * meta-graph by `SchemaResolver` — the same source jarvis validates against
 * — so the writer is a drop-in for jarvis on jarvis-typed data too.
 *
 * Two modes:
 *   - `create`: no-op on an existing node (returns its ref_id) — except a
 *     soft-deleted, non-muted node, which is restored in place
 *     (`schema_node_helper.py:620-657`).
 *   - `upsert`: jarvis `reprocess` semantics — SET everything except the
 *     preserved identity (`ref_id`, `node_key`, `namespace`,
 *     `date_added_to_graph`). Unlike jarvis's reprocess, `Data_Bank` and
 *     the embeddings ARE rebuilt from the new payload, so the resulting
 *     state equals what a fresh create with that payload would produce.
 *
 * Embeddings are computed in-process before the write (no queue, no
 * pending marker — jarvis's "pending" state is simply `text_embeddings IS
 * NULL`). Without an `Embedder` the vectors stay NULL and the boot sweep
 * (`embeddings.ts`) heals them.
 */
import { createHash, randomUUID } from "node:crypto";
import type { ManagedTransaction } from "neo4j-driver";
import { Bolt, int, txRows } from "./bolt.js";
import { SchemaResolver, fromVein, type NodeSchema } from "./schema-resolver.js";
import { GENERIC_NODE_PROPERTIES, VEIN_DOMAIN_LABEL, embeddingColumn, getVeinSchema, typeLabelOf, vectorStem } from "./vein-schemas.js";

// ── Errors ──────────────────────────────────────────────────────────────────

export type GraphValidationCode =
  | "UNKNOWN_TYPE"
  | "MISSING_REQUIRED"
  | "UNKNOWN_ATTRIBUTE"
  | "WRONG_TYPE"
  | "INVALID_LIST"
  | "INVALID_DATETIME"
  | "EMPTY_NODE_KEY_TOKEN"
  | "NOT_FOUND"
  | "DUPLICATE_KEY";

export class GraphValidationError extends Error {
  readonly code: GraphValidationCode;
  readonly type: string;
  readonly attribute?: string;
  constructor(code: GraphValidationCode, type: string, message: string, attribute?: string) {
    super(`${type}${attribute ? `.${attribute}` : ""}: ${message}`);
    this.name = "GraphValidationError";
    this.code = code;
    this.type = type;
    this.attribute = attribute;
  }
}

// ── Validation + normalization (§6) ─────────────────────────────────────────

export interface ValidatedNode {
  schema: NodeSchema;
  /** Attribute values as JS primitives (datetime → epoch seconds), with
   *  null/undefined and empty strings dropped. Used for node_key and
   *  Data_Bank composition. */
  values: Record<string, unknown>;
  /** The same values as Neo4j parameter values (ints wrapped). */
  params: Record<string, unknown>;
}

const isOptionalType = (t: string) => t.startsWith("?");
const baseOf = (t: string) => (isOptionalType(t) ? t.slice(1) : t);

/**
 * The write-time gate. Throws `GraphValidationError` on the first violation
 * and writes nothing. Stricter than jarvis in the safe direction: anything
 * accepted here also passes jarvis's validators. Pass a Vein type name or
 * a resolved `NodeSchema` (from `SchemaResolver`) for any type.
 */
export function validateNode(typeOrSchema: string | NodeSchema, data: Record<string, unknown>): ValidatedNode {
  let schema: NodeSchema;
  if (typeof typeOrSchema === "string") {
    const vein = getVeinSchema(typeOrSchema);
    if (!vein) throw new GraphValidationError("UNKNOWN_TYPE", typeOrSchema, "not a registered Vein type (resolve other types via SchemaResolver)");
    schema = fromVein(vein);
  } else schema = typeOrSchema;
  const type = schema.type;
  const attrs = schema.attributes;

  // 3. Unknown attributes rejected — this is what makes "unexpected" nodes
  //    impossible, not just invalid ones. Generic/system props included.
  for (const key of Object.keys(data)) {
    if (!attrs[key] || GENERIC_NODE_PROPERTIES.has(key)) {
      throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, "attribute is not declared on the schema", key);
    }
  }

  const values: Record<string, unknown> = {};
  const params: Record<string, unknown> = {};
  for (const [name, t] of Object.entries(attrs)) {
    const raw = data[name];
    const base = baseOf(t);
    const present = raw !== null && raw !== undefined && !(typeof raw === "string" && raw.length === 0);
    if (!present) {
      // 2. Required attributes present and non-null. jarvis treats an empty
      //    string as "present" then silently never writes it; we reject it.
      if (!isOptionalType(t)) throw new GraphValidationError("MISSING_REQUIRED", type, `required ${base} attribute is missing`, name);
      continue;
    }
    // 4. Type checks per the grammar.
    const norm = normalizeValue(type, name, base, raw);
    values[name] = norm.value;
    params[name] = norm.param;
  }

  // 5. node_key integrity: every token after the type must be present
  //    (jarvis: "Attribute referenced in node_key not found in node data")
  //    and must not sanitize to nothing.
  for (const field of schema.node_key.split("-").slice(1)) {
    const lower = field.toLowerCase();
    const hit = Object.keys(values).find((k) => k.toLowerCase() === lower);
    if (!hit) throw new GraphValidationError("MISSING_REQUIRED", type, "attribute referenced in node_key is missing", field);
    if (sanitizeKeyValue(values[hit]).length === 0) {
      throw new GraphValidationError("EMPTY_NODE_KEY_TOKEN", type, "node_key attribute sanitizes to an empty token", field);
    }
  }

  // Keep the caller's key order (jarvis iterates the request payload, and
  // the kitchen-sink Data_Bank fallback depends on that order).
  const ordered: Record<string, unknown> = {};
  for (const k of Object.keys(data)) if (k in values) ordered[k] = values[k];
  return { schema, values: ordered, params };
}

function normalizeValue(type: string, name: string, base: string, raw: unknown): { value: unknown; param: unknown } {
  switch (base) {
    case "string":
      if (typeof raw !== "string") throw wrong(type, name, "string", raw);
      return { value: raw, param: raw };
    case "boolean":
      if (typeof raw !== "boolean") throw wrong(type, name, "boolean", raw);
      return { value: raw, param: raw };
    case "int":
      // Booleans are not ints (Python quirk we do not replicate).
      if (typeof raw !== "number" || !Number.isInteger(raw)) throw wrong(type, name, "int", raw);
      return { value: raw, param: int(raw) };
    case "float":
      // int-where-float is accepted (as in jarvis) and written as a FLOAT.
      if (typeof raw !== "number" || !Number.isFinite(raw)) throw wrong(type, name, "float", raw);
      return { value: raw, param: raw };
    case "datetime": {
      const secs = toEpochSeconds(raw);
      if (secs === null) throw new GraphValidationError("INVALID_DATETIME", type, "expected ISO-8601 string or epoch number", name);
      return { value: secs, param: int(secs) };
    }
    case "list": {
      if (!Array.isArray(raw)) throw wrong(type, name, "list", raw);
      // Neo4j lists must be homogeneous primitives.
      const kinds = new Set(raw.map((x) => typeof x));
      if (kinds.size > 1 || (kinds.size === 1 && !["string", "number", "boolean"].includes([...kinds][0]!))) {
        throw new GraphValidationError("INVALID_LIST", type, "list elements must be all strings, all numbers, or all booleans", name);
      }
      return { value: raw, param: raw };
    }
    default:
      // `complex` (jarvis grammar, never used by Vein): passthrough.
      return { value: raw, param: raw };
  }
}

function wrong(type: string, name: string, expected: string, raw: unknown): GraphValidationError {
  return new GraphValidationError("WRONG_TYPE", type, `expected ${expected}, got ${describe(raw)}`, name);
}

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}

/**
 * jarvis `TimeFormatter._convert_to_unix_timestamp`: ISO string (Z accepted)
 * or epoch number; numbers above 10^12 are milliseconds. Always epoch
 * SECONDS as an int. Returns null when unparseable.
 */
export function toEpochSeconds(v: unknown): number | null {
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return null;
    return Math.trunc(v > 1e12 ? v / 1000 : v);
  }
  if (typeof v === "string") {
    const s = v.trim();
    if (/^\d+$/.test(s)) return toEpochSeconds(Number(s));
    const ms = Date.parse(s.replace(/Z$/, "+00:00"));
    if (Number.isNaN(ms)) return null;
    return Math.trunc(ms / 1000);
  }
  if (v instanceof Date && !Number.isNaN(v.getTime())) return Math.trunc(v.getTime() / 1000);
  return null;
}

// ── node_key (§1) ───────────────────────────────────────────────────────────

export const MAX_NODE_KEY_LENGTH = 200;
export const NODE_KEY_HASH_LENGTH = 32;

/** `String(v).trim()` → drop spaces → lowercase → strip `[^a-zA-Z0-9\s]`. */
export function sanitizeKeyValue(v: unknown): string {
  return String(v)
    .trim()
    .replace(/ /g, "")
    .toLowerCase()
    .replace(/[^a-zA-Z0-9\s]/g, "");
}

/**
 * `sanitize_node_key` + `_compose_node_key`, verbatim. Property lookup is
 * case-insensitive; a missing property is an error. If the composed key
 * exceeds 200 chars, the value portion collapses to a 32-hex sha256 prefix.
 */
export function composeNodeKey(schema: { type: string; node_key: string }, values: Record<string, unknown>): string {
  const lower = new Map(Object.entries(values).map(([k, v]) => [k.toLowerCase(), v]));
  const parts: string[] = [];
  const tokens = schema.node_key.split("-");
  tokens.forEach((tok, i) => {
    if (i === 0) {
      parts.push(schema.type.toLowerCase().replace(/[^a-zA-Z0-9\s]/g, ""));
      return;
    }
    if (!lower.has(tok.toLowerCase())) throw new GraphValidationError("MISSING_REQUIRED", schema.type, "node_key property missing", tok);
    parts.push(sanitizeKeyValue(lower.get(tok.toLowerCase())));
  });
  const composed = parts.join("-");
  if (composed.length <= MAX_NODE_KEY_LENGTH || parts.length < 2) return composed;
  const digest = createHash("sha256").update(parts.slice(1).join("-"), "utf8").digest("hex").slice(0, NODE_KEY_HASH_LENGTH);
  return `${parts[0]}-${digest}`;
}

// ── Data_Bank (§2) ──────────────────────────────────────────────────────────

/** jarvis `DATA_BANK_EXCLUDED_FIELDS` — never part of kitchen-sink search text. */
export const DATA_BANK_EXCLUDED_FIELDS = new Set([
  "ref_id", "node_key", "date_added_to_graph", "namespace", "Data_Bank", "text_embeddings", "_search_fields_used",
  "id", "uuid", "tweet_id", "bounty_id", "episode_id", "video_id", "podcast_id", "clip_id", "message_id", "project_id",
  "org_uuid", "workspace_uuid", "phase_uuid", "owner_id", "pub_key", "pubkey",
  "created", "updated", "date", "timestamp", "created_at", "updated_at",
  "image_url", "profile_picture", "source_link", "link", "ticket_url", "media_url", "report_url",
  "old_value", "new_value",
  "contested", "boost", "num_boost", "price", "amount", "commitment_fee", "assigned_hours", "paid", "show", "completed",
  "verified", "impression_count", "sentiment_score", "relevancy_score",
  "start", "end", "line", "column", "operand", "output_data_type",
  "status", "type", "parent", "shape", "icon", "primary_color", "secondary_color", "phase_priority", "number",
]);

/** jarvis priority order for the kitchen-sink fallback. */
const PRIORITY_FIELDS = [
  "name", "title", "episode_title", "show_title", "description", "graph_description", "type_description", "text",
  "content", "body", "code", "docs", "summary", "one_sentence_summary", "deliverables", "github_description",
];

const nonBlank = (v: unknown) => v !== null && v !== undefined && String(v).trim().length > 0;

/**
 * `get_search_fields_for_node`: the schema's explicit `index` fields (in
 * declared order) that are present and non-blank; when the schema has no
 * real index (`["node_key"]`) or none of its index fields are usable, fall
 * through to jarvis's priority list + every other non-excluded property.
 */
export function searchFieldsFor(schema: NodeSchema, values: Record<string, unknown>): string[] {
  if (!(schema.index.length === 1 && schema.index[0] === "node_key")) {
    const focused = schema.index.filter((f) => nonBlank(values[f]));
    if (focused.length) return focused;
  }
  const out: string[] = [];
  const seen = new Set<string>();
  for (const f of PRIORITY_FIELDS) {
    if (f in values && !DATA_BANK_EXCLUDED_FIELDS.has(f) && nonBlank(values[f])) {
      out.push(f);
      seen.add(f);
    }
  }
  for (const [f, v] of Object.entries(values)) {
    if (DATA_BANK_EXCLUDED_FIELDS.has(f) || seen.has(f) || !nonBlank(v)) continue;
    out.push(f);
    seen.add(f);
  }
  return out;
}

/** `build_search_text`: values of the chosen fields, trimmed, joined with
 *  "\n" — no field-name prefixes. Null when nothing qualifies. */
export function buildSearchText(schema: NodeSchema, values: Record<string, unknown>): { text: string | null; fields: string[] } {
  const fields = searchFieldsFor(schema, values);
  const parts = fields.map((f) => String(values[f]).trim()).filter(Boolean);
  return { text: parts.length ? parts.join("\n") : null, fields };
}

/** jarvis `render_schema`: `"Input:\n{text}"` for `input_schema`, etc. */
export function renderVectorField(prop: string, text: string): string | null {
  const t = text.trim();
  if (!t) return null;
  const stem = vectorStem(prop);
  const cleaned = stem.replace(/_/g, " ").trim();
  const label = cleaned ? cleaned[0]!.toUpperCase() + cleaned.slice(1) : "Text";
  return `${label}:\n${t}`;
}

// ── Writer ──────────────────────────────────────────────────────────────────

export interface Embedder {
  /** One 384-float vector per input text, same order. */
  embed(texts: string[]): Promise<number[][]>;
}

export type WriteMode = "create" | "upsert";
export type WriteOutcome = "created" | "existing" | "restored" | "updated";

export interface NodeInput {
  /** Node type — a Vein type (exact) or any jarvis type (resolved
   *  case-insensitively against the live schema). */
  type: string;
  data: Record<string, unknown>;
}

export interface NodeWriteResult {
  ref_id: string;
  node_key: string;
  /** Canonical type label the node was written under. */
  node_type: string;
  outcome: WriteOutcome;
}

export interface NodeWriterOptions {
  embedder?: Embedder;
  /** Shared resolver (one per backend); a private one is created otherwise. */
  resolver?: SchemaResolver;
}

export interface WriteOptions {
  /** Override the backend's default namespace for this write. Must already
   *  be registered when it is not `default` (callers check via
   *  `GraphReader.resolveNamespace`). */
  namespace?: string;
}

export interface NodeUpdate {
  /** Properties to set/overwrite (validated against the schema after
   *  merging over the node's current attributes). */
  set?: Record<string, unknown>;
  /** Attribute names to remove. Required attributes cannot be removed. */
  remove?: string[];
}

export interface NodeUpdateResult {
  ref_id: string;
  node_key: string;
  /** True when the update changed the node_key (identity attrs edited). */
  rekeyed: boolean;
}

/** Identity props never touched after create. */
const PRESERVED = new Set(["ref_id", "node_key", "namespace", "date_added_to_graph"]);

interface PreparedNode {
  schema: NodeSchema;
  node_key: string;
  ref_id: string;
  onCreate: Record<string, unknown>;
  onMatch: Record<string, unknown>;
}

export class NodeWriter {
  readonly resolver: SchemaResolver;
  constructor(
    private readonly bolt: Bolt,
    private readonly opts: NodeWriterOptions = {},
  ) {
    this.resolver = opts.resolver ?? new SchemaResolver(bolt);
  }

  /** Validate + compose everything except the MERGE. Exposed for tests and
   *  the batch path. */
  async prepare(input: NodeInput, opts: WriteOptions = {}): Promise<PreparedNode> {
    const [p] = await this.prepareMany([input], opts);
    return p!;
  }

  async prepareMany(inputs: NodeInput[], opts: WriteOptions = {}): Promise<PreparedNode[]> {
    const namespace = opts.namespace ?? this.bolt.namespace;
    const validated: ValidatedNode[] = [];
    for (const i of inputs) {
      const schema = await this.resolver.schema(i.type);
      if (!schema) throw new GraphValidationError("UNKNOWN_TYPE", i.type, "unknown node type (not a Vein type and no such Schema in the graph)");
      validated.push(validateNode(schema, i.data));
    }
    // Gather every text to embed across the batch → one encoder call.
    const jobs: Array<{ idx: number; column: string; text: string }> = [];
    const prepared: PreparedNode[] = validated.map((v, idx) => {
      const node_key = composeNodeKey(v.schema, v.values);
      const ref_id = randomUUID();
      const { text, fields } = buildSearchText(v.schema, v.values);
      const props: Record<string, unknown> = { ...v.params };
      if (text !== null) {
        props["Data_Bank"] = text;
        props["_search_fields_used"] = fields;
        jobs.push({ idx, column: "text_embeddings", text });
      }
      for (const vi of v.schema.vector_index) {
        const raw = v.values[vi];
        if (typeof raw !== "string") continue;
        const rendered = renderVectorField(vi, raw);
        if (rendered) jobs.push({ idx, column: embeddingColumn(vi), text: rendered });
      }
      const onMatch: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(props)) if (!PRESERVED.has(k)) onMatch[k] = val;
      return {
        schema: v.schema,
        node_key,
        ref_id,
        onCreate: { ...props, ref_id, node_key, namespace, date_added_to_graph: int(Date.now()) },
        onMatch,
      };
    });
    if (this.opts.embedder && jobs.length) {
      const vectors = await this.opts.embedder.embed(jobs.map((j) => j.text));
      jobs.forEach((j, i) => {
        const p = prepared[j.idx]!;
        p.onCreate[j.column] = vectors[i];
        p.onMatch[j.column] = vectors[i];
      });
    }
    return prepared;
  }

  /** Write one node. */
  async write(input: NodeInput, mode: WriteMode = "create", opts: WriteOptions = {}): Promise<NodeWriteResult> {
    const [r] = await this.writeMany([input], mode, opts);
    return r!;
  }

  /**
   * Write many nodes (any mix of types) in one transaction — one UNWIND
   * MERGE per type, same resulting state as the single form. Results are in
   * input order. All-or-nothing: a validation error anywhere writes nothing.
   */
  async writeMany(inputs: NodeInput[], mode: WriteMode = "create", opts: WriteOptions = {}): Promise<NodeWriteResult[]> {
    if (inputs.length === 0) return [];
    const namespace = opts.namespace ?? this.bolt.namespace;
    const prepared = await this.prepareMany(inputs, { namespace });
    const byType = new Map<string, Array<{ i: number; p: PreparedNode }>>();
    prepared.forEach((p, i) => {
      const list = byType.get(p.schema.type) ?? [];
      list.push({ i, p });
      byType.set(p.schema.type, list);
    });
    const results: NodeWriteResult[] = new Array(inputs.length);
    await this.bolt.write(async (tx) => {
      for (const [type, rows] of byType) {
        const out = await mergeBatch(tx, rows[0]!.p.schema, namespace, rows.map((r) => r.p), mode);
        rows.forEach((r, k) => {
          results[r.i] = { ...out[k]!, node_type: type };
        });
      }
    });
    return results;
  }

  /**
   * Partial update by ref_id (jarvis `POST /v2/nodes/:ref_id` with
   * `node_data` / `properties_to_be_deleted`): the patch is merged over the
   * node's current attributes, the merged payload is re-validated as a
   * whole, `node_key` is recomposed (an identity edit that collides with
   * another node fails with DUPLICATE_KEY, nothing written), and
   * `Data_Bank` + vectors are rebuilt. Identity stamps (`ref_id`,
   * `namespace`, `date_added_to_graph`) and `is_deleted` are untouched.
   */
  async update(ref_id: string, patch: NodeUpdate): Promise<NodeUpdateResult> {
    const remove = patch.remove ?? [];
    const set = patch.set ?? {};
    return this.bolt.write(async (tx) => {
      const rows = await txRows(tx, `MATCH (n:Data_Bank {ref_id: $ref_id}) RETURN labels(n) AS labels, properties(n) AS props`, { ref_id });
      if (rows.length === 0) throw new GraphValidationError("NOT_FOUND", "?", `no node with ref_id ${ref_id}`);
      const labels = rows[0]!["labels"] as string[];
      const props = rows[0]!["props"] as Record<string, unknown>;
      const type = typeLabelOf(labels) ?? "?";
      const schema = await this.resolver.schema(type, tx);
      if (!schema) throw new GraphValidationError("UNKNOWN_TYPE", type, "stored node's type has no schema");
      const attrs = schema.attributes;

      const current: Record<string, unknown> = {};
      for (const k of Object.keys(attrs)) if (props[k] !== undefined && props[k] !== null) current[k] = props[k];
      for (const k of remove) {
        const t = attrs[k];
        if (!t) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, "attribute is not declared on the schema", k);
        if (!isOptionalType(t)) throw new GraphValidationError("MISSING_REQUIRED", type, "required attribute cannot be removed", k);
        delete current[k];
      }
      const merged = { ...current, ...set };
      for (const k of remove) if (k in set) delete merged[k];

      const [p] = await this.prepareMany([{ type: schema.type, data: merged }], { namespace: String(props["namespace"]) });
      const node_key = p!.node_key;
      const rekeyed = node_key !== props["node_key"];
      if (rekeyed) {
        const clash = await txRows(
          tx,
          `MATCH (m:\`${schema.type}\` {node_key: $k, namespace: $ns}) WHERE m.ref_id <> $ref_id RETURN m.ref_id AS ref_id LIMIT 1`,
          { k: node_key, ns: props["namespace"], ref_id },
        );
        if (clash.length) throw new GraphValidationError("DUPLICATE_KEY", type, `Node already exists in the graph with node_key: ${node_key}`);
      }

      // Everything the fresh payload would carry, minus identity stamps.
      const setMap: Record<string, unknown> = { ...p!.onMatch, node_key };
      const drop = new Set<string>(remove);
      if (!("Data_Bank" in setMap)) for (const k of ["Data_Bank", "_search_fields_used", "text_embeddings"]) drop.add(k);
      else if (!this.opts.embedder && setMap["Data_Bank"] !== props["Data_Bank"]) drop.add("text_embeddings");
      for (const vi of schema.vector_index) {
        const col = embeddingColumn(vi);
        if (!(vi in merged)) drop.add(col);
        else if (!this.opts.embedder && merged[vi] !== props[vi]) drop.add(col);
      }
      for (const k of drop) delete setMap[k];
      // Attributes present before but absent from the merged payload
      // (e.g. explicitly set to null) go too.
      for (const k of Object.keys(attrs)) if (k in props && !(k in setMap) && !drop.has(k) && !(k in merged)) drop.add(k);
      const removeClause = drop.size ? `REMOVE ${[...drop].map((k) => `n.\`${k}\``).join(", ")}` : "";
      await tx.run(`MATCH (n:Data_Bank {ref_id: $ref_id}) SET n += $set ${removeClause}`, { ref_id, set: setMap });
      return { ref_id, node_key, rekeyed };
    });
  }

  /** Soft delete (`is_deleted = true`). Scoped to Vein's own nodes. */
  async softDelete(ref_id: string): Promise<boolean> {
    const rows = await this.bolt.run(
      `MATCH (n:\`${VEIN_DOMAIN_LABEL}\` {ref_id: $ref_id}) SET n.is_deleted = true RETURN n.ref_id AS ref_id`,
      { ref_id },
    );
    return rows.length > 0;
  }
}

/**
 * The one Cypher template, UNWIND form. Labels = `Type` + Node + Data_Bank
 * + the schema's Domain_* labels; identity = (node_key, namespace). ON
 * CREATE gets the full stamped payload; ON MATCH gets the non-identity
 * payload only when the mode is `upsert` or the node is soft-deleted-and-
 * not-muted (restore). `is_deleted` is cleared only in those cases and is
 * otherwise left exactly as it was (a SET to its own value is a no-op, so
 * absent stays absent).
 *
 * Outcome per row comes from a pre-read in the same transaction; the MERGE
 * itself stays race-safe (a concurrent create just turns into a match).
 */
async function mergeBatch(
  tx: ManagedTransaction,
  schema: NodeSchema,
  namespace: string,
  nodes: PreparedNode[],
  mode: WriteMode,
): Promise<Array<Omit<NodeWriteResult, "node_type">>> {
  const type = schema.type;
  const labels = [`\`${type}\``, "Node", "Data_Bank", ...schema.domainLabels.map((l) => `\`${l}\``)].join(":");
  const before = await txRows(
    tx,
    `UNWIND $keys AS k
     MATCH (n:\`${type}\` {node_key: k, namespace: $ns})
     RETURN k AS node_key, coalesce(n.is_deleted, false) AS deleted, coalesce(n.is_muted, false) AS muted`,
    { keys: nodes.map((n) => n.node_key), ns: namespace },
  );
  const state = new Map(before.map((r) => [r["node_key"] as string, { deleted: r["deleted"] as boolean, muted: r["muted"] as boolean }]));

  const rows = await txRows(
    tx,
    `UNWIND $rows AS row
     MERGE (node:${labels} {node_key: row.node_key, namespace: $ns})
     ON CREATE SET node += row.on_create
     ON MATCH SET
       node += CASE
         WHEN $mode = 'upsert' THEN row.on_match
         WHEN coalesce(node.is_deleted, false) AND NOT coalesce(node.is_muted, false) THEN row.on_match
         ELSE {} END,
       node.is_deleted = CASE
         WHEN coalesce(node.is_deleted, false) AND NOT coalesce(node.is_muted, false) THEN false
         ELSE node.is_deleted END
     RETURN row.node_key AS node_key, node.ref_id AS ref_id, node.ref_id = row.on_create.ref_id AS created`,
    {
      rows: nodes.map((n) => ({ node_key: n.node_key, on_create: n.onCreate, on_match: n.onMatch })),
      ns: namespace,
      mode,
    },
  );
  const byKey = new Map(rows.map((r) => [r["node_key"] as string, r]));
  return nodes.map((n) => {
    const r = byKey.get(n.node_key)!;
    const created = r["created"] as boolean;
    const prior = state.get(n.node_key);
    let outcome: WriteOutcome;
    if (created) outcome = "created";
    else if (prior?.deleted && !prior.muted) outcome = "restored";
    else if (mode === "upsert") outcome = "updated";
    else outcome = "existing";
    return { ref_id: r["ref_id"] as string, node_key: n.node_key, outcome };
  });
}
