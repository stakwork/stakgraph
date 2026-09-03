/**
 * Node-schema registration — the write side of the `:Schema` meta-graph for
 * types that are NOT Vein's own, mirroring jarvis's `POST /v2/schema`
 * (`schema_service.create_schema` + `schema_crud.create_schema`):
 *
 *   - the parent must already be a Schema (default `Thing`); the type must
 *     not (`ERROR_SCHEMA_EXISTS`), except that a soft-deleted one is
 *     replaced, as jarvis's service does;
 *   - attributes are `name → type string` over jarvis's grammar
 *     (`string | boolean | int | float | complex | datetime | list`, `?` =
 *     optional); names are bare identifiers and may not reuse a core schema
 *     key or a generic node property;
 *   - `node_key` is `-`-joined attribute tokens, stored as
 *     `<type lower>-<tokens>`; every token must be a declared attribute
 *     (`name` is inherited from Thing and always available);
 *   - `index` defaults to the node_key tokens and must name attributes;
 *   - the schema is written flat (core keys + attributes as top-level
 *     properties, like `schema-seed.ts`), linked `CHILD_OF` its parent, and
 *     given the per-type `(node_key, namespace)` uniqueness constraint +
 *     node_key index. The domain's search indexes are created when absent
 *     (`ensure_indexes_for_schema`).
 *
 * `createNodeSchema` also EXTENDS an existing non-Vein schema — add-only,
 * the way `seedVeinDomain` reconciles: attributes the schema lacks are
 * added, nothing existing is changed, identity keys (`type`, `parent`,
 * `node_key`, `index`) are never touched. That is what lets a workflow add
 * `verdict` to a jarvis-seeded `Claim` without editing the ontology by hand.
 *
 * Vein's own types are a closed in-code registry (`vein-schemas.ts`) and
 * are refused here.
 */
import { randomUUID } from "node:crypto";
import type { Bolt } from "./bolt.js";
import { txRows } from "./bolt.js";
import { GraphValidationError } from "./node-writer.js";
import { searchableAttributesOf } from "./ontology-seed.js";
import type { SchemaResolver } from "./schema-resolver.js";
import { schemaStatement } from "./schema-seed.js";
import { GENERIC_NODE_PROPERTIES, RESERVED_ATTRIBUTE_NAMES, THING_TYPE, isVeinType } from "./vein-schemas.js";

export interface NodeSchemaInput {
  /** New type label, e.g. `Evidence`. */
  type: string;
  /** Parent type (must exist as a Schema). Default `Thing`. */
  parent?: string;
  /** Attribute name → jarvis type string (`?` prefix = optional). */
  attributes: Record<string, string>;
  /** `-`-joined attribute tokens that identify a node, e.g. `name` or
   *  `claim_text-speaker_name`. Default `name`. A `<type>-` prefix is
   *  accepted and not doubled. */
  node_key?: string;
  /** Searchable attributes; default = the node_key tokens. */
  index?: string[];
  title_key?: string;
  description_key?: string;
  /** Domain the type belongs to (→ `Domain_<domain>` label). Default `entity`. */
  domain?: string;
  type_description?: string;
}

/** A validated, normalized schema ready to write. */
export interface NodeSchemaPlan {
  type: string;
  parent: string;
  node_key: string;
  tokens: string[];
  index: string[];
  domain: string;
  attributes: Record<string, string>;
  /** The flat `:Schema` node properties (core keys + attributes). */
  flat: Record<string, unknown>;
}

export interface NodeSchemaResult {
  /** True when a new Schema node was written; false when an existing one
   *  was extended (or already had everything). */
  created: boolean;
  ref_id: string;
  type: string;
  parent: string;
  node_key: string;
  /** Attributes added to an existing schema (always empty on create). */
  added: string[];
  /** Full effective attribute map after the write (inherited included). */
  attributes: Record<string, string>;
}

const TYPE_LABEL = /^[A-Za-z][A-Za-z0-9_]*$/;
const IDENT = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const TYPE_GRAMMAR = /^\??(string|boolean|int|float|complex|datetime|list)$/;
const VECTOR_OPTIONS = "OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } }";

/** Validate one attribute declaration. Reserved = jarvis's protected schema
 *  keys + the generic node properties the writer stamps. Dual-use names
 *  (`description`, `source_link`, `name`) are allowed, as in jarvis — note
 *  the resolver treats a core-key attribute as optional regardless of its
 *  declared type (`schema-resolver.ts` `fromDb`), so a `description:
 *  "string"` is required only when it is a node_key token. */
function checkAttribute(type: string, name: string, value: unknown): void {
  if (!IDENT.test(name)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, "attribute name is not a bare identifier", name);
  if (RESERVED_ATTRIBUTE_NAMES.has(name) || GENERIC_NODE_PROPERTIES.has(name)) {
    throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, "attribute name is a reserved schema/system property", name);
  }
  if (typeof value !== "string" || !TYPE_GRAMMAR.test(value)) {
    throw new GraphValidationError("WRONG_TYPE", type, `attribute type must be one of string|boolean|int|float|complex|datetime|list (optionally ?-prefixed), got ${JSON.stringify(value)}`, name);
  }
}

/**
 * Pure validation + normalization of a schema input (no DB). Throws a
 * `GraphValidationError` on the first problem. Existence checks (parent,
 * duplicate type) happen in `createNodeSchema`.
 */
export function planNodeSchema(input: NodeSchemaInput): NodeSchemaPlan {
  const type = String(input.type ?? "").trim();
  if (!TYPE_LABEL.test(type)) throw new GraphValidationError("UNKNOWN_TYPE", type || "?", "type must match ^[A-Za-z][A-Za-z0-9_]*$");
  if (type === "*" || type === THING_TYPE) throw new GraphValidationError("UNKNOWN_TYPE", type, "cannot create the root or wildcard schema");
  if (isVeinType(type)) throw new GraphValidationError("UNKNOWN_TYPE", type, "Vein's own types are a closed in-code registry");
  const parent = String(input.parent ?? THING_TYPE).trim() || THING_TYPE;
  if (!TYPE_LABEL.test(parent)) throw new GraphValidationError("UNKNOWN_TYPE", type, `parent must match ^[A-Za-z][A-Za-z0-9_]*$, got ${JSON.stringify(input.parent)}`, "parent");

  const attributes: Record<string, string> = {};
  for (const [k, v] of Object.entries(input.attributes ?? {})) {
    checkAttribute(type, k, v);
    attributes[k] = v as string;
  }
  const declared = new Set(["name", ...Object.keys(attributes)]);

  const prefix = `${type.toLowerCase()}-`;
  let rawKey = String(input.node_key ?? "name").trim();
  if (rawKey.toLowerCase().startsWith(prefix)) rawKey = rawKey.slice(prefix.length);
  const tokens = rawKey.split("-").map((t) => t.trim()).filter(Boolean);
  if (tokens.length === 0) throw new GraphValidationError("MISSING_REQUIRED", type, "node_key needs at least one attribute token", "node_key");
  for (const t of tokens) {
    if (!declared.has(t)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, `node_key token ${t} is not a declared attribute`, "node_key");
  }
  const node_key = `${prefix}${tokens.join("-")}`;

  const index = (input.index ?? tokens).map((f) => String(f).trim()).filter(Boolean);
  for (const f of index) {
    if (!declared.has(f)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, `index field ${f} is not a declared attribute`, "index");
  }
  for (const [key, val] of [["title_key", input.title_key], ["description_key", input.description_key]] as const) {
    if (val !== undefined && !declared.has(val)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", type, `${key} ${val} is not a declared attribute`, key);
  }
  const domain = String(input.domain ?? "entity").trim() || "entity";
  if (!IDENT.test(domain)) throw new GraphValidationError("WRONG_TYPE", type, "domain must be a bare identifier", "domain");

  const flat: Record<string, unknown> = {
    type,
    parent,
    node_key,
    index,
    domain,
    ...(input.title_key ? { title_key: input.title_key } : {}),
    ...(input.description_key ? { description_key: input.description_key } : {}),
    ...(input.type_description ? { type_description: input.type_description } : {}),
    ...attributes,
  };
  return { type, parent, node_key, tokens, index, domain, attributes, flat };
}

/**
 * Create the schema (or add-only extend an existing non-Vein one). The
 * resolver's caches are invalidated so the next write sees the new type.
 */
export async function createNodeSchema(bolt: Bolt, resolver: SchemaResolver, input: NodeSchemaInput): Promise<NodeSchemaResult> {
  const plan = planNodeSchema(input);

  // Adopt the live casing of an existing type / parent (jarvis
  // `resolve_graph_label` / `resolve_canonical_node_type`).
  const existingType = await resolver.resolveType(plan.type);
  const parent = (await resolver.resolveType(plan.parent)) ?? null;
  if (!parent) throw new GraphValidationError("UNKNOWN_TYPE", plan.type, `parent schema ${plan.parent} does not exist`, "parent");
  if (isVeinType(parent)) throw new GraphValidationError("UNKNOWN_TYPE", plan.type, "Vein's own types cannot be extended (closed registry)", "parent");

  const result = await bolt.write(async (tx) => {
    if (existingType) {
      if (isVeinType(existingType)) throw new GraphValidationError("UNKNOWN_TYPE", existingType, "Vein's own types are a closed in-code registry");
      const live = await txRows(tx, `MATCH (s:Schema {type: $t}) WHERE s.is_deleted IS NULL OR s.is_deleted = false RETURN s.ref_id AS ref_id, keys(s) AS ks LIMIT 1`, {
        t: existingType,
      });
      if (live.length) {
        // Extend: add only attributes the schema lacks.
        const have = new Set(live[0]!["ks"] as string[]);
        const missing: Record<string, string> = {};
        for (const [k, v] of Object.entries(plan.attributes)) if (!have.has(k)) missing[k] = v;
        if (Object.keys(missing).length) await tx.run(`MATCH (s:Schema {type: $t}) SET s += $missing`, { t: existingType, missing });
        return { created: false, ref_id: String(live[0]!["ref_id"]), type: existingType, added: Object.keys(missing).sort() };
      }
      // A soft-deleted schema is replaced (jarvis: delete_schema_permanently, then create).
      await tx.run(`MATCH (s:Schema {type: $t}) DETACH DELETE s`, { t: existingType });
    }
    const ref_id = randomUUID();
    await tx.run(`CREATE (s:Schema) SET s = $flat, s.ref_id = $ref_id`, { flat: { ...plan.flat, parent }, ref_id });
    await tx.run(
      `MATCH (child:Schema {type: $c}), (parent:Schema {type: $p})
       MERGE (child)-[r:CHILD_OF]->(parent) ON CREATE SET r.ref_id = $r`,
      { c: plan.type, p: parent, r: randomUUID() },
    );
    return { created: true, ref_id, type: plan.type, added: [] as string[] };
  });

  if (result.created) {
    // DDL cannot run inside the write transaction. Same objects as the seeders.
    const t = result.type;
    await schemaStatement(bolt, `CREATE CONSTRAINT ${`unique_${t.toLowerCase()}_node_key`} IF NOT EXISTS FOR (n:\`${t}\`) REQUIRE (n.node_key, n.namespace) IS UNIQUE`);
    await schemaStatement(bolt, `CREATE INDEX IF NOT EXISTS FOR (n:\`${t}\`) ON (n.node_key)`);
    const d = plan.domain.toLowerCase();
    const props = [...new Set([...searchableAttributesOf(plan.flat), "node_key"])].sort().map((p) => `n.\`${p}\``).join(", ");
    await schemaStatement(
      bolt,
      `CREATE FULLTEXT INDEX \`domain_${d}_attribute_index_v2\` IF NOT EXISTS FOR (n:\`Domain_${d}\`) ON EACH [${props}]
       OPTIONS { indexConfig: { \`fulltext.analyzer\`: 'english' } }`,
    );
    await schemaStatement(bolt, `CREATE VECTOR INDEX \`domain_${d}_vector_index\` IF NOT EXISTS FOR (n:\`Domain_${d}\`) ON n.text_embeddings ${VECTOR_OPTIONS}`);
  }

  resolver.invalidate();
  const schema = await resolver.schema(result.type);
  return {
    created: result.created,
    ref_id: result.ref_id,
    type: result.type,
    parent: schema?.parent ?? parent,
    node_key: schema?.node_key ?? plan.node_key,
    added: result.added,
    attributes: schema?.attributes ?? plan.attributes,
  };
}
