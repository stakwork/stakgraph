/**
 * Live schema resolution — how the backend knows about node types that are
 * NOT Vein's own (Document, EvalSet, Concept, …): exactly the way jarvis
 * does, from the `:Schema` meta-graph in the database.
 *
 *   - type canonicalization (`node_type_helper.resolve_canonical_node_type`):
 *     Vein registry exact match → live label catalog, case-insensitive →
 *     `Schema.type`, case-insensitive;
 *   - the merged schema (`schema_crud.get_schema`): the node's own props plus
 *     every CHILD_OF ancestor's, child wins, ancestors' `type_description`
 *     ignored;
 *   - domain labels (`domain_helper.get_domain_labels_for_node`): the
 *     schema's `domain` → `Domain_<lower>`, dropped when the type or an
 *     ancestor is in `About.hidden_types` or the domain in `hidden_domains`;
 *   - edge schemas (`schema_crud.get_schema_edge_by_edge_type`): exact,
 *     then source-ancestor, target-ancestor, both-ancestor walks (depth 10),
 *     then the `*`→`*` wildcard; short-circuited by jarvis's `EDGE_TYPES`
 *     allowlist;
 *   - `create_schema_if_missing` (`schema_crud.create_edge_schema`).
 *
 * Vein's own types resolve from the in-code registry (`vein-schemas.ts`)
 * without touching the DB. Everything is cached per resolver with a short
 * TTL, like jarvis's in-process caches.
 */
import { randomUUID } from "node:crypto";
import type { ManagedTransaction } from "neo4j-driver";
import { Bolt, txRows, type Row } from "./bolt.js";
import {
  SCHEMA_CORE_PROPERTIES,
  VEIN_DOMAIN,
  VEIN_SCHEMAS,
  effectiveAttributes,
  getVeinSchema,
  type VeinSchema,
} from "./vein-schemas.js";

/** jarvis `ApplicationConstant.EDGE_TYPES` — edge types accepted WITHOUT an
 *  edge schema. */
export const EDGE_TYPES_ALLOWLIST = new Set([
  "RELATED_TO", "CREATED_BY", "PARENT_OF", "CHILD_OF", "ALIAS_OF", "TYPE_OF", "PART_OF", "MENTIONS",
  "ORGANIZES", "BELONGS_TO", "OPINION", "USES", "FEATURES", "MEMBER_OF", "SUMMARIZED_AS", "CONTRIBUTES_TO",
  "DEPENDS_ON", "INFLUENCES", "OWNED_BY", "MANAGES", "REPORTS_TO", "COLLABORATES_WITH", "SUPPORTS", "REPLACES",
  "REQUIRES", "INTEGRATES_WITH", "ASSOCIATED_WITH", "PRODUCED_BY", "CONSISTS_OF", "HOSTED_ON", "DEVELOPED_BY",
  "FUNDED_BY", "REGULATED_BY", "HAS_ATTRIBUTE", "INTERACTS_WITH", "EXPORTS_TO", "IMPORTS_FROM", "SYNONYM_OF",
  "TRANSLATED_AS", "CAUSED_BY", "OFFERS", "IMPACT_ON", "IMPACTED", "REFLECTED_ON", "AFFECTED", "SHAPED",
  "INFORMED_BY", "CONTAINS", "HAS", "CALLS", "TESTS", "IS_PREREQUISITE_FOR", "ANSWERS", "FEEDBACK", "REFERS_TO",
  "WORKS_AT", "HAS_PRICE", "HAS_WEATHER", "HAS_CAPITAL", "BORN_ON", "CAUSE_CHILD_OF", "TARGETS_CAUSE",
  "TARGETS_CONCEPT", "SCRATCHPAD_EDGE",
]);

const MAX_ANCESTOR_WALK_DEPTH = 10;
const TYPE_GRAMMAR = /^\??(string|boolean|int|float|complex|datetime|list)$/;

/** The schema shape the writers validate against — Vein registry entries
 *  and DB-resolved jarvis schemas both reduce to this. */
export interface NodeSchema {
  /** Canonical type label. */
  type: string;
  parent?: string;
  node_key: string;
  /** `get_index_fields`: declared index list (a string becomes a one-item
   *  list); `["node_key"]` when unset. */
  index: string[];
  vector_index: string[];
  /** Attribute name → type string (`?`-prefixed = optional). Includes
   *  everything inherited via CHILD_OF. */
  attributes: Record<string, string>;
  title_key?: string;
  description_key?: string;
  /** `Domain_<x>` labels to stamp on nodes of this type (already filtered
   *  by hidden types/domains). */
  domainLabels: string[];
  /** True for Vein's own types (closed registry rules apply). */
  isVein: boolean;
}

export interface EdgeSchemaMatch {
  source: string;
  target: string;
  edge_type: string;
  properties: Record<string, unknown>;
  via: "exact" | "ancestor" | "wildcard" | "allowlist";
}

interface Cached<T> {
  at: number;
  value: T;
}

export class SchemaResolver {
  private readonly types = new Map<string, Cached<string | null>>();
  private readonly schemas = new Map<string, Cached<NodeSchema | null>>();
  private readonly edges = new Map<string, Cached<EdgeSchemaMatch | null>>();
  private hidden: Cached<{ domains: Set<string>; types: Set<string> }> | undefined;

  constructor(
    private readonly bolt: Bolt,
    private readonly ttlMs = 60_000,
  ) {}

  /** Drop every cache (call after seeding or schema edits). */
  invalidate(): void {
    this.types.clear();
    this.schemas.clear();
    this.edges.clear();
    this.hidden = undefined;
  }

  private fresh<T>(c: Cached<T> | undefined): c is Cached<T> {
    return !!c && Date.now() - c.at < this.ttlMs;
  }

  /**
   * Canonical type for a user-supplied string, or null. Vein types must
   * match exactly (registry); everything else resolves case-insensitively
   * against live labels, then `Schema.type`.
   */
  async resolveType(raw: string, tx?: ManagedTransaction): Promise<string | null> {
    const key = raw.trim();
    if (!key) return null;
    if (getVeinSchema(key)) return key;
    const c = this.types.get(key.toLowerCase());
    if (this.fresh(c)) return c.value;
    let out: string | null = null;
    const labels = await this.rows(tx, `CALL db.labels() YIELD label WHERE toLower(label) = toLower($t) RETURN label LIMIT 1`, { t: key });
    if (labels.length) out = String(labels[0]!["label"]);
    else {
      const s = await this.rows(
        tx,
        `MATCH (n:Schema) WHERE toLower(n.type) = toLower($t) AND (n.is_deleted IS NULL OR n.is_deleted = false) AND n.type <> "*" RETURN n.type AS t LIMIT 1`,
        { t: key },
      );
      if (s.length) out = String(s[0]!["t"]);
    }
    if (out && getVeinSchema(out)) out = getVeinSchema(out)!.type;
    this.types.set(key.toLowerCase(), { at: Date.now(), value: out });
    return out;
  }

  /** The merged schema for a (raw) type, or null when unknown. */
  async schema(raw: string, tx?: ManagedTransaction): Promise<NodeSchema | null> {
    const type = await this.resolveType(raw, tx);
    if (!type) return null;
    const c = this.schemas.get(type);
    if (this.fresh(c)) return c.value;
    const hidden = await this.hiddenSets(tx);
    let out: NodeSchema | null;
    const vein = getVeinSchema(type);
    if (vein) out = fromVein(vein, hidden);
    else out = await this.fromDb(type, hidden, tx);
    this.schemas.set(type, { at: Date.now(), value: out });
    return out;
  }

  private async fromDb(type: string, hidden: { domains: Set<string>; types: Set<string> }, tx?: ManagedTransaction): Promise<NodeSchema | null> {
    const rows = await this.rows(
      tx,
      `MATCH (n:Schema {type: $t})
       OPTIONAL MATCH path = (n)-[:CHILD_OF*1..${MAX_ANCESTOR_WALK_DEPTH}]->(ancestor:Schema)
       RETURN n, [node IN nodes(path) WHERE node <> n | node.type] AS chain, [node IN nodes(path) WHERE node <> n | properties(node)] AS ancestors
       ORDER BY size(chain) DESC LIMIT 1`,
      { t: type },
    );
    if (rows.length === 0) return null;
    const own = (rows[0]!["n"] as { properties: Record<string, unknown> }).properties;
    const chain = ((rows[0]!["chain"] as string[] | null) ?? []).filter(Boolean);
    const merged: Record<string, unknown> = { ...own };
    for (const a of (rows[0]!["ancestors"] as Array<Record<string, unknown>> | null) ?? []) {
      for (const [k, v] of Object.entries(a)) {
        if (k in merged || k === "type_description") continue;
        merged[k] = v;
      }
    }
    // jarvis validates against the flat merged dict: every key whose value
    // is a type string is an attribute — including keys that happen to be
    // in SCHEMA_KNOWN_PROPERTIES (Document's `source_link: "string"`,
    // Thing's `description: "?string"`). Those, plus `name`, are skipped by
    // its required-attribute check, so they are optional here; a node_key
    // token among them is still enforced by the node_key presence check.
    const attributes: Record<string, string> = {};
    for (const [k, v] of Object.entries(merged)) {
      if (typeof v !== "string" || !TYPE_GRAMMAR.test(v)) continue;
      const neverRequired = k === "name" || SCHEMA_CORE_PROPERTIES.has(k);
      attributes[k] = neverRequired && !v.startsWith("?") ? `?${v}` : v;
    }
    const domain = typeof own["domain"] === "string" ? (own["domain"] as string) : undefined;
    const typeHidden = hidden.types.has(type) || chain.some((t) => hidden.types.has(t));
    const domainLabels = !domain || typeHidden || hidden.domains.has(domain.toLowerCase()) ? [] : [`Domain_${domain.toLowerCase()}`];
    const rawIndex = merged["index"];
    const index = Array.isArray(rawIndex) ? rawIndex.map(String).filter(Boolean) : typeof rawIndex === "string" && rawIndex ? [rawIndex] : [];
    const rawVi = merged["vector_index"];
    return {
      type,
      parent: typeof merged["parent"] === "string" ? (merged["parent"] as string) : undefined,
      node_key: String(merged["node_key"] ?? `${type.toLowerCase()}-name`),
      index: index.length ? index : ["node_key"],
      vector_index: Array.isArray(rawVi) ? rawVi.map(String) : [],
      attributes,
      title_key: typeof merged["title_key"] === "string" ? (merged["title_key"] as string) : undefined,
      description_key: typeof merged["description_key"] === "string" ? (merged["description_key"] as string) : undefined,
      domainLabels,
      isVein: false,
    };
  }

  /** `About.hidden_domains` (lowercased) and `hidden_types`. */
  async hiddenSets(tx?: ManagedTransaction): Promise<{ domains: Set<string>; types: Set<string> }> {
    if (this.fresh(this.hidden)) return this.hidden.value;
    const rows = await this.rows(tx, `MATCH (a:About) RETURN a.hidden_domains AS d, a.hidden_types AS t LIMIT 1`);
    const d = rows[0]?.["d"];
    const t = rows[0]?.["t"];
    const value = {
      domains: new Set(Array.isArray(d) ? d.map((x) => String(x).toLowerCase()) : []),
      types: new Set(Array.isArray(t) ? t.map(String) : []),
    };
    this.hidden = { at: Date.now(), value };
    return value;
  }

  /**
   * Is `source -[edge]-> target` allowed? jarvis's allowlist first, then
   * the edge-schema lookup (exact → ancestor walks → wildcard). Types are
   * canonical labels. Returns the match, or null.
   */
  async edgeSchema(edge: string, sourceType: string, targetType: string, tx?: ManagedTransaction): Promise<EdgeSchemaMatch | null> {
    if (EDGE_TYPES_ALLOWLIST.has(edge)) return { source: sourceType, target: targetType, edge_type: edge, properties: {}, via: "allowlist" };
    const key = `${sourceType}|${edge}|${targetType}`;
    const c = this.edges.get(key);
    if (this.fresh(c)) return c.value;
    const p = { s: sourceType, t: targetType, e: edge };
    const pick = (rows: Row[], via: EdgeSchemaMatch["via"]): EdgeSchemaMatch | null =>
      rows.length
        ? { source: String(rows[0]!["source"]), target: String(rows[0]!["target"]), edge_type: String(rows[0]!["edge_type"]), properties: rows[0]!["props"] as Record<string, unknown>, via }
        : null;
    let match =
      pick(
        await this.rows(
          tx,
          `MATCH (source:Schema)-[r]->(target:Schema)
           WHERE toLower(source.type) = toLower($s) AND toLower(target.type) = toLower($t) AND toLower(type(r)) = toLower($e)
           RETURN source.type AS source, target.type AS target, type(r) AS edge_type, properties(r) AS props LIMIT 1`,
          p,
        ),
        "exact",
      ) ??
      pick(
        await this.rows(
          tx,
          `MATCH path = (child:Schema)-[:CHILD_OF*1..${MAX_ANCESTOR_WALK_DEPTH}]->(ancestor:Schema)
           MATCH (ancestor)-[r]->(target:Schema)
           WHERE toLower(child.type) = toLower($s) AND toLower(target.type) = toLower($t) AND toLower(type(r)) = toLower($e)
           RETURN child.type AS source, target.type AS target, type(r) AS edge_type, properties(r) AS props
           ORDER BY length(path) ASC LIMIT 1`,
          p,
        ),
        "ancestor",
      ) ??
      pick(
        await this.rows(
          tx,
          `MATCH path = (child:Schema)-[:CHILD_OF*1..${MAX_ANCESTOR_WALK_DEPTH}]->(ancestor:Schema)
           MATCH (source:Schema)-[r]->(ancestor)
           WHERE toLower(child.type) = toLower($t) AND toLower(source.type) = toLower($s) AND toLower(type(r)) = toLower($e)
           RETURN source.type AS source, child.type AS target, type(r) AS edge_type, properties(r) AS props
           ORDER BY length(path) ASC LIMIT 1`,
          p,
        ),
        "ancestor",
      ) ??
      pick(
        await this.rows(
          tx,
          `MATCH src_path = (src_child:Schema)-[:CHILD_OF*1..${MAX_ANCESTOR_WALK_DEPTH}]->(src_anc:Schema)
           MATCH tgt_path = (tgt_child:Schema)-[:CHILD_OF*1..${MAX_ANCESTOR_WALK_DEPTH}]->(tgt_anc:Schema)
           MATCH (src_anc)-[r]->(tgt_anc)
           WHERE toLower(src_child.type) = toLower($s) AND toLower(tgt_child.type) = toLower($t) AND toLower(type(r)) = toLower($e)
           RETURN src_child.type AS source, tgt_child.type AS target, type(r) AS edge_type, properties(r) AS props
           ORDER BY (length(src_path) + length(tgt_path)) ASC LIMIT 1`,
          p,
        ),
        "ancestor",
      ) ??
      pick(
        await this.rows(
          tx,
          `MATCH (source:Schema {type: "*"})-[r]->(target:Schema {type: "*"})
           WHERE toLower(type(r)) = toLower($e)
           RETURN source.type AS source, target.type AS target, type(r) AS edge_type, properties(r) AS props LIMIT 1`,
          p,
        ),
        "wildcard",
      );
    if (match && match.via === "exact") match = { ...match, source: sourceType, target: targetType };
    this.edges.set(key, { at: Date.now(), value: match });
    return match;
  }

  /**
   * `create_schema_if_missing`: register `source -[EDGE]-> target` between
   * two existing `:Schema` nodes (`*` allowed on either side; the sentinel
   * is ensured). Idempotent. Returns whether a new schema edge was created.
   */
  async createEdgeSchema(sourceType: string, edge: string, targetType: string): Promise<{ created: boolean; ref_id: string }> {
    const edgeType = edge.toUpperCase().replace(/ /g, "_");
    if (!/^[A-Z][A-Z0-9_]*$/.test(edgeType)) throw new Error(`invalid edge type ${edge}`);
    return this.bolt.write(async (tx) => {
      for (const t of [sourceType, targetType]) {
        if (t === "*") {
          await tx.run(`MERGE (s:Schema {type: "*"}) ON CREATE SET s.ref_id = $r, s.is_system = true`, { r: randomUUID() });
          continue;
        }
        const exists = await txRows(tx, `MATCH (s:Schema {type: $t}) RETURN s.type AS t LIMIT 1`, { t });
        if (exists.length === 0) throw new Error(`schema ${t} does not exist`);
      }
      const ref_id = randomUUID();
      const rows = await txRows(
        tx,
        `MATCH (source:Schema {type: $s}), (target:Schema {type: $t})
         MERGE (source)-[r:\`${edgeType}\`]->(target)
         ON CREATE SET r.ref_id = $ref_id
         RETURN r.ref_id = $ref_id AS created, r.ref_id AS ref_id`,
        { s: sourceType, t: targetType, ref_id },
      );
      this.edges.clear();
      return { created: Boolean(rows[0]?.["created"]), ref_id: String(rows[0]?.["ref_id"] ?? ref_id) };
    });
  }

  private rows(tx: ManagedTransaction | undefined, cypher: string, params: Record<string, unknown> = {}): Promise<Row[]> {
    return tx ? txRows(tx, cypher, params) : this.bolt.run(cypher, params);
  }
}

/** A Vein registry schema as a `NodeSchema` (Thing's attributes inherited,
 *  `Domain_vein` unless hidden). */
export function fromVein(vein: VeinSchema, hidden?: { domains: Set<string>; types: Set<string> }): NodeSchema {
  const hide = hidden ? hidden.types.has(vein.type) || hidden.types.has("Thing") || hidden.domains.has(VEIN_DOMAIN.toLowerCase()) : false;
  return {
    type: vein.type,
    parent: vein.parent,
    node_key: vein.node_key,
    index: [...vein.index],
    vector_index: [...(vein.vector_index ?? [])],
    attributes: { ...effectiveAttributes(vein) },
    title_key: vein.title_key,
    description_key: vein.description_key,
    domainLabels: hide ? [] : [`Domain_${VEIN_DOMAIN.toLowerCase()}`],
    isVein: true,
  };
}

/** Every Vein registry schema as `NodeSchema` (no DB, no hidden filtering). */
export function veinNodeSchemas(): NodeSchema[] {
  return VEIN_SCHEMAS.map((s) => fromVein(s));
}
