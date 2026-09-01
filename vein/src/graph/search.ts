/**
 * The read surface (`plans/jarvis-graph-compat.md` §7) — exactly what the
 * `jarvis/*` lab steps touch, ported from jarvis's own read paths so the
 * `graph/*` step twins return the same shapes:
 *
 *   - `getNode`            GET /v2/nodes/:ref_id           (`node_service.py:get_node_by_id`)
 *   - `connectionCounts`   …/connection-counts             (`node_service_v2.py:get_connection_counts`)
 *   - `edgeCounts`         `include_edge_counts` batch      (`_batch_edge_type_counts`)
 *   - `neighbors`          GET /v2/nodes/:ref_id?expand=edges&sort_by=importance (`node_helper_v2.py`)
 *   - `search`             GET /v2/nodes?q&input_q&output_q (`_search_pipeline` + fusion/boost/tiebreak)
 *   - `listSchemas`/`getSchema`   GET /v2/schema[/type]   (`schema_crud.py:get_all_schemas`, `format_single_schema`)
 *   - `registerNamespace`/`listNamespaces`  POST/GET /namespace (`namespace_service.py`)
 *
 * Everything jarvis does beyond these (radar, scratchpad, legal boosts,
 * paid-property stripping, S3 presigning, GDS) is out of scope by design.
 */
import { randomUUID } from "node:crypto";
import type { ManagedTransaction } from "neo4j-driver";
import { Bolt, int, txRows, type Row } from "./bolt.js";
import { typeLabelOf } from "./edge-writer.js";
import { renderVectorField, type Embedder } from "./node-writer.js";
import type { SchemaResolver } from "./schema-resolver.js";
import { SCHEMA_CORE_PROPERTIES, getVeinSchema, vectorIndexedPairs, vectorStem } from "./vein-schemas.js";

// ── Constants (jarvis) ──────────────────────────────────────────────────────

export const RRF_K = 60;
export const RRF_WEIGHTS: Record<string, number> = { fulltext: 1.15, semantic: 1.0 };
export const DEFAULT_VECTOR_WEIGHT = 1.2;
export const VECTOR_Q_K = 50;
export const VECTOR_Q_SIM_FLOOR = 0.4;
export const USAGE_TIEBREAK_EPSILON = 0.02;
export const SEARCH_CANDIDATE_CAP = 5000;
export const DEFAULT_NAMESPACE = "default";

/** `node_visibility_helper.BLOCKED_NODE_STATUSES`. */
export const BLOCKED_NODE_STATUSES = ["halted", "paused", "stopped", "stopping", "error", "failed", "stuck"];

/** jarvis `GENERIC_NODE_PROPERTIES` — stripped from every response
 *  `properties` map. (Distinct from the write-side set in vein-schemas.) */
export const RESPONSE_STRIPPED_NODE_PROPERTIES = new Set([
  "Data_Bank", "namespace", "spelling_verification", "topic_lower", "ref_id", "node_key",
  "relevancy_score", "date_added_to_graph", "updated_at", "text_embeddings", "input_embeddings",
  "output_embeddings", "embeddings", "algo_page_rank", "algo_score", "algo_community_id",
  "algo_embedding", "workflow_author", "weight", "labels", "system_id", "_search_fields_used",
  "unique_source_id",
]);

/** jarvis `GENERIC_EDGE_PROPERTIES`. */
export const RESPONSE_STRIPPED_EDGE_PROPERTIES = new Set([
  "ref_id", "edge_text_embeddings", "edge_text", "edge_key", "weight", "unique_source_id", "algo_similarity",
]);

export { SCHEMA_CORE_PROPERTIES };
const SCHEMA_PARENT_PROPERTIES_TO_IGNORE = new Set(["type_description"]);

const DATA_BANK_FULLTEXT_INDEX_V2 = "data_bank_attribute_index_v2";
const DOMAIN_ALL_VECTOR_INDEX = "domain_all_vector_index";
const LUCENE_SPECIAL = new Set('+-!(){}[]^"~*?:\\/'.split(""));

// ── Types ───────────────────────────────────────────────────────────────────

export interface NodeEnvelope {
  ref_id: string;
  node_type: string | undefined;
  properties: Record<string, unknown>;
  name?: string;
  date_added_to_graph?: number;
  weight?: number;
  score?: number;
  match_type?: string;
  /** `{EDGE_TYPE: count}` when edge counts were requested. */
  edges?: Record<string, number>;
}

export interface EdgeEnvelope {
  source: string;
  target: string;
  ref_id: string;
  edge_type: string;
  weight?: number;
  properties: Record<string, unknown>;
}

export interface SearchParams {
  q?: string;
  input_q?: string;
  output_q?: string;
  /** Node type labels (exact Neo4j labels). */
  types?: string[];
  /** Domain suffixes, e.g. `["vein"]`. Validated against the registry. */
  domains?: string[];
  namespace?: string;
  limit?: number;
  skip?: number;
  include_edge_counts?: boolean;
}

export interface SearchResult {
  nodes: NodeEnvelope[];
  total: number;
  truncated: boolean;
}

export interface NeighborsParams {
  edge_types?: string[];
  node_types?: string[];
  exclude_node_types?: string[];
  limit?: number;
  /** Explicit namespace pins the edge-count map to that partition. */
  namespace?: string;
  include_edge_counts?: boolean;
}

export interface ConnectionCount {
  edge_type: string;
  target_type: string;
  count: number;
}

export interface SchemaEnvelope {
  [core: string]: unknown;
  type: string;
  attributes: Record<string, unknown>;
  inherited_attributes: Record<string, unknown>;
}

export interface OntologyEdge {
  edge_type: string;
  source_type: string;
  target_type: string;
}

export class GraphReadError extends Error {
  constructor(
    readonly code: "INVALID_DOMAIN" | "INVALID_NAMESPACE" | "NOT_FOUND" | "INVALID_INPUT",
    message: string,
  ) {
    super(message);
    this.name = "GraphReadError";
  }
}

// ── Pure helpers (exported for tests) ───────────────────────────────────────

export function escapeLucene(value: string): string {
  let out = "";
  for (const ch of value) out += LUCENE_SPECIAL.has(ch) ? `\\${ch}` : ch;
  return out;
}

/** `_build_fulltext_query` (gs=false): one token escaped as-is; several →
 *  each `+`-prefixed (all required). */
export function buildFulltextQuery(q: string): string {
  const raw = (q ?? "").trim();
  if (!raw) return raw;
  const words = raw.split(/\s+/);
  if (words.length === 1) return escapeLucene(raw);
  return words.map((w) => `+${escapeLucene(w)}`).join(" ");
}

/** `_search_fetch_limit`. */
export function fetchLimit(limit: number | undefined, skip: number | undefined): number {
  const requested = Math.max(limit || 20, 20);
  const offset = Math.max(skip || 0, 0);
  return Math.min(Math.max(requested + offset + 100, requested * 5), SEARCH_CANDIDATE_CAP);
}

/** Python `round()` — half to even. */
export function pyRound(x: number, digits = 0): number {
  const m = 10 ** digits;
  const v = x * m;
  const f = Math.floor(v);
  const diff = v - f;
  let r: number;
  if (diff > 0.5) r = f + 1;
  else if (diff < 0.5) r = f;
  else r = f % 2 === 0 ? f : f + 1;
  return r / m;
}

export interface Hit {
  node: HitNode;
  raw_score: number;
}
export interface HitNode {
  labels: string[];
  properties: Record<string, unknown>;
}
export interface RankedEntry {
  node: HitNode;
  sources: Set<string>;
  raw_scores: Record<string, number>;
  score: number;
  best_rank: number;
  extra: { score: number; match_type: string };
}

function refIdOf(n: HitNode): string {
  return String(n.properties["ref_id"] ?? "");
}

/** `_rank_search_hits`: best raw score per ref_id, sorted desc. */
export function rankHits(hits: Hit[]): Hit[] {
  const best = new Map<string, Hit>();
  for (const h of hits) {
    const id = refIdOf(h.node);
    if (!id) continue;
    const cur = best.get(id);
    if (!cur || h.raw_score > cur.raw_score) best.set(id, h);
  }
  return [...best.values()].sort((a, b) => b.raw_score - a.raw_score);
}

const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** `_fuse_search_hits`: weighted RRF, `(-score, best_rank, ref_id)` order,
 *  score normalized to the top entry. */
export function fuseHits(fulltext: Hit[], semantic: Hit[], buckets: Record<string, Hit[]> = {}): RankedEntry[] {
  const sources: Array<[string, Hit[]]> = [
    ["fulltext", rankHits(fulltext)],
    ["semantic", rankHits(semantic)],
    ...Object.entries(buckets).map(([k, v]): [string, Hit[]] => [k, rankHits(v ?? [])]),
  ];
  const fused = new Map<string, RankedEntry>();
  for (const [source, hits] of sources) {
    const weight = RRF_WEIGHTS[source] ?? DEFAULT_VECTOR_WEIGHT;
    hits.forEach((hit, i) => {
      const rank = i + 1;
      const id = refIdOf(hit.node);
      let e = fused.get(id);
      if (!e) {
        e = { node: hit.node, sources: new Set(), raw_scores: {}, score: 0, best_rank: rank, extra: { score: 0, match_type: "" } };
        fused.set(id, e);
      }
      e.sources.add(source);
      e.raw_scores[source] = hit.raw_score;
      e.score += weight / (RRF_K + rank);
      e.best_rank = Math.min(e.best_rank, rank);
    });
  }
  const ranked = [...fused.values()].sort((a, b) => b.score - a.score || a.best_rank - b.best_rank || cmp(refIdOf(a.node), refIdOf(b.node)));
  if (ranked.length === 0) return [];
  const max = ranked[0]!.score;
  for (const e of ranked) {
    e.extra = {
      score: max ? pyRound(e.score / max, 6) : 0,
      match_type: e.sources.size > 1 ? "hybrid" : [...e.sources][0]!,
    };
  }
  return ranked;
}

export function titleMatchMultiplier(qLower: string, valLower: string): number {
  if (!valLower) return 1.0;
  if (valLower === qLower) return 4.0;
  if (valLower.startsWith(qLower)) return 2.5;
  if (valLower.includes(qLower)) return 2.0;
  return 1.0;
}

/** `_apply_title_key_boost`: multiply by the title-field match tier, re-sort,
 *  re-normalize. `titleKeyFor` resolves a node type's `title_key` (default
 *  `name`). */
export function applyTitleBoost(ranked: RankedEntry[], q: string, titleKeyFor: (type: string | undefined) => string): RankedEntry[] {
  if (!q || ranked.length === 0) return ranked;
  const ql = q.toLowerCase();
  for (const e of ranked) {
    const field = titleKeyFor(typeLabelOf(e.node.labels)) || "name";
    const val = e.node.properties[field];
    if (typeof val === "string") e.score *= titleMatchMultiplier(ql, val.toLowerCase());
  }
  ranked.sort((a, b) => b.score - a.score || a.best_rank - b.best_rank || cmp(refIdOf(a.node), refIdOf(b.node)));
  const max = ranked[0]!.score || 1;
  for (const e of ranked) e.extra.score = max ? pyRound(e.score / max, 6) : 0;
  return ranked;
}

/** `_apply_usage_tiebreak`: bucket normalized score by epsilon, then
 *  usage_count_30d desc → usage_count desc → best rank → ref_id. */
export function applyUsageTiebreak(ranked: RankedEntry[], epsilon = USAGE_TIEBREAK_EPSILON): RankedEntry[] {
  const key = (e: RankedEntry) => {
    const p = e.node.properties;
    return {
      bucket: epsilon ? -pyRound(e.extra.score / epsilon) : -e.extra.score,
      u30: -(Number(p["usage_count_30d"]) || 0),
      ut: -(Number(p["usage_count"]) || 0),
      rank: e.best_rank,
      id: refIdOf(e.node),
    };
  };
  return [...ranked].sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    return ka.bucket - kb.bucket || ka.u30 - kb.u30 || ka.ut - kb.ut || ka.rank - kb.rank || cmp(ka.id, kb.id);
  });
}

/** `_serialize_node`. */
export function serializeNode(n: HitNode, extra?: Record<string, unknown>): NodeEnvelope {
  const props = n.properties;
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) if (!RESPONSE_STRIPPED_NODE_PROPERTIES.has(k)) filtered[k] = v;
  if ("image_url" in filtered && typeof filtered["image_url"] !== "string") filtered["image_url"] = "";
  const out: NodeEnvelope = {
    ref_id: String(props["ref_id"]),
    node_type: typeLabelOf(n.labels),
    date_added_to_graph: Number(props["date_added_to_graph"]) || 0,
    properties: filtered,
  };
  if (props["weight"] !== null && props["weight"] !== undefined) out.weight = Number(props["weight"]);
  if (extra) Object.assign(out, extra);
  return out;
}

function serializeEdge(type: string, props: Record<string, unknown>, source: string, target: string): EdgeEnvelope {
  const filtered: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) if (!RESPONSE_STRIPPED_EDGE_PROPERTIES.has(k)) filtered[k] = v;
  const out: EdgeEnvelope = { source, target, ref_id: String(props["ref_id"] ?? ""), edge_type: type, properties: filtered };
  if (props["weight"] !== null && props["weight"] !== undefined) out.weight = Number(props["weight"]);
  return out;
}

function visibility(alias: string): string {
  return [
    `(${alias}.is_muted IS NULL OR ${alias}.is_muted <> true)`,
    `(${alias}.is_deleted IS NULL OR ${alias}.is_deleted <> true)`,
    `(${alias}.status IS NULL OR NOT ${alias}.status IN $blocked_statuses)`,
  ].join(" AND ");
}

function asHit(r: Row): Hit {
  const n = r["n"] as HitNode;
  return { node: { labels: n.labels, properties: n.properties }, raw_score: Number(r["score"]) };
}

// ── Reader ──────────────────────────────────────────────────────────────────

export interface GraphReaderOptions {
  /** Needed for the semantic and field-scoped retrievers; without it search
   *  is fulltext-only (jarvis logs "TEXT_MODEL not loaded" and does the same). */
  embedder?: Embedder;
  /** Canonicalizes `type` filters case-insensitively (jarvis
   *  `resolve_canonical_node_types`); unresolved names are kept verbatim
   *  (they match nothing — jarvis's silent-empty behaviour). */
  resolver?: SchemaResolver;
}

export class GraphReader {
  constructor(
    private readonly bolt: Bolt,
    private readonly opts: GraphReaderOptions = {},
  ) {}

  // ── domains / namespaces ────────────────────────────────────────────────

  /** Distinct lowercased domains registered by existing Schema nodes. */
  async listDomains(tx?: ManagedTransaction): Promise<string[]> {
    const rows = await this.rows(tx, `MATCH (s:Schema) WHERE s.domain IS NOT NULL RETURN DISTINCT toLower(s.domain) AS d ORDER BY d`);
    return rows.map((r) => String(r["d"]));
  }

  /** `About.hidden_domains` (lowercased), [] when unset. */
  async hiddenDomains(tx?: ManagedTransaction): Promise<string[]> {
    const rows = await this.rows(tx, `MATCH (a:About) WHERE a.hidden_domains IS NOT NULL RETURN a.hidden_domains AS h LIMIT 1`);
    const h = rows[0]?.["h"];
    return Array.isArray(h) ? h.map((d) => String(d).toLowerCase()) : [];
  }

  /** `visible_domain_labels`: `Domain_<d>` for every non-hidden domain. */
  async visibleDomainLabels(tx?: ManagedTransaction): Promise<string[]> {
    const [all, hidden] = await Promise.all([this.listDomains(tx), this.hiddenDomains(tx)]);
    const h = new Set(hidden);
    return all.filter((d) => !h.has(d)).map((d) => `Domain_${d}`);
  }

  /** Registered namespaces (jarvis keeps one `:NameSpace` node holding a
   *  lowercased list). `default` is implicit. */
  async listNamespaces(): Promise<string[]> {
    const rows = await this.bolt.run(`MATCH (n:NameSpace) RETURN n.data AS data LIMIT 1`);
    const data = rows[0]?.["data"];
    return Array.isArray(data) ? data.map(String) : [];
  }

  /** Idempotent `POST /namespace`: append the lowercased name to the single
   *  `:NameSpace` node's `data` list (creating the node on first use). */
  async registerNamespace(name: string): Promise<{ namespace: string; created: boolean }> {
    const ns = String(name).toLowerCase();
    if (!ns) throw new GraphReadError("INVALID_INPUT", "namespace is required");
    return this.bolt.write(async (tx) => {
      const existing = await txRows(tx, `MATCH (n:NameSpace) RETURN n.ref_id AS ref_id, n.data AS data LIMIT 1`);
      const data = Array.isArray(existing[0]?.["data"]) ? (existing[0]!["data"] as string[]) : [];
      if (data.includes(ns)) return { namespace: ns, created: false };
      const ref_id = (existing[0]?.["ref_id"] as string | undefined) ?? randomUUID();
      await tx.run(`MERGE (n:NameSpace {ref_id: $ref_id}) SET n.data = $data`, { ref_id, data: [...data, ns] });
      return { namespace: ns, created: true };
    });
  }

  /** `NameSpaceHelper.get_request_namespace`: `default` always resolves;
   *  anything else must be registered. */
  async resolveNamespace(namespace: string | undefined): Promise<string> {
    const ns = namespace ?? this.bolt.namespace;
    if (ns === DEFAULT_NAMESPACE) return ns;
    const known = await this.listNamespaces();
    if (!known.includes(ns.toLowerCase())) throw new GraphReadError("INVALID_NAMESPACE", `namespace ${ns} is not registered`);
    return ns;
  }

  // ── nodes ───────────────────────────────────────────────────────────────

  /** GET /v2/nodes/:ref_id — `{name, node_type, ref_id, properties, weight?}`
   *  or null when absent/hidden. */
  async getNode(ref_id: string): Promise<NodeEnvelope | null> {
    const rows = await this.bolt.run(
      `MATCH (n:Data_Bank {ref_id: $ref_id}) WHERE ${visibility("n")} RETURN n LIMIT 1`,
      { ref_id, blocked_statuses: BLOCKED_NODE_STATUSES },
    );
    if (rows.length === 0) return null;
    const n = rows[0]!["n"] as HitNode;
    const p = n.properties;
    const env = serializeNode(n);
    delete env.date_added_to_graph;
    const name = p["name"] ?? p["episode_title"] ?? p["show_title"] ?? p["Data_Bank"];
    return { name: name === undefined || name === null ? undefined : String(name), ...env };
  }

  /** GET …/connection-counts: `(edge_type, target_type) → count`, scoped to
   *  the node's own namespace unless one is given. */
  async connectionCounts(ref_id: string, namespace?: string): Promise<ConnectionCount[]> {
    let ns = namespace;
    if (!ns) {
      const r = await this.bolt.run(`MATCH (n:Data_Bank {ref_id: $ref_id}) RETURN n.namespace AS ns LIMIT 1`, { ref_id });
      ns = (r[0]?.["ns"] as string | null | undefined) || DEFAULT_NAMESPACE;
    }
    const visible = await this.visibleDomainLabels();
    const rows = await this.bolt.run(
      `MATCH (n:Data_Bank {ref_id: $ref_id})-[r]-(m)
       WHERE coalesce(n.namespace, $default_ns) = $namespace
         AND coalesce(m.namespace, $default_ns) = $namespace
         ${visible.length ? "AND ANY(lbl IN labels(m) WHERE lbl IN $visible_labels)" : ""}
       RETURN type(r) AS edge_type, labels(m) AS m_labels, count(*) AS cnt`,
      { ref_id, namespace: ns, default_ns: DEFAULT_NAMESPACE, visible_labels: visible },
    );
    const bucket = new Map<string, ConnectionCount>();
    for (const r of rows) {
      const target = typeLabelOf(r["m_labels"] as string[]);
      if (!target) continue;
      const key = `${r["edge_type"]}|${target}`;
      const cur = bucket.get(key) ?? { edge_type: String(r["edge_type"]), target_type: target, count: 0 };
      cur.count += Number(r["cnt"]);
      bucket.set(key, cur);
    }
    return [...bucket.values()].sort((a, b) => b.count - a.count || cmp(a.edge_type, b.edge_type) || cmp(a.target_type, b.target_type));
  }

  /** `_batch_edge_type_counts`: `{ref_id: {EDGE_TYPE: count}}`. */
  async edgeCounts(ref_ids: string[], namespace: string, pinNamespace = false, tx?: ManagedTransaction): Promise<Record<string, Record<string, number>>> {
    if (ref_ids.length === 0) return {};
    const visible = await this.visibleDomainLabels(tx);
    const nsCond = pinNamespace
      ? "coalesce(n.namespace, $default_ns) = $namespace AND coalesce(m.namespace, $default_ns) = $namespace"
      : "coalesce(m.namespace, $default_ns) = coalesce(n.namespace, $default_ns)";
    const rows = await this.rows(
      tx,
      `MATCH (n:Data_Bank)-[r]-(m)
       WHERE n.ref_id IN $ref_ids AND ${nsCond}
         ${visible.length ? "AND ANY(lbl IN labels(m) WHERE lbl IN $visible_labels)" : ""}
       RETURN n.ref_id AS ref_id, type(r) AS edge_type, count(*) AS cnt`,
      { ref_ids, namespace, default_ns: DEFAULT_NAMESPACE, visible_labels: visible },
    );
    const out: Record<string, Record<string, number>> = {};
    for (const r of rows) (out[String(r["ref_id"])] ??= {})[String(r["edge_type"])] = Number(r["cnt"]);
    return out;
  }

  /**
   * GET /v2/nodes/:ref_id?expand=edges&sort_by=importance&limit=…
   * (`node_helper_v2.get_node_edges`, importance branch): 1-hop, ordered by
   * `r.importance` desc before LIMIT, filtered by edge/node type, excluding
   * node types case-insensitively. Returns `{nodes, edges}` with the source
   * node included in `nodes`, like jarvis.
   */
  async neighbors(ref_id: string, p: NeighborsParams = {}): Promise<{ nodes: NodeEnvelope[]; edges: EdgeEnvelope[] }> {
    const where = [visibility("node")];
    const params: Record<string, unknown> = { ref_id, blocked_statuses: BLOCKED_NODE_STATUSES };
    if (p.node_types?.length) {
      params["imp_node_types"] = (await this.canonicalTypes(p.node_types)).map((n) => n.replace(/[-+/>]/g, ""));
      where.push("any(lbl IN labels(node) WHERE lbl IN $imp_node_types)");
    }
    if (p.exclude_node_types?.length) {
      params["exclude_node_types"] = p.exclude_node_types.map((n) => n.toLowerCase());
      where.push("NONE(lbl IN labels(node) WHERE toLower(lbl) IN $exclude_node_types)");
    }
    if (p.edge_types?.length) {
      params["imp_edge_types"] = p.edge_types.map((e) => e.replace(/[<>]/g, ""));
      where.push("type(r) IN $imp_edge_types");
    }
    const limit = p.limit && p.limit > 0 ? `LIMIT $imp_limit` : "";
    if (limit) params["imp_limit"] = int(p.limit!);
    const rows = await this.bolt.run(
      `MATCH (source:Data_Bank {ref_id: $ref_id})-[r]-(node)
       WHERE ${where.join(" AND ")}
       WITH source, r, node
       ORDER BY coalesce(toFloat(r.importance), 0) DESC
       ${limit}
       WITH source, collect(DISTINCT node) AS endNodes,
            collect({type: type(r), props: properties(r), src: startNode(r).ref_id, tgt: endNode(r).ref_id}) AS rels
       RETURN endNodes + [source] AS nodes, rels`,
      params,
    );
    if (rows.length === 0) return { nodes: [], edges: [] };
    const nodes: NodeEnvelope[] = [];
    const seen = new Set<string>();
    for (const n of rows[0]!["nodes"] as HitNode[]) {
      const id = refIdOf(n);
      if (!id || seen.has(id)) continue;
      seen.add(id);
      nodes.push(serializeNode(n));
    }
    const edges: EdgeEnvelope[] = [];
    const seenEdges = new Set<string>();
    for (const e of rows[0]!["rels"] as Array<{ type: string; props: Record<string, unknown>; src: string; tgt: string }>) {
      const id = String(e.props["ref_id"] ?? `${e.src}|${e.type}|${e.tgt}`);
      if (seenEdges.has(id)) continue;
      seenEdges.add(id);
      edges.push(serializeEdge(e.type, e.props, e.src, e.tgt));
    }
    if (p.include_edge_counts && nodes.length) {
      const ns = p.namespace ?? this.bolt.namespace;
      const counts = await this.edgeCounts(nodes.map((n) => n.ref_id), ns, Boolean(p.namespace));
      for (const n of nodes) n.edges = counts[n.ref_id] ?? {};
    }
    return { nodes, edges };
  }

  // ── search ──────────────────────────────────────────────────────────────

  /** GET /v2/nodes?q&input_q&output_q… — hybrid search, ported faithfully. */
  async search(p: SearchParams): Promise<SearchResult> {
    const q = (p.q ?? "").trim();
    const vectorQueries: Record<string, string> = {};
    if (p.input_q?.trim()) vectorQueries["input"] = p.input_q.trim();
    if (p.output_q?.trim()) vectorQueries["output"] = p.output_q.trim();
    if (!q && Object.keys(vectorQueries).length === 0) return { nodes: [], total: 0, truncated: false };

    const namespace = await this.resolveNamespace(p.namespace);
    const limit = p.limit ?? 10;
    const skip = p.skip ?? 0;
    const domains = (p.domains ?? []).map((d) => d.toLowerCase()).filter(Boolean);

    const types = p.types?.length ? await this.canonicalTypes(p.types) : undefined;
    return this.bolt.read(async (tx) => {
      const allDomains = await this.listDomains(tx);
      for (const d of domains) if (!allDomains.includes(d)) throw new GraphReadError("INVALID_DOMAIN", `unknown domain ${d}`);
      const hidden = new Set(await this.hiddenDomains(tx));
      const visibleDomains = allDomains.filter((d) => !hidden.has(d));

      // Filter: namespace + visibility + optional type list; untyped search
      // also excludes types whose domain is hidden.
      const clauses = [`n.namespace = $namespace`, visibility("n")];
      const base: Record<string, unknown> = { namespace, blocked_statuses: BLOCKED_NODE_STATUSES };
      if (types?.length) {
        clauses.push(`ANY(t IN $node_types WHERE t IN labels(n))`);
        base["node_types"] = types;
      } else if (hidden.size) {
        const rows = await txRows(tx, `MATCH (s:Schema) WHERE s.domain IS NOT NULL AND toLower(s.domain) IN $hidden RETURN s.type AS t`, { hidden: [...hidden] });
        const labels = rows.map((r) => String(r["t"]));
        if (labels.length) {
          clauses.push(`NOT ANY(t IN $hidden_type_labels WHERE t IN labels(n))`);
          base["hidden_type_labels"] = labels;
        }
      }
      const filter = clauses.join(" AND ");
      const cap = fetchLimit(limit, skip);

      // Index routing (`_resolve_index_names`): none → global, one → domain,
      // many → per-domain union. When the global index is absent (jarvis
      // never mounted, or Neo4j too old for the multi-label vector index),
      // fall back to the per-domain union over visible domains — jarvis does
      // this for vectors; we do it for fulltext too so a vein-only DB needs
      // no global index. Existence is read once, up front.
      const existing = new Set((await txRows(tx, `SHOW INDEXES YIELD name RETURN name`)).map((r) => String(r["name"])));
      const route = (suffix: string, global: string) =>
        domains.length ? domains.map((d) => `domain_${d}${suffix}`) : existing.has(global) ? [global] : visibleDomains.map((d) => `domain_${d}${suffix}`);
      const ftIndexes = route("_attribute_index_v2", DATA_BANK_FULLTEXT_INDEX_V2).filter((i) => existing.has(i));
      const vecIndexes = route("_vector_index", DOMAIN_ALL_VECTOR_INDEX).filter((i) => existing.has(i));

      let fulltext: Hit[] = [];
      let semantic: Hit[] = [];
      let ftTruncated = false;
      let semTruncated = false;

      const runFulltext = async (indexes: string[], lucene: string): Promise<Hit[]> => {
        const all: Hit[] = [];
        for (const index of indexes) {
          const rows = await txRows(
            tx,
            `CALL db.index.fulltext.queryNodes($index, $q_lucene, $fulltext_options) YIELD node AS n, score WHERE ${filter} RETURN n, score`,
            { ...base, index, q_lucene: lucene, fulltext_options: { limit: int(cap) } },
          );
          all.push(...rows.map(asHit));
        }
        return indexes.length > 1 ? rankHits(all) : all;
      };

      if (q && ftIndexes.length) {
        fulltext = await runFulltext(ftIndexes, buildFulltextQuery(q));
        // Fuzzy fallback on zero hits (no gs operators → always eligible).
        if (fulltext.length === 0) fulltext = await runFulltext(ftIndexes, `${q}~`);
        if (fulltext.length >= cap) ftTruncated = true;
      }

      const runVector = async (indexes: string[], k: number, embedding: number[]): Promise<Hit[]> => {
        const all: Hit[] = [];
        for (const index of indexes) {
          const rows = await txRows(
            tx,
            `CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node AS n, score WHERE ${filter} RETURN n, score`,
            { ...base, index, k: int(k), embedding },
          );
          all.push(...rows.map(asHit));
        }
        return indexes.length > 1 ? rankHits(all) : all;
      };

      if (q && this.opts.embedder && vecIndexes.length) {
        const [embedding] = await this.opts.embedder.embed([q]);
        semantic = await runVector(vecIndexes, cap, embedding!);
        if (semantic.length >= cap) semTruncated = true;
      }

      // Schema-driven `?<stem>_q=` retrievers: per-label index, k=50, floor 0.4.
      const buckets: Record<string, Hit[]> = {};
      if (Object.keys(vectorQueries).length && this.opts.embedder) {
        const stemLabels = new Map<string, string[]>();
        for (const { type, prop } of await this.vectorIndexedPairsLive(tx)) {
          const s = vectorStem(prop);
          stemLabels.set(s, [...(stemLabels.get(s) ?? []), type]);
        }
        for (const [stem, text] of Object.entries(vectorQueries)) {
          const labels = stemLabels.get(stem) ?? [];
          if (labels.length === 0) continue;
          const [emb] = await this.opts.embedder.embed([text]);
          const all: Hit[] = [];
          for (const lbl of labels) {
            const idx = `${lbl.toLowerCase()}_${stem}_vector_index`;
            if (!existing.has(idx)) continue; // migration may not have run yet
            for (const h of await runVector([idx], VECTOR_Q_K, emb!)) {
              if (h.raw_score >= VECTOR_Q_SIM_FLOOR) all.push(h);
            }
          }
          buckets[stem] = rankHits(all);
        }
      }

      let ranked = fuseHits(fulltext, semantic, buckets);
      const titleKeys = new Map<string, string>();
      const titleKeyFor = (t: string | undefined) => (t ? titleKeys.get(t) ?? "name" : "name");
      for (const t of new Set(ranked.map((e) => typeLabelOf(e.node.labels)).filter((t): t is string => !!t))) {
        const vein = getVeinSchema(t);
        if (vein) titleKeys.set(t, vein.title_key);
        else {
          const r = await txRows(tx, `MATCH (s:Schema) WHERE toLower(s.type) = toLower($t) RETURN s.title_key AS k LIMIT 1`, { t });
          if (typeof r[0]?.["k"] === "string") titleKeys.set(t, r[0]["k"] as string);
        }
      }
      ranked = applyTitleBoost(ranked, q, titleKeyFor);
      ranked = applyUsageTiebreak(ranked);

      const total = ranked.length;
      const page = ranked.slice(skip, skip + limit);
      const nodes = page.map((e) => serializeNode(e.node, { score: e.extra.score, match_type: e.extra.match_type }));
      if (p.include_edge_counts && nodes.length) {
        const counts = await this.edgeCounts(nodes.map((n) => n.ref_id), namespace, Boolean(p.namespace), tx);
        for (const n of nodes) n.edges = counts[n.ref_id] ?? {};
      }
      return { nodes, total, truncated: ftTruncated || semTruncated };
    });
  }

  /** `(label, property)` pairs declaring `vector_index`, from live Schema
   *  nodes (jarvis's discovery) — falls back to the Vein registry so a
   *  vein-only DB needs no extra read. */
  private async vectorIndexedPairsLive(tx: ManagedTransaction): Promise<Array<{ type: string; prop: string }>> {
    const rows = await txRows(tx, `MATCH (s:Schema) WHERE s.vector_index IS NOT NULL AND (s.is_deleted IS NULL OR s.is_deleted = false) RETURN s.type AS t, s.vector_index AS v`);
    const out: Array<{ type: string; prop: string }> = [];
    for (const r of rows) for (const prop of (r["v"] as string[]) ?? []) out.push({ type: String(r["t"]), prop });
    return out.length ? out : vectorIndexedPairs();
  }

  // ── ontology ────────────────────────────────────────────────────────────

  /**
   * GET /v2/schema (`get_all_schemas`, non-concise): every live schema with
   * ancestor-merged properties split into core keys + `attributes`, parent
   * attributes moved to `inherited_attributes`; plus `edges` (all Schema→
   * Schema relationships, CHILD_OF included) as concise triples. Optional
   * `domains` filter keeps wildcard endpoints.
   */
  async listSchemas(opts: { domains?: string[]; includeDeleted?: boolean } = {}): Promise<{ schemas: SchemaEnvelope[]; edges: OntologyEdge[] }> {
    const rows = await this.bolt.run(
      `MATCH (n:Schema) ${opts.includeDeleted ? "" : "WHERE (n.is_deleted IS NULL OR n.is_deleted = false)"}
       OPTIONAL MATCH path = (n)-[:CHILD_OF*1..]->(ancestor:Schema)
       OPTIONAL MATCH (n)-[r]->(m:Schema)
       ${opts.includeDeleted ? "" : "WHERE r IS NULL OR (r.is_deleted IS NULL OR r.is_deleted = false)"}
       RETURN n, [node IN nodes(path) WHERE node <> n | node] AS ancestors, r, m`,
    );
    const byRef = new Map<string, Record<string, unknown>>();
    const edges: OntologyEdge[] = [];
    const seenEdges = new Set<string>();
    for (const r of rows) {
      const n = r["n"] as HitNode;
      const ref = String(n.properties["ref_id"]);
      if (!byRef.has(ref)) {
        const merged: Record<string, unknown> = { ...n.properties };
        for (const a of (r["ancestors"] as HitNode[] | null) ?? []) {
          for (const [k, v] of Object.entries(a.properties)) {
            if (k in merged || SCHEMA_PARENT_PROPERTIES_TO_IGNORE.has(k)) continue;
            merged[k] = v;
          }
        }
        byRef.set(ref, merged);
      }
      const rel = r["r"] as { type: string; properties: Record<string, unknown> } | null;
      const m = r["m"] as HitNode | null;
      if (rel && m) {
        const id = String(rel.properties["ref_id"] ?? `${ref}|${rel.type}|${m.properties["ref_id"]}`);
        if (!seenEdges.has(id)) {
          seenEdges.add(id);
          edges.push({ edge_type: rel.type, source_type: String(n.properties["type"]), target_type: String(m.properties["type"]) });
        }
      }
    }
    let schemas: SchemaEnvelope[] = [];
    for (const merged of byRef.values()) {
      if (merged["type"] === "*") continue;
      schemas.push(splitSchema(merged));
    }
    schemas = inheritedAttributes(schemas);
    let outEdges = edges;
    if (opts.domains) {
      const wanted = new Set(opts.domains.map((d) => d.toLowerCase()));
      schemas = schemas.filter((s) => wanted.has(String(s["domain"] ?? "").toLowerCase()));
      const surviving = new Set(schemas.map((s) => s.type));
      const ok = (t: string) => t === "*" || surviving.has(t);
      outEdges = edges.filter((e) => ok(e.source_type) && ok(e.target_type));
    }
    return { schemas, edges: outEdges };
  }

  /** GET /v2/schema/:type (`format_single_schema`): ancestor-merged
   *  `attributes` (own + inherited) plus an `inherited_attributes` view.
   *  Case-insensitive except `Thing`. */
  async getSchema(type: string): Promise<SchemaEnvelope | null> {
    const rows = await this.bolt.run(
      `MATCH (n:Schema) WHERE ${type === "Thing" ? "n.type = $t" : "toLower(n.type) = toLower($t)"}
       OPTIONAL MATCH path = (n)-[:CHILD_OF*1..]->(ancestor:Schema)
       RETURN n, [node IN nodes(path) WHERE node <> n | node] AS ancestors LIMIT 1`,
      { t: type },
    );
    if (rows.length === 0) return null;
    const n = rows[0]!["n"] as HitNode;
    const merged: Record<string, unknown> = { ...n.properties };
    for (const a of (rows[0]!["ancestors"] as HitNode[] | null) ?? []) {
      for (const [k, v] of Object.entries(a.properties)) {
        if (k in merged || SCHEMA_PARENT_PROPERTIES_TO_IGNORE.has(k)) continue;
        merged[k] = v;
      }
    }
    const split = splitSchema(merged);
    const own = new Set(Object.keys(n.properties).filter((k) => !SCHEMA_CORE_PROPERTIES.has(k)));
    const inherited: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(split.attributes)) if (!own.has(k)) inherited[k] = v;
    split.inherited_attributes = inherited;
    return split;
  }

  /** Resolve each type name to its canonical label; unresolved kept as-is. */
  private async canonicalTypes(raw: string[]): Promise<string[]> {
    if (!this.opts.resolver) return raw;
    const out: string[] = [];
    for (const t of raw) out.push((await this.opts.resolver.resolveType(t)) ?? t);
    return out;
  }

  private rows(tx: ManagedTransaction | undefined, cypher: string, params: Record<string, unknown> = {}): Promise<Row[]> {
    return tx ? txRows(tx, cypher, params) : this.bolt.run(cypher, params);
  }
}

/** `_split_schema_properties`. */
export function splitSchema(merged: Record<string, unknown>): SchemaEnvelope {
  const core: Record<string, unknown> = {};
  const attributes: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(merged)) (SCHEMA_CORE_PROPERTIES.has(k) ? core : attributes)[k] = v;
  return { ...core, type: String(merged["type"]), attributes, inherited_attributes: {} };
}

/** `_get_inherited_attributes`: keys the parent also declares move from
 *  `attributes` to `inherited_attributes`. */
export function inheritedAttributes(schemas: SchemaEnvelope[]): SchemaEnvelope[] {
  const byType = new Map(schemas.map((s) => [s.type, s]));
  return schemas.map((s) => {
    const parent = typeof s["parent"] === "string" ? byType.get(s["parent"] as string) : undefined;
    if (!parent) return { ...s, inherited_attributes: {} };
    const attrs = { ...s.attributes };
    const inherited: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(parent.attributes)) {
      if (k in attrs) {
        inherited[k] = v;
        delete attrs[k];
      }
    }
    return { ...s, attributes: attrs, inherited_attributes: inherited };
  });
}
