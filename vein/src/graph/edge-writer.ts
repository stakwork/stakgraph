/**
 * jarvis-dialect edge writes (`plans/jarvis-graph-compat.md` §3, §6).
 *
 * Canonical semantics from jarvis's bulk path (`bulk_edge_helper.py:243-254`
 * + `node_service_v2.py` edge stamping) in plain Cypher — no APOC needed
 * because the edge type is static per statement:
 *
 *   - endpoints matched by `ref_id` on `:Data_Bank` (the only unique ref_id
 *     constraint), never by node_key;
 *   - `IS_ALIAS` COALESCE rewrite (jarvis node-merge parks aliases; the
 *     edge lands on the canonical node);
 *   - `edge_key` = the edge schema's `edge_key` pattern sanitized over the
 *     edge properties when it declares one, else `edgeType.toLowerCase()`;
 *   - one edge per (src, type, tgt): ON CREATE only — existing edges are
 *     never mutated;
 *   - stamps: `ref_id`, `edge_key`, `weight` (1 unless given),
 *     `date_added_to_graph` (epoch ms), and `unique_source_id` when both
 *     endpoints carry the same one; NO `namespace` on edges;
 *   - soft delete = `is_muted = true`.
 *
 * Validation: an edge whose SOURCE is a Vein type must be a row of the
 * closed Vein registry (§6 item 6; `ACCESSED` accepts any target). Any
 * other source type follows jarvis's own rules — the `EDGE_TYPES`
 * allowlist, else an edge schema must exist between the endpoint types or
 * their ancestors (or the `*` wildcard). Both endpoints must resolve — a
 * miss is an error, not a silent zero-row no-op.
 */
import { randomUUID } from "node:crypto";
import type { ManagedTransaction } from "neo4j-driver";
import { Bolt, int, txRows } from "./bolt.js";
import { GraphValidationError } from "./node-writer.js";
import { SchemaResolver } from "./schema-resolver.js";
import { VEIN_EDGES, WILDCARD_TARGET_EDGES, isVeinType, typeLabelOf } from "./vein-schemas.js";

export { typeLabelOf };

export interface EdgeInput {
  edge: string;
  source_ref_id: string;
  target_ref_id: string;
  /** Extra edge attributes — unvalidated passthrough, as in jarvis. Stamp
   *  keys (`ref_id`, `edge_key`, `weight`, `date_added_to_graph`) may not
   *  be supplied. Plain JS numbers are written as FLOAT; wrap with `int()`
   *  from `bolt.ts` for an Integer. */
  properties?: Record<string, unknown>;
  /** Overrides the `weight: 1` stamp on create (jarvis accepts a caller
   *  weight on POST /v2/edges). Written as an Integer when integral. */
  weight?: number;
}

export interface EdgeWriteResult {
  ref_id: string;
  edge_key: string;
  created: boolean;
  /** The ref_ids the edge actually landed on after alias rewrite. */
  source_ref_id: string;
  target_ref_id: string;
}

export interface EdgeWriterOptions {
  resolver?: SchemaResolver;
}

const STAMPS = new Set(["ref_id", "edge_key", "weight", "date_added_to_graph", "unique_source_id"]);
const IDENT = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const EDGE_TYPE = /^[A-Z][A-Z0-9_]*$/;

/** Registry check for one (source type, edge, target type) triple. */
export function isRegisteredEdge(edge: string, sourceType: string, targetType: string): boolean {
  return VEIN_EDGES.some(
    (r) => r.edge === edge && r.source === sourceType && (WILDCARD_TARGET_EDGES.has(edge) || r.target === targetType),
  );
}

export function edgeKeyFor(edge: string): string {
  return edge.toLowerCase();
}

/** jarvis `sanitize_edge_key`: each `-`-token of the schema's edge_key
 *  pattern is looked up (case-insensitively) in the edge properties and
 *  sanitized like node_key values. */
export function composeEdgeKey(pattern: string, properties: Record<string, unknown>): string {
  const lower = new Map(Object.entries(properties).map(([k, v]) => [k.toLowerCase(), v]));
  return pattern
    .split("-")
    .map((tok) => {
      if (!lower.has(tok.toLowerCase())) throw new GraphValidationError("MISSING_REQUIRED", "edge", `edge_key property ${tok} missing from edge data`, tok);
      return String(lower.get(tok.toLowerCase())).trim().replace(/ /g, "").toLowerCase().replace(/[^a-zA-Z0-9\s]/g, "");
    })
    .join("-");
}

interface ResolvedEdge {
  input: EdgeInput;
  edge_key: string;
  unique_source_id?: string;
}

export class EdgeWriter {
  readonly resolver: SchemaResolver;
  constructor(
    private readonly bolt: Bolt,
    opts: EdgeWriterOptions = {},
  ) {
    this.resolver = opts.resolver ?? new SchemaResolver(bolt);
  }

  async write(input: EdgeInput): Promise<EdgeWriteResult> {
    const [r] = await this.writeMany([input]);
    return r!;
  }

  /**
   * Write many edges in one transaction — one UNWIND MERGE per edge type.
   * Results in input order. All-or-nothing: any validation failure
   * (unknown type, unregistered triple, unresolvable endpoint) writes
   * nothing.
   */
  async writeMany(inputs: EdgeInput[]): Promise<EdgeWriteResult[]> {
    if (inputs.length === 0) return [];
    for (const i of inputs) validateEdgeShape(i);
    const results: EdgeWriteResult[] = new Array(inputs.length);
    await this.bolt.write(async (tx) => {
      const resolved = await this.validateEndpoints(tx, inputs);
      const byType = new Map<string, number[]>();
      inputs.forEach((i, idx) => byType.set(i.edge, [...(byType.get(i.edge) ?? []), idx]));
      for (const [edge, idxs] of byType) {
        const out = await mergeEdges(tx, edge, idxs.map((i) => resolved[i]!));
        idxs.forEach((i, k) => {
          results[i] = out[k]!;
        });
      }
    });
    return results;
  }

  /** Edge soft delete (`is_muted = true`), by edge ref_id. */
  async mute(ref_id: string): Promise<boolean> {
    const rows = await this.bolt.run(`MATCH ()-[r {ref_id: $ref_id}]->() SET r.is_muted = true RETURN r.ref_id AS ref_id`, { ref_id });
    return rows.length > 0;
  }

  /** Resolve every endpoint's type label and check each triple: Vein
   *  registry for Vein sources, jarvis's allowlist/edge-schema rules for the
   *  rest. Throws on a missing endpoint or disallowed triple. */
  private async validateEndpoints(tx: ManagedTransaction, inputs: EdgeInput[]): Promise<ResolvedEdge[]> {
    const ids = [...new Set(inputs.flatMap((i) => [i.source_ref_id, i.target_ref_id]))];
    const rows = await txRows(tx, `UNWIND $ids AS id MATCH (n:Data_Bank {ref_id: id}) RETURN id, labels(n) AS labels, n.unique_source_id AS uid`, { ids });
    const nodes = new Map(rows.map((r) => [r["id"] as string, { labels: r["labels"] as string[], uid: r["uid"] as string | null }]));
    const out: ResolvedEdge[] = [];
    for (const i of inputs) {
      const s = nodes.get(i.source_ref_id);
      const t = nodes.get(i.target_ref_id);
      if (!s) throw new GraphValidationError("MISSING_REQUIRED", i.edge, `source ${i.source_ref_id} does not resolve`, "source_ref_id");
      if (!t) throw new GraphValidationError("MISSING_REQUIRED", i.edge, `target ${i.target_ref_id} does not resolve`, "target_ref_id");
      const st = typeLabelOf(s.labels);
      const tt = typeLabelOf(t.labels);
      if (!st) throw new GraphValidationError("WRONG_TYPE", i.edge, "source node has no type label", "source_ref_id");
      let edge_key = edgeKeyFor(i.edge);
      if (isVeinType(st)) {
        if (!isRegisteredEdge(i.edge, st, tt ?? "")) {
          throw new GraphValidationError("WRONG_TYPE", i.edge, `${st}-[${i.edge}]->${tt ?? "?"} is not a registered Vein edge`);
        }
      } else {
        const match = await this.resolver.edgeSchema(i.edge, st, tt ?? "", tx);
        if (!match) throw new GraphValidationError("WRONG_TYPE", i.edge, `Invalid edge type: ${i.edge} (no edge schema ${st}-[${i.edge}]->${tt ?? "?"})`);
        const pattern = match.properties["edge_key"];
        if (typeof pattern === "string" && pattern) edge_key = composeEdgeKey(pattern, i.properties ?? {});
      }
      out.push({ input: i, edge_key, unique_source_id: s.uid && s.uid === t.uid ? s.uid : undefined });
    }
    return out;
  }
}

function validateEdgeShape(i: EdgeInput): void {
  if (!EDGE_TYPE.test(i.edge)) throw new GraphValidationError("UNKNOWN_TYPE", i.edge, "edge type must match ^[A-Z][A-Z0-9_]*$");
  if (!i.source_ref_id || !i.target_ref_id) throw new GraphValidationError("MISSING_REQUIRED", i.edge, "source_ref_id and target_ref_id are required");
  for (const k of Object.keys(i.properties ?? {})) {
    if (STAMPS.has(k)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", i.edge, "edge stamp is system-managed", k);
    if (!IDENT.test(k)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", i.edge, "edge property is not a bare identifier", k);
  }
}

async function mergeEdges(tx: ManagedTransaction, edge: string, edges: ResolvedEdge[]): Promise<EdgeWriteResult[]> {
  const rows = edges.map((e, k) => ({
    k,
    src: e.input.source_ref_id,
    tgt: e.input.target_ref_id,
    edge_key: e.edge_key,
    on_create: {
      ...(e.input.properties ?? {}),
      ref_id: randomUUID(),
      edge_key: e.edge_key,
      weight: e.input.weight === undefined ? int(1) : Number.isInteger(e.input.weight) ? int(e.input.weight) : e.input.weight,
      date_added_to_graph: int(Date.now()),
      ...(e.unique_source_id ? { unique_source_id: e.unique_source_id } : {}),
    },
  }));
  const out = await txRows(
    tx,
    `UNWIND $edges AS e
     MATCH (source:Data_Bank {ref_id: e.src})
     MATCH (target:Data_Bank {ref_id: e.tgt})
     OPTIONAL MATCH (source)-[:IS_ALIAS]->(sa)
     OPTIONAL MATCH (target)-[:IS_ALIAS]->(ta)
     WITH COALESCE(sa, source) AS ns, COALESCE(ta, target) AS nt, e
     MERGE (ns)-[r:\`${edge}\` {edge_key: e.edge_key}]->(nt)
     ON CREATE SET r += e.on_create
     RETURN e.k AS k, r.ref_id AS ref_id, r.ref_id = e.on_create.ref_id AS created,
            ns.ref_id AS source_ref_id, nt.ref_id AS target_ref_id`,
    { edges: rows },
  );
  if (out.length !== edges.length) {
    throw new Error(`edge MERGE for ${edge} returned ${out.length} rows for ${edges.length} inputs`);
  }
  const byK = new Map(out.map((r) => [r["k"] as number, r]));
  return edges.map((e, k) => {
    const r = byK.get(k)!;
    return {
      ref_id: r["ref_id"] as string,
      edge_key: e.edge_key,
      created: r["created"] as boolean,
      source_ref_id: r["source_ref_id"] as string,
      target_ref_id: r["target_ref_id"] as string,
    };
  });
}
