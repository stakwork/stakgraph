/**
 * jarvis-dialect edge writes (`plans/jarvis-graph-compat.md` §3, §6).
 *
 * Canonical semantics from jarvis's bulk path (`bulk_edge_helper.py:243-254`)
 * in plain Cypher — no APOC needed because the edge type is static per
 * statement:
 *
 *   - endpoints matched by `ref_id` on `:Data_Bank` (the only unique ref_id
 *     constraint), never by node_key;
 *   - `IS_ALIAS` COALESCE rewrite (jarvis node-merge parks aliases; the
 *     edge lands on the canonical node);
 *   - `edge_key = edgeType.toLowerCase()` (no Vein edge declares a key
 *     pattern, matching jarvis where effectively none do);
 *   - one edge per (src, type, tgt): ON CREATE only — existing edges are
 *     never mutated;
 *   - stamps: `ref_id`, `edge_key`, `weight: 1`, `date_added_to_graph`
 *     (epoch ms); NO `namespace` on edges;
 *   - soft delete = `is_muted = true`.
 *
 * Validation, stricter than jarvis (§6 item 6): edge type must be in the
 * registry (closed set — no equivalent of jarvis's 63-type bypass), the
 * (source label, edge, target label) triple must match a registry row
 * (`ACCESSED` accepts any target), and both endpoints must resolve — a
 * miss is an error, not a silent zero-row no-op.
 */
import { randomUUID } from "node:crypto";
import type { ManagedTransaction } from "neo4j-driver";
import { Bolt, int, txRows } from "./bolt.js";
import { GraphValidationError } from "./node-writer.js";
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

const STAMPS = new Set(["ref_id", "edge_key", "weight", "date_added_to_graph"]);
const IDENT = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const EDGE_TYPES = new Set(VEIN_EDGES.map((e) => e.edge));
/** Registry check for one (source type, edge, target type) triple. */
export function isRegisteredEdge(edge: string, sourceType: string, targetType: string): boolean {
  return VEIN_EDGES.some(
    (r) => r.edge === edge && r.source === sourceType && (WILDCARD_TARGET_EDGES.has(edge) || r.target === targetType),
  );
}

export function edgeKeyFor(edge: string): string {
  return edge.toLowerCase();
}

export class EdgeWriter {
  constructor(private readonly bolt: Bolt) {}

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
      await validateEndpoints(tx, inputs);
      const byType = new Map<string, number[]>();
      inputs.forEach((i, idx) => byType.set(i.edge, [...(byType.get(i.edge) ?? []), idx]));
      for (const [edge, idxs] of byType) {
        const out = await mergeEdges(tx, edge, idxs.map((i) => inputs[i]!));
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
}

function validateEdgeShape(i: EdgeInput): void {
  if (!EDGE_TYPES.has(i.edge)) throw new GraphValidationError("UNKNOWN_TYPE", i.edge, "not a registered Vein edge type");
  if (!i.source_ref_id || !i.target_ref_id) throw new GraphValidationError("MISSING_REQUIRED", i.edge, "source_ref_id and target_ref_id are required");
  for (const k of Object.keys(i.properties ?? {})) {
    if (STAMPS.has(k)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", i.edge, "edge stamp is system-managed", k);
    if (!IDENT.test(k)) throw new GraphValidationError("UNKNOWN_ATTRIBUTE", i.edge, "edge property is not a bare identifier", k);
  }
}

/** Resolve every endpoint's type label and check each triple against the
 *  registry. Throws on a missing endpoint or unregistered triple. */
async function validateEndpoints(tx: ManagedTransaction, inputs: EdgeInput[]): Promise<void> {
  const ids = [...new Set(inputs.flatMap((i) => [i.source_ref_id, i.target_ref_id]))];
  const rows = await txRows(tx, `UNWIND $ids AS id MATCH (n:Data_Bank {ref_id: id}) RETURN id, labels(n) AS labels`, { ids });
  const labels = new Map(rows.map((r) => [r["id"] as string, r["labels"] as string[]]));
  for (const i of inputs) {
    const sl = labels.get(i.source_ref_id);
    const tl = labels.get(i.target_ref_id);
    if (!sl) throw new GraphValidationError("MISSING_REQUIRED", i.edge, `source ${i.source_ref_id} does not resolve`, "source_ref_id");
    if (!tl) throw new GraphValidationError("MISSING_REQUIRED", i.edge, `target ${i.target_ref_id} does not resolve`, "target_ref_id");
    const st = typeLabelOf(sl);
    const tt = typeLabelOf(tl);
    if (!st || !isVeinType(st)) throw new GraphValidationError("WRONG_TYPE", i.edge, `source type ${st ?? "?"} is not a Vein type`, "source_ref_id");
    if (!isRegisteredEdge(i.edge, st, tt ?? "")) {
      throw new GraphValidationError("WRONG_TYPE", i.edge, `${st}-[${i.edge}]->${tt ?? "?"} is not a registered edge`);
    }
  }
}

async function mergeEdges(tx: ManagedTransaction, edge: string, inputs: EdgeInput[]): Promise<EdgeWriteResult[]> {
  const edge_key = edgeKeyFor(edge);
  const rows = inputs.map((i, k) => ({
    k,
    src: i.source_ref_id,
    tgt: i.target_ref_id,
    edge_key,
    on_create: {
      ...(i.properties ?? {}),
      ref_id: randomUUID(),
      edge_key,
      weight: i.weight === undefined ? int(1) : Number.isInteger(i.weight) ? int(i.weight) : i.weight,
      date_added_to_graph: int(Date.now()),
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
  if (out.length !== inputs.length) {
    throw new Error(`edge MERGE for ${edge} returned ${out.length} rows for ${inputs.length} inputs`);
  }
  const byK = new Map(out.map((r) => [r["k"] as number, r]));
  return inputs.map((_, k) => {
    const r = byK.get(k)!;
    return {
      ref_id: r["ref_id"] as string,
      edge_key,
      created: r["created"] as boolean,
      source_ref_id: r["source_ref_id"] as string,
      target_ref_id: r["target_ref_id"] as string,
    };
  });
}
