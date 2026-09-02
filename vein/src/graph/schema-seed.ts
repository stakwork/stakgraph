/**
 * Vein domain registration — the schema meta-graph, constraints, and
 * indexes jarvis would have created for a domain of its own
 * (`plans/jarvis-graph-compat.md` §4).
 *
 * Runs at every boot of the graph backend. Idempotent and ADD-ONLY: every
 * statement is `IF NOT EXISTS` or a guarded MERGE/CREATE, and an existing
 * Schema node is only ever extended with keys it lacks — never overwritten.
 * Running it twice produces zero graph diff; running it against a
 * jarvis-seeded database changes nothing jarvis owns.
 *
 * Two deployment modes fall out of the same code path: **standalone** (a
 * fresh Neo4j — we create `Thing`, the global constraints, and the global
 * vector index too) and **shared** (jarvis already seeded those; our
 * `IF NOT EXISTS` forms are no-ops).
 */
import { randomUUID } from "node:crypto";
import { Bolt } from "./bolt.js";
import {
  THING_SCHEMA,
  THING_TYPE,
  VEIN_DOMAIN_LABEL,
  VEIN_EDGES,
  VEIN_SCHEMAS,
  embeddingColumn,
  searchableAttributes,
  vectorIndexName,
  vectorIndexedPairs,
  type VeinEdgeDef,
  type VeinSchema,
} from "./vein-schemas.js";

export const VEIN_MIGRATION_ID = "vein_domain_seed_v1";
export const DOMAIN_VECTOR_INDEX = `${VEIN_DOMAIN_LABEL.toLowerCase()}_vector_index`; // domain_vein_vector_index
export const DOMAIN_FULLTEXT_INDEX_V2 = `${VEIN_DOMAIN_LABEL.toLowerCase()}_attribute_index_v2`;
export const GLOBAL_VECTOR_INDEX = "text_embeddings_vector_index";

const VECTOR_OPTIONS =
  "OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } }";

/**
 * Run one schema statement (CREATE CONSTRAINT / INDEX … IF NOT EXISTS),
 * tolerating a pre-existing EQUIVALENT object under a different shape: a
 * jarvis database may carry a plain index on `Data_Bank(ref_id)` where we
 * ask for a uniqueness constraint (Neo4j then refuses: "A constraint cannot
 * be created until the index has been dropped"). jarvis's own seeder logs
 * and continues in that case; so do we — the existing object serves the
 * same purpose and is jarvis-owned. Returns the skip reason, or null.
 */
export async function schemaStatement(bolt: Bolt, cypher: string): Promise<string | null> {
  try {
    await bolt.run(cypher);
    return null;
  } catch (e) {
    const msg = String((e as Error)?.message ?? e);
    if (/already exists|cannot be created until|equivalent/i.test(msg)) return msg.split("\n")[0]!;
    throw e;
  }
}

export interface SeedReport {
  mode: "standalone" | "shared";
  /** Schema types created on this run. */
  createdSchemas: string[];
  /** Existing schemas that were extended with missing keys (add-only). */
  reconciled: Record<string, string[]>;
  /** Edge-schema rows created on this run, as `SRC-[EDGE]->TGT`. */
  createdEdgeSchemas: string[];
  /** Edge-schema rows skipped because an endpoint Schema is absent. */
  skippedEdgeSchemas: string[];
  /** Schema statements skipped because an equivalent jarvis-owned object
   *  already exists under another shape (see `schemaStatement`). */
  skippedSchemaObjects: string[];
}

/**
 * Flatten a schema the way jarvis stores it: core keys and attributes as
 * top-level properties, no `attributes` blob (`schema_crud.py:932,989`).
 * `Thing`'s `name` sits at its top level and flattens the same way.
 */
export function flattenSchema(schema: VeinSchema | typeof THING_SCHEMA): Record<string, unknown> {
  const { attributes, ...core } = schema;
  return { ...core, ...attributes };
}

/** Seed the Vein domain. Safe to call on every boot. */
export async function seedVeinDomain(bolt: Bolt): Promise<SeedReport> {
  const report: SeedReport = {
    mode: "shared",
    createdSchemas: [],
    reconciled: {},
    createdEdgeSchemas: [],
    skippedEdgeSchemas: [],
    skippedSchemaObjects: [],
  };
  const ddl = async (cypher: string) => {
    const skipped = await schemaStatement(bolt, cypher);
    if (skipped) report.skippedSchemaObjects.push(skipped);
  };

  // 1. Thing root — MERGE leaves a jarvis-seeded Thing untouched.
  const thingBefore = await bolt.run(`MATCH (s:Schema {type: $t}) RETURN count(s) AS c`, { t: THING_TYPE });
  const hadThing = Number(thingBefore[0]?.["c"] ?? 0) > 0;
  report.mode = hadThing ? "shared" : "standalone";
  if (!hadThing) {
    await bolt.run(
      `MERGE (s:Schema {type: $type}) ON CREATE SET s = $thing, s.ref_id = $ref_id`,
      { type: THING_TYPE, thing: flattenSchema(THING_SCHEMA), ref_id: randomUUID() },
    );
  }

  // 2. Each Vein schema: guard (case-insensitive, no is_deleted filter — a
  //    soft-deleted schema blocks re-create, same as jarvis's seeder), create
  //    on miss, add-only reconcile on hit.
  for (const schema of VEIN_SCHEMAS) {
    const flat = flattenSchema(schema);
    const hit = await bolt.run(
      `MATCH (n:Schema) WHERE toLower(n.type) = toLower($t) RETURN keys(n) AS ks LIMIT 1`,
      { t: schema.type },
    );
    if (hit.length === 0) {
      await bolt.run(`CREATE (s:Schema) SET s = $flat, s.ref_id = $ref_id`, { flat, ref_id: randomUUID() });
      report.createdSchemas.push(schema.type);
    } else {
      const live = new Set(hit[0]!["ks"] as string[]);
      const missing: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(flat)) {
        if (k === "type" || k === "parent" || k === "ref_id") continue;
        if (!live.has(k)) missing[k] = v;
      }
      if (Object.keys(missing).length > 0) {
        await bolt.run(`MATCH (n:Schema) WHERE toLower(n.type) = toLower($t) SET n += $missing`, {
          t: schema.type,
          missing,
        });
        report.reconciled[schema.type] = Object.keys(missing).sort();
      }
    }

    // 3. CHILD_OF to Thing (type resolution, inheritance, and the edge-schema
    //    ancestor walk all depend on it).
    await bolt.run(
      `MATCH (child:Schema {type: $child}), (parent:Schema {type: $parent})
       MERGE (child)-[r:CHILD_OF]->(parent)
       ON CREATE SET r.ref_id = $ref_id`,
      { child: schema.type, parent: THING_TYPE, ref_id: randomUUID() },
    );

    // 4. Per-type (node_key, namespace) uniqueness + node_key range index.
    await ddl(
      `CREATE CONSTRAINT ${`unique_${schema.type.toLowerCase()}_node_key`} IF NOT EXISTS
       FOR (n:\`${schema.type}\`) REQUIRE (n.node_key, n.namespace) IS UNIQUE`,
    );
    await ddl(`CREATE INDEX IF NOT EXISTS FOR (n:\`${schema.type}\`) ON (n.node_key)`);
  }

  // 5. Global objects (jarvis-owned in shared mode; IF NOT EXISTS is a no-op).
  await ddl(
    `CREATE CONSTRAINT unique_node_key_global IF NOT EXISTS
     FOR (n:Node) REQUIRE (n.node_key, n.namespace) IS UNIQUE`,
  );
  await ddl(`CREATE CONSTRAINT IF NOT EXISTS FOR (n:Data_Bank) REQUIRE n.ref_id IS UNIQUE`);

  // 6. Edge schemas — one relationship per registry row between Schema nodes.
  for (const e of VEIN_EDGES) {
    const r = await seedEdgeSchema(bolt, e);
    if (r === "created") report.createdEdgeSchemas.push(`${e.source}-[${e.edge}]->${e.target}`);
    else if (r === "skipped") report.skippedEdgeSchemas.push(`${e.source}-[${e.edge}]->${e.target}`);
  }

  // 7. Vector indexes — the one thing jarvis does NOT auto-create for a new
  //    domain at startup. Domain, global (standalone), and per-stem.
  await ddl(
    `CREATE VECTOR INDEX \`${DOMAIN_VECTOR_INDEX}\` IF NOT EXISTS
     FOR (n:\`${VEIN_DOMAIN_LABEL}\`) ON n.text_embeddings ${VECTOR_OPTIONS}`,
  );
  await ddl(
    `CREATE VECTOR INDEX \`${GLOBAL_VECTOR_INDEX}\` IF NOT EXISTS
     FOR (n:Data_Bank) ON n.text_embeddings ${VECTOR_OPTIONS}`,
  );
  for (const { type, prop } of vectorIndexedPairs()) {
    await ddl(
      `CREATE VECTOR INDEX \`${vectorIndexName(type, prop)}\` IF NOT EXISTS
       FOR (n:\`${type}\`) ON n.${embeddingColumn(prop)} ${VECTOR_OPTIONS}`,
    );
  }

  // 8. Fulltext (english analyzer `_v2` form — the one jarvis's queries use).
  //    Property list = sorted searchable attrs + node_key, like jarvis's
  //    v2 builder (`attribute_index_helper.py:823,838`).
  const props = [...searchableAttributes(), "node_key"].map((p) => `n.\`${p}\``).join(", ");
  await ddl(
    `CREATE FULLTEXT INDEX \`${DOMAIN_FULLTEXT_INDEX_V2}\` IF NOT EXISTS
     FOR (n:\`${VEIN_DOMAIN_LABEL}\`) ON EACH [${props}]
     OPTIONS { indexConfig: { \`fulltext.analyzer\`: 'english' } }`,
  );

  // 9. Migration ledger stamp so jarvis's runner sees done work, not conflict.
  await ddl(
    `CREATE CONSTRAINT migration_id_unique IF NOT EXISTS
     FOR (m:Migration) REQUIRE m.migration_id IS UNIQUE`,
  );
  await bolt.run(
    `MERGE (m:Migration {migration_id: $id}) ON CREATE SET m.executed_at = timestamp()`,
    { id: VEIN_MIGRATION_ID },
  );

  return report;
}

/**
 * `MERGE (source)-[r:EDGE]->(target)` between two `:Schema` nodes, stamping
 * `ref_id` on create only. A missing endpoint would make MERGE a silent
 * zero-row no-op, so the existence check is explicit: Vein endpoints must
 * exist (bug otherwise); a jarvis-owned endpoint (`Person`) may legitimately
 * be absent in standalone mode → skipped, and `PUBLISHED_BY` stays a
 * property until shared mode.
 */
async function seedEdgeSchema(bolt: Bolt, e: VeinEdgeDef): Promise<"created" | "existing" | "skipped"> {
  const rows = await bolt.run(
    `MATCH (s:Schema {type: $src}), (t:Schema {type: $tgt})
     MERGE (s)-[r:\`${e.edge}\`]->(t)
     ON CREATE SET r.ref_id = $ref_id
     RETURN r.ref_id = $ref_id AS created`,
    { src: e.source, tgt: e.target, ref_id: randomUUID() },
  );
  if (rows.length === 0) {
    if (e.note) return "skipped";
    throw new Error(`seedEdgeSchema: endpoint Schema missing for ${e.source}-[${e.edge}]->${e.target}`);
  }
  return rows[0]!["created"] ? "created" : "existing";
}
