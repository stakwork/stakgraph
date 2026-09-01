/**
 * Seed the bundled jarvis ontology (`fixtures/jarvis-ontology.ts`) into a
 * database — so a STANDALONE vein Neo4j can host the jarvis-typed data the
 * lab pipelines write (Document, EvalSet, Concept, …) without a jarvis
 * process. Same add-only discipline as `seedVeinDomain`: guarded creates,
 * `IF NOT EXISTS` schema objects, zero diff on re-run, and a jarvis-seeded
 * database is left untouched (every schema already exists → nothing
 * written).
 *
 * What jarvis's own seeder + migrations produce for these types and what
 * we reproduce: the flattened `:Schema` nodes (jarvis's ref_ids kept),
 * `CHILD_OF` to each parent, per-type `(node_key, namespace)` constraints
 * and `node_key` indexes, the Schema→Schema edge schemas with their
 * properties, per-domain fulltext (`_v2`, english) and vector indexes, the
 * global `data_bank_attribute_index_v2` + `text_embeddings_vector_index`,
 * and the `*` wildcard sentinel.
 */
import { randomUUID } from "node:crypto";
import { Bolt } from "./bolt.js";
import { JARVIS_ONTOLOGY, type OntologyFixture } from "./fixtures/jarvis-ontology.js";
import { schemaStatement } from "./schema-seed.js";
import { SCHEMA_CORE_PROPERTIES } from "./vein-schemas.js";

export interface OntologySeedReport {
  createdSchemas: string[];
  createdEdgeSchemas: number;
  domains: string[];
  /** Schema statements skipped over an equivalent pre-existing object. */
  skippedSchemaObjects: string[];
}

const VECTOR_OPTIONS =
  "OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } }";

/**
 * `get_searchable_attributes_from_schema`: with an explicit index → index
 * fields + title_key + description_key; otherwise every string-typed
 * non-core attribute (plus index/title/description keys).
 */
export function searchableAttributesOf(schema: Record<string, unknown>): Set<string> {
  const out = new Set<string>();
  const rawIndex = schema["index"];
  const indexFields = Array.isArray(rawIndex) ? rawIndex.map(String).filter(Boolean) : typeof rawIndex === "string" && rawIndex ? [rawIndex] : [];
  const add = (k: unknown) => {
    if (typeof k === "string" && k.trim()) out.add(k.trim());
  };
  if (indexFields.length) {
    for (const f of indexFields) add(f);
    add(schema["title_key"]);
    add(schema["description_key"]);
    return out;
  }
  for (const [k, v] of Object.entries(schema)) {
    if (SCHEMA_CORE_PROPERTIES.has(k)) continue;
    if (v === "string" || v === "?string") out.add(k);
  }
  out.add("node_key");
  add(schema["title_key"]);
  add(schema["description_key"]);
  return out;
}

export async function seedJarvisOntology(bolt: Bolt, fixture: OntologyFixture = JARVIS_ONTOLOGY): Promise<OntologySeedReport> {
  const report: OntologySeedReport = { createdSchemas: [], createdEdgeSchemas: 0, domains: [], skippedSchemaObjects: [] };
  const ddl = async (cypher: string) => {
    const skipped = await schemaStatement(bolt, cypher);
    if (skipped) report.skippedSchemaObjects.push(skipped);
  };

  // 1. Schema nodes (guarded, case-insensitive like jarvis's seeder).
  for (const schema of fixture.schemas) {
    const type = String(schema["type"]);
    const hit = await bolt.run(`MATCH (n:Schema) WHERE toLower(n.type) = toLower($t) RETURN n.type AS t LIMIT 1`, { t: type });
    if (hit.length) continue;
    const props = { ...schema };
    if (typeof props["ref_id"] !== "string") props["ref_id"] = randomUUID();
    await bolt.run(`CREATE (s:Schema) SET s = $props`, { props });
    report.createdSchemas.push(type);
  }

  // 2. CHILD_OF from each schema's `parent`.
  for (const schema of fixture.schemas) {
    const parent = schema["parent"];
    if (typeof parent !== "string" || !parent) continue;
    await bolt.run(
      `MATCH (child:Schema {type: $c}), (parent:Schema {type: $p})
       MERGE (child)-[r:CHILD_OF]->(parent) ON CREATE SET r.ref_id = $ref_id`,
      { c: schema["type"], p: parent, ref_id: randomUUID() },
    );
  }

  // 3. Per-type constraint + index for every schema with a node_key.
  for (const schema of fixture.schemas) {
    const type = String(schema["type"]);
    if (type === "*" || typeof schema["node_key"] !== "string") continue;
    await ddl(
      `CREATE CONSTRAINT ${`unique_${type.toLowerCase()}_node_key`} IF NOT EXISTS FOR (n:\`${type}\`) REQUIRE (n.node_key, n.namespace) IS UNIQUE`,
    );
    await ddl(`CREATE INDEX IF NOT EXISTS FOR (n:\`${type}\`) ON (n.node_key)`);
  }
  await ddl(`CREATE CONSTRAINT unique_node_key_global IF NOT EXISTS FOR (n:Node) REQUIRE (n.node_key, n.namespace) IS UNIQUE`);
  await ddl(`CREATE CONSTRAINT IF NOT EXISTS FOR (n:Data_Bank) REQUIRE n.ref_id IS UNIQUE`);

  // 4. Edge schemas (`create_edge_schema`: MERGE + properties, ref_id kept).
  for (const e of fixture.edge_schemas) {
    if (!/^[A-Z][A-Z0-9_]*$/.test(e.edge)) continue;
    const props = { ...e.props };
    if (typeof props["ref_id"] !== "string") props["ref_id"] = randomUUID();
    // The fixture keeps jarvis's ref_ids, so "did MERGE create?" cannot be
    // read off ref_id — use a transient marker instead.
    const rows = await bolt.run(
      `MATCH (s:Schema {type: $s}), (t:Schema {type: $t})
       MERGE (s)-[r:\`${e.edge}\`]->(t)
       ON CREATE SET r = $props, r.__created = true
       WITH r, coalesce(r.__created, false) AS created
       REMOVE r.__created
       RETURN created`,
      { s: e.source, t: e.target, props },
    );
    if (rows[0]?.["created"]) report.createdEdgeSchemas++;
  }

  // 5. Search indexes: global + per domain, over the union of searchable
  //    attributes (jarvis `_build_domain_indexes` / v2 builder).
  const attrs = new Set<string>();
  const domains = new Set<string>();
  for (const s of fixture.schemas) {
    if (s["type"] === "*") continue;
    for (const a of searchableAttributesOf(s)) attrs.add(a);
    if (typeof s["domain"] === "string" && s["domain"]) domains.add((s["domain"] as string).toLowerCase());
  }
  attrs.add("node_key");
  const props = [...attrs].sort().map((p) => `n.\`${p}\``).join(", ");
  await ddl(
    `CREATE FULLTEXT INDEX data_bank_attribute_index_v2 IF NOT EXISTS FOR (n:Data_Bank) ON EACH [${props}]
     OPTIONS { indexConfig: { \`fulltext.analyzer\`: 'english' } }`,
  );
  await ddl(`CREATE VECTOR INDEX text_embeddings_vector_index IF NOT EXISTS FOR (n:Data_Bank) ON n.text_embeddings ${VECTOR_OPTIONS}`);
  for (const d of [...domains].sort()) {
    await ddl(
      `CREATE FULLTEXT INDEX \`domain_${d}_attribute_index_v2\` IF NOT EXISTS FOR (n:\`Domain_${d}\`) ON EACH [${props}]
       OPTIONS { indexConfig: { \`fulltext.analyzer\`: 'english' } }`,
    );
    await ddl(`CREATE VECTOR INDEX \`domain_${d}_vector_index\` IF NOT EXISTS FOR (n:\`Domain_${d}\`) ON n.text_embeddings ${VECTOR_OPTIONS}`);
  }
  for (const s of fixture.schemas) {
    const vi = s["vector_index"];
    if (!Array.isArray(vi)) continue;
    for (const prop of vi) {
      const stem = String(prop).endsWith("_schema") ? String(prop).slice(0, -"_schema".length) : String(prop);
      await ddl(
        `CREATE VECTOR INDEX \`${String(s["type"]).toLowerCase()}_${stem}_vector_index\` IF NOT EXISTS FOR (n:\`${s["type"]}\`) ON n.${stem}_embeddings ${VECTOR_OPTIONS}`,
      );
    }
  }
  report.domains = [...domains].sort();
  return report;
}
