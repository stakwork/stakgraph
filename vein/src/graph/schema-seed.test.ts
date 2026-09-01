import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { Bolt } from "./bolt.js";
import { seedVeinDomain, flattenSchema, VEIN_MIGRATION_ID, DOMAIN_VECTOR_INDEX, DOMAIN_FULLTEXT_INDEX_V2, GLOBAL_VECTOR_INDEX } from "./schema-seed.js";
import { THING_SCHEMA, VEIN_EDGES, VEIN_SCHEMAS, assertLibraryWellFormed, searchableAttributes, vectorIndexName } from "./vein-schemas.js";
import { graphSnapshot, schemaObjectNames, testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();

describe("vein-schemas (pure)", () => {
  it("library is well-formed", () => {
    assertLibraryWellFormed();
    assert.equal(VEIN_SCHEMAS.length, 9);
    assert.equal(VEIN_EDGES.length, 14);
  });

  it("flattens attributes onto the top level with no attributes blob", () => {
    const flat = flattenSchema(VEIN_SCHEMAS[0]!);
    assert.equal(flat["attributes"], undefined);
    assert.equal(flat["name"], "string");
    assert.equal(flat["type"], "VeinWorkflow");
    assert.deepEqual(flat["index"], ["name", "description"]);
    const thing = flattenSchema(THING_SCHEMA);
    assert.equal(thing["name"], "string");
    assert.equal(thing["description"], "?string");
    assert.equal(thing["attributes"], undefined);
  });

  it("rejects a node_key token that is optional", () => {
    const bad = { ...VEIN_SCHEMAS[0]!, attributes: { ...VEIN_SCHEMAS[0]!.attributes, name: "?string" as const } };
    assert.throws(() => assertLibraryWellFormed([bad], []), /must be required/);
  });

  it("rejects an unknown edge endpoint without a note", () => {
    assert.throws(() => assertLibraryWellFormed(VEIN_SCHEMAS, [{ edge: "X", source: "VeinRun", target: "Nope" }]), /not a Vein type/);
  });
});

describe("seedVeinDomain (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
  });
  after(async () => {
    await bolt?.close();
  });
  beforeEach(async () => {
    await wipeGraph(bolt);
  });

  it("standalone: seeds Thing + 9 schemas + edges + indexes, and is idempotent", async () => {
    const r1 = await seedVeinDomain(bolt);
    assert.equal(r1.mode, "standalone");
    assert.deepEqual(r1.createdSchemas.sort(), VEIN_SCHEMAS.map((s) => s.type).sort());
    // PUBLISHED_BY → Person is skipped (no Person schema in standalone).
    assert.deepEqual(r1.skippedEdgeSchemas, ["VeinStepVersion-[PUBLISHED_BY]->Person"]);
    assert.equal(r1.createdEdgeSchemas.length, VEIN_EDGES.length - 1);

    const snap1 = await graphSnapshot(bolt);

    // Meta-graph shape.
    const thing = snap1.nodes.filter((n) => n.labels.includes("Schema") && n.properties["type"] === "Thing");
    assert.equal(thing.length, 1);
    assert.equal(thing[0]!.properties["name"], "string");
    assert.equal(thing[0]!.properties["node_key"], "thing-name");
    assert.ok(typeof thing[0]!.properties["ref_id"] === "string");
    const schemas = snap1.nodes.filter((n) => n.labels.includes("Schema"));
    assert.equal(schemas.length, 10);
    for (const s of schemas) {
      assert.deepEqual(s.labels, ["Schema"], "Schema nodes carry no other label");
      assert.equal(s.properties["attributes"], undefined);
    }
    const run = schemas.find((s) => s.properties["type"] === "VeinRun")!;
    assert.equal(run.properties["domain"], "Vein");
    assert.equal(run.properties["parent"], "Thing");
    assert.equal(run.properties["run_id"], "string");
    assert.deepEqual(run.properties["index"], ["workflow_name", "status", "summary"]);
    const childOf = snap1.rels.filter((r) => r.type === "CHILD_OF");
    assert.equal(childOf.length, 9);
    for (const r of childOf) {
      assert.equal((r.to as { type: string }).type, "Thing");
      assert.ok(typeof r.properties["ref_id"] === "string");
      assert.deepEqual(Object.keys(r.properties), ["ref_id"]);
    }
    const accessed = snap1.rels.find((r) => r.type === "ACCESSED")!;
    assert.equal((accessed.from as { type: string }).type, "VeinToolCall");
    assert.equal((accessed.to as { type: string }).type, "Thing");
    assert.equal(snap1.rels.filter((r) => r.type === "PUBLISHED_BY").length, 0);
    const ledger = snap1.nodes.find((n) => n.labels.includes("Migration"))!;
    assert.equal(ledger.properties["migration_id"], VEIN_MIGRATION_ID);
    assert.equal(typeof ledger.properties["executed_at"], "number");

    // Domain registered by existence.
    const domains = await bolt.run(`MATCH (s:Schema) WHERE s.domain IS NOT NULL RETURN DISTINCT toLower(s.domain) AS d`);
    assert.deepEqual(domains.map((d) => d["d"]), ["vein"]);

    // Constraints + indexes.
    const names = await schemaObjectNames(bolt);
    for (const s of VEIN_SCHEMAS) assert.ok(names.constraints.includes(`unique_${s.type.toLowerCase()}_node_key`), s.type);
    assert.ok(names.constraints.includes("unique_node_key_global"));
    assert.ok(names.constraints.includes("migration_id_unique"));
    assert.ok(names.indexes.includes(DOMAIN_VECTOR_INDEX));
    assert.ok(names.indexes.includes(GLOBAL_VECTOR_INDEX));
    assert.ok(names.indexes.includes(DOMAIN_FULLTEXT_INDEX_V2));
    assert.ok(names.indexes.includes(vectorIndexName("VeinWorkflowVersion", "input_schema")));
    assert.ok(names.indexes.includes(vectorIndexName("VeinStep", "output_schema")));
    const ft = (snap1.indexes as Array<Record<string, unknown>>).find((i) => i["name"] === DOMAIN_FULLTEXT_INDEX_V2)!;
    assert.deepEqual(ft["labelsOrTypes"], ["Domain_vein"]);
    assert.deepEqual(ft["properties"], [...searchableAttributes(), "node_key"]);
    const vec = (snap1.indexes as Array<Record<string, unknown>>).find((i) => i["name"] === DOMAIN_VECTOR_INDEX)!;
    assert.deepEqual(vec["labelsOrTypes"], ["Domain_vein"]);
    assert.deepEqual(vec["properties"], ["text_embeddings"]);
    const dataBankRefId = (snap1.constraints as Array<Record<string, unknown>>).find(
      (c) => JSON.stringify(c["labelsOrTypes"]) === '["Data_Bank"]' && JSON.stringify(c["properties"]) === '["ref_id"]',
    );
    assert.ok(dataBankRefId, "Data_Bank ref_id uniqueness");

    // Idempotence: second run → nothing created, graph byte-identical.
    const r2 = await seedVeinDomain(bolt);
    assert.equal(r2.mode, "shared");
    assert.deepEqual(r2.createdSchemas, []);
    assert.deepEqual(r2.reconciled, {});
    assert.deepEqual(r2.createdEdgeSchemas, []);
    const snap2 = await graphSnapshot(bolt);
    assert.deepEqual(snap2, snap1);
  });

  it("shared: leaves a jarvis-seeded Thing/Person untouched and add-only reconciles an existing Vein schema", async () => {
    // Simulate what jarvis would already have: its own Thing (different
    // colours, extra key), a Person schema, the global constraints, and a
    // pre-existing VeinRun schema missing one attribute + carrying a
    // jarvis-only key.
    const thingRef = randomUUID();
    const jarvisThing = { ...flattenSchema(THING_SCHEMA), primary_color: "#000000", some_jarvis_key: "x" };
    await bolt.run(`CREATE (s:Schema) SET s = $t, s.ref_id = $r`, { t: jarvisThing, r: thingRef });
    const personRef = randomUUID();
    await bolt.run(
      `CREATE (s:Schema {type: "Person", parent: "Thing", node_key: "person-name", name: "string", index: ["name"], ref_id: $r})
       WITH s MATCH (t:Schema {type: "Thing"}) MERGE (s)-[:CHILD_OF {ref_id: $e}]->(t)`,
      { r: personRef, e: randomUUID() },
    );
    await bolt.run(`CREATE CONSTRAINT unique_node_key_global IF NOT EXISTS FOR (n:Node) REQUIRE (n.node_key, n.namespace) IS UNIQUE`);
    const runSchema = VEIN_SCHEMAS.find((s) => s.type === "VeinRun")!;
    const { summary: _dropped, ...partialAttrs } = runSchema.attributes;
    const staleRun = { ...flattenSchema({ ...runSchema, attributes: partialAttrs }), jarvis_extra: 1, index: ["workflow_name"] };
    const staleRef = randomUUID();
    await bolt.run(`CREATE (s:Schema) SET s = $t, s.ref_id = $r`, { t: staleRun, r: staleRef });

    const before = await graphSnapshot(bolt);
    const r = await seedVeinDomain(bolt);
    assert.equal(r.mode, "shared");
    assert.equal(r.createdSchemas.length, 8);
    assert.ok(!r.createdSchemas.includes("VeinRun"));
    assert.deepEqual(r.reconciled, { VeinRun: ["summary"] });
    assert.deepEqual(r.skippedEdgeSchemas, []);
    assert.ok(r.createdEdgeSchemas.includes("VeinStepVersion-[PUBLISHED_BY]->Person"));

    const after = await graphSnapshot(bolt);
    // jarvis-owned nodes are byte-identical.
    const thingBefore = before.nodes.find((n) => n.properties["type"] === "Thing");
    const thingAfter = after.nodes.find((n) => n.properties["type"] === "Thing");
    assert.deepEqual(thingAfter, thingBefore);
    const personBefore = before.nodes.find((n) => n.properties["type"] === "Person");
    const personAfter = after.nodes.find((n) => n.properties["type"] === "Person");
    assert.deepEqual(personAfter, personBefore);
    // Pre-existing VeinRun: only the missing key was added; ref_id, index,
    // and the jarvis-only key survive.
    const runAfter = after.nodes.find((n) => n.properties["type"] === "VeinRun")!;
    assert.equal(runAfter.properties["ref_id"], staleRef);
    assert.equal(runAfter.properties["summary"], "?string");
    assert.equal(runAfter.properties["jarvis_extra"], 1);
    assert.deepEqual(runAfter.properties["index"], ["workflow_name"]);
    // Every pre-existing schema object still exists.
    const names = await schemaObjectNames(bolt);
    assert.ok(names.constraints.includes("unique_node_key_global"));

    // And a second run is a no-op.
    const r2 = await seedVeinDomain(bolt);
    assert.deepEqual(r2.createdSchemas, []);
    assert.deepEqual(r2.reconciled, {});
    assert.deepEqual(await graphSnapshot(bolt), after);
  });
});
