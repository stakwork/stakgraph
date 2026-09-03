/**
 * Node-schema registration: pure planning rules, and (live) the write —
 * a new type becomes writable through the NodeWriter, an existing jarvis
 * type can be extended add-only, and Vein's closed registry is refused.
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { Bolt } from "./bolt.js";
import { EdgeWriter } from "./edge-writer.js";
import { GraphValidationError, NodeWriter } from "./node-writer.js";
import { seedJarvisOntology } from "./ontology-seed.js";
import { createNodeSchema, planNodeSchema } from "./schema-crud.js";
import { SchemaResolver } from "./schema-resolver.js";
import { seedVeinDomain } from "./schema-seed.js";
import { GraphReader } from "./search.js";
import { schemaObjectNames, testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    if (e instanceof GraphValidationError) return `${e.code}${e.attribute ? `:${e.attribute}` : ""}`;
    throw e;
  }
  return "ok";
}

describe("planNodeSchema (pure)", () => {
  it("normalizes: default parent/domain/index, prefixed node_key, flat props", () => {
    const p = planNodeSchema({ type: "Evidence", attributes: { description: "string", content: "?string" }, node_key: "description" });
    assert.equal(p.parent, "Thing");
    assert.equal(p.domain, "entity");
    assert.equal(p.node_key, "evidence-description");
    assert.deepEqual(p.tokens, ["description"]);
    assert.deepEqual(p.index, ["description"]);
    assert.deepEqual(p.flat, { type: "Evidence", parent: "Thing", node_key: "evidence-description", index: ["description"], domain: "entity", description: "string", content: "?string" });
    // An already-prefixed node_key is not doubled; `name` needs no declaration.
    assert.equal(planNodeSchema({ type: "Claim", attributes: { claim_text: "string" }, node_key: "claim-claim_text-name" }).node_key, "claim-claim_text-name");
    assert.equal(planNodeSchema({ type: "Note", attributes: {} }).node_key, "note-name");
  });

  it("rejects bad types, reserved/unknown attribute names, bad type strings, undeclared key tokens", () => {
    assert.equal(code(() => planNodeSchema({ type: "9x", attributes: {} })), "UNKNOWN_TYPE");
    assert.equal(code(() => planNodeSchema({ type: "Thing", attributes: {} })), "UNKNOWN_TYPE");
    assert.equal(code(() => planNodeSchema({ type: "VeinRun", attributes: {} })), "UNKNOWN_TYPE");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { "bad-name": "string" } })), "UNKNOWN_ATTRIBUTE:bad-name");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { ref_id: "string" } })), "UNKNOWN_ATTRIBUTE:ref_id");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { node_key: "string" } })), "UNKNOWN_ATTRIBUTE:node_key");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { score: "number" } })), "WRONG_TYPE:score");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { a: "string" }, node_key: "b" })), "UNKNOWN_ATTRIBUTE:node_key");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { a: "string" }, index: ["zz"] })), "UNKNOWN_ATTRIBUTE:index");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { a: "string" }, title_key: "zz" })), "UNKNOWN_ATTRIBUTE:title_key");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: {}, node_key: "-" })), "MISSING_REQUIRED:node_key");
    assert.equal(code(() => planNodeSchema({ type: "X", attributes: { a: "?list", b: "datetime", name: "string" } })), "ok");
  });
});

describe("createNodeSchema (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  let resolver: SchemaResolver;
  let nodes: NodeWriter;
  let edges: EdgeWriter;
  let reader: GraphReader;
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
    await wipeGraph(bolt);
    await seedJarvisOntology(bolt);
    await seedVeinDomain(bolt);
    resolver = new SchemaResolver(bolt);
    nodes = new NodeWriter(bolt, { resolver });
    edges = new EdgeWriter(bolt, { resolver });
    reader = new GraphReader(bolt, { resolver });
  });
  after(async () => {
    await bolt?.close();
  });

  it("a new type is writable through the NodeWriter, with constraint + index, CHILD_OF, and inherited attributes", async () => {
    // Unknown before.
    await assert.rejects(nodes.write({ type: "Evidence", data: { name: "e" } }, "create"), (e: any) => e.code === "UNKNOWN_TYPE");
    const r = await createNodeSchema(bolt, resolver, {
      type: "Evidence",
      attributes: { description: "string", content: "?string", evidence_status: "?string", strength: "?float" },
      node_key: "description",
      title_key: "description",
      description_key: "content",
      type_description: "A planned or collected piece of evidence",
    });
    assert.equal(r.created, true);
    assert.equal(r.type, "Evidence");
    assert.equal(r.parent, "Thing");
    assert.equal(r.node_key, "evidence-description");
    assert.deepEqual(r.added, []);
    // `description` is a jarvis dual-use (core) name: the resolver reports it
    // optional, and its presence is enforced through the node_key instead.
    assert.equal(r.attributes.description, "?string");
    assert.equal(r.attributes.evidence_status, "?string");
    assert.equal(r.attributes.name, "?string", "Thing's name inherited (optional, as for every jarvis type)");
    assert.equal(r.attributes.content, "?string");

    const w = await nodes.write({ type: "Evidence", data: { description: "Falkor docs mention vector indexes", evidence_status: "planned" } }, "create");
    assert.equal(w.outcome, "created");
    assert.equal(w.node_key, "evidence-falkordocsmentionvectorindexes");
    const got = await reader.getNode(w.ref_id);
    assert.equal(got?.node_type, "Evidence");
    await assert.rejects(nodes.write({ type: "Evidence", data: { description: "x", bogus: 1 } }, "create"), (e: any) => e.code === "UNKNOWN_ATTRIBUTE");
    await assert.rejects(nodes.write({ type: "Evidence", data: { content: "no description" } }, "create"), (e: any) => e.code === "MISSING_REQUIRED");

    const names = await schemaObjectNames(bolt);
    assert.ok(names.constraints.includes("unique_evidence_node_key"), names.constraints.join(","));
    const chain = await bolt.run(`MATCH (s:Schema {type: "Evidence"})-[:CHILD_OF]->(p:Schema) RETURN p.type AS p, s.domain AS d, s.index AS i`);
    assert.equal(chain[0]!["p"], "Thing");
    assert.equal(chain[0]!["d"], "entity");
    assert.deepEqual(chain[0]!["i"], ["description"]);
    // The ontology read surface sees it.
    const listed = await reader.listSchemas();
    assert.ok(listed.schemas.some((s) => s.type === "Evidence"));
  });

  it("an edge schema between a new type and a seeded one lets the edge write through", async () => {
    const claim = await nodes.write({ type: "Claim", data: { name: "c", claim_text: "Falkor supports vector search", speaker_name: "anon" } }, "create");
    const ev = await nodes.write({ type: "Evidence", data: { description: "release notes" } }, "create");
    await assert.rejects(edges.write({ edge: "EVIDENCED_BY", source_ref_id: claim.ref_id, target_ref_id: ev.ref_id }), (e: any) => e.code === "WRONG_TYPE");
    await resolver.createEdgeSchema("Claim", "EVIDENCED_BY", "Evidence");
    const e = await edges.write({ edge: "EVIDENCED_BY", source_ref_id: claim.ref_id, target_ref_id: ev.ref_id, properties: { strength: 0.6 } });
    assert.equal(e.created, true);
  });

  it("extends an existing jarvis type add-only, with cache invalidation, and refuses Vein types", async () => {
    // Claim (from the seeded ontology) has no verdict.
    await assert.rejects(nodes.write({ type: "Claim", data: { name: "c2", claim_text: "t", speaker_name: "s", verdict: "unknown" } }, "create"), (e: any) => e.code === "UNKNOWN_ATTRIBUTE");
    const before = await bolt.run(`MATCH (s:Schema {type: "Claim"}) RETURN s.ref_id AS r, s.node_key AS k, s.claim_text AS ct`);
    const r = await createNodeSchema(bolt, resolver, {
      type: "claim", // case-insensitive, adopts live casing
      parent: "Content",
      attributes: { verdict: "?string", confidence_score: "?float", claim_text: "?string" /* existing: left alone */ },
      node_key: "name", // ignored on extend
    });
    assert.equal(r.created, false);
    assert.equal(r.type, "Claim");
    assert.equal(r.ref_id, before[0]!["r"]);
    assert.deepEqual(r.added, ["confidence_score", "verdict"]);
    assert.equal(r.node_key, before[0]!["k"], "identity untouched");
    assert.equal(r.attributes.claim_text, "string", "existing attribute not downgraded to optional");
    assert.equal(r.attributes.verdict, "?string");
    const w = await nodes.write({ type: "Claim", data: { name: "c2", claim_text: "t", speaker_name: "s", verdict: "unknown", confidence_score: 0.5 } }, "create");
    assert.equal(w.outcome, "created");
    // Nothing to add → still not created, empty added.
    const again = await createNodeSchema(bolt, resolver, { type: "Claim", attributes: { verdict: "?string" } });
    assert.deepEqual([again.created, again.added], [false, []]);

    await assert.rejects(createNodeSchema(bolt, resolver, { type: "VeinRun", attributes: { x: "string" } }), (e: any) => e.code === "UNKNOWN_TYPE");
    await assert.rejects(createNodeSchema(bolt, resolver, { type: "SubRun", parent: "VeinRun", attributes: {} }), (e: any) => e.code === "UNKNOWN_TYPE" && e.attribute === "parent");
    await assert.rejects(createNodeSchema(bolt, resolver, { type: "Orphan", parent: "NoSuchParent", attributes: {} }), (e: any) => e.code === "UNKNOWN_TYPE" && e.attribute === "parent");
  });

  it("replaces a soft-deleted schema", async () => {
    await createNodeSchema(bolt, resolver, { type: "Draft", attributes: { body: "?string" } });
    await bolt.run(`MATCH (s:Schema {type: "Draft"}) SET s.is_deleted = true`);
    resolver.invalidate();
    const r = await createNodeSchema(bolt, resolver, { type: "Draft", attributes: { text: "string" }, node_key: "text" });
    assert.equal(r.created, true);
    assert.equal(r.node_key, "draft-text");
    const rows = await bolt.run(`MATCH (s:Schema {type: "Draft"}) RETURN count(s) AS c, collect(s.is_deleted) AS d`);
    assert.equal(Number(rows[0]!["c"]), 1);
  });
});
