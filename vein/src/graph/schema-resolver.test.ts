/**
 * Live tests for jarvis-typed data: the bundled ontology seed, the live
 * schema resolver, and node/edge writes of jarvis types (Document, the
 * EvalSet chain the harvey pipeline persists, …) validated the way jarvis
 * validates them — from the `:Schema` meta-graph.
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { Bolt } from "./bolt.js";
import { seedVeinDomain } from "./schema-seed.js";
import { seedJarvisOntology, searchableAttributesOf } from "./ontology-seed.js";
import { JARVIS_ONTOLOGY } from "./fixtures/jarvis-ontology.js";
import { EDGE_TYPES_ALLOWLIST, SchemaResolver } from "./schema-resolver.js";
import { GraphValidationError, NodeWriter, buildSearchText } from "./node-writer.js";
import { EdgeWriter } from "./edge-writer.js";
import { GraphReader } from "./search.js";
import { graphSnapshot, schemaObjectNames, testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();

describe("ontology fixture (pure)", () => {
  it("is the jarvis default library with its wildcard sentinel", () => {
    assert.equal(JARVIS_ONTOLOGY.schemas.length, 151);
    assert.ok(JARVIS_ONTOLOGY.schemas.some((s) => s["type"] === "*"));
    for (const t of ["Thing", "Document", "Concept", "EvalSet", "EvalRequirement", "EvalTrigger", "EvalTriggerOutput", "CriterionResult", "ScratchpadEntry"]) {
      assert.ok(JARVIS_ONTOLOGY.schemas.some((s) => s["type"] === t), t);
    }
    assert.ok(JARVIS_ONTOLOGY.edge_schemas.some((e) => e.source === "EvalSet" && e.edge === "HAS_TRIGGER" && e.target === "EvalTrigger"));
    assert.deepEqual(JARVIS_ONTOLOGY.hidden_domains, ["Scratchpad"]);
  });
  it("searchable attributes follow jarvis's tier-1 / string-scrape rules", () => {
    const doc = JARVIS_ONTOLOGY.schemas.find((s) => s["type"] === "Document")!;
    assert.deepEqual([...searchableAttributesOf(doc)].sort(), ["source_link", "summary", "title"]);
    const noIndex = { type: "X", node_key: "x-name", name: "string", note: "?string", count: "?int", title_key: "name" };
    assert.deepEqual([...searchableAttributesOf(noIndex)].sort(), ["name", "node_key", "note"]);
  });
});

describe("jarvis ontology + resolver + jarvis-typed writes (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  let resolver: SchemaResolver;
  let nodes: NodeWriter;
  let edges: EdgeWriter;
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
    await wipeGraph(bolt);
    const r = await seedJarvisOntology(bolt);
    assert.equal(r.createdSchemas.length, 151);
    assert.ok(r.createdEdgeSchemas >= 300, `edge schemas ${r.createdEdgeSchemas}`);
    assert.ok(r.domains.includes("content") && r.domains.includes("legal"));
    await seedVeinDomain(bolt);
    // jarvis's About node hides the Scratchpad domain by default.
    await bolt.run(`CREATE (:About {hidden_domains: ["Scratchpad"]})`);
    resolver = new SchemaResolver(bolt);
    nodes = new NodeWriter(bolt, { resolver });
    edges = new EdgeWriter(bolt, { resolver });
  });
  after(async () => {
    await bolt?.close();
  });

  it("re-seeding is a graph no-op, and a Thing seeded by the ontology is reused by the Vein domain", async () => {
    const snap = await graphSnapshot(bolt);
    const r = await seedJarvisOntology(bolt);
    assert.deepEqual(r.createdSchemas, []);
    assert.equal(r.createdEdgeSchemas, 0);
    const v = await seedVeinDomain(bolt);
    assert.equal(v.mode, "shared");
    assert.deepEqual(await graphSnapshot(bolt), snap);
    const things = await bolt.run(`MATCH (s:Schema {type: "Thing"}) RETURN s.ref_id AS r`);
    assert.equal(things.length, 1);
    assert.equal(things[0]!["r"], JARVIS_ONTOLOGY.schemas.find((s) => s["type"] === "Thing")!["ref_id"]);
    const names = await schemaObjectNames(bolt);
    for (const i of ["data_bank_attribute_index_v2", "domain_content_attribute_index_v2", "domain_legal_vector_index", "domain_vein_vector_index", "text_embeddings_vector_index"]) {
      assert.ok(names.indexes.includes(i), i);
    }
    assert.ok(names.constraints.includes("unique_document_node_key"));
  });

  it("resolves types case-insensitively (labels → Schema.type), Vein types exactly", async () => {
    assert.equal(await resolver.resolveType("evalset"), "EvalSet");
    assert.equal(await resolver.resolveType(" document "), "Document");
    assert.equal(await resolver.resolveType("VeinRun"), "VeinRun");
    assert.equal(await resolver.resolveType("veinrun"), "VeinRun", "falls through to Schema.type like jarvis");
    assert.equal(await resolver.resolveType("Nope"), null);
    assert.equal(await resolver.resolveType("*"), null);
  });

  it("merges CHILD_OF ancestors, exposes index/domain, forces name optional", async () => {
    const doc = (await resolver.schema("document"))!;
    assert.equal(doc.type, "Document");
    assert.equal(doc.parent, "Content");
    assert.equal(doc.node_key, "document-source_link");
    assert.deepEqual(doc.index, ["source_link", "title"]);
    // `source_link` is in jarvis's SCHEMA_KNOWN_PROPERTIES, so its required
    // check skips it (optional here); presence is enforced via node_key.
    assert.equal(doc.attributes["source_link"], "?string");
    assert.equal(doc.attributes["title"], "?string");
    assert.equal(doc.attributes["name"], "?string", "Thing's `name: string` is never required by jarvis");
    assert.equal(doc.attributes["description"], "?string", "core-key description still validates as an attribute");
    assert.equal(doc.attributes["weight"], "?float");
    assert.ok(!("index" in doc.attributes) && !("icon" in doc.attributes));
    assert.deepEqual(doc.domainLabels, ["Domain_content"]);
    assert.equal(doc.isVein, false);
    const eto = (await resolver.schema("EvalTriggerOutput"))!;
    assert.deepEqual(eto.index, ["evaltriggeroutput-id"], "string index kept verbatim (jarvis get_index_fields)");
    const vein = (await resolver.schema("VeinRun"))!;
    assert.equal(vein.isVein, true);
    assert.deepEqual(vein.domainLabels, ["Domain_vein"]);
    assert.equal(await resolver.schema("Nope"), null);
  });

  it("hidden domains drop the Domain_* label (About.hidden_domains), with cache invalidation", async () => {
    const sp = (await resolver.schema("ScratchpadEntry"))!;
    assert.deepEqual(sp.domainLabels, []);
    await bolt.run(`MATCH (a:About) SET a.hidden_domains = []`);
    assert.deepEqual((await resolver.schema("ScratchpadEntry"))!.domainLabels, [], "cached");
    resolver.invalidate();
    assert.deepEqual((await resolver.schema("ScratchpadEntry"))!.domainLabels, ["Domain_scratchpad"]);
    await bolt.run(`MATCH (a:About) SET a.hidden_domains = ["Scratchpad"]`);
    resolver.invalidate();
  });

  it("edge schemas: exact, ancestor walks, wildcard, allowlist, and rejection", async () => {
    assert.equal((await resolver.edgeSchema("HAS_TRIGGER", "EvalSet", "EvalTrigger"))?.via, "exact");
    assert.equal((await resolver.edgeSchema("has_trigger", "evalset", "EVALTRIGGER"))?.via, "exact", "case-insensitive like jarvis");
    // Thing -[HAS_FLUENT]-> Fluent is declared on the root: any type inherits it.
    const anc = await resolver.edgeSchema("HAS_FLUENT", "Document", "Fluent");
    assert.equal(anc?.via, "ancestor");
    assert.equal(anc?.source, "Document");
    assert.ok(EDGE_TYPES_ALLOWLIST.has("RELATED_TO"));
    assert.equal((await resolver.edgeSchema("RELATED_TO", "Document", "EvalSet"))?.via, "allowlist");
    assert.equal(await resolver.edgeSchema("HAS_CRITERION_RESULT", "EvalSet", "CriterionResult"), null);
    assert.equal(await resolver.edgeSchema("MAPS_TO", "Document", "EvalSet"), null);
    const w = await resolver.createEdgeSchema("*", "MAPS_TO", "*");
    assert.equal(w.created, true);
    assert.equal((await resolver.edgeSchema("MAPS_TO", "Document", "EvalSet"))?.via, "wildcard");
    const c = await resolver.createEdgeSchema("EvalSet", "HAS_NOTE", "Document");
    assert.equal(c.created, true);
    assert.equal((await resolver.createEdgeSchema("EvalSet", "has note", "Document")).created, false, "idempotent, normalized");
    assert.equal((await resolver.edgeSchema("HAS_NOTE", "EvalSet", "Document"))?.via, "exact");
    await assert.rejects(resolver.createEdgeSchema("EvalSet", "HAS_X", "Nope"), /does not exist/);
    await assert.rejects(resolver.createEdgeSchema("EvalSet", "bad-type", "Document"), /invalid edge type/);
  });

  let docRef: string;

  it("writes a Document like jarvis: labels, node_key, explicit-index Data_Bank, case-insensitive type", async () => {
    const r = await nodes.write({ type: "document", data: { source_link: "/tmp/matter/Data Room.pdf", title: "Data Room" } });
    assert.equal(r.outcome, "created");
    assert.equal(r.node_type, "Document");
    assert.equal(r.node_key, "document-tmpmatterdataroompdf");
    docRef = r.ref_id;
    const rows = await bolt.run(`MATCH (n:Data_Bank {ref_id: $r}) RETURN labels(n) AS l, properties(n) AS p`, { r: docRef });
    assert.deepEqual([...(rows[0]!["l"] as string[])].sort(), ["Data_Bank", "Document", "Domain_content", "Node"]);
    const p = rows[0]!["p"] as Record<string, unknown>;
    assert.equal(p["Data_Bank"], "/tmp/matter/Data Room.pdf\nData Room");
    assert.deepEqual(p["_search_fields_used"], ["source_link", "title"]);
    assert.equal(p["namespace"], "default");
    const again = await nodes.write({ type: "Document", data: { source_link: "/tmp/matter/Data Room.pdf" } });
    assert.equal(again.outcome, "existing");
    assert.equal(again.ref_id, docRef);
    await assert.rejects(nodes.write({ type: "Document", data: { source_link: "/x", bogus: 1 } }), (e: unknown) => e instanceof GraphValidationError && e.code === "UNKNOWN_ATTRIBUTE");
    await assert.rejects(nodes.write({ type: "Document", data: { title: "no key" } }), (e: unknown) => e instanceof GraphValidationError && e.code === "MISSING_REQUIRED");
    await assert.rejects(nodes.write({ type: "Nope", data: { name: "x" } }), (e: unknown) => e instanceof GraphValidationError && e.code === "UNKNOWN_TYPE");
    // Completion-marker update (the harvey ingest dedupe): status is a Document attr.
    await nodes.update(docRef, { set: { status: "ingested" } });
    const got = await new GraphReader(bolt).getNode(docRef);
    assert.equal(got?.properties["status"], "ingested");
  });

  it("kitchen-sink Data_Bank when the schema's index is unusable; hidden-domain nodes get no Domain label", async () => {
    const eto = (await resolver.schema("EvalTriggerOutput"))!;
    assert.deepEqual(buildSearchText(eto, { id: "output-1", result: "pass", score: 3, n_total: 5, report_url: "http://x" }), {
      text: "pass\n3\n5",
      fields: ["result", "score", "n_total"],
    });
    const r = await nodes.write({ type: "ScratchpadEntry", data: { intended_type: "Widget", entry_hash: "abc", name: "parked" } });
    const rows = await bolt.run(`MATCH (n:Data_Bank {ref_id: $r}) RETURN labels(n) AS l`, { r: r.ref_id });
    assert.deepEqual([...(rows[0]!["l"] as string[])].sort(), ["Data_Bank", "Node", "ScratchpadEntry"]);
  });

  it("persists the harvey eval chain through DB-validated edges (+ unique_source_id stamp)", async () => {
    const ns = "task-slug";
    const [evalset, req, trigger, output, crit] = await nodes.writeMany(
      [
        { type: "EvalSet", data: { id: ns, name: "Task", description: "desc", recursion: true } },
        { type: "EvalRequirement", data: { id: `${ns}-c1`, name: "Crit 1", description: "must", contested: false } },
        { type: "EvalTrigger", data: { id: "trigger-1", agent: "harvey-deliver", environment: "vein-lab", source: "vein", workflow_id: "harvey-deliver", workflow_input: "{}", run_count: 1 } },
        { type: "EvalTriggerOutput", data: { id: "output-1", result: "pass", verdict: "pass", score: 1, max_score: 1, n_total: 1, n_passed: 1 } },
        { type: "CriterionResult", data: { id: "crit-1", criterion_id: "c1", title: "Crit 1", verdict: "pass", reasoning: "ok" } },
      ],
      "create",
      { namespace: ns },
    );
    assert.ok(evalset && req && trigger && output && crit);
    const rs = await edges.writeMany([
      { edge: "HAS_REQUIREMENT", source_ref_id: evalset!.ref_id, target_ref_id: req!.ref_id },
      { edge: "HAS_TRIGGER", source_ref_id: evalset!.ref_id, target_ref_id: trigger!.ref_id },
      { edge: "HAS_OUTPUT", source_ref_id: trigger!.ref_id, target_ref_id: output!.ref_id },
      { edge: "HAS_CRITERION_RESULT", source_ref_id: output!.ref_id, target_ref_id: crit!.ref_id },
      { edge: "HAS_TRIGGER", source_ref_id: req!.ref_id, target_ref_id: trigger!.ref_id },
    ]);
    assert.deepEqual(rs.map((r) => r.created), [true, true, true, true, true]);
    assert.deepEqual(rs.map((r) => r.edge_key), ["has_requirement", "has_trigger", "has_output", "has_criterion_result", "has_trigger"]);
    await assert.rejects(edges.write({ edge: "HAS_OUTPUT", source_ref_id: evalset!.ref_id, target_ref_id: trigger!.ref_id }), (e: unknown) => e instanceof GraphValidationError && e.code === "WRONG_TYPE");
    // Namespace scoping + counts read back like jarvis.
    const reader = new GraphReader(bolt, { resolver });
    assert.deepEqual(await reader.connectionCounts(evalset!.ref_id), [
      { edge_type: "HAS_REQUIREMENT", target_type: "EvalRequirement", count: 1 },
      { edge_type: "HAS_TRIGGER", target_type: "EvalTrigger", count: 1 },
    ]);
    // unique_source_id lands on the edge when both endpoints share it.
    const [a, b] = await nodes.writeMany([
      { type: "Document", data: { source_link: "/u/a", unique_source_id: "batch:9" } },
      { type: "Document", data: { source_link: "/u/b", unique_source_id: "batch:9" } },
    ]);
    const rel = await edges.write({ edge: "RELATED_TO", source_ref_id: a!.ref_id, target_ref_id: b!.ref_id });
    const props = await bolt.run(`MATCH ()-[r {ref_id: $r}]->() RETURN properties(r) AS p`, { r: rel.ref_id });
    assert.equal((props[0]!["p"] as Record<string, unknown>)["unique_source_id"], "batch:9");
  });

  it("search + ontology cover jarvis types (type filter is case-insensitive)", async () => {
    await bolt.run(`CALL db.awaitIndexes(60)`);
    const reader = new GraphReader(bolt, { resolver });
    const r = await reader.search({ q: "Data Room", types: ["document"], domains: ["content"] });
    assert.equal(r.nodes[0]?.ref_id, docRef);
    assert.equal(r.nodes[0]?.node_type, "Document");
    const onto = await reader.listSchemas({ domains: ["codeartifact"] });
    assert.ok(onto.schemas.some((s) => s.type === "EvalSet"));
    assert.ok(onto.edges.some((e) => e.edge_type === "HAS_TRIGGER" && e.source_type === "EvalSet"));
    const single = await reader.getSchema("evaltrigger");
    assert.equal(single?.attributes["id"], "string");
    assert.equal(single?.attributes["agent"], "?string");
  });
});
