import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { Bolt } from "./bolt.js";
import { seedVeinDomain } from "./schema-seed.js";
import { NodeWriter, type Embedder } from "./node-writer.js";
import { EdgeWriter } from "./edge-writer.js";
import {
  GraphReadError,
  GraphReader,
  applyTitleBoost,
  applyUsageTiebreak,
  buildFulltextQuery,
  escapeLucene,
  fetchLimit,
  fuseHits,
  inheritedAttributes,
  pyRound,
  rankHits,
  serializeNode,
  splitSchema,
  titleMatchMultiplier,
  type Hit,
} from "./search.js";
import { EMBEDDING_DIM } from "./embeddings.js";
import { testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();

/** Deterministic bag-of-words embedder: each lowercase word hashes to a
 *  dimension; cosine = word overlap. Good enough to make vector ranking
 *  predictable without the model. */
const bow: Embedder = {
  async embed(texts) {
    return texts.map((t) => {
      const v = new Array<number>(EMBEDDING_DIM).fill(0);
      for (const w of t.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)) {
        const h = createHash("sha1").update(w).digest();
        v[h.readUInt16BE(0) % EMBEDDING_DIM] += 1;
      }
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      return v.map((x) => x / n);
    });
  },
};

const hit = (ref_id: string, raw_score: number, extra: Record<string, unknown> = {}): Hit => ({
  node: { labels: ["VeinWorkflow", "Node", "Data_Bank", "Domain_vein"], properties: { ref_id, ...extra } },
  raw_score,
});

describe("search helpers (pure)", () => {
  it("escapes Lucene specials and builds required-term queries", () => {
    assert.equal(escapeLucene("a+b:c"), "a\\+b\\:c");
    assert.equal(buildFulltextQuery("DomainSrchEpisode_74aced4f"), "DomainSrchEpisode_74aced4f");
    assert.equal(buildFulltextQuery("  harvey deliver "), "+harvey +deliver");
    assert.equal(buildFulltextQuery(""), "");
  });
  it("fetch limit mirrors jarvis", () => {
    assert.equal(fetchLimit(10, 0), 120);
    assert.equal(fetchLimit(50, 0), 250);
    assert.equal(fetchLimit(2000, 0), 5000);
    assert.equal(fetchLimit(undefined, undefined), 120);
  });
  it("pyRound is half-to-even", () => {
    assert.equal(pyRound(0.5), 0);
    assert.equal(pyRound(1.5), 2);
    assert.equal(pyRound(2.5), 2);
    assert.equal(pyRound(0.123456789, 6), 0.123457);
  });
  it("rankHits keeps the best score per ref_id, descending", () => {
    const r = rankHits([hit("a", 0.2), hit("b", 0.9), hit("a", 0.7)]);
    assert.deepEqual(r.map((h) => [h.node.properties["ref_id"], h.raw_score]), [["b", 0.9], ["a", 0.7]]);
  });
  it("fuseHits: weighted RRF, hybrid beats single-source, scores normalized", () => {
    const ranked = fuseHits([hit("a", 3), hit("b", 2)], [hit("b", 0.9), hit("c", 0.8)], { input: [hit("c", 0.7)] });
    // b: 1.15/61 + 1.0/61 = 0.03525; c: 1.0/62 + 1.2/61 = 0.0358; a: 1.15/61 = 0.01885
    assert.deepEqual(ranked.map((e) => e.node.properties["ref_id"]), ["c", "b", "a"]);
    assert.equal(ranked[0]!.extra.score, 1);
    assert.equal(ranked[0]!.extra.match_type, "hybrid");
    assert.equal(ranked[2]!.extra.match_type, "fulltext");
    assert.ok(Math.abs(ranked[1]!.score - (1.15 / 62 + 1.0 / 61)) < 1e-12, 'b: fulltext rank 2, semantic rank 1');
    assert.equal(ranked[1]!.best_rank, 1);
  });
  it("title boost tiers and re-normalization", () => {
    assert.equal(titleMatchMultiplier("req", "req"), 4);
    assert.equal(titleMatchMultiplier("req", "request"), 2.5);
    assert.equal(titleMatchMultiplier("req", "a request"), 2);
    assert.equal(titleMatchMultiplier("req", "other"), 1);
    const ranked = fuseHits([hit("a", 3, { name: "other" }), hit("b", 2, { name: "Harvey" })], []);
    const boosted = applyTitleBoost(ranked, "harvey", () => "name");
    assert.deepEqual(boosted.map((e) => e.node.properties["ref_id"]), ["b", "a"]);
    assert.equal(boosted[0]!.extra.score, 1);
  });
  it("usage tiebreak reorders only within an epsilon bucket", () => {
    const ranked = fuseHits([hit("a", 3, { usage_count_30d: 1 }), hit("b", 2, { usage_count_30d: 9 })], [hit("a", 1), hit("b", 0.9)]);
    // a and b have identical fused scores (rank 1 & 2 in both → symmetric)? No: a is rank 1 twice, b rank 2 twice.
    assert.equal(ranked[0]!.node.properties["ref_id"], "a");
    const tied = ranked.map((e) => ({ ...e, extra: { ...e.extra, score: 0.995 } }));
    assert.deepEqual(applyUsageTiebreak(tied).map((e) => e.node.properties["ref_id"]), ["b", "a"]);
  });
  it("serializeNode strips generic props and keeps the envelope", () => {
    const env = serializeNode(
      { labels: ["Data_Bank", "VeinRun", "Node", "Domain_vein"], properties: { ref_id: "r", node_key: "k", namespace: "default", Data_Bank: "x", text_embeddings: [1], date_added_to_graph: 5, weight: 2, run_id: "1", image_url: 3 } },
      { score: 1 },
    );
    assert.deepEqual(env, { ref_id: "r", node_type: "VeinRun", date_added_to_graph: 5, weight: 2, properties: { run_id: "1", image_url: "" }, score: 1 });
  });
  it("schema split + inherited attributes", () => {
    const thing = splitSchema({ type: "Thing", ref_id: "t", index: ["name"], name: "string", description: "?string" });
    assert.deepEqual(thing.attributes, { name: "string" }, "description is a core key");
    const run = splitSchema({ type: "VeinRun", parent: "Thing", domain: "Vein", ref_id: "r", run_id: "string", name: "string", description: "?string" });
    const [, runOut] = inheritedAttributes([thing, run]);
    assert.deepEqual(runOut!.attributes, { run_id: "string" });
    // `description` is a jarvis SCHEMA_CORE_PROPERTY, so Thing's
    // `description: "?string"` never surfaces as an attribute (jarvis quirk,
    // mirrored). It is still accepted on node writes.
    assert.deepEqual(runOut!.inherited_attributes, { name: "string" });
    assert.equal(runOut!["description"], "?string");
  });
});

describe("GraphReader (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  let reader: GraphReader;
  const ids: Record<string, string> = {};
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
    await wipeGraph(bolt);
    await seedVeinDomain(bolt);
    const nodes = new NodeWriter(bolt, { embedder: bow });
    const edges = new EdgeWriter(bolt);
    const rs = await nodes.writeMany([
      { type: "VeinWorkflow", data: { name: "harvey-deliver", description: "Deliver legal memos to the client portal", usage_count_30d: 5 } },
      { type: "VeinWorkflow", data: { name: "harvey", description: "Legal research assistant" } },
      { type: "VeinWorkflow", data: { name: "gaia-eval", description: "Benchmark runs against GAIA questions", usage_count_30d: 50 } },
      { type: "VeinWorkflow", data: { name: "hidden-one", description: "Deliver memos" } },
      { type: "VeinRun", data: { run_id: "r1", workflow_name: "harvey-deliver", status: "success", summary: "Delivered 60 of 60 memos", started_at: 1 } },
      { type: "VeinStep", data: { step_type: "video/transcribe", description: "Transcribe audio", input_schema: "{ video_url: string }", output_schema: "{ transcript: string, words: [] }" } },
      { type: "VeinStep", data: { step_type: "email/send", description: "Send an email", input_schema: "{ to: string, body: string }", output_schema: "{ message_id: string }" } },
      { type: "VeinWorkflowVersion", data: { name: "harvey-deliver", content_hash: "c-1", created_at: 1, input_schema: "{ matterId: string }" } },
      { type: "VeinAgentSession", data: { run_id: "r1", path: "harvey-deliver/agent", prompt_preview: "Draft the memo" } },
    ]);
    ["wf1", "wf2", "wf3", "wfHidden", "run", "stepVideo", "stepEmail", "wfv", "session"].forEach((k, i) => (ids[k] = rs[i]!.ref_id));
    await nodes.softDelete(ids["wfHidden"]!);
    await edges.writeMany([
      { edge: "EXECUTED", source_ref_id: ids["run"]!, target_ref_id: ids["wfv"]! },
      { edge: "VERSION_OF", source_ref_id: ids["wfv"]!, target_ref_id: ids["wf1"]! },
      { edge: "ACTIVE_VERSION", source_ref_id: ids["wf1"]!, target_ref_id: ids["wfv"]! },
      { edge: "USES_STEP", source_ref_id: ids["wfv"]!, target_ref_id: ids["stepEmail"]!, properties: { importance: 0.9 } },
      { edge: "IN_RUN", source_ref_id: ids["session"]!, target_ref_id: ids["run"]! },
    ]);
    reader = new GraphReader(bolt, { embedder: bow });
    // Fulltext/vector indexes populate asynchronously; wait for them.
    await bolt.run(`CALL db.awaitIndexes(60)`);
  });
  after(async () => {
    await bolt?.close();
  });

  it("getNode: jarvis envelope, hidden for soft-deleted", async () => {
    const n = await reader.getNode(ids["wf1"]!);
    assert.ok(n);
    assert.equal(n.node_type, "VeinWorkflow");
    assert.equal(n.name, "harvey-deliver");
    assert.equal(n.ref_id, ids["wf1"]);
    assert.equal(n.properties["description"], "Deliver legal memos to the client portal");
    for (const k of ["Data_Bank", "node_key", "namespace", "text_embeddings", "ref_id", "date_added_to_graph"]) assert.ok(!(k in n.properties), k);
    assert.equal(await reader.getNode(ids["wfHidden"]!), null);
    assert.equal(await reader.getNode("nope"), null);
  });

  it("connectionCounts + edgeCounts", async () => {
    assert.deepEqual(await reader.connectionCounts(ids["wfv"]!), [
      { edge_type: "ACTIVE_VERSION", target_type: "VeinWorkflow", count: 1 },
      { edge_type: "EXECUTED", target_type: "VeinRun", count: 1 },
      { edge_type: "USES_STEP", target_type: "VeinStep", count: 1 },
      { edge_type: "VERSION_OF", target_type: "VeinWorkflow", count: 1 },
    ]);
    assert.deepEqual(await reader.edgeCounts([ids["run"]!, ids["wf3"]!], "default"), {
      [ids["run"]!]: { EXECUTED: 1, IN_RUN: 1 },
    });
  });

  it("neighbors: importance order, filters, exclusions, limit, counts; source included", async () => {
    const all = await reader.neighbors(ids["wfv"]!, { include_edge_counts: true });
    assert.equal(all.nodes.length, 4);
    assert.ok(all.nodes.some((n) => n.ref_id === ids["wfv"]));
    assert.equal(all.edges.length, 4);
    const uses = all.edges.find((e) => e.edge_type === "USES_STEP")!;
    assert.equal(uses.source, ids["wfv"]);
    assert.equal(uses.target, ids["stepEmail"]);
    assert.deepEqual(uses.properties, { importance: 0.9, date_added_to_graph: uses.properties["date_added_to_graph"] });
    assert.equal(uses.weight, 1);
    const step = all.nodes.find((n) => n.ref_id === ids["stepEmail"])!;
    assert.deepEqual(step.edges, { USES_STEP: 1 });
    assert.equal(step.properties["step_type"], "email/send");

    const limited = await reader.neighbors(ids["wfv"]!, { limit: 1 });
    assert.equal(limited.edges.length, 1);
    assert.equal(limited.edges[0]!.edge_type, "USES_STEP", "importance-sorted before LIMIT");

    const filtered = await reader.neighbors(ids["wfv"]!, { edge_types: ["VERSION_OF", "EXECUTED"], node_types: ["VeinRun"] });
    assert.deepEqual(filtered.edges.map((e) => e.edge_type), ["EXECUTED"]);
    const excluded = await reader.neighbors(ids["wfv"]!, { exclude_node_types: ["veinrun", "VEINSTEP"] });
    assert.deepEqual(excluded.edges.map((e) => e.edge_type).sort(), ["ACTIVE_VERSION", "VERSION_OF"]);
    assert.deepEqual(await reader.neighbors("nope"), { nodes: [], edges: [] });
  });

  it("search: fulltext + semantic fusion, title boost, envelope, edge counts, soft-delete exclusion", async () => {
    const r = await reader.search({ q: "harvey", include_edge_counts: true });
    assert.ok(r.total >= 2);
    assert.equal(r.nodes[0]!.ref_id, ids["wf2"], "exact title match boosted to the top");
    assert.equal(r.nodes[0]!.score, 1);
    assert.equal(r.nodes[0]!.node_type, "VeinWorkflow");
    assert.ok(["fulltext", "semantic", "hybrid"].includes(r.nodes[0]!.match_type!));
    assert.ok(!("Data_Bank" in r.nodes[0]!.properties));
    assert.ok(r.nodes.every((n) => n.ref_id !== ids["wfHidden"]), "soft-deleted excluded");
    const wf1 = r.nodes.find((n) => n.ref_id === ids["wf1"])!;
    assert.deepEqual(wf1.edges, { ACTIVE_VERSION: 1, VERSION_OF: 1 });
    assert.equal(r.truncated, false);
  });

  it("search: multi-word required terms, fuzzy fallback, type filter, paging", async () => {
    const multi = await reader.search({ q: "legal memos" });
    assert.deepEqual(multi.nodes.map((n) => n.ref_id).slice(0, 1), [ids["wf1"]]);
    const fuzzy = await reader.search({ q: "benchmrk" });
    assert.ok(fuzzy.nodes.some((n) => n.ref_id === ids["wf3"]), "fuzzy ~ fallback finds gaia-eval");
    const typed = await reader.search({ q: "harvey", types: ["VeinRun"] });
    assert.deepEqual(typed.nodes.map((n) => n.node_type), ["VeinRun"]);
    const page1 = await reader.search({ q: "harvey", limit: 1 });
    const page2 = await reader.search({ q: "harvey", limit: 1, skip: 1 });
    assert.equal(page1.nodes.length, 1);
    assert.equal(page2.nodes.length, 1);
    assert.notEqual(page1.nodes[0]!.ref_id, page2.nodes[0]!.ref_id);
    assert.equal(page1.total, page2.total);
  });

  it("search: input_q/output_q hit the per-stem vector indexes (k=50, floor 0.4, weight 1.2)", async () => {
    const byInput = await reader.search({ input_q: "a video url" });
    assert.equal(byInput.nodes[0]!.ref_id, ids["stepVideo"]);
    assert.equal(byInput.nodes[0]!.match_type, "input");
    assert.ok(byInput.nodes.every((n) => ["VeinStep", "VeinWorkflowVersion"].includes(n.node_type!)));
    const byOutput = await reader.search({ output_q: "message id" });
    assert.equal(byOutput.nodes[0]!.ref_id, ids["stepEmail"]);
    const combined = await reader.search({ q: "transcribe", input_q: "video" });
    assert.equal(combined.nodes[0]!.ref_id, ids["stepVideo"]);
    assert.equal(combined.nodes[0]!.match_type, "hybrid");
    assert.deepEqual(await reader.search({}), { nodes: [], total: 0, truncated: false });
  });

  it("search: domains validated against the registry; namespaces must be registered", async () => {
    const vein = await reader.search({ q: "harvey", domains: ["vein"] });
    assert.ok(vein.nodes.length >= 2);
    await assert.rejects(reader.search({ q: "harvey", domains: ["legal"] }), (e: unknown) => e instanceof GraphReadError && e.code === "INVALID_DOMAIN");
    await assert.rejects(reader.search({ q: "harvey", namespace: "tenant-x" }), (e: unknown) => e instanceof GraphReadError && e.code === "INVALID_NAMESPACE");
    assert.deepEqual(await reader.registerNamespace("Tenant-X"), { namespace: "tenant-x", created: true });
    assert.deepEqual(await reader.registerNamespace("tenant-x"), { namespace: "tenant-x", created: false });
    assert.deepEqual(await reader.listNamespaces(), ["tenant-x"]);
    const ns = await bolt.run(`MATCH (n:NameSpace) RETURN count(n) AS c, collect(n.data)[0] AS data`);
    assert.deepEqual(ns[0], { c: 1, data: ["tenant-x"] });
    const empty = await reader.search({ q: "harvey", namespace: "tenant-x" });
    assert.equal(empty.total, 0);
  });

  it("ontology: listSchemas and getSchema mirror jarvis's shapes", async () => {
    const { schemas, edges } = await reader.listSchemas();
    assert.equal(schemas.length, 10);
    const run = schemas.find((s) => s.type === "VeinRun")!;
    assert.equal(run["domain"], "Vein");
    assert.equal(run["parent"], "Thing");
    assert.deepEqual(run["index"], ["workflow_name", "status", "summary"]);
    assert.equal(run.attributes["run_id"], "string");
    assert.ok(!("name" in run.attributes), "parent attrs moved out");
    assert.equal(run.inherited_attributes["name"], "string");
    assert.equal(run["description"], "?string", "description is a core key in jarvis, not an attribute");
    assert.equal(run["type_description"], "One vein workflow run — status, timings, params, and a pointer to its log");
    assert.ok(edges.some((e) => e.edge_type === "CHILD_OF" && e.source_type === "VeinRun" && e.target_type === "Thing"));
    assert.ok(edges.some((e) => e.edge_type === "IN_RUN" && e.source_type === "VeinAgentSession" && e.target_type === "VeinRun"));
    assert.ok(edges.some((e) => e.edge_type === "ACCESSED" && e.target_type === "Thing"));

    const only = await reader.listSchemas({ domains: ["VEIN"] });
    assert.equal(only.schemas.length, 9);
    assert.ok(only.edges.every((e) => e.source_type !== "Thing" && e.target_type !== "Thing"), "Thing is outside the domain and not a wildcard");
    assert.ok(!only.edges.some((e) => e.edge_type === "ACCESSED" || e.edge_type === "CHILD_OF"));
    assert.ok(only.edges.some((e) => e.edge_type === "IN_RUN"));

    const single = await reader.getSchema("veinrun");
    assert.ok(single);
    assert.equal(single.type, "VeinRun");
    assert.equal(single.attributes["run_id"], "string");
    assert.equal(single.attributes["name"], "string", "single form keeps inherited in attributes");
    assert.deepEqual(Object.keys(single.inherited_attributes).sort(), ["image_url", "is_muted", "name", "unique_source_id", "weight"]);
    // jarvis's get_schema only special-cases the exact string "Thing"; any
    // other spelling falls into the case-insensitive walk and still resolves.
    assert.equal((await reader.getSchema("thing"))?.type, "Thing");
    assert.equal((await reader.getSchema("Thing"))?.type, "Thing");
    assert.equal(await reader.getSchema("Nope"), null);
  });
});
