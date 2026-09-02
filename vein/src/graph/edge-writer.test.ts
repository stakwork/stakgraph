import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { Bolt, int } from "./bolt.js";
import { seedVeinDomain } from "./schema-seed.js";
import { GraphValidationError, NodeWriter } from "./node-writer.js";
import { EdgeWriter, edgeKeyFor, isRegisteredEdge, typeLabelOf } from "./edge-writer.js";
import { graphSnapshot, testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();

describe("edge registry checks (pure)", () => {
  it("resolves the type label from jarvis's label set", () => {
    assert.equal(typeLabelOf(["Data_Bank", "Domain_vein", "Node", "VeinRun"]), "VeinRun");
    assert.equal(typeLabelOf(["Node", "Data_Bank", "Concept", "Domain_general"]), "Concept");
    assert.equal(typeLabelOf(["Node", "Data_Bank"]), undefined);
  });
  it("matches registry rows; ACCESSED accepts any target", () => {
    assert.equal(isRegisteredEdge("IN_RUN", "VeinAgentSession", "VeinRun"), true);
    assert.equal(isRegisteredEdge("IN_RUN", "VeinRun", "VeinAgentSession"), false);
    assert.equal(isRegisteredEdge("VERSION_OF", "VeinStepVersion", "VeinStep"), true);
    assert.equal(isRegisteredEdge("VERSION_OF", "VeinStepVersion", "VeinWorkflow"), false);
    assert.equal(isRegisteredEdge("ACCESSED", "VeinToolCall", "Concept"), true);
    assert.equal(isRegisteredEdge("ACCESSED", "VeinToolCall", ""), true);
    assert.equal(isRegisteredEdge("ACCESSED", "VeinRun", "Concept"), false);
    assert.equal(isRegisteredEdge("HAS_TURN", "VeinChat", "VeinTurn"), false);
  });
  it("edge_key is the lowercased type", () => {
    assert.equal(edgeKeyFor("IN_RUN"), "in_run");
  });
});

describe("EdgeWriter (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  let nodes: NodeWriter;
  let edges: EdgeWriter;
  let run: string;
  let session: string;
  let call: string;
  before(async () => {
    bolt = new Bolt(cfg!);
    await bolt.verify();
  });
  after(async () => {
    await bolt?.close();
  });
  beforeEach(async () => {
    await wipeGraph(bolt);
    await seedVeinDomain(bolt);
    nodes = new NodeWriter(bolt);
    edges = new EdgeWriter(bolt);
    const rs = await nodes.writeMany([
      { type: "VeinRun", data: { run_id: "r1", workflow_name: "wf", status: "ok", started_at: 1 } },
      { type: "VeinAgentSession", data: { run_id: "r1", path: "wf/agent" } },
      { type: "VeinToolCall", data: { run_id: "r1", path: "wf/agent", seq: 0, tool_name: "search" } },
    ]);
    [run, session, call] = rs.map((r) => r.ref_id) as [string, string, string];
  });

  async function rel(ref_id: string) {
    const rows = await bolt.run(
      `MATCH (a)-[r {ref_id: $r}]->(b)
       RETURN type(r) AS type, properties(r) AS props, a.ref_id AS src, b.ref_id AS tgt,
              valueType(r.weight) AS weightType, valueType(r.date_added_to_graph) AS dateType`,
      { r: ref_id },
    );
    return rows[0]!;
  }

  it("creates with jarvis stamps, no namespace, and never mutates an existing edge", async () => {
    const t0 = Date.now();
    const a = await edges.write({ edge: "IN_RUN", source_ref_id: session, target_ref_id: run, properties: { note: "x" } });
    assert.equal(a.created, true);
    assert.equal(a.edge_key, "in_run");
    assert.equal(a.source_ref_id, session);
    assert.equal(a.target_ref_id, run);
    const r = await rel(a.ref_id);
    assert.equal(r["type"], "IN_RUN");
    const p = r["props"] as Record<string, unknown>;
    assert.deepEqual(Object.keys(p).sort(), ["date_added_to_graph", "edge_key", "note", "ref_id", "weight"]);
    assert.equal(p["weight"], 1);
    assert.equal(r["weightType"], "INTEGER NOT NULL");
    assert.equal(r["dateType"], "INTEGER NOT NULL");
    assert.ok((p["date_added_to_graph"] as number) >= t0);
    assert.ok(!("namespace" in p));

    const snap = await graphSnapshot(bolt);
    const b = await edges.write({ edge: "IN_RUN", source_ref_id: session, target_ref_id: run, properties: { note: "CHANGED" } });
    assert.equal(b.created, false);
    assert.equal(b.ref_id, a.ref_id);
    assert.deepEqual(await graphSnapshot(bolt), snap);
    const count = await bolt.run(`MATCH (:VeinAgentSession)-[r:IN_RUN]->(:VeinRun) RETURN count(r) AS c`);
    assert.equal(count[0]!["c"], 1);
  });

  it("rejects unknown types, unregistered triples, stamp overrides, and unresolvable endpoints — writing nothing", async () => {
    const snap = await graphSnapshot(bolt);
    const cases: Array<[Parameters<EdgeWriter["write"]>[0], string]> = [
      [{ edge: "HAS_TURN", source_ref_id: session, target_ref_id: run }, "WRONG_TYPE"],
      [{ edge: "in_run", source_ref_id: session, target_ref_id: run }, "UNKNOWN_TYPE"],
      [{ edge: "IN_RUN", source_ref_id: run, target_ref_id: session }, "WRONG_TYPE"],
      [{ edge: "IN_SESSION", source_ref_id: call, target_ref_id: run }, "WRONG_TYPE"],
      [{ edge: "IN_RUN", source_ref_id: session, target_ref_id: run, properties: { ref_id: "x" } }, "UNKNOWN_ATTRIBUTE"],
      [{ edge: "IN_RUN", source_ref_id: session, target_ref_id: run, properties: { "bad-name": 1 } }, "UNKNOWN_ATTRIBUTE"],
      [{ edge: "IN_RUN", source_ref_id: session, target_ref_id: randomUUID() }, "MISSING_REQUIRED"],
      [{ edge: "IN_RUN", source_ref_id: "", target_ref_id: run }, "MISSING_REQUIRED"],
    ];
    for (const [input, code] of cases) {
      await assert.rejects(edges.write(input), (e: unknown) => e instanceof GraphValidationError && e.code === code, `${JSON.stringify(input)} → ${code}`);
    }
    // Batch: one bad row poisons the whole batch.
    await assert.rejects(
      edges.writeMany([
        { edge: "IN_RUN", source_ref_id: session, target_ref_id: run },
        { edge: "IN_SESSION", source_ref_id: call, target_ref_id: run },
      ]),
      GraphValidationError,
    );
    assert.deepEqual(await graphSnapshot(bolt), snap);
  });

  it("ACCESSED may point at any node, including a jarvis-owned one; source must still be a Vein node", async () => {
    // A jarvis-style Concept node: no Vein label, plain Data_Bank ref_id.
    const concept = randomUUID();
    await bolt.run(`CREATE (:Concept:Node:Data_Bank:Domain_general {ref_id: $r, node_key: "concept-x", namespace: "default", name: "x"})`, { r: concept });
    const a = await edges.write({ edge: "ACCESSED", source_ref_id: call, target_ref_id: concept });
    assert.equal(a.created, true);
    assert.equal((await rel(a.ref_id))["tgt"], concept);
    await assert.rejects(edges.write({ edge: "ACCESSED", source_ref_id: concept, target_ref_id: call }), GraphValidationError);
  });

  it("IS_ALIAS rewrite lands the edge on the canonical node", async () => {
    const canonical = (await nodes.write({ type: "VeinRun", data: { run_id: "canon", workflow_name: "wf", status: "ok", started_at: 1 } })).ref_id;
    // Park `run` as an alias of `canonical` (what jarvis node-merge does).
    await bolt.run(`MATCH (a:Data_Bank {ref_id: $a}), (c:Data_Bank {ref_id: $c}) CREATE (a)-[:IS_ALIAS {ref_id: $e}]->(c)`, { a: run, c: canonical, e: randomUUID() });
    const r = await edges.write({ edge: "IN_RUN", source_ref_id: session, target_ref_id: run });
    assert.equal(r.target_ref_id, canonical);
    assert.equal((await rel(r.ref_id))["tgt"], canonical);
    const direct = await bolt.run(`MATCH (:VeinAgentSession)-[r:IN_RUN]->(t:VeinRun {ref_id: $t}) RETURN count(r) AS c`, { t: run });
    assert.equal(direct[0]!["c"], 0);
  });

  it("writeMany: mixed edge types in one tx, input-order results; explicit ints pass through", async () => {
    const rs = await edges.writeMany([
      { edge: "IN_SESSION", source_ref_id: call, target_ref_id: session, properties: { seq: int(3) } },
      { edge: "IN_RUN", source_ref_id: session, target_ref_id: run },
      { edge: "IN_SESSION", source_ref_id: call, target_ref_id: session },
    ]);
    assert.deepEqual(rs.map((r) => [r.edge_key, r.created]), [["in_session", true], ["in_run", true], ["in_session", false]]);
    assert.equal(rs[0]!.ref_id, rs[2]!.ref_id);
    const p = (await rel(rs[0]!.ref_id))["props"] as Record<string, unknown>;
    assert.equal(p["seq"], 3);
    const vt = await bolt.run(`MATCH ()-[r {ref_id: $r}]->() RETURN valueType(r.seq) AS t`, { r: rs[0]!.ref_id });
    assert.equal(vt[0]!["t"], "INTEGER NOT NULL");
  });

  it("caller weight overrides the stamp", async () => {
    const a = await edges.write({ edge: "IN_RUN", source_ref_id: session, target_ref_id: run, weight: 3 });
    const r = await rel(a.ref_id);
    assert.equal((r["props"] as Record<string, unknown>)["weight"], 3);
    assert.equal(r["weightType"], "INTEGER NOT NULL");
    await edges.mute(a.ref_id);
    const b = await edges.write({ edge: "IN_SESSION", source_ref_id: call, target_ref_id: session, weight: 0.5 });
    assert.equal((await rel(b.ref_id))["weightType"], "FLOAT NOT NULL");
  });

  it("mute is the edge soft delete", async () => {
    const a = await edges.write({ edge: "IN_RUN", source_ref_id: session, target_ref_id: run });
    assert.equal(await edges.mute(a.ref_id), true);
    assert.equal(((await rel(a.ref_id))["props"] as Record<string, unknown>)["is_muted"], true);
    assert.equal(await edges.mute("nope"), false);
  });
});
