import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { defineStep } from "../core.js";
import { createRegistry } from "../steps/registry.js";
import { MemoryRunStore } from "../store.js";
import { WorkspaceManager } from "../workspace.js";
import { buildTools } from "../ai/tools.js";
import { compactValue, findWriteKeyword, readQuery, ReadOnlyViolation } from "./query.js";
import { openGraphBackend, type GraphBackend } from "./backend.js";
import { testGraphConfig, wipeGraph } from "./test-util.js";

// ── Pure (no DB) ───────────────────────────────────────────────────────────

describe("findWriteKeyword", () => {
  it("passes read-only queries", () => {
    for (const q of [
      "MATCH (n:Concept {namespace: $ns}) RETURN count(n)",
      "MATCH (a)-[r:RELATED_TO]->(b) RETURN type(r), count(*) ORDER BY count(*) DESC LIMIT 5",
      "CALL db.labels() YIELD label RETURN label",
      "MATCH (n) WHERE n.data_set = 'x' RETURN n.settings, n.offset",
      "MATCH (n) WHERE n.name = 'please SET this' RETURN n // no CREATE here",
      "MATCH (n) RETURN n.`create` /* MERGE in a comment */",
      "UNWIND $ids AS id MATCH (n {ref_id: id}) RETURN n",
      "SHOW INDEXES YIELD name RETURN name",
    ]) assert.equal(findWriteKeyword(q), undefined, q);
  });

  it("flags every write clause, case-insensitively", () => {
    for (const q of [
      "CREATE (n:X)",
      "merge (n:X {k: 1})",
      "MATCH (n) SET n.a = 1",
      "MATCH (n) DETACH DELETE n",
      "MATCH (n) DELETE n",
      "MATCH (n) REMOVE n.a",
      "DROP INDEX foo",
      "MATCH (n) FOREACH (x IN [1] | SET n.a = x)",
      "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
      "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {})",
      "CALL apoc.meta.stats()",
      "CALL db.index.fulltext.createNodeIndex('x', ['A'], ['b'])",
      "CALL { MATCH (n) SET n.a = 1 } RETURN 1",
      "ALTER DATABASE neo4j SET ACCESS READ ONLY",
    ]) assert.notEqual(findWriteKeyword(q), undefined, q);
    // The reported keyword names the offender.
    assert.equal(findWriteKeyword("MATCH (n) SET n.a = 1"), "SET");
    assert.equal(findWriteKeyword("CALL apoc.meta.stats()"), "apoc.*");
  });
});

describe("compactValue", () => {
  it("truncates long strings and collapses vectors, recursively", () => {
    const long = "x".repeat(600);
    const vec = Array.from({ length: 384 }, (_, i) => i / 384);
    const out = compactValue({ s: long, e: vec, short: [1, 2, 3], nested: { e: vec, t: "ok" }, n: 5, nil: null }) as any;
    assert.ok(out.s.startsWith("x".repeat(500)) && out.s.endsWith("[+100 chars]"));
    assert.equal(out.e, "[vector: 384 numbers]");
    assert.deepEqual(out.short, [1, 2, 3]);
    assert.equal(out.nested.e, "[vector: 384 numbers]");
    assert.equal(out.nested.t, "ok");
    assert.equal(out.n, 5);
    assert.equal(out.nil, null);
  });
});

describe("graph_query tool gating", () => {
  async function deps(graph?: GraphBackend) {
    const s = defineStep({ type: "noop", input: z.object({}), output: z.any(), async run() { return 1; } });
    const registry = await createRegistry([s]);
    return { workspace: new WorkspaceManager("/nonexistent-graph-query-test"), registry, store: new MemoryRunStore(), getRegistry: async () => registry, graph };
  }
  it("is absent without a backend and present with one", async () => {
    assert.ok(!("graph_query" in buildTools(await deps())));
    const fake = { cfg: { namespace: "ns-x" } } as unknown as GraphBackend;
    const tools = buildTools(await deps(fake)) as Record<string, { description?: string }>;
    assert.ok("graph_query" in tools);
    assert.match(tools["graph_query"]!.description ?? "", /"ns-x"/);
  });
});

// ── Live (opt-in: VEIN_TEST_NEO4J_URI) ─────────────────────────────────────

const cfg = testGraphConfig();
describe("readQuery (live)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let backend: GraphBackend;
  before(async () => {
    backend = await openGraphBackend(cfg!, { embeddings: false, skipBoot: true });
    await wipeGraph(backend.bolt);
    await backend.bolt.run(
      `UNWIND range(1, 250) AS i
       CREATE (:Widget:Node:Data_Bank {ref_id: 'w' + i, node_key: 'w' + i, namespace: 'default',
               name: 'widget ' + i, blob: $blob, vec: $vec})`,
      { blob: "b".repeat(2000), vec: Array.from({ length: 64 }, (_, i) => i) },
    );
  });
  after(async () => {
    await wipeGraph(backend.bolt);
    await backend.close();
  });

  it("returns columns + rows with params", async () => {
    const r = await readQuery(backend, "MATCH (n:Widget {namespace: $ns}) RETURN count(n) AS n", { params: { ns: "default" } });
    assert.deepEqual(r.columns, ["n"]);
    assert.deepEqual(r.rows, [{ n: 250 }]);
    assert.equal(r.truncated, false);
    assert.equal(r.rowCount, 1);
  });

  it("caps rows (streaming) and compacts values", async () => {
    const r = await readQuery(backend, "MATCH (n:Widget) RETURN n ORDER BY n.name", { maxRows: 10 });
    assert.equal(r.rowCount, 10);
    assert.equal(r.rows.length, 10);
    assert.equal(r.truncated, true);
    const node = r.rows[0]!["n"] as { labels: string[]; properties: Record<string, unknown> };
    assert.ok(node.labels.includes("Widget"));
    assert.equal(node.properties["vec"], "[vector: 64 numbers]");
    assert.match(String(node.properties["blob"]), /\[\+1500 chars\]$/);
  });

  it("default cap is 100", async () => {
    const r = await readQuery(backend, "MATCH (n:Widget) RETURN n.ref_id");
    assert.equal(r.rowCount, 100);
    assert.equal(r.truncated, true);
  });

  it("rejects writes at the pre-check", async () => {
    await assert.rejects(readQuery(backend, "MATCH (n:Widget) SET n.x = 1"), ReadOnlyViolation);
    const n = await readQuery(backend, "MATCH (n:Widget) WHERE n.x = 1 RETURN count(n) AS n");
    assert.deepEqual(n.rows, [{ n: 0 }]);
  });

  it("the READ transaction rejects a write that slips past the pre-check", async () => {
    // Bypass the keyword guard by monkeying with the exported check: run the
    // same session/tx path the tool uses with a write statement.
    const session = backend.bolt.session("READ");
    try {
      await assert.rejects(
        session.executeRead((tx) => tx.run("MATCH (n:Widget) SET n.x = 1 RETURN n")),
        /read access mode|read-only|Writing in read access mode/i,
      );
    } finally {
      await session.close();
    }
    const n = await readQuery(backend, "MATCH (n:Widget) WHERE n.x = 1 RETURN count(n) AS n");
    assert.deepEqual(n.rows, [{ n: 0 }]);
  });

  it("surfaces syntax errors from the driver", async () => {
    await assert.rejects(readQuery(backend, "MATCH (n RETURN n"), /Invalid input|SyntaxError/);
  });

  it("empty result still reports columns", async () => {
    const r = await readQuery(backend, "MATCH (n:Nope) RETURN n.a AS a, n.b AS b");
    assert.deepEqual(r.columns, ["a", "b"]);
    assert.equal(r.rowCount, 0);
  });
});
