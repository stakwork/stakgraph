import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Bolt } from "./bolt.js";
import { seedVeinDomain } from "./schema-seed.js";
import {
  GraphValidationError,
  NodeWriter,
  buildSearchText,
  composeNodeKey,
  renderVectorField,
  toEpochSeconds,
  validateNode,
  type Embedder,
} from "./node-writer.js";
import { getVeinSchema } from "./vein-schemas.js";
import { graphSnapshot, testGraphConfig, wipeGraph } from "./test-util.js";

const cfg = testGraphConfig();
const fixturesDir = new URL("./fixtures/", import.meta.url);

function rejects(fn: () => unknown, code: string, attribute?: string) {
  try {
    fn();
  } catch (e) {
    assert.ok(e instanceof GraphValidationError, `expected GraphValidationError, got ${String(e)}`);
    assert.equal(e.code, code);
    if (attribute) assert.equal(e.attribute, attribute);
    return;
  }
  assert.fail(`expected ${code}`);
}

const RUN = {
  run_id: "1725220000123",
  workflow_name: "harvey-deliver",
  status: "success",
  summary: "Delivered 60 of 60",
  started_at: "2026-09-01T20:00:00Z",
  duration_ms: 1234,
};

describe("validateNode (§6 gate)", () => {
  it("accepts a valid payload and normalizes datetime to epoch seconds", () => {
    const v = validateNode("VeinRun", RUN);
    assert.equal(v.values["started_at"], 1788292800);
    assert.equal(v.values["duration_ms"], 1234);
    assert.equal(v.values["summary"], "Delivered 60 of 60");
    // Neo4j params: ints wrapped as Integer.
    assert.equal(String(v.params["started_at"]), "1788292800");
    assert.equal(typeof v.params["summary"], "string");
  });
  it("rejects unknown types (exact match only)", () => {
    rejects(() => validateNode("Run", RUN), "UNKNOWN_TYPE");
    rejects(() => validateNode("veinrun", RUN), "UNKNOWN_TYPE");
  });
  it("rejects unknown and generic attributes", () => {
    rejects(() => validateNode("VeinRun", { ...RUN, colour: "red" }), "UNKNOWN_ATTRIBUTE", "colour");
    rejects(() => validateNode("VeinRun", { ...RUN, ref_id: "x" }), "UNKNOWN_ATTRIBUTE", "ref_id");
    rejects(() => validateNode("VeinRun", { ...RUN, Data_Bank: "x" }), "UNKNOWN_ATTRIBUTE", "Data_Bank");
    rejects(() => validateNode("VeinRun", { ...RUN, node_key: "x" }), "UNKNOWN_ATTRIBUTE", "node_key");
  });
  it("accepts Thing-inherited attributes", () => {
    const v = validateNode("VeinRun", { ...RUN, unique_source_id: "veinrun:1", is_muted: false, weight: 2 });
    assert.equal(v.values["unique_source_id"], "veinrun:1");
    assert.equal(v.values["weight"], 2);
  });
  it("rejects missing / null / empty required attributes", () => {
    const { status: _s, ...noStatus } = RUN;
    rejects(() => validateNode("VeinRun", noStatus), "MISSING_REQUIRED", "status");
    rejects(() => validateNode("VeinRun", { ...RUN, status: null }), "MISSING_REQUIRED", "status");
    rejects(() => validateNode("VeinRun", { ...RUN, status: "" }), "MISSING_REQUIRED", "status");
  });
  it("drops optional null/undefined/empty values instead of writing them", () => {
    const v = validateNode("VeinRun", { ...RUN, summary: "", log_ref: null, error_message: undefined });
    assert.ok(!("summary" in v.values));
    assert.ok(!("log_ref" in v.values));
    assert.ok(!("error_message" in v.values));
  });
  it("type-checks per the grammar, stricter than Python", () => {
    rejects(() => validateNode("VeinRun", { ...RUN, duration_ms: true }), "WRONG_TYPE", "duration_ms");
    rejects(() => validateNode("VeinRun", { ...RUN, duration_ms: 1.5 }), "WRONG_TYPE", "duration_ms");
    rejects(() => validateNode("VeinRun", { ...RUN, duration_ms: "12" }), "WRONG_TYPE", "duration_ms");
    rejects(() => validateNode("VeinRun", { ...RUN, status: 1 }), "WRONG_TYPE", "status");
    rejects(() => validateNode("VeinRun", { ...RUN, is_muted: "no" }), "WRONG_TYPE", "is_muted");
    rejects(() => validateNode("VeinRun", { ...RUN, started_at: "yesterday" }), "INVALID_DATETIME", "started_at");
    // int-where-float is fine.
    assert.equal(validateNode("VeinRun", { ...RUN, weight: 3 }).values["weight"], 3);
  });
  it("datetime accepts ISO, epoch seconds, epoch ms, numeric strings", () => {
    assert.equal(toEpochSeconds("2026-09-01T20:00:00Z"), 1788292800);
    assert.equal(toEpochSeconds("2026-09-01T20:00:00+00:00"), 1788292800);
    assert.equal(toEpochSeconds(1788292800), 1788292800);
    assert.equal(toEpochSeconds(1788292800123), 1788292800);
    assert.equal(toEpochSeconds("1788292800123"), 1788292800);
    assert.equal(toEpochSeconds(new Date(1788292800000)), 1788292800);
    assert.equal(toEpochSeconds("nope"), null);
    assert.equal(toEpochSeconds({}), null);
  });
  it("rejects node_key attributes that sanitize to nothing", () => {
    rejects(() => validateNode("VeinRun", { ...RUN, run_id: "---" }), "EMPTY_NODE_KEY_TOKEN", "run_id");
  });
});

describe("composeNodeKey (parity with jarvis sanitize_node_key)", () => {
  const cases = JSON.parse(readFileSync(new URL("node-key-parity.json", fixturesDir), "utf8")) as Array<{
    type: string;
    props: Record<string, unknown>;
    expected: string;
  }>;
  for (const c of cases) {
    it(`${c.type} ${JSON.stringify(c.props).slice(0, 60)}`, () => {
      assert.equal(composeNodeKey(getVeinSchema(c.type)!, c.props), c.expected);
    });
  }
  it("property lookup is case-insensitive", () => {
    assert.equal(composeNodeKey(getVeinSchema("VeinRun")!, { RUN_ID: "Abc" }), "veinrun-abc");
  });
});

describe("buildSearchText (Data_Bank)", () => {
  it("joins index fields in declared order with newlines, skipping blanks", () => {
    const s = getVeinSchema("VeinRun")!;
    assert.deepEqual(buildSearchText(s, { summary: " sum ", workflow_name: "wf", status: "  " }), {
      text: "wf\nsum",
      fields: ["workflow_name", "summary"],
    });
    assert.deepEqual(buildSearchText(s, { run_id: "x" }), { text: null, fields: [] });
  });
  it("renders vector fields like jarvis render_schema", () => {
    assert.equal(renderVectorField("input_schema", " {a: 1} "), "Input:\n{a: 1}");
    assert.equal(renderVectorField("output_schema", "x"), "Output:\nx");
    assert.equal(renderVectorField("usage_examples", "x"), "Usage examples:\nx");
    assert.equal(renderVectorField("input_schema", "  "), null);
  });
});

describe("NodeWriter (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let bolt: Bolt;
  let writer: NodeWriter;
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
    writer = new NodeWriter(bolt);
  });

  async function node(ref_id: string) {
    const rows = await bolt.run(
      `MATCH (n:Data_Bank {ref_id: $r})
       RETURN labels(n) AS labels, properties(n) AS props,
              valueType(n.date_added_to_graph) AS dateType, valueType(n.started_at) AS startedType,
              valueType(n.duration_ms) AS durType, valueType(n.weight) AS weightType`,
      { r: ref_id },
    );
    return rows[0]!;
  }

  it("create: jarvis label set, stamps, node_key, Data_Bank, typed ints", async () => {
    const t0 = Date.now();
    const r = await writer.write({ type: "VeinRun", data: { ...RUN, weight: 1, log_ref: "" } });
    assert.equal(r.outcome, "created");
    assert.equal(r.node_key, "veinrun-1725220000123");
    assert.match(r.ref_id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);

    const n = await node(r.ref_id);
    assert.deepEqual([...(n["labels"] as string[])].sort(), ["Data_Bank", "Domain_vein", "Node", "VeinRun"]);
    const p = n["props"] as Record<string, unknown>;
    assert.equal(p["namespace"], "default");
    assert.equal(p["node_key"], "veinrun-1725220000123");
    assert.ok((p["date_added_to_graph"] as number) >= t0 && (p["date_added_to_graph"] as number) <= Date.now() + 1);
    assert.equal(n["dateType"], "INTEGER NOT NULL", "date_added_to_graph is a Neo4j Integer (ms)");
    assert.equal(p["started_at"], 1788292800);
    assert.equal(n["startedType"], "INTEGER NOT NULL", "datetime attrs are Integer seconds");
    assert.equal(n["durType"], "INTEGER NOT NULL");
    assert.equal(n["weightType"], "FLOAT NOT NULL");
    assert.equal(p["Data_Bank"], "harvey-deliver\nsuccess\nDelivered 60 of 60");
    assert.deepEqual(p["_search_fields_used"], ["workflow_name", "status", "summary"]);
    assert.ok(!("log_ref" in p), "empty string never written");
    assert.ok(!("text_embeddings" in p), "no embedder → NULL, not a marker");
    assert.ok(!("is_deleted" in p));
    assert.deepEqual(
      Object.keys(p).sort(),
      ["Data_Bank", "_search_fields_used", "date_added_to_graph", "duration_ms", "namespace", "node_key", "ref_id", "run_id", "started_at", "status", "summary", "weight", "workflow_name"],
    );
  });

  it("create on an existing key is a no-op returning the existing ref_id", async () => {
    const a = await writer.write({ type: "VeinRun", data: RUN });
    const snap = await graphSnapshot(bolt);
    const b = await writer.write({ type: "VeinRun", data: { ...RUN, summary: "CHANGED" } });
    assert.equal(b.outcome, "existing");
    assert.equal(b.ref_id, a.ref_id);
    assert.deepEqual(await graphSnapshot(bolt), snap);
  });

  it("upsert updates everything except the preserved identity and rebuilds Data_Bank", async () => {
    const a = await writer.write({ type: "VeinRun", data: RUN });
    const before = (await node(a.ref_id))["props"] as Record<string, unknown>;
    const b = await writer.write({ type: "VeinRun", data: { ...RUN, status: "error", summary: "boom", error_message: "x" } }, "upsert");
    assert.equal(b.outcome, "updated");
    assert.equal(b.ref_id, a.ref_id);
    const after = (await node(a.ref_id))["props"] as Record<string, unknown>;
    assert.equal(after["date_added_to_graph"], before["date_added_to_graph"]);
    assert.equal(after["node_key"], before["node_key"]);
    assert.equal(after["status"], "error");
    assert.equal(after["error_message"], "x");
    assert.equal(after["Data_Bank"], "harvey-deliver\nerror\nboom");
    assert.ok(!("is_deleted" in after), "upsert on a live node does not add is_deleted");
  });

  it("soft delete + create restores in place; muted+deleted stays put", async () => {
    const a = await writer.write({ type: "VeinRun", data: RUN });
    assert.equal(await writer.softDelete(a.ref_id), true);
    assert.equal(((await node(a.ref_id))["props"] as Record<string, unknown>)["is_deleted"], true);
    const r = await writer.write({ type: "VeinRun", data: { ...RUN, summary: "back" } });
    assert.equal(r.outcome, "restored");
    assert.equal(r.ref_id, a.ref_id);
    const p = (await node(a.ref_id))["props"] as Record<string, unknown>;
    assert.equal(p["is_deleted"], false);
    assert.equal(p["summary"], "back");

    await bolt.run(`MATCH (n:Data_Bank {ref_id: $r}) SET n.is_deleted = true, n.is_muted = true`, { r: a.ref_id });
    const r2 = await writer.write({ type: "VeinRun", data: { ...RUN, summary: "nope" } });
    assert.equal(r2.outcome, "existing");
    const p2 = (await node(a.ref_id))["props"] as Record<string, unknown>;
    assert.equal(p2["is_deleted"], true);
    assert.equal(p2["summary"], "back");
    assert.equal(await writer.softDelete("not-a-ref"), false);
  });

  it("writeMany: mixed types in one tx, input-order results, per-type UNWIND", async () => {
    const rs = await writer.writeMany([
      { type: "VeinChat", data: { chat_id: "c1", title: "Hello", created_at: 1788292800 } },
      { type: "VeinRun", data: RUN },
      { type: "VeinTurn", data: { chat_id: "c1", turn: 0, user_text_preview: "hi" } },
      { type: "VeinRun", data: { ...RUN, run_id: "2" } },
    ]);
    assert.deepEqual(rs.map((r) => r.outcome), ["created", "created", "created", "created"]);
    assert.deepEqual(rs.map((r) => r.node_key), ["veinchat-c1", "veinrun-1725220000123", "veinturn-c1-0", "veinrun-2"]);
    const count = await bolt.run(`MATCH (n:Domain_vein) RETURN count(n) AS c`);
    assert.equal(count[0]!["c"], 4);
    // Turn with no index text → no Data_Bank, no _search_fields_used.
    const turn = (await node(rs[2]!.ref_id))["props"] as Record<string, unknown>;
    assert.equal(turn["Data_Bank"], "hi");
    const rs2 = await writer.writeMany([
      { type: "VeinRun", data: { ...RUN, summary: "again" } },
      { type: "VeinRun", data: { ...RUN, run_id: "3" } },
    ]);
    assert.deepEqual(rs2.map((r) => r.outcome), ["existing", "created"]);
    assert.equal(rs2[0]!.ref_id, rs[1]!.ref_id);
  });

  it("a validation failure anywhere in a batch writes nothing", async () => {
    const snap = await graphSnapshot(bolt);
    await assert.rejects(
      writer.writeMany([
        { type: "VeinRun", data: RUN },
        { type: "VeinRun", data: { ...RUN, run_id: "2", bogus: 1 } },
      ]),
      GraphValidationError,
    );
    assert.deepEqual(await graphSnapshot(bolt), snap);
  });

  it("with an embedder: text_embeddings + per-stem vectors, rebuilt on upsert", async () => {
    const seen: string[] = [];
    const fake: Embedder = {
      async embed(texts) {
        seen.push(...texts);
        return texts.map((t) => Array.from({ length: 384 }, (_, i) => (i === 0 ? t.length : 0)));
      },
    };
    const w = new NodeWriter(bolt, { embedder: fake });
    const r = await w.write({ type: "VeinStep", data: { step_type: "x/y", description: "d", input_schema: "{a}", output_schema: "" } });
    assert.deepEqual(seen, ["x/y\nd", "Input:\n{a}"]);
    const p = (await node(r.ref_id))["props"] as Record<string, unknown>;
    assert.equal((p["text_embeddings"] as number[]).length, 384);
    assert.equal((p["text_embeddings"] as number[])[0], "x/y\nd".length);
    assert.equal((p["input_embeddings"] as number[])[0], "Input:\n{a}".length);
    assert.ok(!("output_embeddings" in p));
    const vt = await bolt.run(`MATCH (n:Data_Bank {ref_id: $r}) RETURN valueType(n.text_embeddings) AS t`, { r: r.ref_id });
    assert.equal(vt[0]!["t"], "LIST<FLOAT NOT NULL> NOT NULL");

    await w.write({ type: "VeinStep", data: { step_type: "x/y", description: "dd", input_schema: "{ab}" } }, "upsert");
    const p2 = (await node(r.ref_id))["props"] as Record<string, unknown>;
    assert.equal((p2["text_embeddings"] as number[])[0], "x/y\ndd".length);
    assert.equal((p2["input_embeddings"] as number[])[0], "Input:\n{ab}".length);
  });

  it("namespace scopes identity", async () => {
    const other = new Bolt({ ...cfg!, namespace: "tenant-b" });
    try {
      const a = await writer.write({ type: "VeinRun", data: RUN });
      const b = await new NodeWriter(other).write({ type: "VeinRun", data: RUN });
      assert.equal(b.outcome, "created");
      assert.notEqual(a.ref_id, b.ref_id);
      const ns = await bolt.run(`MATCH (n:VeinRun) RETURN n.namespace AS ns ORDER BY ns`);
      assert.deepEqual(ns.map((r) => r["ns"]), ["default", "tenant-b"]);
    } finally {
      await other.close();
    }
  });
});
