import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openGraphBackend, type GraphBackend } from "./backend.js";
import { seedVeinDomain } from "./schema-seed.js";
import { testGraphConfig, wipeGraph } from "./test-util.js";
import { Neo4jWorkspaceStore } from "./workspace-store.js";
import { workspaceConformance } from "../test-util/workspace-conformance.js";

/**
 * Live tests (opt-in via VEIN_TEST_NEO4J_URI — see test-util.ts). The
 * graph store passes the same `WorkspaceStore` conformance suite as the
 * file store; the cases below cover what is graph-specific: the edges,
 * soft deletion, and persistence across store instances.
 */

const cfg = testGraphConfig();
let backend: GraphBackend;
let scratch: string;

async function reset() {
  await wipeGraph(backend.bolt);
  await seedVeinDomain(backend.bolt);
  await rm(scratch, { recursive: true, force: true });
}

if (cfg) {
  before(async () => {
    backend = await openGraphBackend(cfg, { embeddings: false, skipBoot: true });
    scratch = await mkdtemp(join(tmpdir(), "vein-graph-ws-"));
  });
  after(async () => {
    await backend.close();
    await rm(scratch, { recursive: true, force: true });
  });
}

workspaceConformance({
  name: "Neo4jWorkspaceStore",
  skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set",
  reset,
  make: () => new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps") }),
});

const STEP = (type: string) => `export default { type: ${JSON.stringify(type)}, input: {}, output: {}, async run() { return 1; } };`;

describe("Neo4jWorkspaceStore (graph-specific)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  let ws: Neo4jWorkspaceStore;
  beforeEach(async () => {
    await reset();
    ws = new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps") });
  });

  const edgesOf = async (edge: string) =>
    backend.bolt.run(`MATCH (a:Data_Bank)-[r:\`${edge}\`]->(b:Data_Bank) RETURN a.node_key AS a, b.node_key AS b ORDER BY a, b`);

  it("writes jarvis-dialect nodes with the Vein labels and VERSION_OF / ACTIVE_VERSION edges", async () => {
    await ws.publishWorkflow("wf", "v1", { steps: [{ id: "a", type: "log", config: { message: "x" } }] }, "first");
    const rows = await backend.bolt.run(
      `MATCH (n:Domain_vein) RETURN labels(n) AS labels, n.node_key AS key, n.namespace AS ns ORDER BY key`,
    );
    assert.deepEqual(
      rows.map((r) => [(r["labels"] as string[]).filter((l) => l.startsWith("Vein"))[0], r["key"], r["ns"]]),
      [
        ["VeinWorkflow", "veinworkflow-wf", cfg!.namespace],
        ["VeinWorkflowVersion", `veinworkflowversion-wf-${await ws.getWorkflowHash("wf")}`, cfg!.namespace],
      ],
    );
    assert.equal((await edgesOf("VERSION_OF")).length, 1);
    assert.equal((await edgesOf("ACTIVE_VERSION")).length, 1);
  });

  it("swaps the ACTIVE_VERSION edge on activation and never duplicates it", async () => {
    await ws.publishWorkflow("wf", "v1", { steps: [{ id: "a", type: "log", config: { message: "1" } }] });
    await ws.publishWorkflow("wf", "v2", { steps: [{ id: "a", type: "log", config: { message: "2" } }] });
    let active = await edgesOf("ACTIVE_VERSION");
    assert.equal(active.length, 1);
    assert.equal(active[0]!["b"], `veinworkflowversion-wf-${await ws.getWorkflowHash("wf", "v2")}`);
    await ws.setActiveVersion("wf", "v1");
    active = await edgesOf("ACTIVE_VERSION");
    assert.equal(active.length, 1);
    assert.equal(active[0]!["b"], `veinworkflowversion-wf-${await ws.getWorkflowHash("wf", "v1")}`);
    assert.equal((await edgesOf("VERSION_OF")).length, 2);
  });

  it("links a version to the custom steps it uses and the workflows it depends on", async () => {
    await ws.publishStep("my/tool", STEP("my/tool"));
    await ws.publishWorkflow("child", "v1", { steps: [{ id: "a", type: "log", config: { message: "c" } }] });
    await ws.publishWorkflow("parent", "v1", {
      steps: [
        { id: "t", type: "my/tool", config: {} },
        { id: "s", type: "subflow", config: { workflow: "child" } },
        { id: "l", type: "loop", config: { steps: [{ id: "inner", type: "my/tool", config: {} }] } },
      ],
    });
    assert.deepEqual(
      (await edgesOf("USES_STEP")).map((r) => [r["a"], r["b"]]),
      [[`veinworkflowversion-parent-${await ws.getWorkflowHash("parent")}`, "veinstep-mytool"]],
    );
    assert.deepEqual(
      (await edgesOf("DEPENDS_ON")).map((r) => [r["a"], r["b"]]),
      [[`veinworkflowversion-parent-${await ws.getWorkflowHash("parent")}`, "veinworkflow-child"]],
    );
  });

  it("re-labels rather than duplicates when the same content is published under a new label", async () => {
    const content = { steps: [{ id: "a", type: "log", config: { message: "same" } }] };
    await ws.publishWorkflow("wf", "v1", content);
    await ws.publishWorkflow("wf", "v9", content);
    const meta = await ws.getWorkflowMetadata("wf");
    assert.deepEqual(Object.keys(meta!.versions), ["v9"]);
    assert.equal(meta!.active, "v9");
    const count = await backend.bolt.run(`MATCH (v:VeinWorkflowVersion) RETURN count(v) AS c`);
    assert.equal(count[0]!["c"], 1);
  });

  it("deleteStep is a soft delete: nodes stay, flagged, and a republish restores the identity", async () => {
    await ws.publishStep("s", STEP("s"), "one");
    const before = await backend.bolt.run(`MATCH (n:VeinStep) RETURN n.ref_id AS ref_id`);
    assert.equal(await ws.deleteStep("s"), true);
    const flagged = await backend.bolt.run(`MATCH (n) WHERE n:VeinStep OR n:VeinStepVersion RETURN n.is_deleted AS d`);
    assert.deepEqual(flagged.map((r) => r["d"]), [true, true]);
    assert.deepEqual(await ws.listSteps(), []);
    assert.equal(await ws.getStepSource("s"), null);

    await ws.publishStep("s", STEP("s"), "again");
    const after = await backend.bolt.run(`MATCH (n:VeinStep) WHERE n.is_deleted = false RETURN n.ref_id AS ref_id`);
    assert.equal(after[0]!["ref_id"], before[0]!["ref_id"], "same node_key → restored, ref_id preserved");
    assert.deepEqual((await ws.listSteps()).map((s) => [s.type, s.description]), [["s", "again"]]);
    assert.deepEqual(await ws.listStepVersions("s"), { active: "v1", versions: ["v1"] });
  });

  it("is persistent: a second store over the same backend sees everything, and materialization prunes stale files", async () => {
    await ws.publishStep("keep", STEP("keep"));
    await ws.publishStep("drop", STEP("drop"));
    const dir = await ws.materializeCustomSteps();
    const other = new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps") });
    assert.deepEqual((await other.listSteps()).map((s) => s.type), ["drop", "keep"]);
    await other.deleteStep("drop");
    assert.equal(await ws.materializeCustomSteps(), dir);
    const { readdir } = await import("node:fs/promises");
    assert.deepEqual((await readdir(dir)).sort(), ["keep.ts"]);
  });

  it("helpers (_-prefixed) are stored and materialized but not listed", async () => {
    await ws.publishStep("ns/_shared", "export const x = 1;");
    await ws.publishStep("ns/real", STEP("ns/real"));
    assert.deepEqual((await ws.listSteps()).map((s) => s.type), ["ns/real"]);
    const dir = await ws.materializeCustomSteps();
    const { readdir } = await import("node:fs/promises");
    assert.deepEqual((await readdir(join(dir, "ns"))).sort(), ["_shared.ts", "real.ts"]);
  });
});
