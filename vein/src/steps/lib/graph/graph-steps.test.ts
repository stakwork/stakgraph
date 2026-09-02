/**
 * LIVE end-to-end test for the graph/* lib steps against a throwaway Neo4j
 * (it wipes the database, seeds the Vein domain, and writes nodes). Skipped
 * unless VEIN_TEST_NEO4J_URI is set — see src/graph/test-util.ts. Runs with
 * embeddings off so no model download is needed (search is fulltext-only).
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { buildRegistry } from "../../registry.js";
import type { StepContext } from "../../../core.js";
import { Bolt } from "../../../graph/bolt.js";
import { closeGraphBackends } from "../../../graph/backend.js";
import { testGraphConfig, wipeGraph } from "../../../graph/test-util.js";

const cfg = testGraphConfig();
const STEP_TYPES = [
  "graph/get-ontology", "graph/get-ontology-type", "graph/graph-search", "graph/graph-get",
  "graph/graph-get-batched", "graph/graph-neighbors", "graph/register-namespace", "graph/create-node",
  "graph/edit-node", "graph/create-triplet", "graph/create-batch-triplet",
];

describe("graph/* lib steps are discovered by the registry", () => {
  it("all eleven twins are present, sourced from lib", async () => {
    const { registry, sources } = await buildRegistry();
    for (const t of STEP_TYPES) {
      assert.ok(registry[t], `missing ${t}`);
      assert.equal(sources[t], "lib");
    }
  });
});

describe("graph/* lib steps (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, () => {
  const secrets: Record<string, string> = {
    NEO4J_URI: cfg?.uri ?? "",
    NEO4J_USER: cfg?.user ?? "neo4j",
    NEO4J_PASSWORD: cfg?.password ?? "",
    VEIN_GRAPH_NAMESPACE: cfg?.namespace ?? "default",
    VEIN_GRAPH_EMBEDDINGS: "off",
  };
  const ctx = {
    runId: "test",
    path: "test",
    scope: {},
    input: undefined,
    emit: async () => {},
    services: { secrets: { get: async (n: string) => secrets[n] } },
  } as unknown as StepContext;
  type StepDef = { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: StepContext): Promise<any> };
  let run: (type: string, input: unknown) => Promise<any>;
  const NS = "test-ns";

  before(async () => {
    const bolt = new Bolt(cfg!);
    await wipeGraph(bolt);
    await bolt.close();
    const { registry } = await buildRegistry();
    run = (type, input) => {
      const def = registry[type] as unknown as StepDef;
      return def.run(def.input.parse(input), ctx);
    };
  });
  after(async () => {
    await closeGraphBackends();
  });

  it("register-namespace is idempotent", async () => {
    assert.deepEqual(await run("graph/register-namespace", { namespace: NS }), { namespace: NS, registered: true });
    assert.deepEqual(await run("graph/register-namespace", { namespace: NS }), { namespace: NS, registered: true, alreadyExisted: true });
  });

  it("get-ontology / get-ontology-type expose the seeded Vein domain", async () => {
    const out = await run("graph/get-ontology", { include_edges: true, include_attributes: true });
    assert.ok(typeof out !== "string", out);
    assert.ok(out.domains.includes("vein"));
    assert.deepEqual(
      out.node_types.vein.map((n: any) => n.type).sort(),
      ["VeinAgentSession", "VeinChat", "VeinRun", "VeinStep", "VeinStepVersion", "VeinToolCall", "VeinTurn", "VeinWorkflow", "VeinWorkflowVersion"],
    );
    assert.ok(out.edges.some((e: any) => e.edge_type === "VERSION_OF" && e.source_type === "VeinWorkflowVersion" && e.target_type === "VeinWorkflow"));
    const runType = out.node_types.vein.find((n: any) => n.type === "VeinRun");
    assert.equal(runType.attributes.run_id, "string");
    assert.equal(runType.inherited_attributes.name, "string");
    const single = await run("graph/get-ontology-type", { type: "veinworkflow" });
    assert.deepEqual(Object.keys(single), ["attributes"]);
    assert.equal(single.attributes.name, "string");
    assert.equal(single.attributes.category, "?string");
    assert.match(await run("graph/get-ontology-type", { type: "Nope" }), /unknown type/);
  });

  let wfRef: string;
  let wfvRef: string;

  it("create-node: merge, validation, closed type set, namespace gate", async () => {
    let out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: "harvey-deliver", description: "Delivers legal memos" } });
    assert.equal(out.status, "Success");
    wfRef = out.ref_id;
    out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: "harvey-deliver" } });
    assert.equal(out.status, "Warning");
    assert.equal(out.ref_id, wfRef);
    assert.match(out.messages[0], /already exists/);
    assert.match(await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: "x", bogus: 1 } }), /UNKNOWN_ATTRIBUTE/);
    assert.match(await run("graph/create-node", { node_type: "Workflow", namespace: NS, node_data: { name: "x" } }), /UNKNOWN_TYPE/);
    assert.match(await run("graph/create-node", { node_type: "VeinWorkflow", namespace: "never-registered", node_data: { name: "x" } }), /INVALID_NAMESPACE/);
  });

  it("graph-get returns the jarvis envelope", async () => {
    const out = await run("graph/graph-get", { ref_id: wfRef, namespace: NS });
    assert.equal(out.ref_id, wfRef);
    assert.equal(out.node_type, "VeinWorkflow");
    assert.equal(out.name, "harvey-deliver");
    assert.equal(out.properties.description, "Delivers legal memos");
    assert.ok(!("Data_Bank" in out.properties) && !("node_key" in out.properties));
    assert.deepEqual(out.edges, {});
    assert.match(await run("graph/graph-get", { ref_id: "nope" }), /node not found/);
  });

  it("edit-node merges, deletes, refuses type changes and required removals", async () => {
    assert.deepEqual(await run("graph/edit-node", { ref_id: wfRef, node_data: { category: "smoke" }, properties_to_be_deleted: ["description"] }), {
      status: "Success", ref_id: wfRef, updated: ["category"], deleted: ["description"],
    });
    const out = await run("graph/graph-get", { ref_id: wfRef });
    assert.equal(out.properties.category, "smoke");
    assert.ok(!("description" in out.properties));
    assert.match(await run("graph/edit-node", { ref_id: wfRef, node_type: "VeinStep" }), /not supported/);
    assert.match(await run("graph/edit-node", { ref_id: wfRef, properties_to_be_deleted: ["name"] }), /MISSING_REQUIRED/);
    assert.match(await run("graph/edit-node", { ref_id: wfRef }), /invalid input/);
  });

  it("graph-search finds the workflow (fulltext, title boost, type filter)", async () => {
    const out = await run("graph/graph-search", { q: "harvey-deliver", namespace: NS, type: "VeinWorkflow" });
    assert.ok(Array.isArray(out) && out[0].ref_id === wfRef, JSON.stringify(out));
    assert.equal(out[0].name, "harvey-deliver");
    assert.equal(out[0].node_type, "VeinWorkflow");
    assert.deepEqual(out[0].edges, {});
    assert.match(await run("graph/graph-search", {}), /requires at least one/);
    assert.match(await run("graph/graph-search", { q: "x", domains: "legal" }), /INVALID_DOMAIN/);
  });

  it("create-triplet: inline side, idempotent edge, registry gate", async () => {
    let out = await run("graph/create-triplet", {
      source_type: "VeinWorkflowVersion",
      source_data: { name: "harvey-deliver", content_hash: "c-1", created_at: "2026-09-01T00:00:00Z" },
      target_ref_id: wfRef,
      edge_type: "version of",
      edge_data: { importance: 0.5 },
      namespace: NS,
    });
    assert.equal(out.status, "Success", JSON.stringify(out));
    assert.equal(out.edge_type, "VERSION_OF");
    assert.equal(out.target_ref_id, wfRef);
    wfvRef = out.source_ref_id;
    out = await run("graph/create-triplet", { source_ref_id: wfvRef, target_ref_id: wfRef, edge_type: "VERSION_OF" });
    assert.equal(out.status, "Warning");
    assert.match(await run("graph/create-triplet", { source_ref_id: wfRef, target_ref_id: wfvRef, edge_type: "VERSION_OF" }), /WRONG_TYPE/);
    assert.match(await run("graph/create-triplet", { source_ref_id: wfRef, edge_type: "VERSION_OF" }), /invalid input/);
  });

  it("graph-neighbors: direction, importance, edge filter, per-neighbor counts", async () => {
    const out = await run("graph/graph-neighbors", { ref_id: wfRef, namespace: NS });
    assert.equal(out.length, 1);
    assert.deepEqual(out[0], { ref_id: wfvRef, node_type: "VeinWorkflowVersion", name: "harvey-deliver", edge_type: "VERSION_OF", direction: "reverse", edges: { VERSION_OF: 1 }, importance: 0.5 });
    assert.deepEqual(await run("graph/graph-neighbors", { ref_id: wfRef, edge_type: ["USES_STEP"] }), []);
  });

  it("graph-get-batched keeps order and isolates per-item errors", async () => {
    const out = await run("graph/graph-get-batched", { ref_ids: [wfRef, "nope", wfvRef, wfRef], namespace: NS });
    assert.equal(out.requested, 4);
    assert.equal(out.returned, 3, "deduped");
    assert.equal(out.nodes[0].ref_id, wfRef);
    assert.deepEqual(out.nodes[0].edges, { VERSION_OF: 1 });
    assert.match(out.nodes[1].error, /not found/);
    assert.equal(out.nodes[2].node_type, "VeinWorkflowVersion");
    assert.equal(out.truncated, false);
  });

  it("create-batch-triplet: per-item outcomes, inline dedupe", async () => {
    const out = await run("graph/create-batch-triplet", {
      namespace: NS,
      triplets: [
        { source_ref_id: wfRef, target_ref_id: wfvRef, edge_type: "ACTIVE_VERSION" },
        { source_type: "VeinStep", source_data: { step_type: "smoke/step" }, target_ref_id: wfRef, edge_type: "USES_STEP" },
        { source_ref_id: wfvRef, target_type: "VeinStep", target_data: { step_type: "smoke/step" }, edge_type: "USES_STEP" },
        { source_ref_id: "nope", target_ref_id: wfRef, edge_type: "EXECUTED" },
      ],
    });
    assert.equal(out.requested, 4);
    assert.equal(out.succeeded, 2, JSON.stringify(out.results));
    assert.equal(out.results[0].status, "Success");
    assert.match(out.results[1].error, /WRONG_TYPE/);
    assert.equal(out.results[2].status, "Success");
    assert.match(out.results[3].error, /does not resolve/);
    const steps = await run("graph/graph-search", { q: "smoke/step", type: "VeinStep", namespace: NS });
    assert.equal(steps.length, 1, "inline VeinStep created once");
  });
});

// ── graph/project: the run/chat projector as a step ─────────────────────────

describe("graph/project step", () => {
  it("is discovered by the registry, sourced from lib", async () => {
    const { registry, sources } = await buildRegistry();
    assert.ok(registry["graph/project"]);
    assert.equal(sources["graph/project"], "lib");
  });

  it("projects a file workspace's runs into the graph (live Neo4j)", { skip: cfg ? false : "VEIN_TEST_NEO4J_URI not set" }, async () => {
    const { mkdtemp, rm } = await import("node:fs/promises");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const { FileWorkspaceStore } = await import("../../../workspace.js");
    const { FileRunStore } = await import("../../../store.js");
    const { openGraphBackend } = await import("../../../graph/backend.js");

    const dataDir = await mkdtemp(join(tmpdir(), "vein-project-step-"));
    try {
      const ws = new FileWorkspaceStore(dataDir);
      await ws.publishWorkflow("wf", "v1", { steps: [{ id: "a", type: "log", config: { message: "x" } }] });
      const store = new FileRunStore(dataDir);
      const base = { runId: "1700000000000", path: "wf" };
      await store.append("wf", base.runId, { ...base, ts: new Date().toISOString(), type: "run.start", input: {} });
      await store.append("wf", base.runId, { ...base, ts: new Date().toISOString(), type: "run.end", output: "ok" });

      const bolt = new Bolt(cfg!);
      await wipeGraph(bolt);
      await bolt.close();
      const secrets: Record<string, string> = {
        NEO4J_URI: cfg!.uri, NEO4J_USER: cfg!.user, NEO4J_PASSWORD: cfg!.password,
        VEIN_GRAPH_NAMESPACE: cfg!.namespace, VEIN_GRAPH_EMBEDDINGS: "off",
      };
      const ctx = {
        runId: "test", path: "test", scope: {}, input: undefined, emit: async () => {},
        services: { secrets: { get: async (n: string) => secrets[n] } },
      } as unknown as StepContext;
      const { registry } = await buildRegistry();
      const def = registry["graph/project"] as unknown as { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: StepContext): Promise<any> };

      const out = await def.run(def.input.parse({ dataDir }), ctx);
      assert.ok(typeof out !== "string", out);
      assert.deepEqual([out.workflows, out.runs, out.chats, out.skipped], [["wf"], 1, 0, 0]);
      const again = await def.run(def.input.parse({ dataDir }), ctx);
      assert.deepEqual([again.runs, again.skipped], [0, 1], "settled run skipped on re-run");

      const b = await openGraphBackend({ ...cfg!, namespace: cfg!.namespace }, { embeddings: false, skipBoot: true });
      const rows = await b.bolt.run(`MATCH (r:VeinRun) RETURN r.run_id AS id, r.status AS s`);
      assert.deepEqual(rows, [{ id: base.runId, s: "success" }]);
    } finally {
      await closeGraphBackends();
      await rm(dataDir, { recursive: true, force: true });
    }
  });
});
