/**
 * LIVE smoke test for the graph/* steps — the vein-native twins of the
 * jarvis/* steps. Needs a throwaway Neo4j (it seeds the Vein domain into it
 * and writes nodes); no vein server, no jarvis. Verifies:
 *   1. seeding: seedGraphSteps publishes into a temp workspace and
 *      buildRegistry discovers every step from disk;
 *   2. step logic end-to-end: register-namespace → get-ontology →
 *      create-node → graph-get → edit-node → graph-search → create-triplet →
 *      graph-neighbors → get-ontology-type → graph-get-batched →
 *      create-batch-triplet, asserting the jarvis-shaped outputs.
 *
 * Run (against the vein graph test container):
 *   docker run -d --name vein-neo4j-test -p 7688:7687 -e NEO4J_AUTH=neo4j/veintest neo4j:5
 *   NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=veintest VEIN_GRAPH_EMBEDDINGS=off \
 *     npx tsx src/lab/graph/smoke.ts
 *
 * VEIN_GRAPH_EMBEDDINGS=off keeps the model download out of the loop
 * (search is fulltext-only then); drop it to exercise the semantic path.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, closeGraphBackends, type StepContext } from "vein";
import { seedGraphSteps } from "./seed.js";

const ctx = {
  runId: "smoke",
  path: "smoke",
  scope: {},
  input: undefined,
  emit: async () => {},
  services: {
    http: async () => ({ status: 500, ok: false, headers: {}, body: "no http in graph smoke" }),
    secrets: { get: async (n: string) => process.env[n] },
  },
} as unknown as StepContext;

type StepDef = { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: StepContext): Promise<unknown> };
const fail = (label: string, out: unknown): never => {
  console.error(`✗ ${label}:`, typeof out === "string" ? out : JSON.stringify(out).slice(0, 600));
  process.exit(1);
};

async function main() {
  if (!process.env["NEO4J_URI"]) {
    console.error("NEO4J_URI not set — point it at a THROWAWAY Neo4j (this smoke seeds and writes).");
    process.exit(2);
  }
  const stamp = Date.now().toString(36);
  const NS = `smoke-${stamp}`;

  // ── 1. seed + discover ───────────────────────────────────────────────────
  const dir = mkdtempSync(join(process.cwd(), ".graph-smoke-"));
  try {
    const workspace = new WorkspaceManager(dir);
    await seedGraphSteps(workspace);
    const { registry } = await buildRegistry(workspace.path);
    const expected = [
      "graph/get-ontology", "graph/get-ontology-type", "graph/graph-search", "graph/graph-get",
      "graph/graph-get-batched", "graph/graph-neighbors", "graph/register-namespace", "graph/create-node",
      "graph/edit-node", "graph/create-triplet", "graph/create-batch-triplet",
    ];
    for (const t of expected) assert.ok(registry[t], `registry missing ${t}`);
    console.log(`✔ seeded + discovered ${expected.length} graph/* steps`);
    const run = async (type: string, input: unknown): Promise<any> => {
      const def = registry[type] as unknown as StepDef;
      return def.run(def.input.parse(input), ctx);
    };

    // ── 2. end-to-end against the live graph ─────────────────────────────
    let out = await run("graph/register-namespace", { namespace: NS });
    if (typeof out === "string" || !out.registered) fail("register-namespace", out);
    out = await run("graph/register-namespace", { namespace: NS });
    if (!out.alreadyExisted) fail("register-namespace idempotence", out);
    console.log(`✔ register-namespace (${NS})`);

    out = await run("graph/get-ontology", { include_edges: true, include_attributes: true });
    if (typeof out === "string") fail("get-ontology", out);
    assert.ok(out.domains.includes("vein"), "vein domain registered");
    const veinTypes = (out.node_types.vein ?? []).map((n: any) => n.type).sort();
    assert.deepEqual(veinTypes, ["VeinAgentSession", "VeinChat", "VeinRun", "VeinStep", "VeinStepVersion", "VeinToolCall", "VeinTurn", "VeinWorkflow", "VeinWorkflowVersion"]);
    assert.ok(out.edges.some((e: any) => e.edge_type === "VERSION_OF" && e.source_type === "VeinWorkflowVersion" && e.target_type === "VeinWorkflow"));
    const runType = out.node_types.vein.find((n: any) => n.type === "VeinRun");
    assert.equal(runType.attributes.run_id, "string");
    assert.equal(runType.inherited_attributes.name, "string");
    console.log("✔ get-ontology (9 Vein types, edges, attributes)");

    out = await run("graph/get-ontology-type", { type: "veinworkflow" });
    if (typeof out === "string") fail("get-ontology-type", out);
    assert.equal(out.attributes.name, "string");
    assert.equal(out.attributes.category, "?string");
    assert.equal(Object.keys(out).join(","), "attributes");
    console.log("✔ get-ontology-type (case-insensitive, attributes only)");

    const wfName = `smoke-wf-${stamp}`;
    out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: wfName, description: "Smoke test workflow that delivers memos" } });
    if (typeof out === "string" || !out.ref_id) fail("create-node", out);
    const wfRef = out.ref_id as string;
    assert.equal(out.status, "Success");
    out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: wfName } });
    assert.equal(out.status, "Warning");
    assert.equal(out.ref_id, wfRef);
    out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: NS, node_data: { name: wfName, bogus: 1 } });
    assert.ok(typeof out === "string" && out.includes("UNKNOWN_ATTRIBUTE"), `validation gate: ${out}`);
    out = await run("graph/create-node", { node_type: "Workflow", namespace: NS, node_data: { name: wfName } });
    assert.ok(typeof out === "string" && out.includes("UNKNOWN_TYPE"), `closed type set: ${out}`);
    out = await run("graph/create-node", { node_type: "VeinWorkflow", namespace: "never-registered", node_data: { name: wfName } });
    assert.ok(typeof out === "string" && out.includes("INVALID_NAMESPACE"), `namespace gate: ${out}`);
    console.log(`✔ create-node (VeinWorkflow ${wfRef}; merge, validation, type, namespace gates)`);

    out = await run("graph/graph-get", { ref_id: wfRef, namespace: NS });
    if (typeof out === "string" || out.ref_id !== wfRef) fail("graph-get", out);
    assert.equal(out.node_type, "VeinWorkflow");
    assert.equal(out.name, wfName);
    assert.equal(out.properties.description, "Smoke test workflow that delivers memos");
    assert.ok(!("Data_Bank" in out.properties) && !("node_key" in out.properties));
    assert.deepEqual(out.edges, {});
    console.log("✔ graph-get");

    out = await run("graph/edit-node", { ref_id: wfRef, namespace: NS, node_data: { category: "smoke" }, properties_to_be_deleted: ["description"] });
    if (typeof out === "string") fail("edit-node", out);
    out = await run("graph/graph-get", { ref_id: wfRef, namespace: NS });
    assert.equal(out.properties.category, "smoke");
    assert.ok(!("description" in out.properties));
    out = await run("graph/edit-node", { ref_id: wfRef, node_type: "VeinStep" });
    assert.ok(typeof out === "string" && out.includes("not supported"));
    out = await run("graph/edit-node", { ref_id: wfRef, properties_to_be_deleted: ["name"] });
    assert.ok(typeof out === "string" && out.includes("MISSING_REQUIRED"));
    console.log("✔ edit-node (merge + delete round-trip; type change and required removal refused)");

    out = await run("graph/graph-search", { q: wfName, namespace: NS, type: "VeinWorkflow" });
    if (typeof out === "string") fail("graph-search", out);
    assert.ok(Array.isArray(out) && out.length >= 1 && out[0].ref_id === wfRef, `search finds the workflow first: ${JSON.stringify(out).slice(0, 300)}`);
    assert.equal(out[0].name, wfName);
    assert.equal(out[0].node_type, "VeinWorkflow");
    out = await run("graph/graph-search", {});
    assert.ok(typeof out === "string" && out.includes("requires at least one"));
    console.log("✔ graph-search (fulltext hit, title boost, type filter)");

    out = await run("graph/create-triplet", {
      source_type: "VeinWorkflowVersion",
      source_data: { name: wfName, content_hash: "c-smoke1", created_at: new Date().toISOString() },
      target_ref_id: wfRef,
      edge_type: "version_of",
      edge_data: { importance: 0.5 },
      namespace: NS,
    });
    if (typeof out === "string" || !out.edge_ref_id) fail("create-triplet", out);
    assert.equal(out.status, "Success");
    assert.equal(out.edge_type, "VERSION_OF");
    const wfvRef = out.source_ref_id as string;
    out = await run("graph/create-triplet", { source_ref_id: wfvRef, target_ref_id: wfRef, edge_type: "VERSION_OF", namespace: NS });
    assert.equal(out.status, "Warning", "idempotent edge");
    out = await run("graph/create-triplet", { source_ref_id: wfRef, target_ref_id: wfvRef, edge_type: "VERSION_OF", namespace: NS });
    assert.ok(typeof out === "string" && out.includes("WRONG_TYPE"), `unregistered triple refused: ${out}`);
    console.log(`✔ create-triplet (inline VeinWorkflowVersion ${wfvRef} -[VERSION_OF]-> workflow; idempotent; registry gate)`);

    out = await run("graph/graph-neighbors", { ref_id: wfRef, namespace: NS });
    if (typeof out === "string") fail("graph-neighbors", out);
    assert.equal(out.length, 1);
    assert.equal(out[0].ref_id, wfvRef);
    assert.equal(out[0].edge_type, "VERSION_OF");
    assert.equal(out[0].direction, "reverse");
    assert.equal(out[0].node_type, "VeinWorkflowVersion");
    assert.equal(out[0].importance, 0.5);
    assert.deepEqual(out[0].edges, { VERSION_OF: 1 });
    out = await run("graph/graph-neighbors", { ref_id: wfRef, edge_type: ["USES_STEP"], namespace: NS });
    assert.equal(out.length, 0);
    console.log("✔ graph-neighbors (direction, importance, edge filter, per-neighbor counts)");

    out = await run("graph/graph-get-batched", { ref_ids: [wfRef, "nope", wfvRef], namespace: NS });
    if (typeof out === "string") fail("graph-get-batched", out);
    assert.equal(out.returned, 3);
    assert.equal(out.nodes[0].ref_id, wfRef);
    assert.deepEqual(out.nodes[0].edges, { VERSION_OF: 1 });
    assert.ok(out.nodes[1].error);
    assert.equal(out.nodes[2].node_type, "VeinWorkflowVersion");
    console.log("✔ graph-get-batched (order kept, per-item errors)");

    out = await run("graph/create-batch-triplet", {
      namespace: NS,
      triplets: [
        { source_ref_id: wfRef, target_ref_id: wfvRef, edge_type: "ACTIVE_VERSION" },
        { source_type: "VeinStep", source_data: { step_type: `smoke/step-${stamp}` }, target_ref_id: wfRef, edge_type: "USES_STEP" },
        { source_ref_id: wfvRef, target_type: "VeinStep", target_data: { step_type: `smoke/step-${stamp}` }, edge_type: "USES_STEP" },
        { source_ref_id: "nope", target_ref_id: wfRef, edge_type: "EXECUTED" },
      ],
    });
    if (typeof out === "string") fail("create-batch-triplet", out);
    assert.equal(out.requested, 4);
    assert.equal(out.succeeded, 2, JSON.stringify(out.results));
    assert.equal(out.results[0].status, "Success");
    assert.ok(out.results[1].error?.includes("WRONG_TYPE"), "VeinStep→VeinWorkflow USES_STEP is not registered");
    assert.equal(out.results[2].status, "Success");
    assert.ok(out.results[3].error?.includes("does not resolve"));
    console.log("✔ create-batch-triplet (per-item outcomes, inline dedupe)");

    console.log("\nall graph/* smoke checks passed");
  } finally {
    await closeGraphBackends();
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
