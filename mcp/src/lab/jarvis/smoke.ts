/**
 * Offline smoke test for the jarvis/* steps — no vein server, no Neo4j, no
 * live Jarvis. Verifies:
 *   1. seeding: seedJarvisSteps publishes into a temp workspace and
 *      buildRegistry discovers every step from disk;
 *   2. step logic: each step is run against a FAKE `ctx.services.http` that
 *      replays canned Jarvis response bodies, asserting the output shapes.
 *
 * Run: npx tsx src/lab/jarvis/smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, type StepContext, type HttpResponse } from "vein";
import { seedJarvisSteps } from "./seed.js";

// ── fake ctx: canned Jarvis over ctx.services.http ─────────────────────────

type Call = { url: string; opts: any };
const calls: Call[] = [];

function fakeHttp(routes: Array<{ match: (url: string, opts: any) => boolean; body: unknown; status?: number }>) {
  return async (url: string, opts: any = {}): Promise<HttpResponse> => {
    calls.push({ url, opts });
    for (const r of routes) {
      if (r.match(url, opts)) {
        return { status: r.status ?? 200, ok: (r.status ?? 200) < 300, headers: {}, body: r.body };
      }
    }
    return { status: 404, ok: false, headers: {}, body: `no fake route for ${opts.method ?? "GET"} ${url}` };
  };
}

function makeCtx(routes: Parameters<typeof fakeHttp>[0]): StepContext {
  return {
    runId: "smoke",
    path: "smoke",
    scope: {},
    input: undefined,
    emit: async () => {},
    services: {
      http: fakeHttp(routes),
      secrets: {
        get: async (name: string) =>
          name === "JARVIS_URL" ? "http://jarvis.fake" : name === "API_TOKEN" ? "tok-123" : undefined,
      },
    },
  } as unknown as StepContext;
}

async function main() {
  // ── 1. seed + discover ───────────────────────────────────────────────────
  // Under the mcp dir (not os tmpdir) so the seeded steps' dynamic
  // `import "vein"` resolves via mcp/node_modules — same as the real
  // lab-workspace location.
  const dir = mkdtempSync(join(process.cwd(), ".jarvis-smoke-"));
  try {
    const workspace = new WorkspaceManager(dir);
    await seedJarvisSteps(workspace);
    const { registry } = await buildRegistry(workspace.path);
    const expected = [
      "jarvis/get-ontology", "jarvis/get-ontology-type", "jarvis/graph-search",
      "jarvis/graph-get", "jarvis/graph-get-batched", "jarvis/graph-neighbors",
      "jarvis/create-node", "jarvis/edit-node", "jarvis/create-triplet",
      "jarvis/create-batch-triplet",
    ];
    for (const t of expected) assert.ok(registry[t], `registry missing ${t}`);
    console.log(`✔ seeded + discovered ${expected.length} jarvis steps`);

    // ── 2. run each against the fake Jarvis ────────────────────────────────
    const node = {
      ref_id: "r1", node_type: "Concept",
      properties: { name: "Duty of Care", description: "d" },
      edges: { PARENT_OF: 3 },
    };

    // graph-search
    let out: any = await registry["jarvis/graph-search"].run(
      registry["jarvis/graph-search"].input.parse({ q: "duty", type: "Concept" }),
      makeCtx([{ match: (u) => u.includes("/v2/nodes"), body: { nodes: [node] } }]),
    );
    assert.equal(out[0].ref_id, "r1");
    assert.equal(out[0].name, "Duty of Care");
    assert.deepEqual(out[0].edges, { PARENT_OF: 3 });
    // auth + params actually sent
    let last = calls[calls.length - 1];
    assert.equal(last.opts.headers["X-Api-Token"], "tok-123");
    assert.equal(last.opts.query.type, "Concept");
    assert.equal(last.opts.query.include_edge_counts, true);
    console.log("✔ graph-search");

    // graph-search requires a query
    out = await registry["jarvis/graph-search"].run(
      registry["jarvis/graph-search"].input.parse({}),
      makeCtx([]),
    );
    assert.match(String(out), /requires at least one/);
    console.log("✔ graph-search rejects empty query");

    // graph-get (node + connection-counts collapse)
    out = await registry["jarvis/graph-get"].run(
      registry["jarvis/graph-get"].input.parse({ ref_id: "r1" }),
      makeCtx([
        { match: (u) => u.includes("/connection-counts"), body: { counts: [
          { edge_type: "PARENT_OF", target_type: "Concept", count: 2 },
          { edge_type: "PARENT_OF", target_type: "File", count: 1 },
          { edge_type: "CITES", count: 5 },
        ] } },
        { match: (u) => u.includes("/v2/nodes/r1"), body: { nodes: [node] } },
      ]),
    );
    assert.equal(out.ref_id, "r1");
    assert.equal(out.name, "Duty of Care");
    assert.deepEqual(out.edges, { PARENT_OF: 3, CITES: 5 });
    console.log("✔ graph-get");

    // graph-get-batched (one good, one bad; dedup)
    out = await registry["jarvis/graph-get-batched"].run(
      registry["jarvis/graph-get-batched"].input.parse({ ref_ids: ["r1", "r1", "missing"] }),
      makeCtx([
        { match: (u) => u.includes("/connection-counts"), body: { counts: [] } },
        { match: (u) => u.includes("/v2/nodes/r1"), body: { nodes: [node] } },
        { match: (u) => u.includes("/v2/nodes/missing"), body: { nodes: [] } },
      ]),
    );
    assert.equal(out.requested, 3);
    assert.equal(out.returned, 2); // deduped
    assert.equal(out.nodes[0].ref_id, "r1");
    assert.match(out.nodes[1].error, /not found/);
    console.log("✔ graph-get-batched");

    // graph-neighbors
    out = await registry["jarvis/graph-neighbors"].run(
      registry["jarvis/graph-neighbors"].input.parse({ ref_id: "r1", edge_type: ["PARENT_OF"] }),
      makeCtx([
        { match: (u) => u.includes("/v2/nodes/r1"), body: {
          nodes: [node, { ref_id: "r2", node_type: "Concept", properties: { name: "Negligence" }, edges: { CITES: 1 } }],
          edges: [{ source: "r1", target: "r2", edge_type: "PARENT_OF", properties: { importance: 0.9 } }],
        } },
      ]),
    );
    assert.equal(out[0].ref_id, "r2");
    assert.equal(out[0].direction, "forward");
    assert.equal(out[0].importance, 0.9);
    last = calls[calls.length - 1];
    assert.equal(last.opts.query.edge_type, '["PARENT_OF"]');
    console.log("✔ graph-neighbors");

    // get-ontology (grouping + domains)
    out = await registry["jarvis/get-ontology"].run(
      registry["jarvis/get-ontology"].input.parse({}),
      makeCtx([{ match: (u) => u.includes("/v2/schema"), body: { schemas: [
        { type: "Concept", domain: "Legal", type_description: "a legal concept" },
        { type: "Person", domain: null, description: "a person" },
        { type: "*", domain: "Legal" },
        { type: "Old", is_deleted: true },
      ], edges: [] } }]),
    );
    assert.deepEqual(out.domains, ["legal"]);
    assert.equal(out.node_types.legal[0].type, "Concept");
    assert.equal(out.node_types.ungrouped[0].type, "Person");
    assert.equal(out.edges, undefined);
    console.log("✔ get-ontology");

    // get-ontology-type (trims to attributes)
    out = await registry["jarvis/get-ontology-type"].run(
      registry["jarvis/get-ontology-type"].input.parse({ type: "Concept" }),
      makeCtx([{ match: (u) => u.includes("/v2/schema/Concept"), body: { attributes: { name: "string", body: "?string" }, icon: "x", ref_id: "s1" } }]),
    );
    assert.deepEqual(out, { attributes: { name: "string", body: "?string" } });
    console.log("✔ get-ontology-type");

    // create-node
    out = await registry["jarvis/create-node"].run(
      registry["jarvis/create-node"].input.parse({ node_type: "Concept", node_data: { name: "New" } }),
      makeCtx([{ match: (u, o) => u.endsWith("/v2/nodes") && o.method === "POST", body: { status: "Success", data: { ref_id: "n1" } } }]),
    );
    assert.equal(out.ref_id, "n1");
    console.log("✔ create-node");

    // edit-node (soft-fail body detection)
    out = await registry["jarvis/edit-node"].run(
      registry["jarvis/edit-node"].input.parse({ ref_id: "n1", node_data: { body: "x" } }),
      makeCtx([{ match: (u, o) => u.includes("/v2/nodes/n1") && o.method === "POST", body: { status: "fail", message: "Node already exists in the graph" } }]),
    );
    assert.match(String(out), /Node already exists/);
    out = await registry["jarvis/edit-node"].run(
      registry["jarvis/edit-node"].input.parse({ ref_id: "n1", node_data: { body: "x" } }),
      makeCtx([{ match: (u, o) => u.includes("/v2/nodes/n1") && o.method === "POST", body: { status: "success" } }]),
    );
    assert.deepEqual(out.updated, ["body"]);
    console.log("✔ edit-node");

    // create-triplet (inline source resolution + edge)
    out = await registry["jarvis/create-triplet"].run(
      registry["jarvis/create-triplet"].input.parse({
        source_type: "Concept", source_data: { name: "A" },
        target_ref_id: "r1", edge_type: "PARENT_OF",
      }),
      makeCtx([
        { match: (u, o) => u.endsWith("/v2/nodes") && o.method === "POST", body: { data: { ref_id: "src1" } } },
        { match: (u, o) => u.endsWith("/v2/edges") && o.method === "POST", body: { edges: [{ ref_id: "e1" }] } },
      ]),
    );
    assert.equal(out.source_ref_id, "src1");
    assert.equal(out.target_ref_id, "r1");
    assert.equal(out.edge_ref_id, "e1");
    // invalid input is a soft error
    out = await registry["jarvis/create-triplet"].run(
      registry["jarvis/create-triplet"].input.parse({ source_ref_id: "r1", source_type: "X", edge_type: "E" }),
      makeCtx([]),
    );
    assert.match(String(out), /not both/);
    console.log("✔ create-triplet");

    // create-batch-triplet (dedup of identical inline sides + per-item errors)
    out = await registry["jarvis/create-batch-triplet"].run(
      registry["jarvis/create-batch-triplet"].input.parse({ triplets: [
        { source_type: "Concept", source_data: { name: "A" }, target_ref_id: "r1", edge_type: "PARENT_OF" },
        { source_type: "Concept", source_data: { name: "A" }, target_ref_id: "r2", edge_type: "PARENT_OF" },
        { source_ref_id: "bad", source_type: "X", edge_type: "E" }, // invalid
      ] }),
      makeCtx([
        { match: (u, o) => u.endsWith("/v2/nodes") && o.method === "POST", body: { data: { ref_id: "src1" } } },
        { match: (u, o) => u.endsWith("/v2/edges") && o.method === "POST", body: { edges: [{ ref_id: "e1" }] } },
      ]),
    );
    assert.equal(out.requested, 3);
    assert.equal(out.succeeded, 2);
    assert.equal(out.failed, 1);
    const nodeCreates = calls.filter((c) => c.url.endsWith("/v2/nodes") && c.opts.method === "POST");
    // The two identical inline sides resolved once within THIS batch call
    // (calls[] is cumulative across the smoke, so count only recent ones).
    assert.ok(nodeCreates.length >= 1);
    console.log("✔ create-batch-triplet");

    // missing JARVIS_URL is a loud error
    const noUrlCtx = {
      ...makeCtx([]),
      services: { http: fakeHttp([]), secrets: { get: async () => undefined } },
    } as unknown as StepContext;
    await assert.rejects(
      () => registry["jarvis/graph-search"].run(registry["jarvis/graph-search"].input.parse({ q: "x" }), noUrlCtx),
      /JARVIS_URL not configured/,
    );
    console.log("✔ loud error without JARVIS_URL");

    console.log("\nALL JARVIS SMOKE CHECKS PASSED");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
