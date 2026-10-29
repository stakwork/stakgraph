import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { accessedNodesOf, composeNodeKey, lintStepSource } from "strut";
import conceptCreate, { CONCEPT_KINDS, KIND_EDGE, conceptSlug } from "./steps/concept-create.js";

// ── a fake graph backend: what the step reaches on ctx.services.graph ──────

type Row = Record<string, unknown>;
interface FakeNode {
  ref_id: string;
  node_type: string;
  properties: Record<string, unknown>;
}
interface FakeOpts {
  nodes?: FakeNode[];
  /** The HiveWorkspace rows the bolt query answers (the step asks for two at most). */
  workspaces?: Array<{ ref_id: string; slug: string }>;
  kind?: "neo4j" | "ladybug";
  /** The node types the graph has a schema for. */
  schemas?: string[];
  /** Make an edge write of this type throw. */
  failEdge?: (edge: string) => Error | undefined;
}

const KEYS: Record<string, string> = { Concept: "concept-name", HiveWorkspace: "hiveworkspace-workspace_id" };

function fakeGraph(o: FakeOpts = {}) {
  const nodes: FakeNode[] = o.nodes ? o.nodes.map((n) => ({ ...n, properties: { ...n.properties } })) : [];
  const edges: Array<{ edge: string; source_ref_id: string; target_ref_id: string }> = [];
  const writes: Array<{ type: string; data: Record<string, unknown>; mode: string; namespace?: string }> = [];
  const deleted: string[] = [];
  const bolt: Array<{ q: string; p?: Row }> = [];
  const schemas = new Set(o.schemas ?? ["Concept", "HiveWorkspace"]);
  const schemaOf = (t: string) => (schemas.has(t) ? { type: t, node_key: KEYS[t]! } : null);
  const keyOf = (n: FakeNode) => composeNodeKey(schemaOf(n.node_type)!, n.properties);
  let seq = 0;
  const g = {
    kind: o.kind ?? "neo4j",
    namespace: "default",
    reader: {
      resolveNamespace: async (ns?: string) => ns ?? "default",
      getNode: async (ref_id: string) => nodes.find((n) => n.ref_id === ref_id) ?? null,
      getNodeByKey: async (type: string, key: string) => nodes.find((n) => n.node_type === type && keyOf(n) === key) ?? null,
    },
    schemas: { schema: async (t: string) => schemaOf(t) },
    nodes: {
      write: async (input: { type: string; data: Record<string, unknown> }, mode: string, opts?: { namespace?: string }) => {
        writes.push({ type: input.type, data: { ...input.data }, mode, namespace: opts?.namespace });
        const node_key = composeNodeKey(schemaOf(input.type)!, input.data);
        const existing = nodes.find((n) => n.node_type === input.type && keyOf(n) === node_key);
        // "create": an existing key is answered, its properties untouched.
        if (existing) return { ref_id: existing.ref_id, node_key, node_type: input.type, outcome: "existing" as const };
        const ref_id = `new-${++seq}`;
        nodes.push({ ref_id, node_type: input.type, properties: { ...input.data } });
        return { ref_id, node_key, node_type: input.type, outcome: "created" as const };
      },
      softDelete: async (ref_id: string) => {
        deleted.push(ref_id);
        return true;
      },
    },
    edges: {
      write: async (e: { edge: string; source_ref_id: string; target_ref_id: string }) => {
        const err = o.failEdge?.(e.edge);
        if (err) throw err;
        edges.push({ edge: e.edge, source_ref_id: e.source_ref_id, target_ref_id: e.target_ref_id });
        return { ref_id: `e-${edges.length}`, edge_key: "", created: true, source_ref_id: e.source_ref_id, target_ref_id: e.target_ref_id };
      },
    },
    bolt: {
      run: async (q: string, p?: Row) => {
        bolt.push({ q, p });
        return (o.workspaces ?? []).slice(0, 2).map((w) => ({ ref_id: w.ref_id, label: w.slug }));
      },
    },
  };
  return { g, nodes, edges, writes, deleted, bolt };
}

const step = conceptCreate as unknown as { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: unknown): Promise<any> };
const ctx = (graph: unknown) => ({ runId: "t", path: "t", scope: {}, emit: async () => {}, services: graph ? { graph } : {} });
const run = (cfg: Record<string, unknown>, graph: unknown) => step.run(step.input.parse(cfg), ctx(graph));

const JOB: FakeNode = { ref_id: "p1", node_type: "Concept", properties: { name: "Job", id: "job", docs: "What a job is." } };
const WS = [{ ref_id: "w1", slug: "acme" }];
const INPUT = { name: "Deploy Checklist", docs: "# Deploy\n\nRun the checks, then ship.", description: "How we deploy.", parent: "p1", kind: "process" };

describe("concept/create", () => {
  it("writes the node, PARENT_OF from the parent (by ref_id) and the kind's anchor edge; answers ref_id, id, the three nodes as provenance", async () => {
    const f = fakeGraph({ nodes: [JOB], workspaces: WS });
    const out = await run(INPUT, f.g);
    assert.deepEqual(out, { status: "Success", ref_id: "new-1", id: "deploy-checklist", name: "Deploy Checklist", parent_ref_id: "p1", anchored: true });
    // The node, create-or-merge, in the resolved namespace: docs whole, never `documentation`, no repo when none was given.
    assert.deepEqual(f.writes, [{ type: "Concept", data: { id: "deploy-checklist", name: "Deploy Checklist", docs: INPUT.docs, description: "How we deploy." }, mode: "create", namespace: "default" }]);
    assert.deepEqual(f.edges, [
      { edge: "PARENT_OF", source_ref_id: "p1", target_ref_id: "new-1" },
      { edge: "PROCESS", source_ref_id: "w1", target_ref_id: "new-1" },
    ]);
    // The workspace lookup: the one HiveWorkspace in the namespace, deleted ones excluded.
    assert.equal(f.bolt.length, 1);
    assert.match(f.bolt[0]!.q, /MATCH \(w:`HiveWorkspace`\)[\s\S]*deleted_at IS NULL[\s\S]*LIMIT 2/);
    assert.deepEqual(f.bolt[0]!.p, { ns: "default" });
    // Provenance for the run graph: the new Concept, its parent, the workspace node.
    assert.deepEqual(accessedNodesOf(out), [
      { ref_id: "new-1", node_type: "Concept", name: "Deploy Checklist" },
      { ref_id: "p1", node_type: "Concept", name: "Job" },
      { ref_id: "w1", node_type: "HiveWorkspace", name: "acme" },
    ]);
  });

  it("resolves the parent by its exact name; a name nobody has, or a ref that is not a Concept, is refused before anything is written", async () => {
    const byName = fakeGraph({ nodes: [JOB], workspaces: WS });
    const out = await run({ ...INPUT, parent: "Job" }, byName.g);
    assert.equal(out.parent_ref_id, "p1");
    assert.deepEqual(byName.edges[0], { edge: "PARENT_OF", source_ref_id: "p1", target_ref_id: "new-1" });

    const missing = fakeGraph({ nodes: [JOB], workspaces: WS });
    const err = await run({ ...INPUT, parent: "Jobs" }, missing.g);
    assert.match(err, /^concept\/create failed — parent Concept not found: "Jobs"/);
    assert.deepEqual([missing.writes, missing.edges], [[], []]);

    const person = fakeGraph({ nodes: [JOB, { ref_id: "u1", node_type: "Person", properties: { name: "Ann" } }], workspaces: WS });
    const notConcept = await run({ ...INPUT, parent: "u1" }, person.g);
    assert.match(notConcept, /^concept\/create failed — the parent "u1" is a Person, not a Concept/);
    assert.deepEqual([person.writes, person.edges], [[], []]);
  });

  it("refuses a Concept as its own parent: by name, by id, and when the merge lands on the parent itself", async () => {
    const byName = fakeGraph({ nodes: [JOB], workspaces: WS });
    assert.match(await run({ ...INPUT, name: "Job", parent: "p1" }, byName.g), /^concept\/create failed — a concept cannot be its own parent$/);
    assert.deepEqual(byName.writes, []);

    // The parent's id is the new Concept's address (another spelling of the name).
    const byId = fakeGraph({ nodes: [{ ...JOB, properties: { name: "Job Kinds", id: "job" } }], workspaces: WS });
    assert.match(await run({ ...INPUT, name: "Job", parent: "p1" }, byId.g), /cannot be its own parent$/);
    assert.deepEqual(byId.writes, []);

    // Name and id differ, but the node_key is the same (keys are lowercased): the create merges onto the parent.
    const landed = fakeGraph({ nodes: [{ ...JOB, properties: { name: "Job", id: "job-kinds" } }], workspaces: WS });
    const out = await run({ ...INPUT, name: "job", parent: "p1" }, landed.g);
    assert.match(out, /^concept\/create failed — a concept cannot be its own parent: "job" is the parent itself \(ref_id p1\)/);
    assert.equal(landed.writes.length, 1);
    assert.deepEqual([landed.edges, landed.deleted], [[], []]);
  });

  it("an existing name merges: Warning with the existing ref_id, its docs untouched, the parent link and the anchor still made", async () => {
    const existing: FakeNode = { ref_id: "c9", node_type: "Concept", properties: { name: "Deploy Checklist", docs: "OLD BODY", description: "Old." } };
    const f = fakeGraph({ nodes: [JOB, existing], workspaces: WS });
    const out = await run(INPUT, f.g);
    assert.equal(out.status, "Warning");
    assert.equal(out.ref_id, "c9");
    assert.equal(out.id, "deploy-checklist");
    assert.equal(out.anchored, true);
    assert.equal(out.messages.length, 1);
    assert.match(out.messages[0], /already exists \(ref_id c9\)/);
    assert.match(out.messages[0], /graph\/edit-node/);
    // Create-mode: the write was asked for, the node's properties were not touched.
    assert.equal(f.writes[0]!.mode, "create");
    assert.deepEqual(f.nodes.find((n) => n.ref_id === "c9")!.properties, existing.properties);
    assert.deepEqual(f.edges, [
      { edge: "PARENT_OF", source_ref_id: "p1", target_ref_id: "c9" },
      { edge: "PROCESS", source_ref_id: "w1", target_ref_id: "c9" },
    ]);
    assert.deepEqual(accessedNodesOf(out)![0], { ref_id: "c9", node_type: "Concept", name: "Deploy Checklist" });
  });

  it("no parent: a top-level Concept, PARENT_OF not written, no parent_ref_id", async () => {
    const f = fakeGraph({ workspaces: WS });
    const out = await run({ name: "Deploy Checklist", docs: "body" }, f.g);
    assert.deepEqual(out, { status: "Success", ref_id: "new-1", id: "deploy-checklist", name: "Deploy Checklist", anchored: true });
    assert.deepEqual(f.edges, [{ edge: "HAS_CONCEPT", source_ref_id: "w1", target_ref_id: "new-1" }]);
    assert.deepEqual(accessedNodesOf(out), [
      { ref_id: "new-1", node_type: "Concept", name: "Deploy Checklist" },
      { ref_id: "w1", node_type: "HiveWorkspace", name: "acme" },
    ]);
  });

  it("no workspace node, two of them, no HiveWorkspace schema, or no Neo4j: not anchored, still a success", async () => {
    for (const o of [
      { nodes: [JOB], workspaces: [] },
      { nodes: [JOB], workspaces: [...WS, { ref_id: "w2", slug: "other" }] },
      { nodes: [JOB], workspaces: WS, schemas: ["Concept"] },
      { nodes: [JOB], workspaces: WS, kind: "ladybug" as const },
    ]) {
      const f = fakeGraph(o);
      const out = await run(INPUT, f.g);
      assert.deepEqual(out, { status: "Success", ref_id: "new-1", id: "deploy-checklist", name: "Deploy Checklist", parent_ref_id: "p1", anchored: false }, JSON.stringify(o));
      assert.deepEqual(f.edges, [{ edge: "PARENT_OF", source_ref_id: "p1", target_ref_id: "new-1" }]);
      assert.deepEqual(accessedNodesOf(out)!.map((n) => n.ref_id), ["new-1", "p1"]);
    }
    // No schema and no Neo4j never ask bolt.
    const ladybug = fakeGraph({ nodes: [JOB], workspaces: WS, kind: "ladybug" });
    await run(INPUT, ladybug.g);
    assert.equal(ladybug.bolt.length, 0);
  });

  it("the anchor edge failing is a Warning with a message, never a failed step: the node and its parent link stand", async () => {
    const f = fakeGraph({ nodes: [JOB], workspaces: WS, failEdge: (e) => (e === "PROCESS" ? Object.assign(new Error("no such edge schema"), { name: "GraphValidationError", code: "WRONG_TYPE" }) : undefined) });
    const out = await run(INPUT, f.g);
    assert.equal(out.status, "Warning");
    assert.equal(out.ref_id, "new-1");
    assert.equal(out.anchored, false);
    assert.deepEqual(out.messages, ["not anchored to the workspace (HiveWorkspace -PROCESS-> Deploy Checklist failed: WRONG_TYPE: no such edge schema)"]);
    assert.deepEqual(f.edges, [{ edge: "PARENT_OF", source_ref_id: "p1", target_ref_id: "new-1" }]);
    assert.deepEqual(f.deleted, []);
    // The workspace node was found: it is provenance even though the edge failed.
    assert.deepEqual(accessedNodesOf(out)!.map((n) => n.ref_id), ["new-1", "p1", "w1"]);
  });

  it("the kind picks the anchor edge, hive's table; HAS_CONCEPT when omitted", async () => {
    assert.deepEqual(KIND_EDGE, { preference: "PREFERENCE", best_practice: "BEST_PRACTICE", gotcha: "GOTCHA", process: "PROCESS", general: "HAS_CONCEPT" });
    for (const kind of [...CONCEPT_KINDS, undefined]) {
      const f = fakeGraph({ nodes: [JOB], workspaces: WS });
      const out = await run({ ...INPUT, kind }, f.g);
      assert.equal(out.anchored, true, String(kind));
      assert.equal(f.edges.at(-1)!.edge, kind ? KIND_EDGE[kind] : "HAS_CONCEPT", String(kind));
    }
    // Not one of hive's kinds: refused by the schema, before the step runs.
    assert.throws(() => step.input.parse({ ...INPUT, kind: "tip" }));
  });

  it("repo prefixes the id with owner/repo and lands on the node", async () => {
    const f = fakeGraph({ nodes: [JOB], workspaces: WS });
    const out = await run({ ...INPUT, repo: "acme/api" }, f.g);
    assert.equal(out.id, "acme/api/deploy-checklist");
    assert.equal(f.writes[0]!.data["repo"], "acme/api");
    assert.equal(f.writes[0]!.data["id"], "acme/api/deploy-checklist");
    // Blank is none.
    const bare = fakeGraph({ nodes: [JOB], workspaces: WS });
    assert.equal((await run({ ...INPUT, repo: "  " }, bare.g)).id, "deploy-checklist");
    assert.ok(!("repo" in bare.writes[0]!.data));
  });

  it("the parent link failing rolls back a Concept this call created, never one a merge landed on", async () => {
    const fail = (e: string) => (e === "PARENT_OF" ? new Error("edge write failed") : undefined);
    const fresh = fakeGraph({ nodes: [JOB], workspaces: WS, failEdge: fail });
    const out = await run(INPUT, fresh.g);
    assert.equal(out, "concept/create failed — could not link Job -PARENT_OF-> Deploy Checklist: edge write failed; the new Concept was removed again");
    assert.deepEqual(fresh.deleted, ["new-1"]);
    assert.equal(fresh.bolt.length, 0, "no anchor after a failed parent link");

    const merged = fakeGraph({ nodes: [JOB, { ref_id: "c9", node_type: "Concept", properties: { name: "Deploy Checklist", docs: "OLD" } }], workspaces: WS, failEdge: fail });
    assert.equal(await run(INPUT, merged.g), "concept/create failed — could not link Job -PARENT_OF-> Deploy Checklist: edge write failed");
    assert.deepEqual(merged.deleted, []);
  });

  it("refuses an empty slug, empty docs, a graph without the Concept schema, and no graph at all, as error strings", async () => {
    const f = fakeGraph({ nodes: [JOB], workspaces: WS });
    assert.match(await run({ ...INPUT, name: "!!! ???" }, f.g), /^concept\/create failed — the name must contain a letter or a digit/);
    assert.match(await run({ ...INPUT, name: "   " }, f.g), /^concept\/create failed — name is required/);
    assert.match(await run({ ...INPUT, docs: "  \n" }, f.g), /^concept\/create failed — docs is required/);
    assert.deepEqual(f.writes, []);
    assert.match(await run(INPUT, fakeGraph({ schemas: ["HiveWorkspace"] }).g), /^concept\/create failed — no "Concept" schema in this graph/);
    assert.match(await run(INPUT, undefined), /^concept\/create failed — no graph on the services bag/);
    // A graph error on the write comes back with its code, in the same shape.
    const broken = fakeGraph({ nodes: [JOB], workspaces: WS });
    broken.g.nodes.write = async () => {
      throw Object.assign(new Error("unknown attribute: id"), { name: "GraphValidationError", code: "UNKNOWN_ATTRIBUTE" });
    };
    assert.equal(await run(INPUT, broken.g), "concept/create failed — UNKNOWN_ATTRIBUTE: unknown attribute: id");
  });

  it("the slug is gitree's: lowercase, runs of anything else become one dash, none at the ends", () => {
    assert.equal(conceptSlug("Deploy Checklist"), "deploy-checklist");
    assert.equal(conceptSlug("  Hello, World!  "), "hello-world");
    assert.equal(conceptSlug("Café-Über"), "caf-ber");
    assert.equal(conceptSlug("---"), "");
    assert.equal(conceptSlug("v2.0 API"), "v2-0-api");
  });

  it("the source is a seeded step: imports only strut, and passes strut's step lint", async () => {
    const src = await readFile(join(dirname(fileURLToPath(import.meta.url)), "steps", "concept-create.ts"), "utf-8");
    assert.deepEqual(lintStepSource(src), []);
    const imports = [...src.matchAll(/^import .* from "([^"]+)";?$/gm)].map((m) => m[1]);
    assert.deepEqual(imports, ["strut"]);
    assert.doesNotMatch(src, /openGraphBackend|process\.env|\bfetch\(/);
  });
});
