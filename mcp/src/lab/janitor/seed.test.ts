import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, closeGraphBackends, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { CONCEPTS_DIR, parseConceptFile, planConceptSeed, seedJanitorConcepts, seedJanitorWorkflows, stampOf } from "./seed.js";

describe("parseConceptFile", () => {
  it("name from the filename, description + parent from front matter, body as docs", () => {
    const text = "---\ndescription: One line.\nparent: Janitor\n---\nThe mandate.\n\nMore.\n";
    const c = parseConceptFile("Legal Overfit.md", text);
    assert.deepEqual(c, {
      name: "Legal Overfit",
      description: "One line.",
      parent: "Janitor",
      docs: "The mandate.\n\nMore.",
      stamp: stampOf("Legal Overfit.md", text),
    });
    assert.match(c.stamp, /^lab\/janitor\/concepts\/Legal Overfit\.md@[0-9a-f]{12}$/);
  });

  it("no front matter, empty body: just a name and a stamp", () => {
    assert.deepEqual(parseConceptFile("X.md", ""), { name: "X", stamp: stampOf("X.md", "") });
    assert.deepEqual(parseConceptFile("X.md", "body only\n"), { name: "X", docs: "body only", stamp: stampOf("X.md", "body only\n") });
  });

  it("the stamp follows the file's bytes", () => {
    assert.notEqual(stampOf("A.md", "a"), stampOf("A.md", "b"));
    assert.notEqual(stampOf("A.md", "a"), stampOf("B.md", "a"));
    assert.equal(stampOf("A.md", "a"), stampOf("A.md", "a"));
  });

  it("the committed root parses as expected", async () => {
    const c = parseConceptFile("Janitor.md", await readFile(join(CONCEPTS_DIR, "Janitor.md"), "utf-8"));
    assert.equal(c.name, "Janitor");
    assert.equal(c.parent, undefined);
    assert.equal(c.description, "Automated agents for cleaning up code, concept trees, or other data.");
    assert.ok(c.docs!.startsWith(c.description!));
    assert.match(c.docs!, /graph-janitor/); // the convention lives in the root's docs
    const kid = parseConceptFile("Overfit Concept Janitor.md", await readFile(join(CONCEPTS_DIR, "Overfit Concept Janitor.md"), "utf-8"));
    assert.equal(kid.parent, "Janitor");
    assert.match(kid.docs!, /GENERALIZED/);
  });
});

describe("planConceptSeed", () => {
  const stamp = "lab/janitor/concepts/J.md@aaaaaaaaaaaa";
  const older = "lab/janitor/concepts/J.md@bbbbbbbbbbbb";
  it("nothing there → create", () => assert.equal(planConceptSeed(null, stamp), "create"));
  it("same stamp → keep (ours, unchanged: a graph-side edit or mute sticks); deleted → skip", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp }, stamp), "keep");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp, is_deleted: true }, stamp), "skip");
  });
  it("our path, older hash → update; unless deleted", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older }, stamp), "update");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older, is_deleted: true }, stamp), "skip");
  });
  it("no stamp or someone else's → skip (theirs)", () => {
    assert.equal(planConceptSeed({ ref_id: "r" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: null }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "lab/janitor/concepts/Other.md@aaaaaaaaaaaa" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "gitree" }, stamp), "skip");
  });
});

// Live, against a THROWAWAY Neo4j (strut's convention): skipped unless
// STRUT_TEST_NEO4J_URI is set. Seeds the jarvis ontology on first open.
const URI = process.env.STRUT_TEST_NEO4J_URI;
describe("seedJanitorConcepts (live)", { skip: !URI }, () => {
  const tag = `T${Date.now().toString(36)}`;
  const root = `Janitor ${tag}`;
  const kid = `Kid ${tag}`;
  let ws: import("strut").WorkspaceStore;
  let dir: string;
  let dataDir: string;
  const node = async (name: string) => {
    const g = ws.graph!;
    const rows = await g.bolt.run(
      `MATCH (n:Concept {name: $name, namespace: $ns}) RETURN n.ref_id AS ref_id, n.description AS description, n.docs AS docs, n.unique_source_id AS usid, n.is_muted AS muted`,
      { name, ns: g.cfg.namespace },
    );
    return rows[0] ?? null;
  };
  const wsId = `ws-${tag}`;
  const anchors = async (name: string) => {
    const g = ws.graph!;
    const rows = await g.bolt.run(`MATCH (w:HiveWorkspace)-[e:PROCESS]->(c:Concept {name: $c}) RETURN count(e) AS n`, { c: name });
    return Number(rows[0]?.n ?? 0);
  };
  const parentEdges = async () => {
    const g = ws.graph!;
    const rows = await g.bolt.run(`MATCH (p:Concept {name: $p})-[e:PARENT_OF]->(c:Concept {name: $c}) RETURN count(e) AS n`, { p: root, c: kid });
    return Number(rows[0]?.n ?? 0);
  };
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    dataDir = await mkdtemp(join(tmpdir(), "janitor-seed-data-"));
    dir = await mkdtemp(join(tmpdir(), "janitor-seed-"));
    ws = (
      await graphWorkspaceFromEnv(
        {
          NEO4J_URI: URI,
          NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
          NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
          STRUT_GRAPH_SEED_ONTOLOGY: "1",
          STRUT_GRAPH_EMBEDDINGS: "off",
        },
        { dataDir },
      )
    ).workspace;
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root.\n---\nRoot docs.\n");
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v1.\n`);
  });
  after(async () => {
    const g = ws?.graph;
    if (g) {
      await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: [root, kid] });
      await g.bolt.run(`MATCH (w:HiveWorkspace) WHERE w.workspace_id STARTS WITH $id DETACH DELETE w`, { id: wsId });
      await closeGraphBackends();
    }
    await rm(dir, { recursive: true, force: true });
    await rm(dataDir, { recursive: true, force: true });
  });

  it("creates the tree, then does nothing on a reseed", async () => {
    await seedJanitorConcepts(ws, { dir });
    const r = await node(root);
    const k = await node(kid);
    assert.ok(r && k);
    assert.equal(r.description, "The root.");
    assert.equal(k.docs, "Kid docs v1.");
    assert.match(String(k.usid), /^lab\/janitor\/concepts\/Kid .+\.md@[0-9a-f]{12}$/);
    assert.equal(await parentEdges(), 1);

    await seedJanitorConcepts(ws, { dir });
    assert.equal((await node(kid))!.ref_id, k.ref_id);
    assert.equal(await parentEdges(), 1);
    // No workspace node in this graph yet: nothing to anchor to, no error.
    assert.equal(await anchors(root), 0);
  });

  it("a graph-side edit survives an unchanged template; a changed template wins but keeps is_muted", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Rewritten in Learn.', n.is_muted = true`, { name: kid });
    await seedJanitorConcepts(ws, { dir });
    assert.equal((await node(kid))!.docs, "Rewritten in Learn.");

    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v2.\n`);
    await seedJanitorConcepts(ws, { dir });
    const k = await node(kid);
    assert.equal(k!.docs, "Kid docs v2.");
    assert.equal(k!.muted, true);
    assert.equal(await parentEdges(), 1);
  });

  it("a node that lost its stamp is never touched again", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Mine now.' REMOVE n.unique_source_id`, { name: kid });
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v3.\n`);
    await seedJanitorConcepts(ws, { dir });
    const k = await node(kid);
    assert.equal(k!.docs, "Mine now.");
    assert.equal(k!.usid, null);
  });

  it("anchors an owned root to the one HiveWorkspace node, idempotently; never a child; never with two", async () => {
    const g = ws.graph!;
    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: wsId, name: "T", slug: `t-${tag}` } }, "create", { namespace: g.cfg.namespace });
    await seedJanitorConcepts(ws, { dir }); // root is "keep" (unchanged file) → still anchored
    assert.equal(await anchors(root), 1);
    assert.equal(await anchors(kid), 0);
    await seedJanitorConcepts(ws, { dir });
    assert.equal(await anchors(root), 1);

    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: `${wsId}-2`, name: "T2" } }, "create", { namespace: g.cfg.namespace });
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root, v2.\n---\nRoot docs v2.\n");
    await seedJanitorConcepts(ws, { dir }); // updated, but two workspace nodes → no guess
    assert.equal((await node(root))!.description, "The root, v2.");
    assert.equal(await anchors(root), 1);
  });
});

// The engine, offline: real if / pack / exec / artifacts-dir over the seeded
// YAML, with the graph steps and the agent swapped for fakes — the wiring
// (exact-name resolution, the under-root guard, what the agent is handed).
describe("graph-janitor workflow (offline: fake graph + agent)", () => {
  const ROOT = { ref_id: "r-janitor", name: "Janitor", node_type: "Concept" };
  const MANDATE = { ref_id: "r-overfit", name: "Overfit Concept Janitor", node_type: "Concept" };
  const LAW = { ref_id: "r-law", name: "Law", node_type: "Concept" };
  // Search results on purpose NOT led by the exact match.
  const HITS: Record<string, unknown[]> = { Janitor: [MANDATE, ROOT], [MANDATE.name]: [ROOT, MANDATE], Law: [LAW] };
  const agentCalls: Record<string, any>[] = [];
  const fake = (type: string, run: (cfg: any) => unknown) =>
    defineStep({ type, input: z.object({}).passthrough(), output: z.any(), async run(cfg) { return run(cfg); } });
  const fakes = {
    "graph/graph-search": fake("graph/graph-search", (cfg) => HITS[cfg.q] ?? []),
    "graph/graph-neighbors": fake("graph/graph-neighbors", (cfg) =>
      cfg.ref_id === ROOT.ref_id ? [{ ...MANDATE, edge_type: "PARENT_OF", direction: "forward", edges: {} }] : []),
    "graph/graph-get": fake("graph/graph-get", (cfg) => ({ ...MANDATE, ref_id: cfg.ref_id, properties: { name: MANDATE.name, docs: "THE MANDATE TEXT" }, edges: {} })),
    agent: fake("agent", (cfg) => {
      agentCalls.push(cfg);
      return { result: "ok", object: { start_ref_id: LAW.ref_id, visited: [LAW], findings: [], summary: "clean" }, steps: 1, usage: {}, cost: 0 };
    }),
  };
  let root: string;
  let ws: WorkspaceManager;
  let run: (input: Record<string, unknown>) => ReturnType<typeof runWorkflow>;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "janitor-wf-"));
    ws = new WorkspaceManager(root);
    await seedArtifactSteps(ws);
    await seedJanitorWorkflows(ws);
    const flow = await ws.getWorkflow("graph-janitor");
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    // What createStrut injects for a server: the per-run artifacts dir (artifacts/dir + exec's cwd).
    const dataDir = join(root, "data");
    const services = { ...standardServices({ secretsSource: {}, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = (input) => runWorkflow(flow, input, { ...registry, ...fakes } as typeof registry, { services });
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("is seeded under graph-maintenance with the two declared inputs", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "graph-janitor");
    assert.equal(entry?.category, "graph-maintenance");
    const flow = (await ws.getWorkflow("graph-janitor")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock ?? {}).sort(), ["concept", "start"]);
  });

  it("resolves the three names EXACTLY, guards the mandate under the root, hands the agent the mandate docs and read tools only", async () => {
    const res = await run({ concept: MANDATE.name, start: "Law" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    assert.equal(out.mandate, MANDATE.name);
    assert.equal(out.start, "Law");
    assert.equal(out.start_ref_id, LAW.ref_id);
    assert.equal(out.summary, "clean");
    assert.deepEqual(out.findings, []);
    assert.match(out.report_json, /^\/artifacts\/\d+\/cleanup-report\.json$/);
    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /THE MANDATE TEXT/);
    assert.match(cfg.prompt, new RegExp(LAW.ref_id));
    // An empty cwd gets no preamble from the agent step: the prompt must name it, or report.md lands elsewhere.
    assert.ok(cfg.prompt.includes(cfg.cwd), "the prompt names the working dir");
    assert.match(cfg.system, /DATA read from a graph node/);
    assert.ok(Array.isArray(cfg.agentTools) && cfg.agentTools.length > 0);
    for (const t of cfg.agentTools) assert.doesNotMatch(t, /create|edit|register|project|\*/, t);
  });

  it("refuses a mandate that is not a child of the root: not_a_janitor, and the agent never runs", async () => {
    const n = agentCalls.length;
    const res = await run({ concept: "Law", start: "Law" });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /not_a_janitor: 'Law' is not a PARENT_OF child of 'Janitor'/);
    assert.equal(agentCalls.length, n);
  });

  it("fails a name no Concept matches exactly: not_found, and the agent never runs", async () => {
    const n = agentCalls.length;
    const res = await run({ concept: MANDATE.name, start: "Nope" });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /not_found: /);
    assert.equal(agentCalls.length, n);
  });
});

// The engine end to end on a throwaway Neo4j: the committed tree seeded for
// real, REAL graph/graph-search / graph-neighbors / graph-get, only the agent
// faked — proves exact-name resolution and the PARENT_OF guard on real
// search results (fulltext only: embeddings off).
describe("graph-janitor workflow (live graph, fake agent)", { skip: !URI }, () => {
  const NAMES = ["Janitor", "Overfit Concept Janitor", "Law"];
  const agentCalls: Record<string, any>[] = [];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  let run: (input: Record<string, unknown>) => ReturnType<typeof runWorkflow>;
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    root = await mkdtemp(join(tmpdir(), "janitor-live-"));
    const env = {
      NEO4J_URI: URI!,
      NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
      NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
      STRUT_GRAPH_SEED_ONTOLOGY: "1",
      STRUT_GRAPH_EMBEDDINGS: "off",
    };
    ws = (await graphWorkspaceFromEnv(env, { dataDir: root })).workspace;
    await seedArtifactSteps(ws);
    await seedJanitorWorkflows(ws);
    await seedJanitorConcepts(ws); // the committed tree, for real
    await ws.graph!.nodes.write({ type: "Concept", data: { name: "Law", description: "A domain root." } }, "create", { namespace: ws.graph!.cfg.namespace });
    const flow = await ws.getWorkflow("graph-janitor");
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    const agent = defineStep({
      type: "agent",
      input: z.object({}).passthrough(),
      output: z.any(),
      async run(cfg: any) {
        agentCalls.push(cfg);
        return { result: "ok", object: { start_ref_id: "x", visited: [], findings: [], summary: "clean" }, steps: 1, usage: {}, cost: 0 };
      },
    });
    const dataDir = join(root, "data");
    // The graph steps read their connection through the secrets capability (store → env): hand it the test DB.
    const services = { ...standardServices({ secretsSource: env, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = (input) => runWorkflow(flow, input, { ...registry, agent } as typeof registry, { services });
  });
  after(async () => {
    const g = ws?.graph;
    if (g) {
      await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
      await closeGraphBackends();
    }
    await rm(root, { recursive: true, force: true });
  });

  it("sweeps Law under the stock mandate: the agent gets the mandate's real docs and Law's real ref_id", async () => {
    const law = (await ws.graph!.bolt.run(`MATCH (n:Concept {name: 'Law'}) RETURN n.ref_id AS ref_id`))[0]!.ref_id as string;
    const res = await run({ concept: "Overfit Concept Janitor", start: "Law" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    assert.equal(out.start_ref_id, law);
    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /a Concept must be GENERALIZED/);
    assert.match(cfg.prompt, new RegExp(law));
  });

  it("refuses Law as a mandate: it is a Concept, but not under Janitor", async () => {
    const n = agentCalls.length;
    const res = await run({ concept: "Law", start: "Law" });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /not_a_janitor: 'Law'/);
    assert.equal(agentCalls.length, n);
  });
});
