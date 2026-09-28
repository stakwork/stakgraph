import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, closeGraphBackends, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { parseConceptFile, planConceptSeed, seedConcepts, stampOf } from "../concept-seed.js";
import { BUILDER_CONCEPTS, BUILDER_ENTRY, builderSystem, renderBuilderSystem } from "../builder/system.js";
import { JANITOR_CONCEPTS, seedJanitorWorkflows } from "./seed.js";

const P = JANITOR_CONCEPTS.prefix;

describe("parseConceptFile", () => {
  it("name from the filename, description + parent from front matter, body as docs", () => {
    const text = "---\ndescription: One line.\nparent: Janitor\n---\nThe mandate.\n\nMore.\n";
    const c = parseConceptFile("Legal Overfit.md", text, P);
    assert.deepEqual(c, {
      name: "Legal Overfit",
      description: "One line.",
      parent: "Janitor",
      docs: "The mandate.\n\nMore.",
      stamp: stampOf("Legal Overfit.md", text, P),
    });
    assert.match(c.stamp, /^lab\/janitor\/concepts\/Legal Overfit\.md@[0-9a-f]{12}$/);
  });

  it("no front matter, empty body: just a name and a stamp", () => {
    assert.deepEqual(parseConceptFile("X.md", "", P), { name: "X", stamp: stampOf("X.md", "", P) });
    assert.deepEqual(parseConceptFile("X.md", "body only\n", P), { name: "X", docs: "body only", stamp: stampOf("X.md", "body only\n", P) });
  });

  it("the stamp follows the file's bytes, and its set", () => {
    assert.notEqual(stampOf("A.md", "a", P), stampOf("A.md", "b", P));
    assert.notEqual(stampOf("A.md", "a", P), stampOf("B.md", "a", P));
    assert.equal(stampOf("A.md", "a", P), stampOf("A.md", "a", P));
    assert.notEqual(stampOf("A.md", "a", P), stampOf("A.md", "a", BUILDER_CONCEPTS.prefix));
  });

  it("the committed tree: Workflow Builder > Janitor > the stock mandate", async () => {
    const read = (set: { dir: string; prefix: string }, file: string) =>
      readFile(join(set.dir, file), "utf-8").then((text) => parseConceptFile(file, text, set.prefix));
    const entry = await read(BUILDER_CONCEPTS, `${BUILDER_ENTRY}.md`);
    assert.equal(entry.name, "Workflow Builder");
    assert.equal(entry.parent, undefined);
    assert.match(entry.stamp, /^lab\/builder\/concepts\/Workflow Builder\.md@[0-9a-f]{12}$/);
    assert.ok(entry.docs!.startsWith(entry.description!));
    assert.doesNotMatch(entry.docs!, /janitor/i, "the entry page knows no kind by name");

    const c = await read(JANITOR_CONCEPTS, "Janitor.md");
    assert.equal(c.name, "Janitor");
    assert.equal(c.parent, "Workflow Builder");
    assert.match(c.stamp, /^lab\/janitor\/concepts\/Janitor\.md@/, "the file did not move: its nodes stay ours");
    assert.equal(c.description, "Automated agents for cleaning up code, concept trees, or other data.");
    assert.ok(c.docs!.startsWith(c.description!));
    // The convention lives in its docs: the engine, and the three steps to add one.
    for (const needle of [/graph-janitor/, /graph\/create-node/, /graph\/create-triplet/, /PARENT_OF/, /automation/]) assert.match(c.docs!, needle);
    const kid = await read(JANITOR_CONCEPTS, "Overfit Concept Janitor.md");
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
describe("seedConcepts (live)", { skip: !URI }, () => {
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
    await seedConcepts(ws, [{ dir, prefix: P }]);
    const r = await node(root);
    const k = await node(kid);
    assert.ok(r && k);
    assert.equal(r.description, "The root.");
    assert.equal(k.docs, "Kid docs v1.");
    assert.match(String(k.usid), /^lab\/janitor\/concepts\/Kid .+\.md@[0-9a-f]{12}$/);
    assert.equal(await parentEdges(), 1);

    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal((await node(kid))!.ref_id, k.ref_id);
    assert.equal(await parentEdges(), 1);
    // No workspace node in this graph yet: nothing to anchor to, no error.
    assert.equal(await anchors(root), 0);
  });

  it("a graph-side edit survives an unchanged template; a changed template wins but keeps is_muted", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Rewritten in Learn.', n.is_muted = true`, { name: kid });
    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal((await node(kid))!.docs, "Rewritten in Learn.");

    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v2.\n`);
    await seedConcepts(ws, [{ dir, prefix: P }]);
    const k = await node(kid);
    assert.equal(k!.docs, "Kid docs v2.");
    assert.equal(k!.muted, true);
    assert.equal(await parentEdges(), 1);
  });

  it("a node that lost its stamp is never touched again", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Mine now.' REMOVE n.unique_source_id`, { name: kid });
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v3.\n`);
    await seedConcepts(ws, [{ dir, prefix: P }]);
    const k = await node(kid);
    assert.equal(k!.docs, "Mine now.");
    assert.equal(k!.usid, null);
  });

  it("anchors an owned root to the one HiveWorkspace node, idempotently; never a child; never with two", async () => {
    const g = ws.graph!;
    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: wsId, name: "T", slug: `t-${tag}` } }, "create", { namespace: g.cfg.namespace });
    await seedConcepts(ws, [{ dir, prefix: P }]); // root is "keep" (unchanged file) → still anchored
    assert.equal(await anchors(root), 1);
    assert.equal(await anchors(kid), 0);
    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal(await anchors(root), 1);

    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: `${wsId}-2`, name: "T2" } }, "create", { namespace: g.cfg.namespace });
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root, v2.\n---\nRoot docs v2.\n");
    await seedConcepts(ws, [{ dir, prefix: P }]); // updated, but two workspace nodes → no guess
    assert.equal((await node(root))!.description, "The root, v2.");
    assert.equal(await anchors(root), 1);
  });

  it("a parent may come from another set; a root that gains a parent keeps the anchor it has", async () => {
    const top = `Top ${tag}`;
    const other = await mkdtemp(join(tmpdir(), "concept-seed-other-"));
    try {
      await writeFile(join(other, `${top}.md`), "---\ndescription: Above the root.\n---\nTop docs.\n");
      await writeFile(join(dir, `${root}.md`), `---\ndescription: The root, v3.\nparent: ${top}\n---\nRoot docs v3.\n`);
      // The child's set first: edges are written after every node, so order is free.
      await seedConcepts(ws, [{ dir, prefix: P }, { dir: other, prefix: "lab/other/concepts/" }]);
      const g = ws.graph!;
      const rows = await g.bolt.run(`MATCH (p:Concept {name: $p})-[e:PARENT_OF]->(c:Concept {name: $c}) RETURN count(e) AS n`, { p: top, c: root });
      assert.equal(Number(rows[0]?.n ?? 0), 1);
      assert.match(String((await node(top))!.usid), /^lab\/other\/concepts\//);
      assert.equal((await node(root))!.description, "The root, v3.");
      assert.equal(await anchors(root), 1);
    } finally {
      await ws.graph!.bolt.run(`MATCH (n:Concept {name: $name}) DETACH DELETE n`, { name: top });
      await rm(other, { recursive: true, force: true });
    }
  });
});

// The engine, offline: real if / pack / exec / artifacts-dir over the seeded
// YAML, with graph-get and the agent swapped for fakes — the wiring (the
// three reads by key, the under-Janitor guard, what the agent is handed).
describe("graph-janitor workflow (offline: fake graph + agent)", () => {
  const MANDATE = { ref_id: "r-overfit", node_type: "Concept", name: "Overfit Concept Janitor", properties: { docs: "THE MANDATE TEXT" }, edges: {} };
  const LAW = { ref_id: "r-law", node_type: "Concept", name: "Law", properties: { description: "Legal knowledge." }, edges: {} };
  const JANITOR = { ref_id: "r-janitor", node_type: "Concept", name: "Janitor", properties: {}, edges: {} };
  const kid = { ref_id: MANDATE.ref_id, node_type: "Concept", name: MANDATE.name, description: "Flags overfit Concepts." };
  let truncated = false;
  const gets: Record<string, any>[] = [];
  const agentCalls: Record<string, any>[] = [];
  const fake = (type: string, run: (cfg: any) => unknown) =>
    defineStep({ type, input: z.object({}).passthrough(), output: z.any(), async run(cfg) { return run(cfg); } });
  const fakes = {
    "graph/graph-get": fake("graph/graph-get", (cfg) => {
      gets.push(cfg);
      // Like the real step: children only when asked for, a sentence when there is no such node.
      if (cfg.name === "Janitor") return { ...JANITOR, ...(cfg.children ? { children: [kid], ...(truncated ? { children_truncated: true } : {}) } : {}) };
      if (cfg.name === MANDATE.name) return MANDATE;
      if (cfg.name === "Law") return LAW;
      return `node not found: Concept "${cfg.name}"`;
    }),
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

  it("is seeded under graph-maintenance with the two declared inputs; its description is no how-to", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "graph-janitor");
    assert.equal(entry?.category, "graph-maintenance");
    const flow = (await ws.getWorkflow("graph-janitor")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock ?? {}).sort(), ["concept", "start"]);
    // How to ADD a janitor is the Janitor Concept's to say (its docs), not the workflow's.
    for (const d of [entry?.description ?? "", String(flow.description ?? "")]) assert.doesNotMatch(d, /create-node|create-triplet/);
  });

  it("reads the three names by key, guards the mandate under Janitor, hands the agent the mandate docs and read tools only", async () => {
    gets.length = 0;
    const res = await run({ concept: MANDATE.name, start: "Law" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(
      [...gets].sort((a, b) => a.name.localeCompare(b.name)),
      [
        { node_type: "Concept", name: "Janitor", children: "PARENT_OF" },
        { node_type: "Concept", name: "Law" },
        { node_type: "Concept", name: MANDATE.name },
      ],
      "three reads by key: no search, no ref_id",
    );
    const out = res.output as Record<string, any>;
    assert.equal(out.mandate, MANDATE.name);
    assert.equal(out.start, "Law");
    assert.equal(out.start_ref_id, LAW.ref_id);
    assert.equal(out.summary, "clean");
    assert.deepEqual(out.findings, []);
    assert.match(out.report_json, /^\/artifacts\/\d+\/cleanup-report\.json$/);
    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /THE MANDATE TEXT/);
    assert.match(cfg.prompt, /Law — Legal knowledge\./);
    assert.match(cfg.prompt, new RegExp(LAW.ref_id));
    // An empty cwd gets no preamble from the agent step: the prompt must name it, or report.md lands elsewhere.
    assert.ok(cfg.prompt.includes(cfg.cwd), "the prompt names the working dir");
    assert.match(cfg.system, /DATA read from a graph node/);
    assert.ok(Array.isArray(cfg.agentTools) && cfg.agentTools.length > 0);
    for (const t of cfg.agentTools) assert.doesNotMatch(t, /create|edit|register|project|\*/, t);
  });

  it("refuses a mandate that is not a child of Janitor: not_a_janitor, and the agent never runs", async () => {
    const n = agentCalls.length;
    const res = await run({ concept: "Law", start: "Law" });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /not_a_janitor: 'Law' is not a PARENT_OF child of 'Janitor'/);
    assert.doesNotMatch(JSON.stringify(res.error), /first 50 children/);
    assert.equal(agentCalls.length, n);
  });

  it("says so when Janitor has more children than were listed", async () => {
    truncated = true;
    try {
      const res = await run({ concept: "Law", start: "Law" });
      assert.match(JSON.stringify(res.error), /not_a_janitor: 'Law'.*only its first 50 children by name were checked/);
    } finally {
      truncated = false;
    }
  });

  it("fails a name no Concept has: not_found, and the agent never runs", async () => {
    const n = agentCalls.length;
    for (const input of [{ concept: MANDATE.name, start: "Nope" }, { concept: "Nope", start: "Law" }]) {
      const res = await run(input);
      assert.equal(res.status, "error");
      assert.match(JSON.stringify(res.error), /not_found: /);
    }
    assert.equal(agentCalls.length, n);
  });
});

// The builder's section of the system prompt (strut's chatSystem hook).
describe("builder system section", () => {
  it("renders the entry page: the notes, then its kinds as a table of contents and how to open one", () => {
    const text = renderBuilderSystem({
      name: "Workflow Builder",
      docs: "What is built here.",
      children: [{ name: "Janitor", description: "Cleans things up." }, { name: "Bare" }],
    });
    assert.match(text, /^Workspace knowledge — the "Workflow Builder" Concept/);
    assert.match(text, /never replace them/);
    assert.match(text, /\n\nWhat is built here\.\n\nKinds recorded under it:\n- Janitor — Cleans things up\.\n- Bare\n/);
    assert.match(text, /run_step\("graph\/graph-get", \{ node_type: "Concept", name: "<name>", children: "PARENT_OF" \}\)/);
    assert.doesNotMatch(text, /and more/);
  });

  it("caps the notes, says when the list is cut, and has no table of contents without children", () => {
    const long = renderBuilderSystem({ name: "W", docs: "x".repeat(7000), children: [{ name: "A" }], more: true });
    assert.match(long, /x{6000}…\n/);
    assert.doesNotMatch(long, /x{6001}/);
    assert.match(long, /- … and more/);
    const bare = renderBuilderSystem({ name: "W", children: [] });
    assert.match(bare, /\(no docs\)$/);
    assert.doesNotMatch(bare, /Kinds recorded|run_step/);
  });

  it("is nothing on a filesystem workspace, and no step runs", async () => {
    const dir = await mkdtemp(join(tmpdir(), "builder-fs-"));
    try {
      const getRegistry = () => assert.fail("no graph, no read");
      assert.equal(await builderSystem({ workspace: new WorkspaceManager(dir), getRegistry, services: {} }), undefined);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

// End to end on a throwaway Neo4j: the committed tree seeded for real, REAL
// graph/graph-get, only the agent faked — the reads by key and the PARENT_OF
// guard on real nodes, and the builder's section read off the same graph.
describe("graph-janitor workflow + builder section (live graph, fake agent)", { skip: !URI }, () => {
  const NAMES = ["Workflow Builder", "Janitor", "Overfit Concept Janitor", "Law"];
  const agentCalls: Record<string, any>[] = [];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  let registry: Awaited<ReturnType<typeof buildRegistry>>["registry"];
  let services: Record<string, unknown>;
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
    await ws.graph!.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await seedArtifactSteps(ws);
    await seedJanitorWorkflows(ws);
    await seedConcepts(ws, [BUILDER_CONCEPTS, JANITOR_CONCEPTS]); // the committed tree, for real
    await ws.graph!.nodes.write({ type: "Concept", data: { name: "Law", description: "A domain root." } }, "create", { namespace: ws.graph!.cfg.namespace });
    const flow = await ws.getWorkflow("graph-janitor");
    registry = (await buildRegistry(await ws.materializeCustomSteps())).registry;
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
    services = { ...standardServices({ secretsSource: env, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
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

  it("the tree is Workflow Builder > Janitor > the stock mandate", async () => {
    const rows = await ws.graph!.bolt.run(
      `MATCH (a:Concept {name: 'Workflow Builder'})-[:PARENT_OF]->(b:Concept {name: 'Janitor'})-[:PARENT_OF]->(c:Concept {name: 'Overfit Concept Janitor'}) RETURN count(*) AS n`,
    );
    assert.equal(Number(rows[0]!.n), 1);
  });

  it("the builder's section is the Workflow Builder page: its docs, and Janitor in its table of contents", async () => {
    const text = await builderSystem({ workspace: ws, getRegistry: () => registry, services });
    assert.ok(text, "a section");
    assert.match(text!, /Each child Concept of this node \(PARENT_OF from here\) is one KIND of thing/);
    assert.match(text!, /\n- Janitor — Automated agents for cleaning up code, concept trees, or other data\.\n/);
    // One level only: a kind's instances and its convention are on ITS page.
    assert.doesNotMatch(text!, /Overfit Concept Janitor|create-triplet/);
  });

  it("the page the builder opens next carries the convention and the mandates", async () => {
    const def = registry["graph/graph-get"] as unknown as { input: { parse(v: unknown): unknown }; run(cfg: unknown, ctx: unknown): Promise<any> };
    const page = await def.run(def.input.parse({ node_type: "Concept", name: "Janitor", children: "PARENT_OF" }), { runId: "t", path: "t", scope: {}, emit: async () => {}, services });
    assert.match(page.properties.docs, /graph\/create-triplet/);
    assert.deepEqual(page.children.map((c: any) => c.name), ["Overfit Concept Janitor"]);
    assert.match(page.children[0].description, /^Flags Concepts overfit/);
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

  it("refuses Law as a mandate: it is a Concept, but not under Janitor; and Janitor itself: it is under Workflow Builder", async () => {
    const n = agentCalls.length;
    for (const concept of ["Law", "Janitor"]) {
      const res = await run({ concept, start: "Law" });
      assert.equal(res.status, "error");
      assert.match(JSON.stringify(res.error), new RegExp(`not_a_janitor: '${concept}'`));
    }
    assert.equal(agentCalls.length, n);
  });
});
