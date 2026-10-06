import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, closeGraphBackends, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { parseConceptFile, planConceptSeed, seedConcepts, stampOf } from "../concept-seed.js";
import { CODE_CONCEPTS } from "../code/seed.js";
import { BUILDER_CONCEPTS, BUILDER_ENTRY, builderSystem, renderBuilderSystem } from "../builder/system.js";
import { JANITOR_CONCEPTS, JANITOR_FIXTURES, seedJanitorWorkflows } from "./seed.js";

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

  const read = (set: { dir: string; prefix: string }, file: string) =>
    readFile(join(set.dir, file), "utf-8").then((text) => parseConceptFile(file, text, set.prefix));

  it("the committed tree: Workflow Builder > Janitor, and no mandate", async () => {
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

    // The entry page tells the job agent it is read for what to RUN, naming no kind.
    assert.match(entry.docs!, /job agent/);
    assert.doesNotMatch(entry.docs!, /code change|code-change/i, "the entry page knows no kind by name");

    // Code Change: the kind page the job agent follows to code-change-pr (lab/code).
    const cc = await read(CODE_CONCEPTS, "Code Change.md");
    assert.equal(cc.name, "Code Change");
    assert.equal(cc.parent, "Workflow Builder");
    assert.match(cc.stamp, /^lab\/code\/concepts\/Code Change\.md@[0-9a-f]{12}$/);
    assert.equal(cc.description, "A change to a repository's source code, delivered as a pull request.");
    assert.ok(cc.docs!.startsWith(cc.description!));
    // The convention: the workflow and how to read it, the follow-up rule, the credential failure, the artifact, how to extend.
    for (const needle of [/code-change-pr/, /meta\/get-workflow/, /same `branch`/, /no_push_permission:/, /pull_request/, /graph\/create-node/, /graph\/create-triplet/, /PARENT_OF/]) assert.match(cc.docs!, needle);
    assert.doesNotMatch(cc.docs!, /pod-pr|hive\//, "workspace-general: no one swarm's workflow or step by name");

    // What counts as dirty is each workspace's to say: the seed ships the kind, never a mandate.
    assert.deepEqual(await readdir(JANITOR_CONCEPTS.dir), ["Janitor.md"]);
    assert.deepEqual(JANITOR_CONCEPTS.retired, ["Overfit Concept Janitor.md"]);
    const fixture = await read(JANITOR_FIXTURES, "Overfit Concept Janitor.md");
    assert.equal(fixture.parent, "Janitor");
    assert.match(fixture.stamp, /^lab\/janitor\/fixtures\//, "a fixture never carries a seed's stamp");
  });

  it("what ships to every workspace says nothing about one workspace's domain", async () => {
    const domain = /\b(law|legal|rubric|evals?|benchmark|exam|grading|overfit)\b/i;
    for (const set of [BUILDER_CONCEPTS, JANITOR_CONCEPTS]) {
      for (const file of await readdir(set.dir)) {
        const c = await read(set, file);
        for (const text of [c.name, c.description ?? "", c.docs ?? ""]) assert.doesNotMatch(text, domain, `${file}`);
      }
    }
    // The engine, whole: its descriptions, its prompts, its output schema.
    const text = await readFile(join(JANITOR_CONCEPTS.dir, "..", "workflows", "graph-janitor.yaml"), "utf-8");
    assert.doesNotMatch(text, domain);
    // What kinds of problem there are is the mandate's to say: a label, not a list the engine knows.
    const flow = (await import("js-yaml")).load(text) as any;
    const issue = flow.steps.find((s: any) => s.id === "janitor").config.schema.properties.findings.items.properties.issue;
    assert.equal(issue.type, "string");
    assert.equal(issue.enum, undefined);
    assert.match(flow.steps.find((s: any) => s.id === "janitor").config.prompt, /the mandate's own label/);
    for (const needle of [/LABEL/, /`issue`/]) assert.match((await read(JANITOR_CONCEPTS, "Janitor.md")).docs!, needle);
    assert.match((await read(JANITOR_FIXTURES, "Overfit Concept Janitor.md")).docs!, /Label such a finding `overfit`/);
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
  const leaf = `Leaf ${tag}`;
  const late = `Late ${tag}`;
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
  const edge = async (p: string, c: string) => {
    const g = ws.graph!;
    const rows = await g.bolt.run(`MATCH (p:Concept {name: $p})-[e:PARENT_OF]->(c:Concept {name: $c}) RETURN count(e) AS n`, { p, c });
    return Number(rows[0]?.n ?? 0);
  };
  const parentEdges = () => edge(root, kid);
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
      await g.bolt.run(`MATCH (n:Concept) WHERE n.name ENDS WITH $tag DETACH DELETE n`, { tag: ` ${tag}` });
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

  it("anchors EVERY owned Concept to the one HiveWorkspace node, idempotently; never someone else's; never with two", async () => {
    const g = ws.graph!;
    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: wsId, name: "T", slug: `t-${tag}` } }, "create", { namespace: g.cfg.namespace });
    await writeFile(join(dir, `${leaf}.md`), `---\ndescription: A leaf.\nparent: ${root}\n---\nLeaf docs.\n`);
    await seedConcepts(ws, [{ dir, prefix: P }]); // root is "keep" (unchanged file) → anchored all the same
    assert.equal(await anchors(root), 1);
    assert.equal(await anchors(leaf), 1, "a parent does not matter");
    assert.equal(await anchors(kid), 0, "it lost its stamp: theirs");
    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal(await anchors(root), 1);
    assert.equal(await anchors(leaf), 1);

    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: `${wsId}-2`, name: "T2" } }, "create", { namespace: g.cfg.namespace });
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root, v2.\n---\nRoot docs v2.\n");
    await writeFile(join(dir, `${late}.md`), "---\ndescription: Seeded late.\n---\nLate docs.\n");
    await seedConcepts(ws, [{ dir, prefix: P }]); // two workspace nodes → no guess
    assert.equal((await node(root))!.description, "The root, v2.");
    assert.equal(await anchors(late), 0);
    assert.equal(await anchors(root), 1, "what is there stays");
    await g.bolt.run(`MATCH (w:HiveWorkspace {workspace_id: $id}) DETACH DELETE w`, { id: `${wsId}-2` });
  });

  // The seeded graph follows the files, not what was seeded before.
  it("PARENT_OF follows the files: a root gains a parent (from another set), a parent changes, a hand-made edge goes", async () => {
    const g = ws.graph!;
    const top = `Top ${tag}`;
    const other = await mkdtemp(join(tmpdir(), "concept-seed-other-"));
    const sets = [{ dir, prefix: P }, { dir: other, prefix: "lab/other/concepts/" }]; // the child's set first: order is free
    try {
      assert.equal(await edge(root, leaf), 1);
      await writeFile(join(other, `${top}.md`), "---\ndescription: Above the root.\n---\nTop docs.\n");
      await writeFile(join(dir, `${root}.md`), `---\ndescription: The root, v3.\nparent: ${top}\n---\nRoot docs v3.\n`);
      await seedConcepts(ws, sets);
      assert.match(String((await node(top))!.usid), /^lab\/other\/concepts\//);
      assert.equal(await edge(top, root), 1, "a root that gains a parent");
      assert.equal(await edge(root, leaf), 1, "still declared");
      // Anchored like every owned Concept — as on a graph where it was seeded with its parent from the start.
      assert.deepEqual([await anchors(top), await anchors(root), await anchors(leaf), await anchors(late)], [1, 1, 1, 1]);

      await writeFile(join(dir, `${leaf}.md`), `---\ndescription: A leaf.\nparent: ${top}\n---\nLeaf docs.\n`);
      await g.edges.write({ edge: "PARENT_OF", source_ref_id: (await node(leaf))!.ref_id as string, target_ref_id: (await node(late))!.ref_id as string });
      assert.equal(await edge(leaf, late), 1);
      await seedConcepts(ws, sets);
      assert.equal(await edge(top, leaf), 1, "the parent the file names now");
      assert.equal(await edge(root, leaf), 0, "the parent it used to name");
      assert.equal(await edge(leaf, late), 0, "between two owned Concepts, no file declares it");
      assert.equal(await edge(top, root), 1);

      await seedConcepts(ws, sets);
      assert.deepEqual([await edge(top, leaf), await edge(top, root), await edge(root, leaf), await edge(leaf, late)], [1, 1, 0, 0], "a fixed point");
    } finally {
      await rm(other, { recursive: true, force: true });
    }
  });

  it("a retired file's node is soft-deleted while it carries that file's stamp; someone's own is left", async () => {
    const g = ws.graph!;
    const gone = `Gone ${tag}`;
    const taken = `Taken ${tag}`;
    const other = `Other ${tag}`;
    const mk = (name: string, usid?: string) =>
      g.nodes.write({ type: "Concept", data: { name, description: "Once shipped.", ...(usid ? { unique_source_id: usid } : {}) } }, "create", { namespace: g.cfg.namespace });
    const deleted = async (name: string) =>
      (await g.bolt.run(`MATCH (n:Concept {name: $name}) RETURN coalesce(n.is_deleted, false) AS d`, { name }))[0]?.d;
    await mk(gone, `${P}${gone}.md@0123456789ab`);
    await mk(taken); // seeded once, then someone cleared the stamp
    await mk(other, `lab/other/concepts/${other}.md@0123456789ab`); // same name, another set's file
    const set = { dir, prefix: P, retired: [`${gone}.md`, `${taken}.md`, `${other}.md`, `Never Seeded ${tag}.md`] };
    await seedConcepts(ws, [set]);
    assert.deepEqual([await deleted(gone), await deleted(taken), await deleted(other)], [true, false, false]);
    await seedConcepts(ws, [set]);
    assert.equal(await deleted(gone), true);
    // Hidden from every read a workflow makes.
    assert.equal(await g.reader.getNode((await node(gone))!.ref_id as string), null);
  });

  it("an edge with someone else's Concept at either end is never touched: not removed, not written", async () => {
    const g = ws.graph!;
    const theirs = `Theirs ${tag}`;
    const ref = async (name: string) => (await node(name))!.ref_id as string;
    await g.nodes.write({ type: "Concept", data: { name: theirs, description: "A person's Concept." } }, "create", { namespace: g.cfg.namespace });
    // Their Concept above one of ours, by hand; and the kid they took over, still under our root.
    await g.edges.write({ edge: "PARENT_OF", source_ref_id: await ref(theirs), target_ref_id: await ref(late) });
    assert.equal(await edge(root, kid), 1);
    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal(await edge(theirs, late), 1);
    assert.equal(await edge(root, kid), 1);
    // They detach it: the kid's file still names the root, but the kid is theirs — the seed does not link it back.
    await g.bolt.run(`MATCH (:Concept {name: $p})-[e:PARENT_OF]->(:Concept {name: $c}) DELETE e`, { p: root, c: kid });
    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.equal(await edge(root, kid), 0);
  });
});

// The engine, offline: real if / pack / exec / artifacts-dir over the seeded
// YAML, with graph-get and the agent swapped for fakes — the wiring (the
// three reads by key, the under-Janitor guard, what the agent is handed).
describe("graph-janitor workflow (offline: fake graph + agent)", () => {
  const MANDATE = { ref_id: "r-mandate", node_type: "Concept", name: "Some Mandate", properties: { docs: "THE MANDATE TEXT" }, edges: {} };
  const LAW = { ref_id: "r-law", node_type: "Concept", name: "Law", properties: { description: "Legal knowledge." }, edges: {} };
  const JANITOR = { ref_id: "r-janitor", node_type: "Concept", name: "Janitor", properties: {}, edges: {} };
  const kid = { ref_id: MANDATE.ref_id, node_type: "Concept", name: MANDATE.name, description: "What dirty means here." };
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
    assert.match(text, /graph_get\(\{ node_type: "Concept", name: "<name>", children: "PARENT_OF" \}\)/);
    assert.doesNotMatch(text, /run_step/);
    assert.doesNotMatch(text, /and more/);
  });

  it("caps the notes, says when the list is cut, and has no table of contents without children", () => {
    const long = renderBuilderSystem({ name: "W", docs: "x".repeat(7000), children: [{ name: "A" }], more: true });
    assert.match(long, /x{6000}…\n/);
    assert.doesNotMatch(long, /x{6001}/);
    assert.match(long, /- … and more/);
    const bare = renderBuilderSystem({ name: "W", children: [] });
    assert.match(bare, /\(no docs\)$/);
    assert.doesNotMatch(bare, /Kinds recorded|graph_get/);
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
    // The committed tree, for real — and a mandate, which no seed ships: the test plants the fixture.
    await seedConcepts(ws, [BUILDER_CONCEPTS, JANITOR_CONCEPTS, JANITOR_FIXTURES]);
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

  it("the tree is Workflow Builder > Janitor > the planted mandate", async () => {
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

  it("sweeps Law under the planted mandate: the agent gets the mandate's real docs and Law's real ref_id", async () => {
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
