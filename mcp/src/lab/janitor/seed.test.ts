import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile, readFile, readdir } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, type Neo4jBackend, closeGraphBackends, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { parseConceptFile, planConceptSeed, retireConcept, seedConcepts, stampOf } from "../concept-seed.js";
import { CODE_CONCEPTS } from "../code/seed.js";
import { JOB_CONCEPTS } from "../job/seed.js";
import { POD_CONCEPTS } from "../pods/seed.js";
import { BUILDER_CONCEPTS, BUILDER_ENTRY, builderSystem, renderBuilderSystem } from "../builder/system.js";
import { JANITOR_CONCEPTS, JANITOR_FIXTURES, seedJanitorSteps, seedJanitorWorkflows } from "./seed.js";
import { SEED_OPTS } from "../seed-opts.js";

// The live suites share one persistent test graph: whatever version an earlier run (or another
// checkout) left active, the COMMITTED YAML must be the one that runs here.
const PIN_COMMITTED = { ...SEED_OPTS, reactivateKnown: true };

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

    // The builder's tree is what gets BUILT; what a job agent runs is under `Job`.
    assert.doesNotMatch(entry.docs!, /job agent|code change|code-change|plan mode/i, "the entry page knows no kind by name");

    // Job: the job agent's root page, naming no kind.
    const job = await read(JOB_CONCEPTS, "Job.md");
    assert.equal(job.name, "Job");
    assert.equal(job.parent, undefined);
    assert.match(job.stamp, /^lab\/job\/concepts\/Job\.md@[0-9a-f]{12}$/);
    assert.ok(job.docs!.startsWith(job.description!));
    assert.doesNotMatch(job.docs!, /code change|code-change|plan mode/i, "the entry page knows no kind by name");

    // Plan Mode: a way of working under Job — one HTML file, the technical part collapsed, ask before going on.
    const pm = await read(JOB_CONCEPTS, "Plan Mode.md");
    assert.equal(pm.parent, "Job");
    assert.ok(pm.docs!.startsWith(pm.description!));
    for (const needle of [/plan\.html/, /<details>/, /`ask`/, /scripts never run/]) assert.match(pm.docs!, needle);
    // Explore: a way of working under Job — explore here or on a peer by @slug, and keep what was found as a note;
    // another job's note used as it is is reported by its link, never named only in the reply.
    const ex = await read(JOB_CONCEPTS, "Explore.md");
    assert.equal(ex.name, "Explore");
    assert.equal(ex.parent, "Job");
    assert.ok(ex.docs!.startsWith(ex.description!));
    for (const needle of [/meta\/run-workflow/, /strut\/run-workflow/, /"explore"/, /notes\/<topic>\.md/, /@<slug>/, /kind: "markdown"/, /`url` job\/get gave it/]) assert.match(ex.docs!, needle);
    // Verify: a way of working under Job — claims about the job, an instrument per surface, verify before done
    // (strut plans/job-claims.md); a person's refutation stands on the turn they looked at.
    const vf = await read(JOB_CONCEPTS, "Verify.md");
    assert.equal(vf.name, "Verify");
    assert.equal(vf.parent, "Job");
    assert.ok(vf.docs!.startsWith(vf.description!));
    for (const needle of [/meta\/add-claim/, /kind: "job"/, /meta\/add-check/, /browser\/verify/, /\{\{ input\.urls\.pod \}\}/, /http\/verify/, /cwd: "\{\{ input\.dir \}\}"/, /pod\/run-tests/, /external check/, /meta\/verify-job/, /`refuted`/, /`claims` on `job\/get`/, /\[claim-event\]/, /does not\s+lift it/]) {
      assert.match(vf.docs!, needle);
    }
    // The entry page knows the workspaces and the peer step, still no kind.
    for (const needle of [/strut\/run-workflow/, /@<slug>/, /own agent/]) assert.match(job.docs!, needle);
    assert.doesNotMatch(job.docs!, /\bverify\b/i, "the entry page knows no kind by name");
    assert.deepEqual(await readdir(JOB_CONCEPTS.dir), ["Explore.md", "Job.md", "Plan Mode.md", "Verify.md"]);

    // Code Change: the kind page the job agent follows to code-change-pr (lab/code).
    const cc = await read(CODE_CONCEPTS, "Code Change.md");
    assert.equal(cc.name, "Code Change");
    assert.equal(cc.parent, "Job");
    assert.match(cc.stamp, /^lab\/code\/concepts\/Code Change\.md@[0-9a-f]{12}$/);
    assert.equal(cc.description, "A change to a repository's source code, delivered as a pull request.");
    assert.ok(cc.docs!.startsWith(cc.description!));
    // The convention: the workflow and how to read it, the follow-up rule, the credential failure, the artifact, how to extend.
    for (const needle of [/code-change-pr/, /meta\/get-workflow/, /same `branch`/, /no_push_permission:/, /pull_request/, /graph\/create-node/, /graph\/create-triplet/, /PARENT_OF/, /cannot read your directory/, /`Explore`/]) assert.match(cc.docs!, needle);
    assert.doesNotMatch(cc.docs!, /pod-pr|hive\//, "workspace-general: no one swarm's workflow or step by name");

    // Pod Config: under Pod, how a pod is made to run the app, and how a fix there reaches every pod.
    const pod = await read(POD_CONCEPTS, "Pod.md");
    assert.equal(pod.parent, "Code Change");
    assert.match(pod.docs!, /child `Pod Config`/);
    const pc = await read(POD_CONCEPTS, "Pod Config.md");
    assert.equal(pc.name, "Pod Config");
    assert.equal(pc.parent, "Pod");
    assert.match(pc.stamp, /^lab\/pods\/concepts\/Pod Config\.md@[0-9a-f]{12}$/);
    assert.ok(pc.docs!.startsWith(pc.description!));
    // Look first, the in-pod knowledge, ask before promoting, verify on a fresh pod, report what a later job finds.
    for (const needle of [/job\/list \{ q: "pod config <workspace>" \}/, /staklink reload/, /pm2 stop all/, /0\.0\.0\.0/, /pod\/config \{ workspace, control, sealed \}/, /`ask`/, /pod\/config-set/, /fresh: true/, /pull request merges/, /id: "pod-config"/, /devcontainer\.json/, /No credential is promoted/]) assert.match(pc.docs!, needle);
    assert.deepEqual(await readdir(POD_CONCEPTS.dir), ["Pod Config.md", "Pod.md"]);

    // What counts as dirty is each workspace's to say: the seed ships the kind, never a mandate.
    assert.deepEqual(await readdir(JANITOR_CONCEPTS.dir), ["Janitor.md"]);
    assert.deepEqual(JANITOR_CONCEPTS.retired, ["Overfit Concept Janitor.md"]);
    const fixture = await read(JANITOR_FIXTURES, "Overfit Concept Janitor.md");
    assert.equal(fixture.parent, "Janitor");
    assert.match(fixture.stamp, /^lab\/janitor\/fixtures\//, "a fixture never carries a seed's stamp");
  });

  it("what ships to every workspace says nothing about one workspace's domain", async () => {
    const domain = /\b(law|legal|rubric|evals?|benchmark|exam|grading|overfit)\b/i;
    for (const set of [BUILDER_CONCEPTS, JANITOR_CONCEPTS, JOB_CONCEPTS, CODE_CONCEPTS, POD_CONCEPTS]) {
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
  it("same stamp → keep (ours, unchanged: a graph-side edit or mute sticks); deleted → update (restored)", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp }, stamp), "keep");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp, deleted: true }, stamp), "update");
  });
  it("our path, older hash → update; deleted too", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older }, stamp), "update");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older, deleted: true }, stamp), "update");
  });
  it("no stamp or someone else's → skip (theirs); deleted, it stays deleted", () => {
    assert.equal(planConceptSeed({ ref_id: "r" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", deleted: true }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: null }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "lab/janitor/concepts/Other.md@aaaaaaaaaaaa" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "gitree" }, stamp), "skip");
  });
});

describe("retireConcept (jarvis)", () => {
  const env = { url: process.env.JARVIS_URL, token: process.env.API_TOKEN };
  const seen: { method?: string; url?: string; token?: unknown; admin?: unknown }[] = [];
  let reply: { status: number; body: unknown } = { status: 200, body: {} };
  let server: Server;
  // The jarvis path never touches bolt.
  const graph = { kind: "neo4j", cfg: { namespace: "ns1" }, bolt: { run: () => assert.fail("bolt used") } } as unknown as Neo4jBackend;
  before(async () => {
    server = createServer((req: IncomingMessage, res) => {
      seen.push({ method: req.method, url: req.url, token: req.headers["x-api-token"], admin: req.headers["x-is-admin"] });
      res.writeHead(reply.status, { "Content-Type": "application/json" }).end(JSON.stringify(reply.body));
    });
    await new Promise<void>((ok) => server.listen(0, "127.0.0.1", ok));
    process.env.JARVIS_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
    process.env.API_TOKEN = "tok";
  });
  after(async () => {
    for (const [k, v] of [["JARVIS_URL", env.url], ["API_TOKEN", env.token]] as const) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    await new Promise((ok) => server.close(ok));
  });

  it("deletes through DELETE /v2/nodes/<ref_id>/single in the graph's namespace, as admin", async () => {
    reply = { status: 200, body: { status: "success", ref_id: "r 1", deleted_at: 1, deleted_edge_count: 2 } };
    await retireConcept(graph, "r 1");
    assert.deepEqual(seen.at(-1), { method: "DELETE", url: "/v2/nodes/r%201/single?namespace=ns1", token: "tok", admin: "true" });
  });

  it("409 ALREADY_DELETED is done", async () => {
    reply = { status: 409, body: { errorCode: "ALREADY_DELETED" } };
    await retireConcept(graph, "r");
  });

  it("any other failure throws", async () => {
    reply = { status: 404, body: { errorCode: "NOT_FOUND" } };
    await assert.rejects(retireConcept(graph, "r"), /HTTP 404/);
    reply = { status: 409, body: { errorCode: "SOMETHING_ELSE" } };
    await assert.rejects(retireConcept(graph, "r"), /HTTP 409/);
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
  const jarvisUrl = process.env.JARVIS_URL;
  before(async () => {
    delete process.env.JARVIS_URL; // a standalone strut: retire runs over bolt
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
    if (jarvisUrl !== undefined) process.env.JARVIS_URL = jarvisUrl;
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

  it("a retired file's node is deleted while it carries that file's stamp; someone's own is left", async () => {
    const g = ws.graph!;
    const gone = `Gone ${tag}`;
    const taken = `Taken ${tag}`;
    const other = `Other ${tag}`;
    const dead = `Dead ${tag}`;
    const mk = (name: string, usid?: string) =>
      g.nodes.write({ type: "Concept", data: { name, description: "Once shipped.", ...(usid ? { unique_source_id: usid } : {}) } }, "create", { namespace: g.cfg.namespace });
    const state = async (name: string) =>
      (await g.bolt.run(`MATCH (n:Concept {name: $name}) RETURN n.deleted_at AS at, coalesce(n.is_deleted, false) AS d`, { name }))[0];
    const deleted = async (name: string) => (await state(name))?.d;
    await mk(gone, `${P}${gone}.md@0123456789ab`);
    await mk(taken); // seeded once, then someone cleared the stamp
    await mk(other, `lab/other/concepts/${other}.md@0123456789ab`); // same name, another set's file
    await mk(dead, `${P}${dead}.md@0123456789ab`);
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.deleted_at = 1`, { name: dead }); // deleted_at only: already retired
    await g.edges.write({ edge: "PARENT_OF", source_ref_id: (await node(taken))!.ref_id as string, target_ref_id: (await node(gone))!.ref_id as string });
    const set = { dir, prefix: P, retired: [`${gone}.md`, `${taken}.md`, `${other}.md`, `${dead}.md`, `Never Seeded ${tag}.md`] };
    await seedConcepts(ws, [set]);
    assert.deepEqual([await deleted(gone), await deleted(taken), await deleted(other)], [true, false, false]);
    const at = Number((await state(gone))!.at);
    assert.ok(at > 1_000_000_000_000, "deleted_at in epoch ms");
    assert.equal(await edge(taken, gone), 0, "its edges are removed");
    const d = (await state(dead))!;
    assert.deepEqual([Number(d.at), d.d], [1, false], "a node with only deleted_at counts as retired: untouched");
    await seedConcepts(ws, [set]);
    assert.equal(await deleted(gone), true);
    assert.equal(Number((await state(gone))!.at), at, "the first deleted_at is kept");
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

  // Prod, 2026-10-09: `Workflow Builder` was deleted in Learn; the builder lost
  // its host section, and every later boot parented Janitor under the dead node.
  it("a deleted Concept of ours comes back on the next seed, linked and anchored; theirs stays deleted and is nobody's parent", async () => {
    const g = ws.graph!;
    const top = `Top ${tag}`;
    const orphan = `Orphan ${tag}`;
    const live = async (name: string) =>
      (await g.bolt.run(`MATCH (n:Concept {name: $name}) RETURN n.deleted_at IS NULL AND n.is_deleted IS NULL AS live, n.docs AS docs`, { name }))[0]!;
    // jarvis's delete: both stamps, the node's edges removed.
    const del = (name: string) =>
      g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.deleted_at = $now, n.is_deleted = true WITH n OPTIONAL MATCH (n)-[r]-() DELETE r`, { name, now: Date.now() });
    await del(root); // ours (its file names `top` as parent)
    await del(late); // ours, no parent
    await del(kid); // theirs: the stamp was cleared
    assert.deepEqual([(await live(root)).live, await anchors(root), await edge(top, root)], [false, 0, 0]);
    await writeFile(join(dir, `${orphan}.md`), `---\ndescription: Under a dead node.\nparent: ${kid}\n---\nOrphan docs.\n`);

    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.deepEqual([(await live(root)).live, (await live(late)).live], [true, true], "restored");
    assert.equal((await live(root)).docs, "Root docs v3.");
    assert.deepEqual([await anchors(root), await anchors(late)], [1, 1], "anchored again");
    assert.equal(await edge(top, root), 1, "its parent from the file");
    assert.equal((await live(kid)).live, false, "theirs stays deleted");
    assert.ok(await node(orphan), "the orphan is seeded");
    assert.equal(await edge(kid, orphan), 0, "but never under a deleted node");
    assert.equal(await anchors(orphan), 1);

    await seedConcepts(ws, [{ dir, prefix: P }]);
    assert.deepEqual([(await live(root)).live, await anchors(root), await edge(top, root), await edge(kid, orphan)], [true, 1, 1, 0], "a fixed point");
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
  const queues: Record<string, any>[] = [];
  const fakes = {
    // The rotation itself is pure (apply.test.ts) and reads over bolt: here, what it is handed and what it hands on.
    "janitor/queue": fake("janitor/queue", (cfg) => {
      queues.push(cfg);
      return {
        batch: [{ ref_id: LAW.ref_id, name: "Law", node_type: "Concept", description: "Legal knowledge.", docs: null }],
        ref_ids: [LAW.ref_id],
        visited: [{ ref_id: LAW.ref_id, name: "Law", node_type: "Concept" }],
        text: "### 1. Law\nTHE BATCH TEXT",
        next_cursor: "Law",
        from: 0,
        total: 12,
        wrapped: false,
        truncated: false,
      };
    }),
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
    await seedJanitorSteps(ws); // janitor/plan runs for real: with no findings it never opens the graph
    await seedJanitorWorkflows(ws);
    const flow = await ws.getWorkflow("graph-janitor");
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    // What createStrut injects for a server: the per-run artifacts dir (artifacts/dir + exec's cwd).
    const dataDir = join(root, "data");
    const services = { ...standardServices({ secretsSource: {}, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = (input) => runWorkflow(flow, input, { ...registry, ...fakes } as typeof registry, { services });
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("is seeded under graph-maintenance with its declared inputs (apply off by default, cursor optional); its description is no how-to", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "graph-janitor");
    assert.equal(entry?.category, "graph-maintenance");
    const flow = (await ws.getWorkflow("graph-janitor")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock ?? {}).sort(), ["apply", "concept", "cursor", "start"]);
    assert.equal(flow.inputBlock.apply.default, false);
    assert.equal(flow.inputBlock.cursor.required, false);
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
    // A dry run by default: the plan ran (empty), nothing was applied, and the result says so.
    assert.equal(out.apply, false);
    assert.deepEqual(out.plan.counts, { findings: 0, edits: 0, moves: 0, skipped: 0, manual: 0 });
    assert.equal(out.applied, undefined);
    assert.match(out.report_plan, /^\/artifacts\/\d+\/plan\.json$/);
    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /THE MANDATE TEXT/);
    // The agent reviews the batch code picked, handed to it verbatim — it does not choose what to read.
    assert.match(cfg.prompt, /THE BATCH TEXT/);
    assert.match(cfg.prompt, /a BATCH of 1 Concept nodes from the subtree under "Law" \(ref_id r-law\), which holds 12/);
    assert.deepEqual(queues.at(-1), { start_ref_id: LAW.ref_id, cursor: undefined, size: 10, node_type: "Concept" }, "no cursor given: none passed on");
    assert.deepEqual(out.visited, [{ ref_id: LAW.ref_id, name: "Law", node_type: "Concept" }], "visited is the batch");
    assert.deepEqual(out.queue, { from: 0, size: 1, total: 12, wrapped: false });
    assert.equal(out.next_cursor, "Law");
    // An empty cwd gets no preamble from the agent step: the prompt must name it, or report.md lands elsewhere.
    assert.ok(cfg.prompt.includes(cfg.cwd), "the prompt names the working dir");
    assert.match(cfg.system, /DATA read from a graph node/);
    assert.ok(Array.isArray(cfg.agentTools) && cfg.agentTools.length > 0);
    for (const t of cfg.agentTools) assert.doesNotMatch(t, /create|edit|register|project|\*/, t);
  });

  it("hands the queue the cursor the automation passed", async () => {
    const res = await run({ concept: MANDATE.name, start: "Law", cursor: "Hematology" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.equal(queues.at(-1)!.cursor, "Hematology");
  });

  it("refuses a mandate that is not a child of Janitor: not_a_janitor, and the agent never runs", async () => {
    const n = agentCalls.length;
    const res = await run({ concept: "Law", start: "Law" });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /not_a_janitor: 'Law' is not a PARENT_OF child of 'Janitor'/);
    // The names that would have been accepted ride along: a typo is self-correcting.
    assert.match(JSON.stringify(res.error), new RegExp(`; its children: ${MANDATE.name}`));
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
    await seedJanitorSteps(ws);
    await seedJanitorWorkflows(ws, PIN_COMMITTED);
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

// A deployed graph has `Code Change` under `Workflow Builder` (the tree before
// `Job`): the next boot moves it under `Job` and drops the old edge.
describe("the Job tree on a graph seeded before it (live graph)", { skip: !URI }, () => {
  const NAMES = ["Workflow Builder", "Janitor", "Job", "Plan Mode", "Code Change", "Explore", "Verify"];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  const edges = async (parent: string) =>
    (
      await ws.graph!.bolt.run(`MATCH (:Concept {name: $parent})-[:PARENT_OF]->(c:Concept) RETURN c.name AS name ORDER BY name`, { parent })
    ).map((r) => String(r.name));
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    root = await mkdtemp(join(tmpdir(), "job-tree-live-"));
    ws = (
      await graphWorkspaceFromEnv(
        {
          NEO4J_URI: URI!,
          NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
          NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
          STRUT_GRAPH_SEED_ONTOLOGY: "1",
          STRUT_GRAPH_EMBEDDINGS: "off",
        },
        { dataDir: root },
      )
    ).workspace;
    await ws.graph!.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
  });
  after(async () => {
    await ws?.graph?.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await closeGraphBackends();
    await rm(root, { recursive: true, force: true });
  });

  it("Code Change moves from Workflow Builder to Job; Plan Mode, Explore and Verify join it; Janitor stays", async () => {
    const old = join(root, "old-code");
    const { mkdir } = await import("node:fs/promises");
    await mkdir(old);
    const cc = await readFile(join(CODE_CONCEPTS.dir, "Code Change.md"), "utf-8");
    await writeFile(join(old, "Code Change.md"), cc.replace(/^parent: Job$/m, "parent: Workflow Builder"));
    await seedConcepts(ws, [BUILDER_CONCEPTS, JANITOR_CONCEPTS, { dir: old, prefix: CODE_CONCEPTS.prefix }]);
    assert.deepEqual(await edges("Workflow Builder"), ["Code Change", "Janitor"], "the tree as deployed");

    await seedConcepts(ws, [BUILDER_CONCEPTS, JANITOR_CONCEPTS, JOB_CONCEPTS, CODE_CONCEPTS]);
    assert.deepEqual(await edges("Workflow Builder"), ["Janitor"]);
    assert.deepEqual(await edges("Job"), ["Code Change", "Explore", "Plan Mode", "Verify"]);
  });
});

// The apply branch and the revert over the REAL steps (janitor/plan,
// graph/edit-node, graph/move-node, janitor/readback, janitor/revert-plan)
// with a fake agent, on the committed tree: a dry run leaves the graph
// alone; apply=true writes, reads back and logs; the revert puts it back; a
// node changed since is left alone. Here, not in apply.test.ts: test files
// run in parallel processes, and the live suites share the Janitor tree.
// apply=false leaves the graph alone; apply=true writes, reads back and logs;
// the revert puts it back; a node changed since is left alone.
describe("graph-janitor apply + revert (live graph, fake agent)", { skip: !URI }, () => {
  const tag = `A${Date.now().toString(36)}`;
  const NAMES = ["Workflow Builder", "Janitor", "Overfit Concept Janitor", `Law ${tag}`, `Tort ${tag}`, `Crime ${tag}`];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  let services: Record<string, unknown>;
  let registry: Awaited<ReturnType<typeof buildRegistry>>["registry"];
  let agentAnswer: () => Record<string, unknown>;
  let agentPrompt = "";
  const ids: Record<string, string> = {};
  const read = async (name: string) => {
    const rows = await ws.graph!.bolt.run(
      `MATCH (n:Concept {name: $name}) OPTIONAL MATCH (p:Concept)-[:PARENT_OF]->(n) RETURN n.ref_id AS ref_id, n.description AS description, n.docs AS docs, n.unique_source_id AS usid, collect(p.name) AS parents`,
      { name },
    );
    return rows[0]!;
  };
  const run = async (name: string, input: Record<string, unknown>) => runWorkflow(await ws.getWorkflow(name), input, registry, { services });
  const artifact = (runId: string, file: string) => readFile(join(root, "data", "artifacts", runId, file), "utf-8");
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    root = await mkdtemp(join(tmpdir(), "janitor-apply-live-"));
    const env = {
      NEO4J_URI: URI!,
      NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
      NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
      STRUT_GRAPH_SEED_ONTOLOGY: "1",
      STRUT_GRAPH_EMBEDDINGS: "off",
    };
    ws = (await graphWorkspaceFromEnv(env, { dataDir: root })).workspace;
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await seedArtifactSteps(ws);
    await seedJanitorSteps(ws);
    await seedJanitorWorkflows(ws, PIN_COMMITTED);
    await seedConcepts(ws, [BUILDER_CONCEPTS, JANITOR_CONCEPTS, JANITOR_FIXTURES]);
    const ns = g.cfg.namespace;
    for (const [name, description] of [[`Law ${tag}`, "Legal knowledge."], [`Tort ${tag}`, "A civil wrong."], [`Crime ${tag}`, "An offence."]] as const) {
      ids[name] = (await g.nodes.write({ type: "Concept", data: { name, description } }, "create", { namespace: ns })).ref_id;
    }
    for (const kid of [`Tort ${tag}`, `Crime ${tag}`]) await g.edges.write({ edge: "PARENT_OF", source_ref_id: ids[`Law ${tag}`]!, target_ref_id: ids[kid]! });
    for (const name of ["Janitor", "Workflow Builder"]) ids[name] = String((await read(name)).ref_id);
    registry = (await buildRegistry(await ws.materializeCustomSteps())).registry;
    const agent = defineStep({
      type: "agent",
      input: z.object({}).passthrough(),
      output: z.any(),
      async run(cfg) {
        agentPrompt = String((cfg as Record<string, unknown>).prompt ?? "");
        return { result: "ok", object: agentAnswer(), steps: 1, usage: {}, cost: 0 };
      },
    });
    registry = { ...registry, agent } as typeof registry;
    const dataDir = join(root, "data");
    services = { ...standardServices({ secretsSource: env, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    const visited = NAMES.filter((n) => n !== "Overfit Concept Janitor").map((n) => ({ ref_id: ids[n], name: n, node_type: "Concept" }));
    const f = (name: string, action: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({ ref_id: ids[name], name, issue: "vague", problem: "p", evidence: "x", suggestion: "s", severity: "medium", action, ...extra });
    agentAnswer = () => ({
      start_ref_id: ids[`Law ${tag}`],
      visited,
      findings: [
        f(`Tort ${tag}`, { kind: "set_property", property: "description", old_value: "A civil wrong.", new_value: "A wrong between people." }),
        f(`Crime ${tag}`, { kind: "move", new_parent_ref_id: ids[`Tort ${tag}`] }),
        f(`Crime ${tag}`, { kind: "replace_text", property: "description", find: "An offence", replace: "A public wrong" }),
        f("Janitor", { kind: "set_property", property: "description", old_value: "", new_value: "Hijacked." }, { severity: "high" }),
        f("Workflow Builder", { kind: "set_property", property: "docs", old_value: "", new_value: "Rewritten." }, { evidence: "" }),
        f(`Law ${tag}`, { kind: "set_property", property: "description", old_value: "Something else.", new_value: "Nope." }),
        f(`Law ${tag}`, { kind: "none" }, { suggestion: "merge" }),
      ],
      summary: "two real changes",
    });
  });
  after(async () => {
    const g = ws?.graph;
    if (g) {
      await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
      await closeGraphBackends();
    }
    await rm(root, { recursive: true, force: true });
  });

  let applyRun = "";
  it("dry run: the plan keeps the two real changes and skips the rest with reasons — a node outside the batch untouched whatever it is — and the graph is untouched", async () => {
    const res = await run("graph-janitor", { concept: "Overfit Concept Janitor", start: `Law ${tag}` });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    assert.equal(out.apply, false);
    assert.deepEqual(out.plan.edits.map((e: any) => [e.id, e.name, e.old_value, e.new_value, e.via]), [
      ["e1", `Tort ${tag}`, "A civil wrong.", "A wrong between people.", undefined],
      ["e2", `Crime ${tag}`, "An offence.", "A public wrong.", { find: "An offence", replace: "A public wrong" }],
    ]);
    assert.deepEqual(out.plan.moves.map((m: any) => [m.id, m.from_ref_id, m.to_ref_id]), [["m1", ids[`Law ${tag}`], ids[`Tort ${tag}`]]]);
    // The REAL queue: the batch is Law's subtree in tree order, the start first, its children by name.
    assert.deepEqual(out.visited.map((v: any) => v.name), [`Law ${tag}`, `Crime ${tag}`, `Tort ${tag}`]);
    assert.deepEqual([out.queue.total, out.next_cursor], [3, `Law ${tag} › Tort ${tag}`]);
    // Where each node sits, read live: its path, its parent, its siblings.
    assert.ok(agentPrompt.includes(`### 1. Law ${tag}\nref_id: ${ids[`Law ${tag}`]}\npath: Law ${tag}\nparent: none — the start of the sweep\nsiblings: none\n`), agentPrompt);
    assert.ok(agentPrompt.includes(`### 3. Tort ${tag}\nref_id: ${ids[`Tort ${tag}`]}\npath: Law ${tag} › Tort ${tag}\nparent: Law ${tag} (ref_id ${ids[`Law ${tag}`]})\nsiblings: Crime ${tag}\n`), agentPrompt);
    // Workflow Builder is outside the batch (git_seeded is planChanges' to test, apply.test.ts).
    assert.deepEqual(out.plan.skipped.map((s: any) => [s.name, s.reason]), [["Janitor", "protected"], ["Workflow Builder", "not_in_batch"], [`Law ${tag}`, "stale"]]);
    assert.equal(out.plan.manual.length, 1);
    assert.equal(out.applied, undefined);
    const plan = JSON.parse(await artifact(res.runId, "plan.json"));
    assert.equal(plan.counts.edits, 2);
    assert.equal((await read(`Tort ${tag}`)).description, "A civil wrong.");
    assert.equal((await read(`Crime ${tag}`)).description, "An offence.");
    assert.deepEqual((await read(`Crime ${tag}`)).parents, [`Law ${tag}`]);
    assert.match(String((await read("Workflow Builder")).usid), /^lab\/builder\/concepts\/Workflow Builder\.md@/);
  });

  it("apply=true: the edit and the move land, read back, and the undo log names what they replaced", async () => {
    const res = await run("graph-janitor", { concept: "Overfit Concept Janitor", start: `Law ${tag}`, apply: true });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    applyRun = res.runId;
    const out = res.output as Record<string, any>;
    assert.equal(out.apply, true);
    assert.deepEqual(out.applied.map((c: any) => c.id), ["e1", "e2", "m1"]);
    assert.equal(out.applied_json, `/artifacts/${applyRun}/applied.json`);
    assert.equal((await read(`Tort ${tag}`)).description, "A wrong between people.");
    assert.equal((await read(`Crime ${tag}`)).description, "A public wrong.");
    assert.deepEqual((await read(`Crime ${tag}`)).parents, [`Tort ${tag}`]);
    assert.equal((await read("Janitor")).description, "Automated agents for cleaning up code, concept trees, or other data.");
    const log = JSON.parse(await artifact(applyRun, "applied.json"));
    assert.equal(log.workflow, "graph-janitor");
    assert.equal(log.run_id, applyRun);
    assert.deepEqual(log.counts, { planned: 3, applied: 3, failed: 0, not_verified: 0 });
    // The undo log holds whole values for both edit shapes: a revert never needs to know one was a replace_text.
    assert.deepEqual(log.applied.map((c: any) => [c.id, c.old_value ?? c.from_ref_id, c.new_value ?? c.to_ref_id]), [
      ["e1", "A civil wrong.", "A wrong between people."],
      ["e2", "An offence.", "A public wrong."],
      ["m1", ids[`Law ${tag}`], ids[`Tort ${tag}`]],
    ]);
    const report = await artifact(applyRun, "report.md");
    assert.match(report, new RegExp(`## Applied \\(run ${applyRun}\\)`));
    assert.match(report, /graph-janitor-revert/);
  });

  it("the revert puts both back and is logged as a revert of the apply run", async () => {
    const res = await run("graph-janitor-revert", { run_id: applyRun });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    assert.equal(out.revert_of, applyRun);
    assert.equal(out.source.workflow, "graph-janitor");
    assert.deepEqual(out.applied.map((c: any) => [c.id, c.reverts]), [["e1", "e1"], ["e2", "e2"], ["m1", "m1"]]);
    assert.equal((await read(`Tort ${tag}`)).description, "A civil wrong.");
    assert.equal((await read(`Crime ${tag}`)).description, "An offence.");
    assert.deepEqual((await read(`Crime ${tag}`)).parents, [`Law ${tag}`]);
    const log = JSON.parse(await artifact(res.runId, "applied.json"));
    assert.deepEqual([log.workflow, log.revert_of], ["graph-janitor-revert", applyRun]);
  });

  it("a second revert of the same run touches nothing: the nodes no longer hold what the janitor wrote", async () => {
    const res = await run("graph-janitor-revert", { run_id: applyRun, changes: ["e1", "e2", "m1", "zz"] });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    assert.deepEqual(out.applied, []);
    assert.deepEqual(out.plan.skipped.map((s: any) => [s.id, s.reason]), [["e1", "moved_on"], ["e2", "moved_on"], ["m1", "moved_on"], ["zz", "unknown_change"]]);
  });

  it("a run that never applied anything has no undo log: no_applied_log", async () => {
    const dry = await run("graph-janitor", { concept: "Overfit Concept Janitor", start: `Law ${tag}` });
    assert.equal(dry.status, "success");
    const res = await run("graph-janitor-revert", { run_id: dry.runId });
    assert.equal(res.status, "error");
    assert.match(JSON.stringify(res.error), /no_applied_log: /);
  });
});
