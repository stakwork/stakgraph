import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import {
  STRUT_CONCEPTS,
  WorkspaceManager,
  buildRegistry,
  builderSystem,
  closeGraphBackends,
  defineStep,
  fileArtifactsCapability,
  parseConceptFile,
  runWorkflow,
  seedConcepts,
  standardServices,
} from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { anchorConcepts } from "../concept-anchor.js";
import { CODE_CONCEPTS } from "../code/seed.js";
import { JOB_CONCEPTS, JOB_SEED } from "../job/seed.js";
import { POD_CONCEPTS } from "../pods/seed.js";
import { POD_SETUP_CONCEPTS } from "../pod-setup/seed.js";
import { JANITOR_CONCEPTS, JANITOR_FIXTURES, seedJanitorSteps, seedJanitorWorkflows } from "./seed.js";
import { SEED_OPTS } from "../seed-opts.js";

// The live suites share one persistent test graph: whatever version an earlier run (or another
// checkout) left active, the COMMITTED YAML and steps must be the ones that run here — together,
// or the engine hands a step a config its other version does not take.
const PIN_COMMITTED = { ...SEED_OPTS, reactivateKnown: true };

/** Every set the lab seeds at boot, after strut's own (createLabStrut.ts). */
const LAB_CONCEPTS = [JANITOR_CONCEPTS, JOB_CONCEPTS, CODE_CONCEPTS, POD_CONCEPTS, POD_SETUP_CONCEPTS];

// The seeder is strut's (`seedConcepts`, tested there on both engines); what is the lab's is its files.
describe("the lab's Concept files", () => {
  const read = (set: { dir: string; prefix: string }, file: string) =>
    readFile(join(set.dir, file), "utf-8").then((text) => parseConceptFile(file, text, set.prefix));

  it("hang under strut's two pages: Janitor under Workflow Builder, the job's kinds under Job, and no mandate", async () => {
    const c = await read(JANITOR_CONCEPTS, "Janitor.md");
    assert.equal(c.name, "Janitor");
    assert.equal(c.parent, "Workflow Builder");
    assert.match(c.stamp, /^lab\/janitor\/concepts\/Janitor\.md@/, "the file did not move: its nodes stay ours");
    assert.equal(c.description, "Automated agents for cleaning up code, concept trees, or other data.");
    assert.ok(c.docs!.startsWith(c.description!));
    // The convention lives in its docs: the engine, and the three steps to add one.
    for (const needle of [/graph-janitor/, /graph\/create-node/, /graph\/create-triplet/, /PARENT_OF/, /automation/]) assert.match(c.docs!, needle);

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
    // Verify: a way of working under Job — one claim per change, a check that uses the deliverable as its user
    // would (an instrument per surface) or the person's look when nothing here runs it, never a check over the
    // agent's own diff, verify before done (strut plans/job-claims.md); a person's rejection stands on the turn they looked at.
    const vf = await read(JOB_CONCEPTS, "Verify.md");
    assert.equal(vf.name, "Verify");
    assert.equal(vf.parent, "Job");
    assert.ok(vf.docs!.startsWith(vf.description!));
    for (const needle of [/meta\/add-claim/, /kind: "job"/, /meta\/add-check/, /ONE claim for a change/, /browser\/verify/, /\{\{ input\.urls\.pod \}\}/, /screenshot/, /http\/verify/, /cwd: "\{\{ input\.dir \}\}"/, /pod\/run-tests/, /external check/, /Never a check that reads your own diff/, /meta\/verify-job/, /`refuted`/, /`claims` on `job\/get`/, /\[claim-event\]/, /does not\s+lift it/]) {
      assert.match(vf.docs!, needle);
    }
    // Onboarding: a kind under Job — draft with identify-services, write it to the pool without waiting, run it in the
    // pod the job holds, check it, fix it there; the fresh-pod check only after a fix made by hand (strut plans/onboarding.md).
    const ob = await read(JOB_CONCEPTS, "Onboarding.md");
    assert.equal(ob.parent, "Job");
    assert.ok(ob.docs!.startsWith(ob.description!));
    for (const needle of [/pod\/claim \{ workspace \}/, /"identify-services"/, /pod\/config \{ workspace, control, sealed \}/, /npx -y hell0-w0rld/, /wait: false/, /pod\/config-apply/, /Do not ask first/, /Any\s+other pool config is not a first setup/, /meta\/verify-job/, /fresh: true/, /`Pod Config`/]) {
      assert.match(ob.docs!, needle);
    }
    assert.deepEqual(await readdir(JOB_CONCEPTS.dir), ["Explore.md", "Onboarding.md", "Plan Mode.md", "Verify.md"]);

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
    for (const needle of [/job\/list \{ q: "pod config <workspace>" \}/, /staklink reload/, /pm2 stop all/, /`Pod Setup`/, /pod\/config \{ workspace, control, sealed \}/, /`ask`/, /pod\/config-set/, /fresh: true/, /pull request merges/, /id: "pod-config"/, /devcontainer\.json/, /No credential is promoted/]) assert.match(pc.docs!, needle);
    // One set of rules: what the files must say is Pod Setup's, never copied here.
    assert.doesNotMatch(pc.docs!, /0\.0\.0\.0|hell0-w0rld|staklink-android/, "the setup contract is Pod Setup's");
    assert.deepEqual(await readdir(POD_CONCEPTS.dir), ["Pod Config.md", "Pod.md"]);

    // What counts as dirty is each workspace's to say: the seed ships the kind, never a mandate.
    assert.deepEqual(await readdir(JANITOR_CONCEPTS.dir), ["Janitor.md"]);
    assert.deepEqual(JANITOR_CONCEPTS.retired, ["Overfit Concept Janitor.md"]);
    const fixture = await read(JANITOR_FIXTURES, "Overfit Concept Janitor.md");
    assert.equal(fixture.parent, "Janitor");
    assert.match(fixture.stamp, /^lab\/janitor\/fixtures\//, "a fixture never carries a seed's stamp");
  });

  it("ship no page under one of strut's names: strut owns Workflow Builder and Job", async () => {
    const strut = await readdir(STRUT_CONCEPTS.dir);
    assert.deepEqual(strut.sort(), ["Job.md", "Workflow Builder.md"]);
    for (const set of LAB_CONCEPTS) for (const file of await readdir(set.dir)) assert.ok(!strut.includes(file), `${set.dir}/${file}`);
  });

  it("what ships to every workspace says nothing about one workspace's domain", async () => {
    const domain = /\b(law|legal|rubric|evals?|benchmark|exam|grading|overfit)\b/i;
    for (const set of LAB_CONCEPTS) {
      for (const file of await readdir(set.dir)) {
        const c = await read(set, file);
        for (const text of [c.name, c.description ?? "", c.docs ?? ""]) assert.doesNotMatch(text, domain, `${file}`);
      }
    }
    assert.doesNotMatch(JOB_SEED.system!, domain);
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

  // What the `Job` page said before strut shipped a generic one: the lab's, so it rides on the job's prompt.
  it("JOB_SEED's section: the Hive workspace and its peers, and browser-explore for looking at a page", () => {
    const sys = JOB_SEED.system!;
    for (const needle of [/one Hive workspace/, /@<slug>/, /strut\/run-workflow \{ peer: "<slug>", workflow, input \}/, /that is where its code lives/]) assert.match(sys, needle);
    for (const needle of [/meta\/run-workflow \{ name: "browser-explore"/, /shots\/NNN\.png/, /Never install or launch a browser/]) assert.match(sys, needle);
    // Strut's template already opens the Job page; the lab says nothing of it twice.
    assert.doesNotMatch(sys, /graph\/graph-get|`Job` Concept/);
  });
});

// Live, against a THROWAWAY Neo4j (strut's convention): skipped unless
// STRUT_TEST_NEO4J_URI is set. Seeds the jarvis ontology on first open.
const URI = process.env.STRUT_TEST_NEO4J_URI;
const liveEnv = () => ({
  NEO4J_URI: URI!,
  NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
  NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
  STRUT_GRAPH_SEED_ONTOLOGY: "1",
  STRUT_GRAPH_EMBEDDINGS: "off",
});

describe("anchorConcepts (live)", { skip: !URI }, () => {
  const tag = `T${Date.now().toString(36)}`;
  const P = `lab/test-${tag}/concepts/`;
  const root = `Root ${tag}`;
  const kid = `Kid ${tag}`;
  const theirs = `Theirs ${tag}`;
  const late = `Late ${tag}`;
  const wsId = `ws-${tag}`;
  let ws: import("strut").WorkspaceStore;
  let dir: string;
  let dataDir: string;
  const g = () => ws.graph! as import("strut").Neo4jBackend;
  const anchors = async (name: string) =>
    Number((await g().bolt.run(`MATCH (w:HiveWorkspace)-[e:PROCESS]->(c:Concept {name: $c}) RETURN count(e) AS n`, { c: name }))[0]?.n ?? 0);
  const seed = async () => {
    const outcomes = await seedConcepts(g(), [{ dir, prefix: P }]);
    await anchorConcepts(g(), outcomes);
    return outcomes;
  };
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    dataDir = await mkdtemp(join(tmpdir(), "concept-anchor-data-"));
    dir = await mkdtemp(join(tmpdir(), "concept-anchor-"));
    ws = (await graphWorkspaceFromEnv(liveEnv(), { dataDir })).workspace;
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root.\n---\nRoot docs.\n");
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs.\n`);
    await writeFile(join(dir, `${theirs}.md`), "---\ndescription: Shipped, but a person's.\n---\nTheirs.\n");
    // A person's Concept under a shipped name: the seed skips it, so it is never anchored.
    await g().nodes.write({ type: "Concept", data: { name: theirs, description: "A person's." } }, "create", { namespace: g().namespace });
  });
  after(async () => {
    if (ws?.graph) {
      await g().bolt.run(`MATCH (n:Concept) WHERE n.name ENDS WITH $tag DETACH DELETE n`, { tag: ` ${tag}` });
      await g().bolt.run(`MATCH (w:HiveWorkspace) WHERE w.workspace_id STARTS WITH $id DETACH DELETE w`, { id: wsId });
      await closeGraphBackends();
    }
    await rm(dir, { recursive: true, force: true });
    await rm(dataDir, { recursive: true, force: true });
  });

  it("no workspace node: nothing anchored, no error", async () => {
    const out = await seed();
    assert.deepEqual(out.map((o) => [o.name, o.action]).sort(), [[kid, "create"], [root, "create"], [theirs, "skip"]].sort());
    assert.deepEqual([await anchors(root), await anchors(kid)], [0, 0]);
  });

  it("every owned Concept, kept ones too, to the one HiveWorkspace node, idempotently; never someone else's; never with two", async () => {
    await g().nodes.write({ type: "HiveWorkspace", data: { workspace_id: wsId, name: "T", slug: `t-${tag}` } }, "create", { namespace: g().namespace });
    await seed(); // both "keep" (unchanged files) → anchored all the same
    assert.deepEqual([await anchors(root), await anchors(kid), await anchors(theirs)], [1, 1, 0], "a parent does not matter; theirs is never anchored");
    await seed();
    assert.deepEqual([await anchors(root), await anchors(kid)], [1, 1]);

    await g().nodes.write({ type: "HiveWorkspace", data: { workspace_id: `${wsId}-2`, name: "T2" } }, "create", { namespace: g().namespace });
    await writeFile(join(dir, `${late}.md`), "---\ndescription: Seeded late.\n---\nLate docs.\n");
    await seed(); // two workspace nodes → no guess
    assert.equal(await anchors(late), 0);
    assert.equal(await anchors(root), 1, "what is there stays");
    await g().bolt.run(`MATCH (w:HiveWorkspace {workspace_id: $id}) DETACH DELETE w`, { id: `${wsId}-2` });
    await seed();
    assert.equal(await anchors(late), 1);
  });

  // Prod, 2026-10-09: `Workflow Builder` was deleted in Learn. The seed restores a page of its own; the anchor follows.
  it("a deleted page of ours comes back anchored", async () => {
    await g().bolt.run(`MATCH (n:Concept {name: $name}) SET n.deleted_at = $now, n.is_deleted = true WITH n OPTIONAL MATCH (n)-[r]-() DELETE r`, { name: root, now: Date.now() });
    assert.equal(await anchors(root), 0);
    const out = await seed();
    assert.equal(out.find((o) => o.name === root)?.action, "update");
    assert.equal(await anchors(root), 1);
  });

  it("is nothing without a Neo4j graph", async () => {
    await anchorConcepts(undefined, [{ name: "X", action: "create", ref_id: "r" }]);
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

// End to end on a throwaway Neo4j: strut's pages and the lab's Janitor seeded
// for real, REAL graph/graph-get, only the agent faked — the reads by key and
// the PARENT_OF guard on real nodes, and the builder's section (strut's
// default chatSystem) read off the same graph with Janitor among its kinds.
describe("graph-janitor workflow + builder section (live graph, fake agent)", { skip: !URI }, () => {
  const NAMES = ["Workflow Builder", "Job", "Janitor", "Overfit Concept Janitor", "Law"];
  const agentCalls: Record<string, any>[] = [];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  let registry: Awaited<ReturnType<typeof buildRegistry>>["registry"];
  let services: Record<string, unknown>;
  let run: (input: Record<string, unknown>) => ReturnType<typeof runWorkflow>;
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    root = await mkdtemp(join(tmpdir(), "janitor-live-"));
    const env = liveEnv();
    ws = (await graphWorkspaceFromEnv(env, { dataDir: root })).workspace;
    await ws.graph!.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await seedArtifactSteps(ws);
    await seedJanitorSteps(ws, PIN_COMMITTED);
    await seedJanitorWorkflows(ws, PIN_COMMITTED);
    // The committed tree, for real — and a mandate, which no seed ships: the test plants the fixture.
    await seedConcepts(ws.graph, [STRUT_CONCEPTS, JANITOR_CONCEPTS, JANITOR_FIXTURES]);
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
    const text = await builderSystem({ graph: ws.graph, registry, services });
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

// A deployed graph has the lab's own `Workflow Builder` and `Job` pages,
// stamped by the lab's old seeder. The first boot on a strut that ships them
// takes both over by their legacy stamps: the same Workflow Builder text kept
// as it is, the lab's Job page replaced by strut's generic one — the same
// nodes, so the tree under them and their anchors stay.
describe("strut takes over the lab's root pages (live graph)", { skip: !URI }, () => {
  const NAMES = ["Workflow Builder", "Janitor", "Job", "Plan Mode", "Code Change", "Explore", "Verify", "Onboarding", "Pod", "Pod Config"];
  let root: string;
  let ws: import("strut").WorkspaceStore;
  const g = () => ws.graph! as import("strut").Neo4jBackend;
  const node = async (name: string) =>
    (await g().bolt.run(`MATCH (n:Concept {name: $name}) RETURN n.ref_id AS ref_id, n.docs AS docs, n.unique_source_id AS usid`, { name }))[0]!;
  const kids = async (parent: string) =>
    (await g().bolt.run(`MATCH (:Concept {name: $parent})-[:PARENT_OF]->(c:Concept) RETURN c.name AS name ORDER BY name`, { parent })).map((r) => String(r.name));
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    root = await mkdtemp(join(tmpdir(), "strut-roots-live-"));
    ws = (await graphWorkspaceFromEnv(liveEnv(), { dataDir: root })).workspace;
    await g().bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
  });
  after(async () => {
    await ws?.graph?.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await closeGraphBackends();
    await rm(root, { recursive: true, force: true });
  });

  it("Workflow Builder is kept, Job becomes strut's, the lab's pages stay under them, and a second boot writes nothing", async () => {
    // The tree as the lab's old seeder left it.
    const oldBuilder = join(root, "old-builder");
    const oldJob = join(root, "old-job");
    await mkdir(oldBuilder);
    await mkdir(oldJob);
    await writeFile(join(oldBuilder, "Workflow Builder.md"), await readFile(join(STRUT_CONCEPTS.dir, "Workflow Builder.md")));
    await writeFile(join(oldJob, "Job.md"), "---\ndescription: The lab's job page.\n---\nThis strut belongs to one Hive workspace.\n");
    await seedConcepts(g(), [{ dir: oldBuilder, prefix: "lab/builder/concepts/" }, { dir: oldJob, prefix: "lab/job/concepts/" }, ...LAB_CONCEPTS]);
    const before = { builder: await node("Workflow Builder"), job: await node("Job") };
    assert.match(String(before.job.usid), /^lab\/job\/concepts\/Job\.md@/);

    // The boot: strut's own call inside createStrut, then the lab's (createLabStrut.ts).
    await seedConcepts(g(), [STRUT_CONCEPTS]);
    const outcomes = await seedConcepts(g(), [STRUT_CONCEPTS, ...LAB_CONCEPTS]);
    const builder = await node("Workflow Builder");
    const job = await node("Job");
    assert.equal(builder.ref_id, before.builder.ref_id);
    assert.equal(builder.usid, before.builder.usid, "the same text: kept, stamp and all");
    assert.equal(job.ref_id, before.job.ref_id, "the same node");
    assert.match(String(job.usid), /^strut\/concepts\/Job\.md@/);
    assert.doesNotMatch(String(job.docs), /Hive workspace/, "strut's generic page");
    assert.deepEqual(await kids("Workflow Builder"), ["Janitor"]);
    assert.deepEqual(await kids("Job"), ["Code Change", "Explore", "Onboarding", "Plan Mode", "Verify"]);
    assert.deepEqual(await kids("Code Change"), ["Pod"]);
    assert.deepEqual(await kids("Pod"), ["Pod Config"]);
    assert.ok(outcomes.every((o) => o.action === "keep"), JSON.stringify(outcomes.filter((o) => o.action !== "keep")));

    // The next boot, both calls: nothing to write.
    assert.deepEqual(new Set((await seedConcepts(g(), [STRUT_CONCEPTS])).map((o) => o.action)), new Set(["keep"]));
    assert.deepEqual(new Set((await seedConcepts(g(), [STRUT_CONCEPTS, ...LAB_CONCEPTS])).map((o) => o.action)), new Set(["keep"]));
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
  const NAMES = ["Workflow Builder", "Job", "Janitor", "Overfit Concept Janitor", `Law ${tag}`, `Tort ${tag}`, `Crime ${tag}`];
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
    const env = liveEnv();
    ws = (await graphWorkspaceFromEnv(env, { dataDir: root })).workspace;
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    await seedArtifactSteps(ws);
    await seedJanitorSteps(ws, PIN_COMMITTED);
    await seedJanitorWorkflows(ws, PIN_COMMITTED);
    await seedConcepts(g, [STRUT_CONCEPTS, JANITOR_CONCEPTS, JANITOR_FIXTURES]);
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
    assert.match(String((await read("Workflow Builder")).usid), /^strut\/concepts\/Workflow Builder\.md@/);
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
