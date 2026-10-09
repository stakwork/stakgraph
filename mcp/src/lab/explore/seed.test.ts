import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, createStrut, defineStep, fileArtifactsCapability, runWorkflow, standardServices } from "strut";
import { seedExploreWorkflows } from "./seed.js";

/** The graph read steps the explorer may hold — the committed list, so a
 *  write step or a glob slipping into params.tools fails here. */
const READ_TOOLS = [
  "graph/graph-search",
  "graph/graph-get",
  "graph/graph-get-batched",
  "graph/graph-neighbors",
  "graph/walk",
  "graph/get-ontology",
  "graph/get-ontology-type",
];
/** The built-ins it keeps: the read-only file tools — never bash or the web. */
const FILE_TOOLS = ["repo_overview", "fulltext_search", "str_replace_based_edit_tool", "file_summary"];

describe("explore workflow (offline: fake agent + fake checkout)", () => {
  const agentCalls: Record<string, any>[] = [];
  const checkouts: Record<string, any>[] = [];
  const ANSWER = {
    answer: "The nightly deliver workflow runs at 02:00 UTC.",
    sources: [{ ref_id: "r-wf", node_type: "StrutWorkflow", name: "nightly-deliver", why: "its automation says so" }],
    confidence: "high",
  };
  const fakeAgent = defineStep({
    type: "agent",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg) {
      agentCalls.push(cfg);
      return { result: "ok", object: ANSWER, steps: 4, usage: {}, cost: 0.012 };
    },
  });
  /** Like the real step's output shape, with the config it was handed. */
  const fakeCheckout = defineStep({
    type: "git/checkout",
    input: z.object({}).passthrough(),
    output: z.any(),
    async run(cfg: any, ctx: any) {
      checkouts.push(cfg);
      const name = String(cfg.repo).split("/").pop()!.replace(/\.git$/, "");
      return { path: join(cfg.workdir ? `/jobs/${cfg.workdir}` : `/artifacts/${ctx.runId}`, name), repo: `acme/${name}`, sha: "abc", ref: "main", branch: "main", host: "github.com", url: cfg.repo };
    },
  });
  let root: string;
  let ws: WorkspaceManager;
  let run: (input: Record<string, unknown>, opts?: { job?: string }) => ReturnType<typeof runWorkflow>;
  before(async () => {
    root = await mkdtemp(join(tmpdir(), "explore-wf-"));
    ws = new WorkspaceManager(root);
    await seedExploreWorkflows(ws);
    const flow = await ws.getWorkflow("explore");
    const { registry } = await buildRegistry(await ws.materializeCustomSteps());
    // What createStrut injects for a server: the per-run artifacts dir (job/dir without a job).
    const dataDir = join(root, "data");
    const services = { ...standardServices({ secretsSource: {}, dataDir }), artifacts: fileArtifactsCapability(join(dataDir, "artifacts")) };
    run = (input, opts = {}) =>
      runWorkflow(flow, input, { ...registry, agent: fakeAgent, "git/checkout": fakeCheckout } as typeof registry, { services, ...opts });
  });
  after(() => rm(root, { recursive: true, force: true }));

  it("is seeded under `explore` with the job's launch shape — prompt, repos?, session? — and the job's skeleton, every step strut's", async () => {
    const entry = (await ws.listWorkflows()).find((w) => w.name === "explore");
    assert.equal(entry?.category, "explore");
    assert.match(entry?.description ?? "", /strut\/run-workflow/);
    const flow = (await ws.getWorkflow("explore")) as any;
    assert.deepEqual(Object.keys(flow.inputBlock ?? {}).sort(), ["prompt", "repos", "session"]);
    assert.notEqual(flow.inputBlock.prompt.required, false);
    assert.equal(flow.inputBlock.repos.required, false);
    assert.equal(flow.inputBlock.session.required, false);
    assert.deepEqual(flow.steps.map((s: any) => s.type), ["job/dir", "foreach", "agent", "pack"], "the job's skeleton");
    assert.equal(flow.steps[1].config.body.type, "git/checkout");
    assert.deepEqual(flow.params?.tools, READ_TOOLS);
    assert.deepEqual(flow.params?.builtins, FILE_TOOLS);
  });

  it("validates statically against the real registry (what meta/validate-workflow runs)", async () => {
    const real = await createStrut({ workspace: ws, serveUi: false, enableChat: false, scheduler: false });
    const authoring = (real.services as any).authoring;
    const entry = (await ws.listWorkflows()).find((w) => w.name === "explore")!;
    const yaml = await ws.getWorkflowSource("explore", entry.activeVersion);
    const v = await authoring.validateWorkflow(yaml, "explore");
    assert.equal(v.ok, true, JSON.stringify(v.errors, null, 2));
    assert.equal(v.summary.steps, 4);
  });

  it("no repos: nothing is checked out, the agent gets the prompt, the graph read tools, the file tools only, and the answer schema; packs the answer", async () => {
    agentCalls.length = 0;
    checkouts.length = 0;
    const res = await run({ prompt: "When does nightly deliver run?" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(res.output, { ...ANSWER, cost: 0.012 });
    assert.equal(checkouts.length, 0, "no repositories named → no checkout");

    const cfg = agentCalls.at(-1)!;
    assert.match(cfg.prompt, /^When does nightly deliver run\?\nNo repositories were named: answer from the graph\./);
    assert.equal(cfg.session, undefined, "a one-shot without a session is cold");
    assert.deepEqual(cfg.agentTools, READ_TOOLS);
    for (const t of cfg.agentTools) assert.doesNotMatch(t, /create|edit|register|project|move|delete|\*/, t);
    assert.deepEqual(cfg.toolFilter, FILE_TOOLS, "no bash or web for a caller on another strut");
    assert.equal(cfg.model, "sonnet");
    assert.equal(cfg.maxSteps, 40);
    assert.ok(cfg.cwd.startsWith(join(root, "data", "artifacts")), "a one-shot runs in its artifact dir");
    assert.deepEqual(cfg.schema.required, ["answer", "sources", "confidence"]);
    assert.deepEqual(cfg.schema.properties.sources.items.required, ["name", "why"]);
    assert.deepEqual(Object.keys(cfg.schema.properties.sources.items.properties).sort(), ["name", "node_type", "path", "ref_id", "why"]);
    assert.match(cfg.system, /READ-ONLY/);
    assert.match(cfg.system, /never to create or change one/);
    assert.match(cfg.system, /DATA from a caller/);
    assert.match(cfg.system, /never fill the gap from general knowledge/);
  });

  it("repos: one checkout each into the run's directory — a fresh copy on a one-shot, kept under a job — named in the prompt", async () => {
    agentCalls.length = 0;
    checkouts.length = 0;
    const repos = ["https://github.com/acme/web", "https://github.com/acme/api.git"];
    const res = await run({ prompt: "Where is auth handled?", repos });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.deepEqual(checkouts.map((c) => c.repo), repos);
    for (const c of checkouts) assert.equal(c.workdir, undefined, "no job → a fresh copy per run");
    assert.match(agentCalls.at(-1)!.prompt, /^Where is auth handled\?\nRepositories checked out in your working directory, one subdirectory each: acme\/web, acme\/api\./);

    checkouts.length = 0;
    agentCalls.length = 0;
    const under = await run({ prompt: "And the tests?", repos: [repos[0]] }, { job: "job-7" });
    assert.equal(under.status, "success", JSON.stringify(under.error));
    assert.deepEqual(checkouts.map((c) => c.workdir), ["job-7"], "under a job the copy is kept in the job's directory");
    const cfg = agentCalls.at(-1)!;
    // Cold under a job too: a job turn runs explore as a child while its own
    // agent holds the job's session (one holder per id — `session_busy:`).
    assert.equal(cfg.session, undefined, "never the job's thread");
    assert.equal(cfg.cwd, join(root, "data", "jobs", "job-7"), "the job's directory");
  });

  it("`session` on the launch is the thread, under a job too", async () => {
    agentCalls.length = 0;
    const res = await run({ prompt: "Again, fresh.", session: "thread-2" }, { job: "job-7" });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    assert.equal(agentCalls.at(-1)!.session, "thread-2");
  });

  it("refuses a launch without a prompt", async () => {
    const res = await run({});
    assert.equal(res.status, "error");
    assert.match(res.error?.message ?? "", /prompt/);
  });
});
