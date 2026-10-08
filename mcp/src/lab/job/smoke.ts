/**
 * job OFFLINE smoke — no LLM, no network. Seeds `job` into a throwaway
 * workspace, checks the steps it leans on are discoverable, static-validates
 * it through the authoring capability (what meta/validate-workflow runs)
 * against the REAL registry, then RUNS it the way hive does — over HTTP,
 * with `job` on the launch and a local callback server — with the `agent`
 * step swapped for a fake that writes `plan.md` into its cwd and answers in
 * the step's output shape:
 *
 *  - two launches with the SAME job: both 202 `{ callback: true }`, both
 *    callbacks carry `artifacts[0]` as `{ id: "plan", kind: "markdown",
 *    title: "Plan", url: "/jobs/<job>/files/plan.md" }`, the file behind
 *    that link is the SECOND turn's, served with the sandbox headers; the
 *    fake saw the job's directory as `cwd` and the job id as `session`;
 *  - a launch WITHOUT a job resolves to `/artifacts/<runId>/plan.md` and
 *    the fake saw no `session` (a cold, one-shot agent);
 *  - a launch with the same job and `session` on the input: the same
 *    directory and link, the fake saw THAT session — the thread and the
 *    files are tied by the YAML's default, not by strut.
 *
 *   npx tsx src/lab/job/smoke.ts
 *
 * With JOB_SMOKE_LIVE=1 (and a provider key in env) it also runs the REAL
 * agent once, with `job` on the launch, and prints the callback.
 */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, createRegistry, createStrut, defineStep, type Strut } from "strut";
import { seedJobWorkflows } from "./seed.js";

const JOB = "job";
const STEPS = ["job/dir", "agent", "pack"];
// Registry steps granted ON TOP of the agent's built-ins (files, bash,
// web_search, web_fetch — which a filter must never narrow): graph reads for
// the Concept tree, the meta/* read + run tools (a turn runs another
// workflow — `code-change-pr` — as a child run under the job), and the
// pod/* tools (a hive sandbox the job holds between turns).
const TOOLS: string[] = [
  "graph/graph-search",
  "graph/graph-get",
  "graph/graph-neighbors",
  "graph/get-ontology",
  "meta/list-workflows",
  "meta/get-workflow",
  "meta/run-workflow",
  "meta/get-run",
  "pod/*",
];

/** What the fake agent was handed each turn — the resolved step config. */
interface Turn {
  cwd: string;
  session: string | undefined;
  prompt: string;
  config: Record<string, unknown>;
}

/** Stands in for the core `agent` step: writes the plan into its cwd and
 *  answers in the step's output shape (`object` is the structured answer). */
function fakeAgent(turns: Turn[]) {
  return defineStep({
    type: "agent",
    input: z.object({ cwd: z.string(), prompt: z.string(), session: z.string().optional() }).passthrough(),
    output: z.any(),
    async run(cfg) {
      turns.push({ cwd: cfg.cwd, session: cfg.session, prompt: cfg.prompt, config: cfg });
      writeFileSync(join(cfg.cwd, "plan.md"), `# Plan (turn ${turns.length})\n\n${cfg.prompt}\n`);
      return {
        result: "wrote the plan",
        object: { text: "wrote the plan", artifacts: [{ id: "plan", title: "Plan", path: "plan.md" }] },
        steps: 1,
        usage: {},
        cost: 0,
      };
    },
  });
}

/** A stand-in host: collects what strut posts to the callback URL. */
async function callbackHost(): Promise<{ url: string; posts: any[]; close: () => void }> {
  const posts: any[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      posts.push(JSON.parse(body));
      res.writeHead(204).end();
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as { port: number }).port;
  return { url: `http://127.0.0.1:${port}/hook?token=s3cret`, posts, close: () => server.close() };
}

async function until(cond: () => boolean, what: string, ms = 30_000): Promise<void> {
  const t0 = Date.now();
  while (!cond() && Date.now() - t0 < ms) await new Promise((r) => setTimeout(r, 25));
  assert.ok(cond(), `timed out waiting for ${what}`);
}

/** `strut.app.request` with the key the deployment expects (none in dev). */
function apiFor(strut: Strut<any>) {
  const key = process.env["STRUT_API_KEY"];
  const auth: Record<string, string> = key ? { authorization: `Bearer ${key}` } : {};
  return async (path: string, body?: unknown) => {
    const res = await strut.app.request(path, {
      method: body ? "POST" : "GET",
      headers: { "content-type": "application/json", ...auth },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { status: res.status, headers: res.headers, text: await res.text() };
  };
}
const asJson = (r: { text: string }) => JSON.parse(r.text);

async function main() {
  const dir = mkdtempSync(join(process.cwd(), "job-smoke-"));
  const host = await callbackHost();
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedJobWorkflows(workspace);
    const { registry } = await buildRegistry(await workspace.materializeCustomSteps());
    for (const t of STEPS) assert.ok(registry[t], `registry missing ${t}`);
    const entry = (await workspace.listWorkflows()).find((w) => w.name === JOB);
    assert.ok(entry, `${JOB} not seeded`);
    assert.equal(entry!.category, "job");
    console.log(`✔ seeded ${JOB} (category job); ${STEPS.join(", ")} discoverable`);

    // ── 2. static validation against the REAL registry (what
    //       meta/validate-workflow runs) ────────────────────────────────────
    const real = await createStrut({ workspace, serveUi: false, enableChat: false, scheduler: false });
    const authoring = (real.services as any).authoring;
    const yaml = await workspace.getWorkflowSource(JOB, entry!.activeVersion);
    const v = await authoring.validateWorkflow(yaml, JOB);
    assert.equal(v.ok, true, `${JOB}: ${JSON.stringify(v.errors, null, 2)}`);
    if (v.warnings.length) console.log(`  warnings:`, v.warnings.map((w: any) => `${w.path}: ${w.message}`));
    const flow = await workspace.getWorkflow(JOB);
    assert.deepEqual(flow.params?.["tools"], TOOLS);
    assert.ok(flow.inputBlock?.["prompt"], "the input block declares prompt");
    assert.equal(flow.inputBlock?.["workspace"]?.required, false, "the input block declares workspace, optional");
    assert.equal(flow.inputBlock?.["session"]?.required, false, "the input block declares session, optional");
    console.log(`✔ ${JOB} validates (${v.summary.steps} steps); params.tools = ${JSON.stringify(TOOLS)}`);

    // ── 3. the run, as hive does it: over HTTP with `job` + a callback, the
    //       agent a fake ───────────────────────────────────────────────────
    const turns: Turn[] = [];
    const strut = await createStrut({
      workspace,
      registry: await createRegistry([fakeAgent(turns)]),
      serveUi: false,
      enableChat: false,
      scheduler: false,
    });
    const api = apiFor(strut);
    const job = randomUUID();

    const first = await api(`/workflows/${JOB}/run`, { job, input: { prompt: "Plan dark mode.", workspace: "ws-1" }, callback: { url: host.url } });
    assert.equal(first.status, 202, first.text);
    const firstBody = asJson(first);
    assert.equal(firstBody.callback, true, "the 202 carries callback: true");
    assert.equal(typeof firstBody.runId, "string");
    await until(() => host.posts.length === 1, "the first callback");
    const post1 = host.posts[0];
    assert.equal(post1.event, "run.end");
    assert.equal(post1.workflow, JOB);
    assert.equal(post1.runId, firstBody.runId);
    assert.equal(post1.status, "success", JSON.stringify(post1));
    assert.deepEqual(post1.artifacts, [{ id: "plan", kind: "markdown", title: "Plan", url: `/jobs/${job}/files/plan.md` }]);
    // `output` is what the workflow packed — the agent's own list, unresolved.
    assert.equal(post1.output.text, "wrote the plan");
    assert.deepEqual(post1.output.artifacts, [{ id: "plan", title: "Plan", path: "plan.md" }]);
    assert.equal(post1.output.ask, undefined);
    assert.equal(post1.output.cost, 0);
    assert.equal(turns.length, 1);
    assert.equal(turns[0]!.cwd, join(dir, "jobs", job), "cwd is the job's directory");
    assert.equal(turns[0]!.session, job, "session is the job id");
    assert.equal(turns[0]!.prompt, "Hive workspace: ws-1. Plan dark mode.", "the workspace heads the prompt");
    // The params reached the step with their types intact.
    const cfg = turns[0]!.config;
    assert.deepEqual(cfg["agentTools"], TOOLS, "params.tools reaches the step as agentTools");
    assert.equal(cfg["toolFilter"], undefined, "no filter on the built-ins");
    assert.equal(cfg["maxSteps"], 500);
    assert.equal(cfg["cacheTtl"], "1h");
    assert.equal(cfg["model"], "claude-sonnet-5-5");
    assert.match(String(cfg["system"]), /^You are working on a JOB/);
    assert.equal((cfg["schema"] as any)?.required?.join(","), "text,artifacts");
    console.log(`✔ turn 1: 202 { runId: ${firstBody.runId}, callback: true } → callback artifacts[0] = ${JSON.stringify(post1.artifacts[0])}`);

    const second = await api(`/workflows/${JOB}/run`, { job, input: { prompt: "Split step 2 in two." }, callback: { url: host.url } });
    assert.equal(second.status, 202, second.text);
    const secondBody = asJson(second);
    assert.equal(secondBody.callback, true);
    await until(() => host.posts.length === 2, "the second callback");
    const post2 = host.posts[1];
    assert.equal(post2.status, "success", JSON.stringify(post2));
    assert.deepEqual(post2.artifacts, post1.artifacts, "the same id behind the same link");
    assert.equal(turns.length, 2);
    assert.equal(turns[1]!.cwd, turns[0]!.cwd, "the same directory every turn");
    assert.equal(turns[1]!.session, job);
    assert.equal(turns[1]!.prompt, "Split step 2 in two.", "no workspace on the launch: the prompt as given");
    console.log(`✔ turn 2: the same artifact { id: plan, url: ${post2.artifacts[0].url} }`);

    // The file behind the link is the second turn's, served sandboxed.
    const file = await api(`/jobs/${job}/files/plan.md`);
    assert.equal(file.status, 200, file.text);
    assert.match(file.text, /^# Plan \(turn 2\)/);
    assert.match(file.text, /Split step 2 in two\./);
    assert.equal(file.headers.get("content-type"), "text/markdown; charset=utf-8");
    assert.equal(file.headers.get("content-security-policy"), "sandbox");
    assert.equal(file.headers.get("x-content-type-options"), "nosniff");
    const list = await api(`/jobs/${job}/files`);
    assert.deepEqual(asJson(list), { job, files: ["plan.md"] });
    // The same list for a host that missed the callback; the summary is stamped.
    const refs = asJson(await api(`/workflows/${JOB}/runs/${secondBody.runId}/artifacts`));
    assert.equal(refs.job, job);
    assert.deepEqual(refs.artifacts, post2.artifacts);
    assert.equal(asJson(await api(`/workflows/${JOB}/runs/${secondBody.runId}`)).job, job);
    console.log(`✔ GET /jobs/${job}/files/plan.md serves turn 2 with content-security-policy: sandbox + x-content-type-options: nosniff; GET …/runs/${secondBody.runId}/artifacts matches the callback`);

    // ── 4. no job: the same YAML is a one-shot in the run's artifact dir ──
    const plain = await api(`/workflows/${JOB}/run`, { input: { prompt: "One-shot." }, callback: { url: host.url } });
    assert.equal(plain.status, 202, plain.text);
    const plainBody = asJson(plain);
    await until(() => host.posts.length === 3, "the plain run's callback");
    const post3 = host.posts[2];
    assert.equal(post3.status, "success", JSON.stringify(post3));
    assert.deepEqual(post3.artifacts, [{ id: "plan", kind: "markdown", title: "Plan", url: `/artifacts/${plainBody.runId}/plan.md` }]);
    assert.equal(turns.length, 3);
    assert.equal(turns[2]!.session, undefined, "no job → no session (a cold agent)");
    assert.equal(turns[2]!.cwd, join(dir, "artifacts", plainBody.runId), "cwd is the run's own artifact directory");
    const art = await api(`/artifacts/${plainBody.runId}/plan.md`);
    assert.equal(art.status, 200);
    assert.match(art.text, /^# Plan \(turn 3\)/);
    assert.ok(existsSync(join(dir, "jobs", job, "plan.md")), "the job's file is untouched by the plain run");
    assert.match(readFileSync(join(dir, "jobs", job, "plan.md"), "utf8"), /^# Plan \(turn 2\)/);
    console.log(`✔ without a job: artifacts[0].url = ${post3.artifacts[0].url}, no session`);

    // ── 5. the same job, another thread: `session` on the input ──────────
    const other = await api(`/workflows/${JOB}/run`, { job, input: { prompt: "Fresh eyes.", session: "thread-b" }, callback: { url: host.url } });
    assert.equal(other.status, 202, other.text);
    await until(() => host.posts.length === 4, "the other thread's callback");
    const post4 = host.posts[3];
    assert.equal(post4.status, "success", JSON.stringify(post4));
    assert.deepEqual(post4.artifacts, post1.artifacts, "the same job's file behind the same link");
    assert.equal(turns.length, 4);
    assert.equal(turns[3]!.cwd, turns[0]!.cwd, "the job's directory, as every turn");
    assert.equal(turns[3]!.session, "thread-b", "the session named on the launch, not the job id");
    assert.match(readFileSync(join(dir, "jobs", job, "plan.md"), "utf8"), /^# Plan \(turn 4\)/);
    console.log(`✔ same job, session on the input: cwd unchanged, session = thread-b`);

    // ── 6. optional: the real agent, one turn ────────────────────────────
    if (process.env["JOB_SMOKE_LIVE"] === "1") {
      const liveApi = apiFor(real);
      const liveJob = randomUUID();
      const t0 = Date.now();
      const launched = await liveApi(`/workflows/${JOB}/run`, {
        job: liveJob,
        input: { prompt: "Write a three-step plan for adding dark mode to a web app" },
        callback: { url: host.url },
      });
      assert.equal(launched.status, 202, launched.text);
      const { runId } = asJson(launched);
      console.log(`live: launched ${JOB} run ${runId} under job ${liveJob}`);
      await until(() => host.posts.length === 5, "the live callback", 15 * 60_000);
      const live = host.posts[4];
      console.log(`live run ${live.status} in ${Date.now() - t0} ms — the callback:`);
      console.log(JSON.stringify({ ...live, transcripts: live.transcripts }, null, 2));
      const url: string | undefined = live.artifacts?.[0]?.url;
      if (url) {
        const body = await liveApi(url);
        console.log(`\n${url} (HTTP ${body.status}):\n${body.text.slice(0, 2000)}`);
      }
    }
    console.log("\njob smoke: all good");
  } finally {
    host.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("job smoke FAILED:", e);
  process.exit(1);
});
