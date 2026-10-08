/**
 * pods OFFLINE smoke — no hive, no pod, no LLM. Seeds the pod/* steps into a
 * throwaway workspace, checks every one is discoverable, then runs the
 * job-shaped path over HTTP against a stand-in hive + staklink on
 * localhost — claim → latest → agent → branch-diff → push, with `job` on
 * the launch — and a revision turn (agent-start → agent-status → push onto
 * the same branch), and checks:
 *
 *  - the claim registered a HOLD on the job naming pod/release;
 *  - the pod's password is nowhere: not in the run's events, the output,
 *    the callback or the artifacts (only `sealed`), while every staklink
 *    call carried it;
 *  - the agent in the pod was started with the job as its session and this
 *    run's LLM gateway grant; the push went as the run's GitHub identity,
 *    named no base (the pod's default branch is the base) and opened a pull
 *    request; the revision pushed to the current branch
 *    and got the same pull request back, no second one;
 *  - `DELETE /jobs/:id` released the pod through pod/release (the stand-in
 *    hive saw drop-pod) and dropped the hold.
 *
 *   npx tsx src/lab/pods/smoke.ts
 */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, createStrut, jobsCapability, standardServices, type Strut } from "strut";
import { SEED_OPTS } from "../seed-opts.js";
import { SEED_STEPS, seedPodSteps } from "./seed.js";

const PASSWORD = "pw-s3cret-never-shown";
const KEY = "hiveorg_test_key";
const FRONTEND = "https://pod-1-3000.workspaces.test";
const IDE = "https://pod-1.workspaces.test";

interface Seen {
  method: string;
  path: string;
  auth?: string | undefined;
  token?: string | undefined;
  body?: any;
}

/** A stand-in hive (the pool manager) and staklink (under /pod). */
async function standIn(): Promise<{ base: string; seen: Seen[]; posts: any[]; close: () => void }> {
  const seen: Seen[] = [];
  const posts: any[] = [];
  const polls = new Map<string, number>();
  let agents = 0;
  let base = "";
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      const url = new URL(req.url ?? "/", "http://stand-in");
      const body = raw ? JSON.parse(raw) : undefined;
      const json = (status: number, b: unknown) => {
        res.writeHead(status, { "content-type": "application/json" });
        res.end(JSON.stringify(b));
      };
      // strut's run callback
      if (url.pathname === "/hook") {
        posts.push(body);
        return res.writeHead(204).end();
      }
      const auth = req.headers.authorization;
      const token = req.headers["x-api-token"] as string | undefined;
      seen.push({ method: req.method ?? "", path: url.pathname + url.search, auth, token, body });
      // hive's pool manager
      if (url.pathname.startsWith("/api/pool-manager/")) {
        if (token !== KEY) return json(401, { error: "Unauthorized" });
        if (url.pathname.startsWith("/api/pool-manager/claim-pod/")) {
          return json(200, { success: true, podId: "pod-1", pod_url: IDE, frontend: FRONTEND, ide: IDE, control: `${base}/pod`, password: PASSWORD });
        }
        if (url.pathname.startsWith("/api/pool-manager/drop-pod/")) return json(200, { success: true });
      }
      // staklink, as the pod
      if (url.pathname.startsWith("/pod/")) {
        if (auth !== `Bearer ${PASSWORD}`) return json(401, { error: "unauthorized" });
        const p = url.pathname.slice("/pod".length);
        if (p === "/latest") return json(200, { request_id: "req-latest", status: "pending" });
        if (p === "/agent") return json(200, { request_id: `req-agent-${++agents}`, status: "pending" });
        if (p === "/script_progress") {
          const id = url.searchParams.get("request_id") ?? "";
          const n = (polls.get(id) ?? 0) + 1;
          polls.set(id, n);
          if (n < 2) return json(200, { status: "pending" });
          if (id === "req-latest") return json(200, { status: "completed", result: { success: true } });
          return json(200, { status: "completed", result: { success: true, result: `Changed the title (${id}).`, summary: "Title changed", usage: { total_tokens: 10 }, model: "sonnet" } });
        }
        if (p === "/branch-diff") return json(200, [{ file: "README.md", action: "modify", content: "# App\n", repoName: "app", errors: [] }]);
        if (p === "/push") {
          const pr = url.searchParams.get("pr") === "true";
          return json(200, { success: true, commits: ["abc123"], branches: { app: "strut/title" }, ...(pr ? { prs: { app: "https://github.com/o/app/pull/7" } } : {}) });
        }
      }
      json(404, { error: `no stand-in for ${req.method} ${url.pathname}` });
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  return { base, seen, posts, close: () => server.close() };
}

/** `strut.app.request` with the key the deployment expects (none in dev). */
function apiFor(strut: Strut<any>) {
  const key = process.env["STRUT_API_KEY"];
  const auth: Record<string, string> = key ? { authorization: `Bearer ${key}` } : {};
  return async (path: string, init?: { method?: string; body?: unknown }) => {
    const res = await strut.app.request(path, {
      method: init?.method ?? (init?.body ? "POST" : "GET"),
      headers: { "content-type": "application/json", ...auth },
      ...(init?.body ? { body: JSON.stringify(init.body) } : {}),
    });
    const text = await res.text();
    return { status: res.status, text, json: text ? JSON.parse(text) : null };
  };
}

async function until(cond: () => boolean, what: string, ms = 30_000): Promise<void> {
  const t0 = Date.now();
  while (!cond() && Date.now() - t0 < ms) await new Promise((r) => setTimeout(r, 25));
  assert.ok(cond(), `timed out waiting for ${what}`);
}

const POD_SMOKE = `name: pod-smoke
input:
  workspace: { type: string }
steps:
  - id: dir
    type: job/dir
  - id: claim
    type: pod/claim
    config: { workspace: "{{ input.workspace }}" }
  - id: latest
    type: pod/latest
    config:
      control: "{{ claim.control }}"
      sealed: "{{ claim.sealed }}"
      repos: [{ url: "https://github.com/o/app.git", base_branch: main }]
      pollMs: 10
  - id: agent
    type: pod/agent
    config: { control: "{{ claim.control }}", sealed: "{{ claim.sealed }}", prompt: "Change the title.", repoName: app, pollMs: 10 }
  - id: diff
    type: pod/branch-diff
    config: { control: "{{ claim.control }}", sealed: "{{ claim.sealed }}", base: main }
  - id: push
    type: pod/push
    config:
      control: "{{ claim.control }}"
      sealed: "{{ claim.sealed }}"
      repo_url: "https://github.com/o/app.git"
      branch_name: strut/title
      commit_message: "Change the title"
  - id: result
    type: pack
    config:
      podId: "{{ claim.podId }}"
      control: "{{ claim.control }}"
      sealed: "{{ claim.sealed }}"
      agent: "{{ agent.output }}"
      session: "{{ agent.session }}"
      files: "{{ diff.total }}"
      branch: "{{ push.branch }}"
      pr: "{{ push.pr_url }}"
      artifacts:
        - { id: pod, kind: url, title: Pod, url: "{{ claim.frontend }}" }
`;

const POD_REVISE = `name: pod-revise
input:
  control: { type: string }
  sealed: { type: string }
steps:
  - id: start
    type: pod/agent-start
    config: { control: "{{ input.control }}", sealed: "{{ input.sealed }}", prompt: "Also fix the subtitle." }
  - id: look
    type: pod/agent-status
    config: { control: "{{ input.control }}", sealed: "{{ input.sealed }}", request_id: "{{ start.request_id }}" }
  - id: done
    type: pod/agent-status
    config: { control: "{{ input.control }}", sealed: "{{ input.sealed }}", request_id: "{{ start.request_id }}" }
  - id: push
    type: pod/push
    config:
      control: "{{ input.control }}"
      sealed: "{{ input.sealed }}"
      repo_url: "https://github.com/o/app.git"
      branch_name: strut/title
      commit_message: "Fix the subtitle"
      base_branch: main
      stay_on_branch: true
  - id: result
    type: pack
    config:
      session: "{{ start.session }}"
      first: "{{ look.status }}"
      status: "{{ done.status }}"
      output: "{{ done.output }}"
      branch: "{{ push.branch }}"
      pr: "{{ push.pr_url }}"
`;

async function main() {
  const dir = mkdtempSync(join(process.cwd(), "pods-smoke-"));
  const hive = await standIn();
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedPodSteps(workspace);
    const { registry } = await buildRegistry(await workspace.materializeCustomSteps());
    for (const name of SEED_STEPS) {
      if (name.startsWith("_")) continue;
      assert.ok(registry[`pod/${name}`], `registry missing pod/${name}`);
    }
    console.log(`✔ seeded ${SEED_STEPS.length - 1} pod/* steps, all discoverable`);
    await workspace.publishWorkflowByContent("pod-smoke", POD_SMOKE, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-revise", POD_REVISE, undefined, "pods", undefined, SEED_OPTS);

    // ── 2. a strut whose secrets name the stand-in hive, whose GitHub
    //       identity lookup is answered locally, and whose LLM gateway
    //       grant is a fake ────────────────────────────────────────────────
    const fetchImpl: typeof fetch = async (url, init) =>
      String(url).startsWith("https://api.github.com/user")
        ? new Response(JSON.stringify({ login: "octo" }), { status: 200, headers: { "content-type": "application/json" } })
        : fetch(url, init);
    const grant = { apiKey: "sk-bf-test", baseUrl: `${hive.base}/llm`, headers: { "x-macaroon": "mac-test", "x-bf-dim-session-id": "job" } };
    const strut = await createStrut({
      workspace,
      services: {
        ...standardServices({ fetchImpl: fetchImpl as any, secretsSource: { HIVE_URL: hive.base, HIVE_API_KEY: KEY, GITHUB_TOKEN: "ghp_test" }, dataDir: dir }),
        llmAuth: async () => grant,
      },
      serveUi: false,
      enableChat: false,
      scheduler: false,
    });
    const api = apiFor(strut);
    const job = randomUUID();
    const callback = { url: `${hive.base}/hook` };

    // ── 3. the first turn: claim → latest → agent → diff → push ──────────
    const first = await api("/workflows/pod-smoke/run", { body: { job, input: { workspace: "ws-1" }, callback } });
    assert.equal(first.status, 202, first.text);
    await until(() => hive.posts.length === 1, "the first callback");
    const post1 = hive.posts[0];
    assert.equal(post1.status, "success", JSON.stringify(post1));
    const out1 = post1.output;
    assert.equal(out1.podId, "pod-1");
    assert.equal(out1.control, `${hive.base}/pod`);
    assert.match(out1.sealed, /^v1\.[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+$/, "the password rides sealed");
    assert.equal(out1.session, job, "the agent's session is the job");
    assert.equal(out1.agent, "Changed the title (req-agent-1).");
    assert.equal(out1.files, 1);
    assert.deepEqual([out1.branch, out1.pr], ["strut/title", "https://github.com/o/app/pull/7"]);
    assert.deepEqual(post1.artifacts, [{ id: "pod", kind: "url", title: "Pod", url: FRONTEND }]);
    console.log(`✔ turn 1: claimed pod-1, reset, the agent worked, pushed ${out1.branch} → ${out1.pr}`);

    // The hold, naming pod/release with what it needs.
    const holds = await jobsCapability(dir).holds(job);
    assert.equal(holds.length, 1);
    assert.deepEqual({ ...holds[0], since: undefined }, { id: "pod-1", kind: "pod", release: { type: "pod/release", input: { workspace: "ws-1", podId: "pod-1" } }, note: FRONTEND, since: undefined });
    console.log(`✔ the job holds pod-1, release = pod/release ${JSON.stringify(holds[0]!.release.input)}`);

    // The password is nowhere strut keeps: events, output, callback.
    const events = await api(`/workflows/pod-smoke/runs/${first.json.runId}/events`);
    assert.equal(events.status, 200);
    assert.ok(!events.text.includes(PASSWORD), "the password is not in the run's events");
    assert.ok(!JSON.stringify(post1).includes(PASSWORD), "nor in the callback");
    assert.ok(events.text.includes(out1.sealed), "the sealed form is (it is what the agent carries)");
    // …while every call to the pod carried it, and every call to hive the org key.
    const pod = hive.seen.filter((s) => s.path.startsWith("/pod/"));
    assert.ok(pod.length >= 6, `calls to the pod: ${pod.length}`);
    assert.ok(pod.every((s) => s.auth === `Bearer ${PASSWORD}`), "every staklink call as the pod");
    assert.ok(hive.seen.filter((s) => s.path.startsWith("/api/")).every((s) => s.token === KEY), "every hive call with the org key");
    console.log(`✔ the password reached only the pod (${pod.length} calls); strut's record carries \`sealed\``);

    // What the pod was told.
    const latest = hive.seen.find((s) => s.path === "/pod/latest")!;
    assert.equal(latest.method, "PUT");
    assert.deepEqual(latest.body, { tasks: [], repos: [{ url: "https://github.com/o/app.git", base_branch: "main" }], git_credentials: { provider: "github", auth_type: "pat", auth_data: { token: "ghp_test", username: "octo" } } });
    const agent = hive.seen.find((s) => s.path === "/pod/agent")!;
    // The macaroon rides inside the key (`<vk>.<macaroon>`, split by the gateway's
    // wrapper), never as a header goose would drop; the dims still go as headers.
    assert.deepEqual(agent.body, {
      prompt: "Change the title.",
      apiKey: "sk-bf-test.mac-test",
      baseUrl: `${hive.base}/llm`,
      repoName: "app",
      session: job,
      headers: { "x-bf-dim-session-id": "job" },
    });
    assert.equal(hive.seen.find((s) => s.path.startsWith("/pod/branch-diff"))!.path, "/pod/branch-diff?base=main");
    const push = hive.seen.find((s) => s.path.startsWith("/pod/push"))!;
    assert.equal(push.path, "/pod/push?commit=true&pr=true");
    assert.deepEqual(push.body.repos, [{ url: "https://github.com/o/app.git", branch_name: "strut/title", commit_name: "Change the title" }]);
    assert.equal(push.body.git_credentials.auth_data.username, "octo");
    console.log(`✔ latest as octo on main; the agent with session=${job.slice(0, 8)}… and the run's gateway grant; push opened the PR`);

    // ── 4. a revision turn: start → status (pending, then done) → push onto
    //       the current branch; `create_pr` stays on and the SAME PR comes back ─
    const second = await api("/workflows/pod-revise/run", { body: { job, input: { control: out1.control, sealed: out1.sealed }, callback } });
    assert.equal(second.status, 202, second.text);
    await until(() => hive.posts.length === 2, "the second callback");
    const post2 = hive.posts[1];
    assert.equal(post2.status, "success", JSON.stringify(post2));
    assert.deepEqual(post2.output, { session: job, first: "pending", status: "completed", output: "Changed the title (req-agent-2).", branch: "strut/title", pr: "https://github.com/o/app/pull/7" });
    const push2 = hive.seen.filter((s) => s.path.startsWith("/pod/push"))[1]!;
    assert.equal(push2.path, "/pod/push?commit=true&pr=true&stayOnCurrentBranch=true");
    console.log(`✔ turn 2: agent-start → agent-status (pending, completed) → push stayed on ${post2.output.branch}, the same PR came back`);

    // ── 5. closing the job releases the pod ──────────────────────────────
    const drops = () => hive.seen.filter((s) => s.path.startsWith("/api/pool-manager/drop-pod/"));
    assert.equal(drops().length, 0);
    const gone = await api(`/jobs/${job}`, { method: "DELETE" });
    assert.equal(gone.status, 200, gone.text);
    assert.deepEqual(gone.json, { ok: true, job, released: ["pod-1"] });
    assert.equal(drops().length, 1);
    assert.equal(drops()[0]!.path, "/api/pool-manager/drop-pod/ws-1?podId=pod-1");
    assert.deepEqual(await jobsCapability(dir).holds(job), []);
    assert.equal((await api(`/jobs/${job}/files`)).status, 404);
    console.log(`✔ DELETE /jobs/${job.slice(0, 8)}… released pod-1 through pod/release (hive saw drop-pod) and the job is gone`);

    console.log("\npods smoke: all good");
  } finally {
    hive.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("pods smoke FAILED:", e);
  process.exit(1);
});
