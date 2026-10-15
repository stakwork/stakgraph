/**
 * pods OFFLINE smoke — no hive, no pod, no LLM. Seeds the pod/* steps into a
 * throwaway workspace, checks every one is discoverable, then runs the
 * job-shaped path over HTTP against a stand-in hive + staklink on
 * localhost — claim → latest → agent → branch-diff → push, with `job` on
 * the launch — and a revision turn (agent-start → agent-status → push onto
 * the same branch), and checks:
 *
 *  - the claim went to hive AS the job (`?job=&run=`) and registered a HOLD
 *    on the job naming pod/release; the push noted the pull request on it;
 *  - the pod's password is nowhere: not in the run's events, the output,
 *    the callback or the artifacts (only `sealed`), while every staklink
 *    call carried it;
 *  - the agent in the pod was started with the job as its session and this
 *    run's LLM gateway grant; the push went as the run's GitHub identity,
 *    named no base (the pod's default branch is the base) and opened a pull
 *    request; the revision pushed to the current branch
 *    and got the same pull request back, no second one;
 *  - `DELETE /jobs/:id` released the pod through pod/release (the stand-in
 *    hive saw drop-pod, as the job) and dropped the hold; a pod hive has let
 *    go (404) or handed to someone else (409) counts as released;
 *  - the pool config (strut plans/pod-config.md): pod/config reads hive's
 *    config alone and beside the pod's files, diffed, every env value in the
 *    pm2 files masked (a changed value that masks alike still shows);
 *    pod/config-set writes only what changed by name onto a fresh read —
 *    masked values round-tripping, both lists sent, either compose name as
 *    docker-compose.yml — and waits for the rollout; what would corrupt the
 *    config is refused before anything is written; `pod/claim { fresh: true }`
 *    waits out a 409, holds the second pod, and gives back a pod a hive that
 *    predates fresh claims hands out; no stored env value reaches strut's
 *    record.
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
import { mask } from "./steps/_pool.js";

const PASSWORD = "pw-s3cret-never-shown";
const KEY = "hiveorg_test_key";
const FRONTEND = "https://pod-1-3000.workspaces.test";
const IDE = "https://pod-1.workspaces.test";
/** The pod's control URL, pod-shaped like hive's (`<podId>-<port>.<domain>`): pod/push finds the hold from its hostname. The strut's fetch answers it from the stand-in. */
const CONTROL = "http://pod-1-15552.workspaces.test";
/** The fresh pod a config change is checked on. */
const FRONTEND2 = "https://pod-2-3000.workspaces.test";
const IDE2 = "https://pod-2.workspaces.test";
const CONTROL2 = "http://pod-2-15552.workspaces.test";
const PR = "https://github.com/o/app/pull/7";

// ── the pool config the stand-in hive keeps for ws-1 ─────────────────────
const b64 = (text: string) => Buffer.from(text, "utf-8").toString("base64");
/** A stored env value: strut's record must never hold it. */
const SECRET = "sk-live-SECRET-value-1234";
/** hive's generated pm2.config.js (its formatPM2Apps): values in clear. */
const STORED_PM2 = `module.exports = {
  apps: [
    {
      name: "frontend",
      script: "npm run dev",
      cwd: "/workspaces/app",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        API_KEY: "${SECRET}",
        API_URL: "http://localhost:3000",
        PORT: "3000",
        INSTALL_COMMAND: "npm install"
      }
    }
  ],
};
`;
/** The pod's copy, as its agent left it: the URL's port changed (it masks the
 *  same), a flag added. */
const POD_PM2 = STORED_PM2.replace("http://localhost:3000", "http://localhost:4000").replace('        PORT: "3000",', '        USE_MOCKS: "true",\n        PORT: "3000",');
const COMPOSE = "services:\n  db:\n    image: postgres:16\n";
const POD_COMPOSE = `${COMPOSE}  rustfs:\n    image: rustfs/rustfs:latest\n`;
const DOCKERFILE = "FROM ghcr.io/stakwork/staklink-js:latest\n";
const STORED_FILES = { "pm2.config.js": b64(STORED_PM2), "docker-compose.yml": b64(COMPOSE), Dockerfile: b64(DOCKERFILE), "devcontainer.json": b64("{}\n") };
/** `services` and the env list as hive serves them to a key caller: masked. */
const STORED_SERVICES = [
  { name: "frontend", port: 3000, scripts: { start: "npm run dev", install: "npm install" }, env: { API_KEY: mask(SECRET) }, cwd: "app" },
  { name: "worker", port: 3001, scripts: { start: "npm run worker" }, env: {} },
];
const STORED_ENV = [
  { name: "API_URL", value: mask("http://localhost:3000") },
  { name: "OLD_FLAG", value: mask("on") },
];
const POD_FILES = [
  { name: "pm2.config.js", path: "/workspaces/.pod-config/.user-dockerfile/pm2.config.js", content: POD_PM2 },
  { name: "docker-compose.yaml", path: "/workspaces/.pod-config/.user-dockerfile/docker-compose.yaml", content: POD_COMPOSE },
  { name: "Dockerfile", path: "/workspaces/.pod-config/.user-dockerfile/Dockerfile", content: DOCKERFILE },
];

interface Seen {
  method: string;
  path: string;
  auth?: string | undefined;
  token?: string | undefined;
  body?: any;
}

/** A stand-in hive (the pool manager) and staklink (under /pod). */
interface Pool {
  /** Bodies of the config PUTs, as sent. */
  puts: any[];
  changeId: string | null;
  readsSincePut: number;
  freshClaims: number;
}

async function standIn(): Promise<{ base: string; seen: Seen[]; posts: any[]; pool: Pool; close: () => void }> {
  const seen: Seen[] = [];
  const posts: any[] = [];
  const pool: Pool = { puts: [], changeId: null, readsSincePut: 0, freshClaims: 0 };
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
      // hive's pool config: read (env masked, files in clear) and write
      if (url.pathname === "/api/workspaces/ws-1/stakgraph") {
        if (token !== KEY) return json(401, { error: "Unauthorized" });
        if (req.method === "GET") {
          if (pool.changeId) pool.readsSincePut++;
          // the third read after a PUT finds a new pod running
          const ready = pool.readsSincePut >= 3;
          const rollout = pool.changeId
            ? { state: ready ? "ready" : "rolling", changeId: pool.changeId, requestedAt: "2026-10-14T12:00:00Z", pushedAt: "2026-10-14T12:00:01Z", error: null, pods: { new: ready ? 1 : 0, old: 2, oldInUse: 1, pending: ready ? 0 : 1, failed: 0, minimumVms: 2 } }
            : { state: "idle", changeId: null, requestedAt: null, pushedAt: null, error: null, pods: { new: 0, old: 2, oldInUse: 1, pending: 0, failed: 0, minimumVms: 2 } };
          return json(200, { success: true, data: { services: STORED_SERVICES, environmentVariables: STORED_ENV, containerFiles: STORED_FILES, rollout } });
        }
        if (req.method === "PUT") {
          pool.puts.push(body);
          pool.changeId = `chg-${pool.puts.length}`;
          pool.readsSincePut = 0;
          // hive answers with its stored state in `data`, values in clear
          return json(200, {
            success: true,
            data: { services: [{ ...STORED_SERVICES[0], env: { API_KEY: SECRET } }], containerFiles: STORED_FILES },
            changeId: pool.changeId,
            requestedAt: "2026-10-14T12:00:00Z",
            infraChanged: true,
            rollout: { state: "rolling", error: null },
          });
        }
      }
      // hive's pool manager
      if (url.pathname.startsWith("/api/pool-manager/")) {
        if (token !== KEY) return json(401, { error: "Unauthorized" });
        if (url.pathname.startsWith("/api/pool-manager/claim-pod/")) {
          if (url.searchParams.get("generation") === "new") {
            // a hive that predates fresh claims: any pod, no `generation`
            if (url.pathname.endsWith("/ws-old")) return json(200, { success: true, podId: "pod-9", frontend: FRONTEND, ide: IDE, control: CONTROL, password: PASSWORD });
            if (++pool.freshClaims === 1) return json(409, { success: false, error: "No pod created after the last pool push is ready yet" });
            return json(200, { success: true, podId: "pod-2", pod_url: IDE2, frontend: FRONTEND2, ide: IDE2, control: CONTROL2, password: PASSWORD, generation: "new" });
          }
          return json(200, { success: true, podId: "pod-1", pod_url: IDE, frontend: FRONTEND, ide: IDE, control: CONTROL, password: PASSWORD, generation: "old" });
        }
        if (url.pathname.startsWith("/api/pool-manager/drop-pod/")) {
          const podId = url.searchParams.get("podId");
          if (podId === "pod-taken") return json(409, { error: "Pod has been reassigned", reassigned: true });
          if (podId === "pod-gone") return json(404, { error: "Pod not found" });
          return json(200, { success: true });
        }
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
        if (p === "/pod-config") return json(200, { files: POD_FILES });
        if (p === "/push") {
          const pr = url.searchParams.get("pr") === "true";
          return json(200, { success: true, commits: ["abc123"], branches: { app: "strut/title" }, ...(pr ? { prs: { app: PR } } : {}) });
        }
      }
      json(404, { error: `no stand-in for ${req.method} ${url.pathname}` });
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  return { base, seen, posts, pool, close: () => server.close() };
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

const POD_LET_GO = `name: pod-let-go
input:
  podId: { type: string }
steps:
  - id: release
    type: pod/release
    config: { workspace: ws-1, podId: "{{ input.podId }}" }
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

const POD_PROMOTE = `name: pod-promote
input:
  control: { type: string }
  sealed: { type: string }
steps:
  - id: alone
    type: pod/config
    config: { workspace: ws-1 }
  - id: before
    type: pod/config
    config: { workspace: ws-1, control: "{{ input.control }}", sealed: "{{ input.sealed }}" }
  - id: set
    type: pod/config-set
    config:
      workspace: ws-1
      services:
        frontend:
          port: 3000
          scripts: { start: "npm run dev", install: "npm install", postStart: "node scripts/make-bucket.js" }
          env: { API_KEY: "{{ before.services[0].env.API_KEY }}" }
          cwd: app
        worker: null
      env: { API_URL: "http://localhost:4000", USE_MOCKS: "true", OLD_FLAG: null }
      containerFiles: { docker-compose.yaml: "{{ before.files[1].pod }}" }
      pollMs: 10
  - id: result
    type: pack
    config:
      alone: "{{ alone }}"
      before: "{{ before }}"
      set: "{{ set }}"
`;

const POD_FRESH = `name: pod-fresh
input:
  workspace: { type: string }
steps:
  - id: claim
    type: pod/claim
    config: { workspace: "{{ input.workspace }}", fresh: true }
`;

const POD_CONFIG_BAD = `name: pod-config-bad
input:
  services: { type: json, required: false }
  env: { type: json, required: false }
  files: { type: json, required: false }
steps:
  - id: set
    type: pod/config-set
    config: { workspace: ws-1, services: "{{ input.services }}", env: "{{ input.env }}", containerFiles: "{{ input.files }}", pollMs: 10 }
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
    console.log(`✔ seeded ${SEED_STEPS.filter((n) => !n.startsWith("_")).length} pod/* steps, all discoverable`);
    await workspace.publishWorkflowByContent("pod-smoke", POD_SMOKE, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-revise", POD_REVISE, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-let-go", POD_LET_GO, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-promote", POD_PROMOTE, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-fresh", POD_FRESH, undefined, "pods", undefined, SEED_OPTS);
    await workspace.publishWorkflowByContent("pod-config-bad", POD_CONFIG_BAD, undefined, "pods", undefined, SEED_OPTS);

    // ── 2. a strut whose secrets name the stand-in hive, whose GitHub
    //       identity lookup is answered locally, and whose LLM gateway
    //       grant is a fake ────────────────────────────────────────────────
    const fetchImpl: typeof fetch = async (url, init) => {
      const u = String(url);
      if (u.startsWith("https://api.github.com/user")) {
        return new Response(JSON.stringify({ login: "octo" }), { status: 200, headers: { "content-type": "application/json" } });
      }
      // A pod-shaped control URL: the stand-in's staklink answers for it.
      const pod = /^http:\/\/pod-\d+-15552\.workspaces\.test(.*)$/.exec(u);
      if (pod) return fetch(`${hive.base}/pod${pod[1]}`, init);
      return fetch(url, init);
    };
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
    assert.equal(out1.control, CONTROL);
    assert.match(out1.sealed, /^v1\.[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]+$/, "the password rides sealed");
    assert.equal(out1.session, job, "the agent's session is the job");
    assert.equal(out1.agent, "Changed the title (req-agent-1).");
    assert.equal(out1.files, 1);
    assert.deepEqual([out1.branch, out1.pr], ["strut/title", PR]);
    assert.deepEqual(post1.artifacts, [{ id: "pod", kind: "url", title: "Pod", url: FRONTEND }]);
    console.log(`✔ turn 1: claimed pod-1, reset, the agent worked, pushed ${out1.branch} → ${out1.pr}`);

    // The claim went to hive AS the job, with the run as the reason.
    const claim = hive.seen.find((s) => s.path.startsWith("/api/pool-manager/claim-pod/"))!;
    assert.equal(claim.path, `/api/pool-manager/claim-pod/ws-1?job=${job}&run=${first.json.runId}`);

    // The hold, naming pod/release with what it needs — and, since the push,
    // the pull request on its note under the frontend URL the claim wrote.
    const holds = await jobsCapability(dir).holds(job);
    assert.equal(holds.length, 1);
    assert.deepEqual({ ...holds[0], since: undefined }, { id: "pod-1", kind: "pod", release: { type: "pod/release", input: { workspace: "ws-1", podId: "pod-1" } }, note: `${FRONTEND}\n${PR}`, url: FRONTEND, since: undefined });
    console.log(`✔ claimed as job ${job.slice(0, 8)}…; the job holds pod-1 (release = pod/release ${JSON.stringify(holds[0]!.release.input)}), the PR noted on it`);

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

    // The agent's transcript is in the run log as strut's `messages` marker —
    // what makes a goose run a thread in the job timeline (strut
    // plans/jobs-sidebar.md, ruling 4): the events endpoint serves it as a
    // link, the link serves the messages, and the password is in neither.
    const agentEnd = events.json.find((e: any) => e.type === "step.end" && e.path === "pod-smoke/agent");
    assert.ok(agentEnd?.transcript, "pod/agent's step.end carries a transcript link");
    assert.ok(!("messages" in agentEnd), "the messages are a link in /events, never inline");
    const tr = await api(agentEnd.transcript);
    assert.equal(tr.status, 200, tr.text);
    assert.deepEqual(tr.json.map((m: any) => m.role), ["system", "user", "assistant"]);
    assert.equal(tr.json[1].content, "Change the title.");
    assert.match(tr.json[2].content, /Changed the title/);
    assert.ok(!tr.text.includes(PASSWORD), "the password is not in the transcript");
    console.log(`✔ pod/agent's transcript is in the run log: ${agentEnd.transcript}`);

    // ── 4. a revision turn: start → status (pending, then done) → push onto
    //       the current branch; `create_pr` stays on and the SAME PR comes back ─
    const second = await api("/workflows/pod-revise/run", { body: { job, input: { control: out1.control, sealed: out1.sealed }, callback } });
    assert.equal(second.status, 202, second.text);
    await until(() => hive.posts.length === 2, "the second callback");
    const post2 = hive.posts[1];
    assert.equal(post2.status, "success", JSON.stringify(post2));
    assert.deepEqual(post2.output, { session: job, first: "pending", status: "completed", output: "Changed the title (req-agent-2).", branch: "strut/title", pr: PR });
    const push2 = hive.seen.filter((s) => s.path.startsWith("/pod/push"))[1]!;
    assert.equal(push2.path, "/pod/push?commit=true&pr=true&stayOnCurrentBranch=true");
    // The same pull request again is noted once.
    assert.equal((await jobsCapability(dir).holds(job))[0]!.note, `${FRONTEND}\n${PR}`);
    console.log(`✔ turn 2: agent-start → agent-status (pending, completed) → push stayed on ${post2.output.branch}, the same PR came back (noted once)`);
    // A pending look carries no transcript; the completed one does — the
    // resumed agent is its own thread in the timeline.
    const ev2 = await api(`/workflows/pod-revise/runs/${second.json.runId}/events`);
    const look = ev2.json.find((e: any) => e.type === "step.end" && e.path === "pod-revise/look");
    const done = ev2.json.find((e: any) => e.type === "step.end" && e.path === "pod-revise/done");
    assert.ok(look && !look.transcript, "a pending status has no transcript");
    assert.ok(done?.transcript, "a completed status has one");
    const tr2 = await api(done.transcript);
    assert.deepEqual(tr2.json.map((m: any) => m.role), ["system", "user", "assistant"]);
    assert.equal(tr2.json[2].content, "Changed the title (req-agent-2).");
    console.log(`✔ pod/agent-status's transcript once completed: ${done.transcript}`);

    // ── 4b. a pod hive has let go, or handed to someone else, counts as released ─
    for (const podId of ["pod-gone", "pod-taken"]) {
      const n = hive.posts.length;
      const run = await api("/workflows/pod-let-go/run", { body: { job, input: { podId }, callback } });
      assert.equal(run.status, 202, run.text);
      await until(() => hive.posts.length === n + 1, `the ${podId} callback`);
      const post = hive.posts[n];
      assert.equal(post.status, "success", JSON.stringify(post));
      assert.deepEqual(post.output, { podId, released: true });
    }
    console.log("✔ pod/release: a 404 (hive let it go) and a 409 reassigned (hive gave it away) both count as released");

    // ── 5. closing the job releases the pod ──────────────────────────────
    const drops = () => hive.seen.filter((s) => s.path.startsWith("/api/pool-manager/drop-pod/"));
    assert.equal(drops().length, 2, "the two let-go runs above");
    const gone = await api(`/jobs/${job}`, { method: "DELETE" });
    assert.equal(gone.status, 200, gone.text);
    assert.deepEqual(gone.json, { ok: true, job, released: ["pod-1"] });
    assert.equal(drops().length, 3);
    // As the job: hive drops it only while this job is still its claimant.
    assert.equal(drops()[2]!.path, `/api/pool-manager/drop-pod/ws-1?podId=pod-1&job=${job}`);
    assert.deepEqual(await jobsCapability(dir).holds(job), []);
    assert.equal((await api(`/jobs/${job}/files`)).status, 404);
    console.log(`✔ DELETE /jobs/${job.slice(0, 8)}… released pod-1 through pod/release (hive saw drop-pod as the job) and the job is gone`);

    // ── 6. the pool config: read it beside the pod, promote a fix, check
    //       it on a fresh pod (strut plans/pod-config.md) ──────────────────
    const job2 = randomUUID();
    const runFor = async (workflow: string, input: unknown) => {
      const n = hive.posts.length;
      const run = await api(`/workflows/${workflow}/run`, { body: { job: job2, input, callback } });
      assert.equal(run.status, 202, run.text);
      await until(() => hive.posts.length === n + 1, `the ${workflow} callback`);
      return { runId: run.json.runId as string, post: hive.posts[n] };
    };

    const promote = await runFor("pod-promote", { control: out1.control, sealed: out1.sealed });
    assert.equal(promote.post.status, "success", JSON.stringify(promote.post));
    const { alone, before, set } = promote.post.output;

    // hive's config alone: the files, pm2's env masked, the lists as hive masks them.
    assert.deepEqual(alone.files.map((f: any) => [f.name, f.status, f.path, f.diff]), [
      ["pm2.config.js", null, null, null],
      ["docker-compose.yml", null, null, null],
      ["Dockerfile", null, null, null],
    ]);
    assert.match(alone.files[0].hive, /API_KEY: "sk\*+34#[0-9a-f]{6}"/);
    assert.match(alone.files[0].hive, /PORT: "3000",\n {8}INSTALL_COMMAND: "npm install"/, "a service's config vars stay readable");
    assert.equal(alone.files[1].hive, COMPOSE);
    assert.deepEqual(alone.services, STORED_SERVICES);
    assert.deepEqual(alone.env, { API_URL: mask("http://localhost:3000"), OLD_FLAG: "****" });
    assert.equal(alone.rollout.state, "idle");

    // beside the pod: a diff per file, the compose file under hive's name.
    const [pm2, compose, dockerfile] = before.files;
    assert.deepEqual([pm2.status, compose.status, dockerfile.status], ["changed", "changed", "same"]);
    assert.equal(compose.path, "/workspaces/.pod-config/.user-dockerfile/docker-compose.yaml");
    assert.match(compose.diff, /^--- hive\/docker-compose\.yml\n\+\+\+ pod\/docker-compose\.yml\n@@ -1,3 \+1,5 @@\n services:\n {3}db:\n {5}image: postgres:16\n\+ {2}rustfs:\n\+ {4}image: rustfs\/rustfs:latest$/);
    assert.equal(dockerfile.diff, null);
    // The URL's port changed and masks the same: the tags tell them apart.
    const urls = [...pm2.diff.matchAll(/^([-+]) {8}API_URL: "(ht\*+00)#([0-9a-f]{6})",$/gm)];
    assert.deepEqual(urls.map((u) => [u[1], u[2]]), [["-", mask("http://localhost:3000")], ["+", mask("http://localhost:4000")]]);
    assert.notEqual(urls[0]![3], urls[1]![3]);
    assert.match(pm2.diff, /^\+ {8}USE_MOCKS: "\*{4}#[0-9a-f]{6}",$/m);
    assert.doesNotMatch(pm2.diff, /^[-+] {8}API_KEY/m, "an unchanged value diffs equal");
    console.log("✔ pod/config: hive's config alone, then beside the pod — pm2 env masked (a changed port shows), compose .yaml read as docker-compose.yml");

    // The write: only the edits, by name, on a fresh read — both lists, the
    // masked value as read, the compose file under hive's name, base64.
    assert.equal(hive.pool.puts.length, 1);
    assert.deepEqual(hive.pool.puts[0], {
      services: [{ name: "frontend", port: 3000, scripts: { start: "npm run dev", install: "npm install", postStart: "node scripts/make-bucket.js" }, env: { API_KEY: mask(SECRET) }, cwd: "app" }],
      environmentVariables: [
        { name: "API_URL", value: "http://localhost:4000" },
        { name: "USE_MOCKS", value: "true" },
      ],
      containerFiles: { "docker-compose.yml": b64(POD_COMPOSE) },
    });
    assert.deepEqual({ ...set, rollout: undefined }, {
      workspace: "ws-1",
      changeId: "chg-1",
      state: "ready",
      settledBy: null,
      written: { services: ["frontend"], env: ["API_URL", "USE_MOCKS"], containerFiles: ["docker-compose.yml"] },
      removed: { services: ["worker"], env: ["OLD_FLAG"] },
      rollout: undefined,
    });
    assert.equal(set.rollout.state, "ready");
    assert.equal(hive.pool.readsSincePut, 3, "polled until a new pod was ready");
    console.log(`✔ pod/config-set: wrote services ${set.written.services}, env ${set.written.env}, ${set.written.containerFiles}; removed ${set.removed.services}, ${set.removed.env}; waited for ${set.changeId} to be ready`);

    // No stored value reached strut's record: not the events, not the output.
    const events2 = await api(`/workflows/pod-promote/runs/${promote.runId}/events`);
    for (const [where, text] of [["events", events2.text], ["callback", JSON.stringify(promote.post)]] as const) {
      assert.ok(!text.includes(SECRET), `the stored API_KEY is in the ${where}`);
      assert.ok(!text.includes("http://localhost:3000"), `the stored API_URL is in the ${where}`);
    }
    console.log("✔ no stored env value in the run's events or callback (pm2 files masked, hive's PUT answer dropped)");

    // What would corrupt the config is refused before hive is called.
    const refusals: Array<[unknown, RegExp]> = [
      [{ files: { "pm2.config.js": "module.exports = {}" } }, /pm2\.config\.js is generated from `services`/],
      [{ files: { "devcontainer.json": "{}" } }, /devcontainer\.json is never edited/],
      [{ files: { "docker-compose.yml": "a: 1", "docker-compose.yaml": "a: 2" } }, /two compose files given/],
      [{ env: { API_URL: pm2.diff.match(/ht\*+00#[0-9a-f]{6}/)![0] } }, /API_URL is a masked value/],
      [{ env: { PORT: "3000" } }, /PORT is generated from the service's port/],
      [{ services: { frontend: { port: 3000, scripts: { start: "x" }, env: { POST_START_COMMAND: "y" } } } }, /services\.frontend\.env: POST_START_COMMAND is generated from the service's scripts\.postStart/],
      [{ services: { ghost: null } }, /no service "ghost" to remove/],
      [{}, /nothing to change/],
    ];
    for (const [input, why] of refusals) {
      const { post } = await runFor("pod-config-bad", input);
      assert.equal(post.status, "error", JSON.stringify(post));
      assert.match(post.error.message, why);
    }
    assert.equal(hive.pool.puts.length, 1, "no refused change reached hive");
    console.log(`✔ pod/config-set refused ${refusals.length} changes that would corrupt the config, before writing`);

    // A fresh pod: 409 while none is ready, then the pod built after the push,
    // a second hold on the job.
    const wait = await runFor("pod-fresh", { workspace: "ws-1" });
    assert.equal(wait.post.status, "error");
    assert.match(wait.post.error.message, /no fresh pod ready yet in workspace "ws-1" — No pod created after the last pool push is ready yet; .*try again/);
    const fresh = await runFor("pod-fresh", { workspace: "ws-1" });
    assert.equal(fresh.post.status, "success", JSON.stringify(fresh.post));
    assert.deepEqual({ ...fresh.post.output, sealed: undefined }, { podId: "pod-2", workspace: "ws-1", frontend: FRONTEND2, ide: IDE2, control: CONTROL2, sealed: undefined, generation: "new" });
    const freshClaim = hive.seen.filter((s) => s.path.startsWith("/api/pool-manager/claim-pod/")).at(-1)!;
    assert.equal(freshClaim.path, `/api/pool-manager/claim-pod/ws-1?job=${job2}&run=${fresh.runId}&generation=new`);
    const freshHolds = await jobsCapability(dir).holds(job2);
    assert.deepEqual(freshHolds.map((h: any) => h.id), ["pod-2"]);
    // Borrowed, not the job's pod: no url, so the job's `urls.pod` never points at it.
    assert.equal(freshHolds[0]!.url, undefined);
    assert.equal(freshHolds[0]!.note, FRONTEND2);
    // A hive that predates fresh claims hands out any pod: given back, never used.
    const old = await runFor("pod-fresh", { workspace: "ws-old" });
    assert.equal(old.post.status, "error");
    assert.match(old.post.error.message, /did not say it is new .*the pod was released/);
    assert.equal(drops().at(-1)!.path, `/api/pool-manager/drop-pod/ws-old?podId=pod-9&job=${job2}`);
    assert.deepEqual((await jobsCapability(dir).holds(job2)).map((h: any) => h.id), ["pod-2"], "no hold on the pod given back");
    // Done with it: release the fresh pod.
    const letGo = await runFor("pod-let-go", { podId: "pod-2" });
    assert.deepEqual(letGo.post.output, { podId: "pod-2", released: true });
    assert.deepEqual(await jobsCapability(dir).holds(job2), []);
    console.log("✔ pod/claim { fresh: true }: waited out a 409, held pod-2 (generation new), gave back a pod an older hive handed out, released pod-2");

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
