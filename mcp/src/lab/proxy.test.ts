import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Request as ExpressRequest, Response as ExpressResponse } from "express";
import type { RunEvent, RunResult } from "strut";

// The request registry writes under REQS_DIR at module load, so it — and the
// proxy module over it — are imported after the env is set.
const REQS_DIR = await mkdtemp(join(tmpdir(), "lab-proxy-reqs-"));
process.env.REQS_DIR = REQS_DIR;
const API_TOKEN = "test-api-token";
const prevToken = process.env.API_TOKEN;
process.env.API_TOKEN = API_TOKEN;

const asyncReqs = await import("../graph/reqs.js");
const {
  actorOf,
  createProxies,
  isRepoInFlight,
  parseOwnerRepo,
  parseDescribeBody,
  mapDescribeResult,
  DESCRIBE_MODEL,
  DESCRIBE_WORKFLOW,
  PROCESS_WORKFLOW,
  _resetInFlight,
} = await import("./proxy.js");
type ProxyStrut = import("./proxy.js").ProxyStrut;

after(async () => {
  if (prevToken === undefined) delete process.env.API_TOKEN;
  else process.env.API_TOKEN = prevToken;
  await rm(REQS_DIR, { recursive: true, force: true });
});

// ── Stand-ins ─────────────────────────────────────────────────────────────

function req(o: { headers?: Record<string, string>; query?: Record<string, unknown>; body?: unknown } = {}): ExpressRequest {
  const h = Object.fromEntries(Object.entries(o.headers ?? {}).map(([k, v]) => [k.toLowerCase(), v]));
  return { header: (n: string) => h[n.toLowerCase()], query: o.query ?? {}, body: o.body } as unknown as ExpressRequest;
}

function res() {
  const r = {
    code: 200,
    body: undefined as unknown,
    status(c: number) {
      r.code = c;
      return r;
    },
    json(b: unknown) {
      r.body = b;
      return r;
    },
  };
  return r as typeof r & ExpressResponse;
}

const hive = (actor = "alice") => ({ "x-api-token": API_TOKEN, "x-strut-actor": actor });

type RunOpts = Parameters<ProxyStrut["run"]>[2];

/** A strut whose runs the test settles, with the run list and event logs it
 *  should report. */
function fakeStrut(opts: { runs?: Array<{ runId: string; status?: string; finishedAt?: string }>; events?: Record<string, RunEvent[]> } = {}) {
  const puts: Array<{ path: string; body: unknown }> = [];
  const launches: Array<{ workflow: string; input: unknown; opts: RunOpts }> = [];
  const pending: Array<(r: RunResult) => void> = [];
  const strut: ProxyStrut = {
    app: {
      async fetch(request) {
        const url = new URL(request.url);
        if (request.method === "PUT") {
          puts.push({ path: url.pathname, body: await request.json() });
          return Response.json({ ok: true });
        }
        if (url.pathname.endsWith("/runs")) return Response.json(opts.runs ?? []);
        return new Response("not found", { status: 404 });
      },
    },
    store: {
      async getRunEvents(_wf, runId) {
        return opts.events?.[runId] ?? [];
      },
      async *tailEvents(_wf, runId) {
        for (const e of opts.events?.[runId] ?? []) yield e;
      },
    },
    run: (workflow, input, o) => {
      launches.push({ workflow, input, opts: o });
      return new Promise<RunResult>((resolve) => pending.push(resolve));
    },
  };
  const settle = async (r: RunResult) => {
    pending.shift()?.(r);
    await new Promise((f) => setImmediate(f));
  };
  return { strut, puts, launches, settle };
}

function proxies(strut: ProxyStrut, extra: { hasOpenRouterKey?: () => boolean } = {}) {
  const calls = { strut: 0, process: 0, describe: 0 };
  const p = createProxies({
    strut: async () => {
      calls.strut += 1;
      return strut;
    },
    fallback: {
      process: () => {
        calls.process += 1;
      },
      describe: () => {
        calls.describe += 1;
      },
    },
    ...extra,
  });
  return { ...p, calls };
}

const event = (type: RunEvent["type"], runId: string, extra: Partial<RunEvent> = {}): RunEvent => ({
  ts: new Date().toISOString(),
  runId,
  path: PROCESS_WORKFLOW,
  type,
  ...extra,
});

// ── The actor ─────────────────────────────────────────────────────────────

describe("actorOf", () => {
  it("is the x-strut-actor beside a matching x-api-token", () => {
    assert.equal(actorOf(req({ headers: hive("bob-1") })), "bob-1");
    assert.equal(actorOf(req({ headers: { "x-api-token": API_TOKEN, "x-strut-actor": "  " } })), undefined);
    assert.equal(actorOf(req({ headers: { "x-api-token": API_TOKEN } })), undefined);
  });

  it("names nobody on any other credential", () => {
    assert.equal(actorOf(req({ headers: { "x-api-token": "wrong", "x-strut-actor": "bob-1" } })), undefined);
    assert.equal(actorOf(req({ headers: { authorization: "Bearer jwt", "x-strut-actor": "bob-1" } })), undefined);
    assert.equal(actorOf(req({ headers: { "x-strut-actor": "bob-1" } })), undefined);
  });

  it("names nobody in dev mode (no API_TOKEN)", () => {
    delete process.env.API_TOKEN;
    try {
      assert.equal(actorOf(req({ headers: hive() })), undefined);
    } finally {
      process.env.API_TOKEN = API_TOKEN;
    }
  });
});

describe("parseOwnerRepo", () => {
  it("takes owner/repo, or parses repo_url in its forms", () => {
    assert.deepEqual(parseOwnerRepo({ owner: "o", repo: "r" }), { owner: "o", repo: "r" });
    assert.deepEqual(parseOwnerRepo({ repo_url: "https://github.com/o/r.git" }), { owner: "o", repo: "r" });
    assert.deepEqual(parseOwnerRepo({ repo_url: "git@github.com:o/r.git" }), { owner: "o", repo: "r" });
    assert.deepEqual(parseOwnerRepo({ repo_url: "o/r" }), { owner: "o", repo: "r" });
    assert.deepEqual(parseOwnerRepo({ repo_url: "https://gitlab.com/o/r/-/tree/main" }), { owner: "o", repo: "r" });
    assert.equal(parseOwnerRepo({ owner: "o" }), undefined);
    assert.equal(parseOwnerRepo({}), undefined);
  });
});

// ── /gitree/process ───────────────────────────────────────────────────────

describe("gitree/process proxy", () => {
  beforeEach(() => _resetInFlight());

  it("sends a request with no actor to the old handler and never builds the strut", async () => {
    const { strut } = fakeStrut();
    const p = proxies(strut);
    await p.process(req({ query: { owner: "o", repo: "r", token: "ghp_x" } }), res());
    await p.process(req({ headers: { authorization: "Bearer spa-jwt" }, query: { owner: "o", repo: "r" } }), res());
    assert.equal(p.calls.process, 2);
    assert.equal(p.calls.strut, 0);
  });

  it("stores the token as the actor's secret, launches as the actor, and settles /progress from the run", async () => {
    const { strut, puts, launches, settle } = fakeStrut();
    const p = proxies(strut);
    const r = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "one", token: "ghp_x" } }), r);

    const body = r.body as { request_id: string; status: string; run_id: string; attached?: boolean };
    assert.equal(r.code, 200);
    assert.equal(body.status, "pending");
    assert.equal(body.attached, undefined);
    assert.deepEqual(puts, [{ path: "/actors/alice/secrets/GITHUB_TOKEN", body: { value: "ghp_x" } }]);
    assert.equal(launches.length, 1);
    assert.equal(launches[0]!.workflow, PROCESS_WORKFLOW);
    assert.deepEqual(launches[0]!.input, { owner: "o", repo: "one" });
    assert.equal(launches[0]!.opts?.actor, "alice");
    assert.equal(launches[0]!.opts?.runId, body.run_id);
    // The token is in the secret, not the launch.
    assert.ok(!JSON.stringify(launches).includes("ghp_x"));
    assert.equal(isRepoInFlight("o/one"), true);
    assert.equal(isRepoInFlight(), true);
    assert.equal(isRepoInFlight("o/other"), false);
    assert.equal(asyncReqs.checkReq(body.request_id).status, "pending");

    await settle({ runId: body.run_id, status: "success", output: { count: 2 } });
    const done = asyncReqs.checkReq(body.request_id);
    assert.equal(done.status, "completed");
    assert.deepEqual(done.result, { status: "success", message: "Processed o/one", run_id: body.run_id, output: { count: 2 } });
    assert.equal(isRepoInFlight("o/one"), false);
  });

  it("answers a second call for the same repo with the sweep already running", async () => {
    const { strut, puts, launches, settle } = fakeStrut();
    const p = proxies(strut);
    const first = res();
    await p.process(req({ headers: hive(), query: { repo_url: "https://github.com/o/two", token: "ghp_x" } }), first);
    const second = res();
    await p.process(req({ headers: hive("bob"), query: { owner: "o", repo: "two", token: "ghp_y" } }), second);

    const a = first.body as { request_id: string; run_id: string };
    const b = second.body as { request_id: string; run_id: string; attached?: boolean };
    assert.equal(b.request_id, a.request_id);
    assert.equal(b.run_id, a.run_id);
    assert.equal(b.attached, true);
    assert.equal(launches.length, 1);
    assert.equal(puts.length, 1);

    await settle({ runId: a.run_id, status: "success" });
    const third = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "two" } }), third);
    assert.equal(launches.length, 2);
    assert.equal(puts.length, 1, "no token in the query, nothing stored");
  });

  it("attaches to a run strut resumed after a restart instead of launching another", async () => {
    const events: Record<string, RunEvent[]> = {
      resumed: [
        event("run.start", "resumed", { input: { owner: "o", repo: "three" }, actor: "alice", principal: "alice" }),
        event("run.end", "resumed", { output: { count: 1 } }),
      ],
      stale: [event("run.start", "stale", { input: { owner: "o", repo: "four" } })],
    };
    const { strut, launches, puts } = fakeStrut({
      runs: [
        { runId: "finished", status: "success", finishedAt: "2026-10-13T00:00:00Z" },
        { runId: "resumed", status: "running" },
        { runId: "stale", status: "stale" },
      ],
      events,
    });
    const p = proxies(strut);

    const r = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "three", token: "ghp_x" } }), r);
    const body = r.body as { request_id: string; run_id: string; attached?: boolean };
    assert.equal(body.run_id, "resumed");
    assert.equal(body.attached, true);
    assert.equal(launches.length, 0);
    assert.equal(puts.length, 0);
    await new Promise((f) => setImmediate(f));
    const done = asyncReqs.checkReq(body.request_id);
    assert.equal(done.status, "completed");
    assert.deepEqual(done.result, { status: "success", message: "Processed o/three", run_id: "resumed", output: { count: 1 } });
    assert.equal(isRepoInFlight("o/three"), false);

    // A stale run (cut off, not resumed) holds nothing: launch.
    const r2 = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "four" } }), r2);
    assert.equal(launches.length, 1);
    assert.equal((r2.body as { attached?: boolean }).attached, undefined);
  });

  it("fails the request when the run errors or is cancelled", async () => {
    const { strut, settle } = fakeStrut();
    const p = proxies(strut);
    const r = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "five" } }), r);
    const a = r.body as { request_id: string; run_id: string };
    await settle({ runId: a.run_id, status: "error", error: { message: "clone failed" } });
    assert.equal(asyncReqs.checkReq(a.request_id).status, "failed");
    assert.deepEqual(asyncReqs.checkReq(a.request_id).error, { message: "clone failed", run_id: a.run_id });

    const r2 = res();
    await p.process(req({ headers: hive(), query: { owner: "o", repo: "five" } }), r2);
    const b = r2.body as { request_id: string; run_id: string };
    assert.notEqual(b.run_id, a.run_id);
    await settle({ runId: b.run_id, status: "cancelled" });
    assert.deepEqual(asyncReqs.checkReq(b.request_id).error, { message: "run cancelled", run_id: b.run_id });
  });

  it("rejects a request naming no repo", async () => {
    const { strut, launches } = fakeStrut();
    const p = proxies(strut);
    const r = res();
    await p.process(req({ headers: hive(), query: { token: "ghp_x" } }), r);
    assert.equal(r.code, 400);
    assert.equal(launches.length, 0);
  });
});

// ── /repo/describe ────────────────────────────────────────────────────────

describe("repo/describe proxy", () => {
  it("maps the body to the describe-code launch, defaulting the model to the workflow's", () => {
    const none = parseDescribeBody({}, () => false);
    assert.ok(none.ok);
    assert.deepEqual(none.launch, { input: { file_paths: [], cost_limit: 0.5 }, params: { batch_size: 50, concurrency: 5 } });

    const full = parseDescribeBody(
      { repo_url: "https://github.com/o/r", file_paths: ["src/a.ts", 7], cost_limit: "2", batch_size: 10, concurrency: "3", embed: false },
      () => true,
    );
    assert.ok(full.ok);
    assert.deepEqual(full.launch, {
      input: { repo_url: "https://github.com/o/r", file_paths: ["src/a.ts"], cost_limit: 2 },
      params: { batch_size: 10, concurrency: 3, model: DESCRIBE_MODEL },
    });

    const chosen = parseDescribeBody({ model: "sonnet" }, () => true);
    assert.ok(chosen.ok && chosen.launch.params.model === "sonnet");
  });

  it("refuses apiKey and bad numbers", () => {
    assert.match((parseDescribeBody({ apiKey: "sk-x" }, () => false) as { error: string }).error, /apiKey/);
    assert.ok(parseDescribeBody({ apiKey: "  " }, () => false).ok, "a blank key is no key");
    assert.match((parseDescribeBody({ cost_limit: "0" }, () => false) as { error: string }).error, /cost_limit/);
    assert.match((parseDescribeBody({ batch_size: "x" }, () => false) as { error: string }).error, /batch_size/);
    assert.match((parseDescribeBody({ concurrency: -1 }, () => false) as { error: string }).error, /concurrency/);
  });

  it("maps the run output to the old result shape", () => {
    assert.deepEqual(mapDescribeResult({ runId: "x", status: "success", output: { processed: 3, failed: 1, cost: 0.02, stopped: "cost_limit" } }), {
      success: true,
      processed: 3,
      total_cost: 0.02,
      failed: 1,
      stopped: "cost_limit",
      run_id: "x",
    });
  });

  it("sends a request with no actor to the old handler, and launches one with", async () => {
    const { strut, launches, settle } = fakeStrut();
    const p = proxies(strut, { hasOpenRouterKey: () => false });
    await p.describe(req({ headers: { authorization: "Bearer spa-jwt" }, body: { repo_url: "https://github.com/o/r", apiKey: "sk-user" } }), res());
    assert.equal(p.calls.describe, 1);
    assert.equal(launches.length, 0);

    const bad = res();
    await p.describe(req({ headers: hive(), body: { apiKey: "sk-user" } }), bad);
    assert.equal(bad.code, 400);

    const r = res();
    await p.describe(req({ headers: hive("carol"), body: { repo_url: "https://github.com/o/r", cost_limit: 1 } }), r);
    const body = r.body as { request_id: string; status: string; run_id: string };
    assert.equal(r.code, 200);
    assert.equal(body.status, "pending");
    assert.equal(launches.length, 1);
    assert.equal(launches[0]!.workflow, DESCRIBE_WORKFLOW);
    assert.deepEqual(launches[0]!.input, { repo_url: "https://github.com/o/r", file_paths: [], cost_limit: 1 });
    assert.deepEqual(launches[0]!.opts?.params, { batch_size: 50, concurrency: 5 });
    assert.equal(launches[0]!.opts?.actor, "carol");
    assert.equal(launches[0]!.opts?.runId, body.run_id);

    await settle({ runId: body.run_id, status: "success", output: { processed: 2, failed: 0, cost: 0.01, stopped: "done" } });
    const done = asyncReqs.checkReq(body.request_id);
    assert.equal(done.status, "completed");
    assert.deepEqual(done.result, { success: true, processed: 2, total_cost: 0.01, failed: 0, stopped: "done", run_id: body.run_id });
  });
});
