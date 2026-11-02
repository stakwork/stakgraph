import { describe, it } from "node:test";
import assert from "node:assert/strict";
import claim, { IDLE_MINUTES } from "./steps/claim.js";
import diff from "./steps/diff.js";
import { podAccess, seal, type PodCtx } from "./steps/_shared.js";

// The pod steps over a job's holds: pod/claim freeing an idle job's pod when
// the pool is empty, and every other step finding its pod's way in on the
// hold instead of in what the model copied. A stand-in jobs capability (the
// shape strut's job index serves) and a stand-in hive + pod.

const MIN = 60_000;
const ago = (ms: number) => new Date(Date.now() - ms).toISOString();

interface Run {
  workflow: string;
  runId: string;
  at: string;
  status: "running" | "success" | "error" | "cancelled";
  durationMs?: number;
}
interface Job {
  job: string;
  usedAt: string;
  busy?: string;
  runs: Run[];
  holds: any[];
}

const podHold = (id: string, workspace: string, extra: Record<string, unknown> = {}, url?: string) => ({
  id,
  kind: "pod",
  since: ago(3 * 60 * MIN),
  release: { type: "pod/release", input: { workspace, podId: id, ...extra } },
  ...(url ? { url } : {}),
});

/** A job last used `idle` ago: one run that started then and is over. */
const idleJob = (job: string, idle: number, holds: any[], over: Partial<Job> = {}): Job => ({
  job,
  usedAt: ago(idle),
  runs: [{ workflow: "job", runId: `${job}-1`, at: ago(idle), status: "success", durationMs: 0 }],
  holds,
  ...over,
});

function jobsOf(jobs: Job[]) {
  const released: string[] = [];
  const held: Array<{ job: string; hold: any }> = [];
  const find = (id: string) => jobs.find((j) => j.job === id);
  return {
    released,
    held,
    cap: {
      list: async () => jobs.map((j) => ({ job: j.job, usedAt: j.usedAt, ...(j.busy ? { busy: j.busy } : {}), runs: j.runs.length, holds: j.holds.length })),
      get: async (id: string) => {
        const j = find(id);
        return j ? { job: j.job, usedAt: j.usedAt, ...(j.busy ? { busy: j.busy } : {}), holds: j.holds, runs: j.runs, urls: {}, repos: [], files: [], sessions: [], artifacts: [] } : null;
      },
      holds: async (id: string) => find(id)?.holds ?? [],
      release: async (id: string, podId: string) => {
        released.push(`${id}/${podId}`);
        const j = find(id);
        if (j) j.holds = j.holds.filter((h) => h.id !== podId);
      },
      hold: async (id: string, hold: any) => {
        held.push({ job: id, hold });
      },
    },
  };
}

interface Seen {
  method: string;
  url: string;
  auth?: string;
}

/** A stand-in hive: claim-pod answers from `claims` in order, drop-pod by podId (200 unless named). */
function hiveCtx(o: { job?: string; jobs?: unknown; claims: Array<{ status: number; body?: unknown }>; drops?: Record<string, number> }) {
  const seen: Seen[] = [];
  const claims = [...o.claims];
  const ctx = {
    runId: "run-me",
    path: "p",
    ...(o.job ? { job: o.job } : {}),
    services: {
      secrets: { get: async (n: string) => ({ HIVE_URL: "https://hive.example", HIVE_API_KEY: "hiveorg_key" })[n] },
      ...(o.jobs ? { jobs: o.jobs } : {}),
      http: async (url: string, init: any) => {
        seen.push({ method: init.method, url, auth: init.headers?.Authorization });
        if (url.startsWith("https://hive.example/api/pool-manager/claim-pod/")) {
          const r = claims.shift() ?? { status: 503, body: { error: "No available pods" } };
          return { status: r.status, ok: r.status < 400, headers: {}, body: r.body };
        }
        if (url.startsWith("https://hive.example/api/pool-manager/drop-pod/")) {
          const status = o.drops?.[new URL(url).searchParams.get("podId") ?? ""] ?? 200;
          return { status, ok: status < 400, headers: {}, body: status < 400 ? { success: true } : { error: "nope" } };
        }
        if (url.startsWith("https://pod-")) return { status: 200, ok: true, headers: {}, body: [] };
        throw new Error(`unexpected request: ${init.method} ${url}`);
      },
    },
  } as unknown as PodCtx;
  return { ctx, seen };
}

const GOT = (podId: string) => ({ status: 200, body: { podId, frontend: `https://${podId}-3000.example`, ide: `https://${podId}.example`, control: `https://${podId}-15552.example`, password: "pod-password" } });
const EMPTY = { status: 503, body: { success: false, error: "No available pods" } };
const drops = (seen: Seen[]) => seen.filter((s) => s.url.includes("/drop-pod/")).map((s) => s.url.replace("https://hive.example/api/pool-manager/drop-pod/", ""));
const runClaim = (ctx: PodCtx, workspace = "ws-1") => claim.run(claim.input.parse({ workspace }), ctx);

describe("pod/claim on an empty pool", () => {
  it("frees the pod of the job idle longest — as that job — and claims once more", async () => {
    const jobs = jobsOf([
      idleJob("b", 30 * MIN, [podHold("pod-b", "ws-1")]),
      idleJob("a", 120 * MIN, [podHold("pod-a", "ws-1")]),
      // never: a turn holding the directory, a run with no summary yet, a run
      // that ended inside the window (however long ago it began), another
      // workspace's pod, this job's own pod
      idleJob("busy", 300 * MIN, [podHold("pod-busy", "ws-1")], { busy: "run-busy" }),
      idleJob("running", 300 * MIN, [podHold("pod-running", "ws-1")], { runs: [{ workflow: "job", runId: "r", at: ago(300 * MIN), status: "running" }] }),
      idleJob("recent", 300 * MIN, [podHold("pod-recent", "ws-1")], { runs: [{ workflow: "job", runId: "r", at: ago(300 * MIN), status: "success", durationMs: 295 * MIN }] }),
      idleJob("elsewhere", 300 * MIN, [podHold("pod-elsewhere", "ws-2")]),
      idleJob("me", 300 * MIN, [podHold("pod-mine", "ws-1")]),
    ]);
    const { ctx, seen } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [EMPTY, GOT("pod-a")] });
    const out = await runClaim(ctx);
    assert.equal(out.podId, "pod-a");
    assert.deepEqual(out.took, { job: "a", podId: "pod-a" });
    assert.deepEqual(drops(seen), ["ws-1?podId=pod-a&job=a"], "one pod freed, as the job that held it");
    assert.deepEqual(jobs.released, ["a/pod-a"], "and its hold off that job");
    assert.equal(seen.filter((s) => s.url.includes("/claim-pod/")).length, 2);
  });

  it("takes a hold hive no longer honors (404, 409) off its job and goes on to the next", async () => {
    const jobs = jobsOf([idleJob("a", 300 * MIN, [podHold("pod-a", "ws-1")]), idleJob("b", 200 * MIN, [podHold("pod-b", "ws-1")]), idleJob("c", 100 * MIN, [podHold("pod-c", "ws-1")])]);
    const { ctx, seen } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [EMPTY, GOT("pod-c")], drops: { "pod-a": 409, "pod-b": 404 } });
    const out = await runClaim(ctx);
    assert.deepEqual(out.took, { job: "c", podId: "pod-c" });
    assert.deepEqual(drops(seen), ["ws-1?podId=pod-a&job=a", "ws-1?podId=pod-b&job=b", "ws-1?podId=pod-c&job=c"]);
    assert.deepEqual(jobs.released, ["a/pod-a", "b/pod-b", "c/pod-c"]);
  });

  it("stops at a drop hive refuses, keeps the hold, and says no pod is available", async () => {
    const jobs = jobsOf([idleJob("a", 300 * MIN, [podHold("pod-a", "ws-1")]), idleJob("b", 200 * MIN, [podHold("pod-b", "ws-1")])]);
    const { ctx, seen } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [EMPTY], drops: { "pod-a": 500 } });
    await assert.rejects(runClaim(ctx), new RegExp(`no pod available in workspace "ws-1" right now — and no pod another job here has held idle for ${IDLE_MINUTES} minutes could be freed`));
    assert.deepEqual(drops(seen), ["ws-1?podId=pod-a&job=a"]);
    assert.deepEqual(jobs.released, []);
  });

  it("frees nothing when every holder is busy or recent, and claims once", async () => {
    const jobs = jobsOf([idleJob("a", (IDLE_MINUTES - 1) * MIN, [podHold("pod-a", "ws-1")]), idleJob("b", 300 * MIN, [podHold("pod-b", "ws-1")], { busy: "run-b" })]);
    const { ctx, seen } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [EMPTY] });
    await assert.rejects(runClaim(ctx), /no pod available/);
    assert.deepEqual(drops(seen), []);
    assert.equal(seen.filter((s) => s.url.includes("/claim-pod/")).length, 1);
  });

  it("says, when the freed pod went to another claim first", async () => {
    const jobs = jobsOf([idleJob("a", 300 * MIN, [podHold("pod-a", "ws-1")])]);
    const { ctx } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [EMPTY, EMPTY] });
    await assert.rejects(runClaim(ctx), /pod pod-a, freed from idle job a, went to another claim first/);
  });

  it("leaves a 503 as it was where the jobs capability cannot list jobs", async () => {
    const { ctx, seen } = hiveCtx({ job: "me", jobs: { hold: async () => {}, release: async () => {}, holds: async () => [] }, claims: [EMPTY] });
    await assert.rejects(runClaim(ctx), /no pod available/);
    assert.deepEqual(drops(seen), []);
  });
});

describe("pod/claim's hold", () => {
  it("keeps the pod's way in — control and sealed — on the hold, beside pod/release's own input", async () => {
    const jobs = jobsOf([]);
    const { ctx } = hiveCtx({ job: "me", jobs: jobs.cap, claims: [GOT("pod-1")] });
    const out = await runClaim(ctx);
    assert.equal(out.took, null);
    assert.equal(jobs.held.length, 1);
    const { hold } = jobs.held[0]!;
    assert.deepEqual(hold.release, { type: "pod/release", input: { workspace: "ws-1", podId: "pod-1", control: out.control, sealed: out.sealed } });
    assert.equal(hold.url, out.frontend);
  });
});

describe("podAccess", () => {
  const C1 = "https://pod-1-15552.example";
  const C2 = "https://pod-2-15552.example";

  async function onJob(holds: (ctx: PodCtx, s: string) => any[]) {
    const base = hiveCtx({ claims: [] }).ctx;
    const sealed = await seal(base, "pod-password");
    const jobs = jobsOf([{ job: "me", usedAt: ago(0), runs: [], holds: holds(base, sealed) }]);
    return { ...hiveCtx({ job: "me", jobs: jobs.cap, claims: [] }), sealed };
  }

  it("on a job, the hold's way in wins over what the model copied", async () => {
    const { ctx, seen, sealed } = await onJob((_, s) => [podHold("pod-1", "ws-1", { control: C1, sealed: s }, "https://pod-1-3000.example")]);
    const broken = sealed.slice(0, -2);
    assert.deepEqual(await podAccess(ctx, { podId: "pod-1", control: C1, sealed: broken }), { control: C1, sealed });
    assert.deepEqual(await podAccess(ctx, { control: `${C1}/`, sealed: broken }), { control: C1, sealed }, "matched by control alone");
    // A step run that way reaches the pod with its real password (swarm38
    // run 1793579091586: one character dropped from `sealed` failed the step).
    await diff.run(diff.input.parse({ control: C1, sealed: broken }), ctx);
    assert.deepEqual(seen.map((s) => [s.url, s.auth]), [[`${C1}/diff`, "Bearer pod-password"]]);
  });

  it("with nothing named, drives the job's own pod — the hold with a url, never a borrowed fresh one", async () => {
    const { ctx, sealed } = await onJob((_, s) => [podHold("pod-2", "ws-1", { control: C2, sealed: s }), podHold("pod-1", "ws-1", { control: C1, sealed: s }, "https://pod-1-3000.example")]);
    assert.deepEqual(await podAccess(ctx, {}), { control: C1, sealed });
    assert.deepEqual(await podAccess(ctx, { podId: "pod-2" }), { control: C2, sealed }, "the fresh one by podId");
    const only = await onJob((_, s) => [podHold("pod-2", "ws-1", { control: C2, sealed: s })]);
    await assert.rejects(podAccess(only.ctx, {}), /holds no pod of its own, only pod-2 .*name it by podId/);
  });

  it("says when the job does not hold the pod — released, or taken while it sat idle", async () => {
    const { ctx } = await onJob((_, s) => [podHold("pod-1", "ws-1", { control: C1, sealed: s }, "https://pod-1-3000.example")]);
    await assert.rejects(podAccess(ctx, { podId: "pod-9" }), /job "me" does not hold pod pod-9: it was released \(pod\/release\), or taken for another job's claim while this job sat idle .*claim again/);
    const none = await onJob(() => []);
    await assert.rejects(podAccess(none.ctx, {}), /job "me" holds no pod: none was claimed, or it was released/);
  });

  it("on a hold from before it kept the way in, takes control and sealed as given, else asks for them", async () => {
    const { ctx, sealed } = await onJob(() => [podHold("pod-1", "ws-1", {}, "https://pod-1-3000.example")]);
    assert.deepEqual(await podAccess(ctx, { podId: "pod-1", control: C1, sealed }), { control: C1, sealed });
    await assert.rejects(podAccess(ctx, { podId: "pod-1" }), /pod pod-1: its hold on job "me" was recorded before holds kept the pod's way in — pass control and sealed/);
    await assert.rejects(podAccess(ctx, {}), /pod pod-1: its hold .* recorded before/);
  });

  it("without a job, takes control and sealed as given, and needs both", async () => {
    const { ctx } = hiveCtx({ claims: [] });
    assert.deepEqual(await podAccess(ctx, { control: C1, sealed: "v1.a.b" }), { control: C1, sealed: "v1.a.b" });
    await assert.rejects(podAccess(ctx, { control: C1 }), /name the pod: `control` and `sealed` from pod\/claim/);
    await assert.rejects(podAccess(ctx, { podId: "pod-1" }), /name the pod/);
  });
});
