/** pod/file: bytes from staklink's GET /file into the job dir or the run's artifacts. */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { jobRoot } from "strut";
import file from "./steps/file.js";
import { seal, type PodCtx } from "./steps/_shared.js";

const CONTROL = "https://pod.example:3355";
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 1, 2, 3]);

let realFetch: typeof fetch;
let calls: Array<{ url: string; headers: Record<string, string> }>;
let reply: { status: number; body: Buffer | string };
let dir: string;
let written: Array<{ runId: string; path: string; data: Buffer }>;

beforeEach(async () => {
  realFetch = globalThis.fetch;
  calls = [];
  written = [];
  reply = { status: 200, body: PNG };
  dir = await mkdtemp(join(tmpdir(), "pod-file-"));
  globalThis.fetch = (async (url: string, init: any) => {
    calls.push({ url: String(url), headers: init.headers });
    return new Response(reply.body, { status: reply.status });
  }) as typeof fetch;
});
afterEach(async () => {
  globalThis.fetch = realFetch;
  await rm(dir, { recursive: true, force: true });
});

function ctxOf(job?: string): PodCtx {
  return {
    runId: "r1",
    path: "p",
    ...(job ? { job } : {}),
    services: {
      dataDir: dir,
      secrets: { get: async (n: string) => (n === "HIVE_API_KEY" ? "hiveorg_key" : undefined) },
      artifacts: { write: async (runId: string, path: string, data: Buffer) => void written.push({ runId, path, data }) },
    },
  } as unknown as PodCtx;
}

async function run(ctx: PodCtx, over: Record<string, string> = {}) {
  const cfg = file.input.parse({ control: CONTROL, sealed: await seal(ctx, "pod-password"), path: "captcha/round-1.png", to: "captcha/round-1.png", ...over });
  return file.run(cfg, ctx);
}

test("under a job the bytes land in the job dir and the url is the job file url", async () => {
  const out = await run(ctxOf("job-1"));
  assert.deepEqual(out, { path: "captcha/round-1.png", bytes: PNG.length, url: "/jobs/job-1/files/captcha/round-1.png" });
  assert.deepEqual(await readFile(join(jobRoot(dir, "job-1"), "captcha", "round-1.png")), PNG);
  assert.equal(calls[0].url, `${CONTROL}/file?path=captcha%2Fround-1.png`);
  assert.equal(calls[0].headers.Authorization, "Bearer pod-password");
});

test("with no job the bytes go through the artifacts capability", async () => {
  const out = await run(ctxOf());
  assert.deepEqual(out, { path: "captcha/round-1.png", bytes: PNG.length, url: "/artifacts/r1/captcha/round-1.png" });
  assert.equal(written[0].runId, "r1");
  assert.equal(written[0].path, "captcha/round-1.png");
  assert.deepEqual(written[0].data, PNG);
});

test("404, 400 and 413 are clear errors that never carry the password", async () => {
  for (const [status, re] of [[404, /404 — no file/], [400, /400 — the pod refused the path/], [413, /413 .*10 MB/]] as const) {
    reply = { status, body: "nope" };
    await assert.rejects(run(ctxOf("job-1")), (e: Error) => re.test(e.message) && !e.message.includes("pod-password"));
  }
});

test("a `to` or `path` that escapes is rejected before any request", async () => {
  for (const over of [{ to: "../x.png" }, { to: "/etc/x" }, { to: "a/../../x" }, { path: "../secret" }]) {
    await assert.rejects(run(ctxOf("job-1"), over), /stays inside/);
  }
  assert.equal(calls.length, 0);
});
