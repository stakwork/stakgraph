import { describe, it } from "node:test";
import assert from "node:assert/strict";
import configApply from "./steps/config-apply.js";
import { seal, type PodCtx } from "./steps/_shared.js";

// pod/config-apply over a stand-in hive (the stored pool config) and a
// stand-in pod (staklink's PUT /pod-config + /script_progress).

const SECRET = "sk-live-0123456789";
const PM2 = `module.exports = { apps: [{ name: "frontend", script: "npm run dev", env: { API_KEY: "${SECRET}", PORT: "3000" } }] };\n`;
const COMPOSE = "services:\n  postgres:\n    image: postgres:15\n";
const DOCKERFILE = "FROM ghcr.io/stakwork/staklink-js:latest\n";
const b64 = (s: string) => Buffer.from(s).toString("base64");
const CONTROL = "https://pod-15552.example";

const APPLIED = {
  success: true,
  written: ["pm2.config.js", "docker-compose.yml"],
  unchanged: ["Dockerfile"],
  changed: { pm2: true, compose: true, dockerfile: false },
  compose: { up: true, project: "pod-abc", projectFrom: "pod", services: ["postgres"] },
  reload: { ok: true },
  rebuildNeeded: false,
  processes: [
    { name: "staklink-proxy", status: "online" },
    { name: "frontend", status: "online" },
  ],
};

interface Call {
  method: string;
  url: string;
  headers: Record<string, string>;
  body?: any;
}

function setup(opts: { stored?: Record<string, string>; put?: { status: number; body: any }; progress?: any[] } = {}) {
  const calls: Call[] = [];
  const progress = [...(opts.progress ?? [{ status: "completed", result: APPLIED }])];
  const ctx = {
    runId: "r",
    path: "p",
    services: {
      secrets: { get: async (n: string) => ({ HIVE_URL: "https://hive.example", HIVE_API_KEY: "hiveorg_key" })[n] },
      http: async (url: string, init: any) => {
        calls.push({ method: init.method, url, headers: init.headers ?? {}, body: init.body });
        if (url === "https://hive.example/api/workspaces/acme/stakgraph") {
          const files = opts.stored ?? { "pm2.config.js": PM2, "docker-compose.yml": COMPOSE, Dockerfile: DOCKERFILE, "devcontainer.json": "{}" };
          const containerFiles = Object.fromEntries(Object.entries(files).map(([k, v]) => [k, b64(v)]));
          return { status: 200, ok: true, headers: {}, body: { success: true, data: { services: [], environmentVariables: [], containerFiles } } };
        }
        if (url === `${CONTROL}/pod-config`) {
          const r = opts.put ?? { status: 200, body: { request_id: "req-1", status: "pending" } };
          return { status: r.status, ok: r.status < 400, headers: {}, body: r.body };
        }
        if (url.startsWith(`${CONTROL}/script_progress`)) {
          return { status: 200, ok: true, headers: {}, body: progress.shift() ?? { status: "pending" } };
        }
        throw new Error(`unexpected request: ${init.method} ${url}`);
      },
    },
  } as unknown as PodCtx;
  return { ctx, calls };
}

const run = async (ctx: PodCtx) =>
  configApply.run(
    configApply.input.parse({ workspace: "acme", control: CONTROL, sealed: await seal(ctx, "pod-password"), pollMs: 1 }),
    ctx,
  );

describe("pod/config-apply", () => {
  it("sends hive's stored files to the pod, as the pod, and reports what the pod did — never the files", async () => {
    const { ctx, calls } = setup({ progress: [{ status: "pending" }, { status: "completed", result: APPLIED }] });
    const out = await run(ctx);

    const put = calls.find((c) => c.method === "PUT")!;
    assert.equal(put.url, `${CONTROL}/pod-config`);
    assert.equal(put.headers.Authorization, "Bearer pod-password");
    assert.deepEqual(put.body, {
      files: [
        { name: "pm2.config.js", content: PM2 },
        { name: "docker-compose.yml", content: COMPOSE },
        { name: "Dockerfile", content: DOCKERFILE },
      ],
    }, "hive's three files as stored, values in clear; devcontainer.json never");
    assert.equal(calls.filter((c) => c.url.includes("/script_progress")).length, 2, "polled until it settled");

    assert.deepEqual(out, {
      workspace: "acme",
      applied: ["pm2.config.js", "docker-compose.yml", "Dockerfile"],
      written: ["pm2.config.js", "docker-compose.yml"],
      unchanged: ["Dockerfile"],
      success: true,
      compose: { up: true, project: "pod-abc", services: ["postgres"] },
      reload: { ok: true },
      rebuildNeeded: false,
      processes: APPLIED.processes,
      request_id: "req-1",
    });
    assert.ok(!JSON.stringify(out).includes(SECRET), "the env values stay out of the output");
    configApply.output.parse(out);
  });

  it("passes a compose or reload failure through as the result, not an error", async () => {
    const failed = {
      ...APPLIED,
      success: false,
      compose: { up: false, project: "pod-abc", projectFrom: "pod", services: ["postgres"], error: "docker compose up: port is already allocated" },
      rebuildNeeded: true,
    };
    const { ctx } = setup({ progress: [{ status: "completed", result: failed }] });
    const out = await run(ctx);
    assert.equal(out.success, false);
    assert.equal(out.compose.error, "docker compose up: port is already allocated");
    assert.equal(out.rebuildNeeded, true);
  });

  it("says what to do on a pod whose staklink predates the route", async () => {
    const { ctx } = setup({ put: { status: 404, body: "Cannot PUT /pod-config" } });
    await assert.rejects(run(ctx), /staklink predates PUT \/pod-config.*pod\/config-set \{ wait: true \}, then pod\/claim \{ fresh: true \}/);
  });

  it("fails when the pod could not apply it, or the pod is busy", async () => {
    const failed = setup({ progress: [{ status: "failed", error: { message: "EACCES: permission denied" } }] });
    await assert.rejects(run(failed.ctx), /could not apply the config: EACCES/);
    const busy = setup({ put: { status: 409, body: { error: "A pod config write is already in progress" } } });
    await assert.rejects(run(busy.ctx), /409 — the pod is busy/);
  });

  it("refuses before touching the pod when hive stores no pm2.config.js", async () => {
    const { ctx, calls } = setup({ stored: { "docker-compose.yml": COMPOSE } });
    await assert.rejects(run(ctx), /hive stores no pm2.config.js for workspace "acme"/);
    assert.ok(!calls.some((c) => c.url.startsWith(CONTROL)));
  });
});
