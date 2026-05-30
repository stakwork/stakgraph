import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { createVein } from "./createVein.js";
import { createRegistry } from "./steps/registry.js";
import { defineStep, flow, step } from "./core.js";
import { WorkspaceManager } from "./workspace.js";
import { MemoryRunStore } from "./store.js";

// ── createRegistry ─────────────────────────────────────────────────────────

describe("createRegistry", () => {
  it("layers user steps on top of built-in core and lib steps", async () => {
    const myStep = defineStep({
      type: "my-step",
      input: z.object({}),
      output: z.string(),
      async run() {
        return "hi";
      },
    });
    const reg = await createRegistry([myStep]);
    assert.equal(typeof reg["my-step"], "object");
    // Core steps are present.
    assert.equal(typeof reg["http"], "object");
    assert.equal(typeof reg["log"], "object");
    assert.equal(typeof reg["foreach"], "object");
    // Lib steps are present too.
    assert.equal(typeof reg["github/fetch-pr"], "object");
  });

  it("returns core + lib when called with no user steps", async () => {
    const reg = await createRegistry([]);
    assert.ok("http" in reg);
    assert.ok("if" in reg);
    assert.ok("github/fetch-pr" in reg);
  });

  it("throws on duplicate step types in the input", async () => {
    const a = defineStep({
      type: "dup",
      input: z.object({}),
      output: z.any(),
      async run() {
        return 1;
      },
    });
    const b = defineStep({
      type: "dup",
      input: z.object({}),
      output: z.any(),
      async run() {
        return 2;
      },
    });
    await assert.rejects(
      () => createRegistry([a, b]),
      /duplicate step type "dup"/,
    );
  });

  it("throws when a step is missing required fields", async () => {
    await assert.rejects(
      () => createRegistry([{ type: "x" } as any]),
      /missing "type" or "run"/,
    );
  });

  it("lets a user step shadow a core step", async () => {
    const myHttp = defineStep({
      type: "http",
      input: z.object({}),
      output: z.string(),
      async run() {
        return "shadowed";
      },
    });
    const reg = await createRegistry([myHttp]);
    assert.equal(reg["http"], myHttp);
  });
});

// ── createVein basics ──────────────────────────────────────────────────────

describe("createVein", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-factory-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("builds an instance with sensible defaults", async () => {
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    assert.ok(vein.app, "app should be a Hono instance");
    assert.equal(vein.workspace.path, tempDir);
    assert.deepEqual(vein.services, {}); // default services
    const reg = vein.getRegistry();
    assert.ok("http" in reg);
  });

  it("uses an injected registry as-is", async () => {
    const myStep = defineStep({
      type: "ping",
      input: z.object({}),
      output: z.string(),
      async run() {
        return "pong";
      },
    });

    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    const reg = vein.getRegistry();
    assert.ok("ping" in reg);
  });

  it("threads services into vein.run()", async () => {
    interface Services {
      tag: string;
      readings: number[];
    }

    const recordStep = defineStep<"record", z.ZodObject<{ value: z.ZodNumber }>, z.ZodAny, Services>({
      type: "record",
      input: z.object({ value: z.number() }),
      output: z.any(),
      async run(cfg, ctx) {
        ctx.services.readings.push(cfg.value);
        return { tag: ctx.services.tag, total: ctx.services.readings.length };
      },
    });

    const readings: number[] = [];
    const vein = await createVein<Services>({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([recordStep]),
      store: new MemoryRunStore(),
      services: { tag: "test-env", readings },
      serveUi: false,
      enableChat: false,
    });

    const wf = flow("record-test", {
      input: z.object({}),
      steps: [step("r", "record", { value: 42 })],
    });

    const result = await vein.run(wf, {});
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { tag: "test-env", total: 1 });
    assert.deepEqual(readings, [42]);
  });

  it("allows per-run services overrides", async () => {
    interface Services {
      label: string;
    }

    const labelStep = defineStep<"label", z.ZodObject<{}>, z.ZodString, Services>({
      type: "label",
      input: z.object({}),
      output: z.string(),
      async run(_cfg, ctx) {
        return ctx.services.label;
      },
    });

    const vein = await createVein<Services>({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([labelStep]),
      store: new MemoryRunStore(),
      services: { label: "default" },
      serveUi: false,
      enableChat: false,
    });

    const wf = flow("label-test", {
      input: z.object({}),
      steps: [step("l", "label", {})],
    });

    const a = await vein.run(wf);
    assert.equal(a.output, "default");

    const b = await vein.run(wf, {}, { services: { label: "override" } });
    assert.equal(b.output, "override");
  });

  it("resolves workflows by name from the workspace", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("hello", "v1", {
      steps: [
        { id: "g", type: "log", config: { message: "hi" } },
      ],
    });

    const vein = await createVein({
      workspace: ws,
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    const result = await vein.run("hello", {});
    assert.equal(result.status, "success");
  });

  it("exposes a working /health endpoint", async () => {
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await vein.app.request("/health");
    assert.equal(res.status, 200);
    const body = (await res.json()) as { ok: boolean; stepCount: number };
    assert.equal(body.ok, true);
    assert.ok(body.stepCount > 0);
  });

  it("exposes /steps with registered types", async () => {
    const myStep = defineStep({
      type: "custom-thing",
      input: z.object({}),
      output: z.any(),
      async run() {
        return null;
      },
    });
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await vein.app.request("/steps");
    assert.equal(res.status, 200);
    const body = (await res.json()) as { core: Array<{ type: string }> };
    const types = body.core.map((s) => s.type);
    assert.ok(types.includes("custom-thing"));
    assert.ok(types.includes("http"));
  });

  it("rejects POST /steps when the registry was injected", async () => {
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await vein.app.request("/steps", {
      method: "POST",
      body: JSON.stringify({ name: "x", code: "export default {}" }),
      headers: { "content-type": "application/json" },
    });
    assert.equal(res.status, 409);
  });

  it("runs a registered workflow over HTTP via SSE", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("echo-flow", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "hi" } }],
    });
    const vein = await createVein({
      workspace: ws,
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await vein.app.request("/workflows/echo-flow/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: {} }),
    });
    assert.equal(res.status, 200);
    const text = await res.text();
    // SSE stream contains at least one event and a final done event.
    assert.ok(text.includes("run.start"));
    assert.ok(text.includes("event: done"));
  });

  it("rebuildRegistry is a no-op when the registry was injected", async () => {
    const myStep = defineStep({
      type: "fixed",
      input: z.object({}),
      output: z.any(),
      async run() {
        return 1;
      },
    });
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const before = Object.keys(vein.getRegistry()).sort();
    await vein.rebuildRegistry();
    const after = Object.keys(vein.getRegistry()).sort();
    assert.deepEqual(before, after);
  });

  it("works with an in-memory store (no FileRunStore methods called)", async () => {
    const memStore = new MemoryRunStore();
    const vein = await createVein({
      workspace: new WorkspaceManager(tempDir),
      store: memStore,
      serveUi: false,
      enableChat: false,
    });

    const wf = flow("mem-test", {
      input: z.object({}),
      steps: [step("g", "log", { message: "hello" })],
    });

    const result = await vein.run(wf, {});
    assert.equal(result.status, "success");

    // The /runs endpoint should refuse, not crash, when the store
    // doesn't support listing.
    const res = await vein.app.request("/workflows/mem-test/runs");
    assert.equal(res.status, 501);
  });
});
