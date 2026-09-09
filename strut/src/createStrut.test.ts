import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { createStrut } from "./createStrut.js";
import { createRegistry } from "./steps/registry.js";
import { defineStep, flow, step } from "./core.js";
import { pathlessWorkspace } from "./test-util/pathless-workspace.js";
import { WorkspaceManager } from "./workspace.js";
import { MemoryRunStore, FileRunStore } from "./store.js";

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

// ── createStrut basics ──────────────────────────────────────────────────────

describe("createStrut", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `strut-factory-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("builds an instance with sensible defaults", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    assert.ok(strut.app, "app should be a Hono instance");
    assert.equal(strut.workspace.path, tempDir);
    // Default services provide the standard adapter capabilities out of the box.
    const svc = strut.services as { http?: unknown; secrets?: unknown };
    assert.equal(typeof svc.http, "function");
    assert.equal(typeof svc.secrets, "object");
    const reg = strut.getRegistry();
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

    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    const reg = strut.getRegistry();
    assert.ok("ping" in reg);
  });

  it("threads services into strut.run()", async () => {
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
    const strut = await createStrut<Services>({
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

    const result = await strut.run(wf, {});
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

    const strut = await createStrut<Services>({
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

    const a = await strut.run(wf);
    assert.equal(a.output, "default");

    const b = await strut.run(wf, {}, { services: { label: "override" } });
    assert.equal(b.output, "override");
  });

  it("resolves workflows by name from the workspace", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("hello", "v1", {
      steps: [
        { id: "g", type: "log", config: { message: "hi" } },
      ],
    });

    const strut = await createStrut({
      workspace: ws,
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });

    const result = await strut.run("hello", {});
    assert.equal(result.status, "success");
  });

  it("serves a partial summary for a run with events but no run.json", async () => {
    const ws = new WorkspaceManager(tempDir);
    const strut = await createStrut({
      workspace: ws,
      serveUi: false,
      enableChat: false,
    });
    // Simulate a run orphaned before finalize: events on disk, no run.json.
    const ev = (over: Record<string, unknown>) => ({
      ts: "2026-01-01T00:00:00.000Z",
      runId: "9999",
      path: "dead-wf",
      type: "run.start",
      ...over,
    });
    await strut.store.append("dead-wf", "9999", ev({ input: { taskId: "t1" } }) as never);
    await strut.store.append(
      "dead-wf",
      "9999",
      ev({ type: "step.end", path: "dead-wf/first", output: { n: 1 }, ts: "2026-01-01T00:01:00.000Z" }) as never,
    );

    const res = await strut.app.request("/workflows/dead-wf/runs/9999");
    assert.equal(res.status, 200);
    const body = (await res.json()) as Record<string, unknown>;
    assert.equal(body.partial, true);
    assert.equal(body.status, "stale");
    assert.deepEqual(body.input, { taskId: "t1" });
    assert.deepEqual(body.steps, { first: { n: 1 } });

    // A run with no events at all is still a 404.
    const missing = await strut.app.request("/workflows/dead-wf/runs/1234");
    assert.equal(missing.status, 404);
  });

  it("exposes a working /health endpoint", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await strut.app.request("/health");
    assert.equal(res.status, 200);
    const body = (await res.json()) as { ok: boolean; stepCount: number };
    assert.equal(body.ok, true);
    assert.ok(body.stepCount > 0);
  });

  it("runs a single step via POST /steps/:type/run", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await strut.app.request("/steps/log/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config: { message: "hello {{ input.name }}" },
        input: { name: "world" },
      }),
    });
    assert.equal(res.status, 200);
    const body = (await res.json()) as { status: string; output: unknown };
    assert.equal(body.status, "success");
    assert.equal(body.output, "hello world");

    const missing = await strut.app.request("/steps/nope/run", { method: "POST" });
    assert.equal(missing.status, 404);
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
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await strut.app.request("/steps");
    assert.equal(res.status, 200);
    const body = (await res.json()) as { core: Array<{ type: string }> };
    const types = body.core.map((s) => s.type);
    assert.ok(types.includes("custom-thing"));
    assert.ok(types.includes("http"));
  });

  it("rejects POST /steps when the registry was injected", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const res = await strut.app.request("/steps", {
      method: "POST",
      body: JSON.stringify({ name: "x", code: "export default {}" }),
      headers: { "content-type": "application/json" },
    });
    assert.equal(res.status, 409);
  });

  it("exposes step version endpoints (list / get-version / set-active)", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const headers = { "content-type": "application/json" };

    // Publish v1, then a changed v2 of the same step.
    const pub1 = await strut.app.request("/steps", {
      method: "POST",
      headers,
      body: JSON.stringify({ name: "scorer", code: "// v1" }),
    });
    const { version: v1 } = (await pub1.json()) as { version: string };
    await strut.app.request("/steps", {
      method: "POST",
      headers,
      body: JSON.stringify({ name: "scorer", code: "// v2" }),
    });

    // versions lists both, active is v2
    const verRes = await strut.app.request("/steps/scorer/versions");
    assert.equal(verRes.status, 200);
    const ver = (await verRes.json()) as { active: string; versions: string[] };
    assert.equal(ver.versions.length, 2);
    assert.notEqual(ver.active, v1);

    // archived source for v1 is retrievable
    const srcRes = await strut.app.request(`/steps/scorer/version/${v1}`);
    assert.equal(srcRes.status, 200);
    const src = (await srcRes.json()) as { source: string };
    assert.equal(src.source, "// v1");

    // set active back to v1
    const actRes = await strut.app.request("/steps/scorer/active", {
      method: "PUT",
      headers,
      body: JSON.stringify({ version: v1 }),
    });
    assert.equal(actRes.status, 200);
    const ver2Res = await strut.app.request("/steps/scorer/versions");
    const ver2 = (await ver2Res.json()) as { active: string };
    assert.equal(ver2.active, v1);
  });

  it("launches a run detached over HTTP, returning a runId immediately", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("echo-flow", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "hi" } }],
    });
    const strut = await createStrut({
      workspace: ws,
      store: new FileRunStore(tempDir),
      serveUi: false,
      enableChat: false,
    });
    const res = await strut.app.request("/workflows/echo-flow/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: {} }),
    });
    // Detached: 202 + { runId }, no streamed body.
    assert.equal(res.status, 202);
    const { runId } = (await res.json()) as { runId: string };
    assert.ok(runId, "should return a runId");
    // Drain to completion so the background writes settle before teardown.
    const drain = await strut.app.request(`/workflows/echo-flow/runs/${runId}/stream`);
    await drain.text();
  });

  it("reattaches to a run via SSE tail (history + final done)", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("echo-flow", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "hi" } }],
    });
    const strut = await createStrut({
      workspace: ws,
      store: new FileRunStore(tempDir),
      serveUi: false,
      enableChat: false,
    });

    // Launch detached, then tail its log to completion.
    const launch = await strut.app.request("/workflows/echo-flow/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input: {} }),
    });
    const { runId } = (await launch.json()) as { runId: string };

    const res = await strut.app.request(`/workflows/echo-flow/runs/${runId}/stream`);
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.ok(text.includes("run.start"), "replays history");
    assert.ok(text.includes("run.end"), "follows to the terminal event");
    assert.ok(text.includes("event: done"), "sends a final done with the result");
  });

  it("streams a run reattached after it already finished", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("echo-flow", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "hi" } }],
    });
    const strut = await createStrut({
      workspace: ws,
      store: new FileRunStore(tempDir),
      serveUi: false,
      enableChat: false,
    });

    // Run to completion *first* (so the log is fully written), then attach.
    const result = await strut.run("echo-flow", {});
    assert.equal(result.status, "success");

    const res = await strut.app.request(
      `/workflows/echo-flow/runs/${result.runId}/stream`,
    );
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.ok(text.includes("run.start"));
    assert.ok(text.includes("event: done"));
  });

  it("streams a completed run's events from a MemoryRunStore (no capability gate)", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const wf = flow("mem-stream", {
      input: z.object({}),
      steps: [step("g", "log", { message: "hello" })],
    });
    const result = await strut.run(wf, {});
    const res = await strut.app.request(`/workflows/mem-stream/runs/${result.runId}/stream`);
    assert.equal(res.status, 200);
    const text = await res.text();
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
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      registry: await createRegistry([myStep]),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const before = Object.keys(strut.getRegistry()).sort();
    await strut.rebuildRegistry();
    const after = Object.keys(strut.getRegistry()).sort();
    assert.deepEqual(before, after);
  });

  it("serves run history from an in-memory store (list, lookup, events)", async () => {
    const memStore = new MemoryRunStore();
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: memStore,
      serveUi: false,
      enableChat: false,
    });

    const wf = flow("mem-test", {
      input: z.object({}),
      steps: [step("g", "log", { message: "hello" })],
    });

    const result = await strut.run(wf, {});
    assert.equal(result.status, "success");

    // Every read endpoint works over the memory store — it is a complete
    // ephemeral backend, not a write-only stub.
    const list = await strut.app.request("/workflows/mem-test/runs");
    assert.equal(list.status, 200);
    const runs = (await list.json()) as { runId: string; status: string }[];
    assert.deepEqual(runs.map((r) => [r.runId, r.status]), [[result.runId, "success"]]);

    const one = await strut.app.request(`/workflows/mem-test/runs/${result.runId}`);
    assert.equal(one.status, 200);
    assert.equal(((await one.json()) as { status: string }).status, "success");

    const events = await strut.app.request(`/workflows/mem-test/runs/${result.runId}/events`);
    assert.equal(events.status, 200);
    const types = ((await events.json()) as { type: string }[]).map((e) => e.type);
    assert.ok(types.includes("run.start") && types.includes("run.end"));

    const missing = await strut.app.request("/workflows/mem-test/runs/nope");
    assert.equal(missing.status, 404);
  });

  it("custom steps can `import \"strut\"` from a workspace outside the package tree", async () => {
    // tempDir is under the OS tmpdir — no `strut` package is reachable by
    // walking up from it. The resolve hook (strut-resolver.ts) maps the bare
    // specifier to this running strut, so the step gets the same defineStep/z.
    const ws = new WorkspaceManager(tempDir);
    await ws.publishStep(
      "hook-step",
      `import { z, defineStep } from "strut";
       export default defineStep({
         type: "hook-step",
         input: z.object({ name: z.string() }),
         output: z.string(),
         async run({ input }) { return "hi " + input.name; },
       });`,
    );
    const strut = await createStrut({ workspace: ws, store: new MemoryRunStore(), serveUi: false, enableChat: false, stt: false });
    assert.ok("hook-step" in strut.getRegistry(), "step importing strut loads from an out-of-tree workspace");
    const { z: ourZ } = await import("zod");
    const def = strut.getRegistry()["hook-step"]!;
    assert.ok(def.input instanceof ourZ.ZodObject, "the step's zod is this process's zod (one module instance)");
  });

  it("custom steps published before the rename can still `import \"vein\"`", async () => {
    // Pre-#1664 step versions live in the graph verbatim (their content hash
    // is their identity), so the resolve hook keeps the old bare specifier.
    const ws = new WorkspaceManager(tempDir);
    await ws.publishStep(
      "legacy-hook-step",
      `import { z, defineStep } from "vein";
       export default defineStep({
         type: "legacy-hook-step",
         input: z.object({}),
         output: z.string(),
         async run() { return "still here"; },
       });`,
    );
    const strut = await createStrut({ workspace: ws, store: new MemoryRunStore(), serveUi: false, enableChat: false, stt: false });
    assert.ok("legacy-hook-step" in strut.getRegistry(), "step importing the old package name loads");
  });

  it("a non-file WorkspaceStore gets in-memory store defaults and still loads custom steps", async () => {
    const ws = pathlessWorkspace(new WorkspaceManager(tempDir));
    // Import-free step source (the temp dir sits outside the project tree,
    // so `import "strut"` wouldn't resolve — same trick as registry.test.ts).
    await ws.publishStep(
      "conf-step",
      `export default {
        type: "conf-step",
        input: { _def: { typeName: 'ZodObject', shape: () => ({}) } },
        output: { _def: { typeName: 'ZodAny' } },
        async run() { return "ok"; },
      };`,
    );
    const strut = await createStrut({ workspace: ws, dataDir: join(tempDir, "data"), serveUi: false, enableChat: false });
    assert.ok(strut.store instanceof MemoryRunStore, "run store defaults to memory for a non-file workspace");
    assert.equal(strut.dataDir, join(tempDir, "data"));
    assert.ok("conf-step" in strut.getRegistry(), "custom steps load via materializeCustomSteps()");
    const health = (await (await strut.app.request("/health")).json()) as { dataDir: string };
    assert.equal(health.dataDir, join(tempDir, "data"));
    const meta = await strut.app.request("/workflows/nope");
    assert.equal(meta.status, 404);
  });

  it("dataDir defaults to the file workspace root and is overridable", async () => {
    const a = await createStrut({ workspace: new WorkspaceManager(tempDir), serveUi: false, enableChat: false });
    assert.equal(a.dataDir, tempDir);
    const b = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      dataDir: join(tempDir, "elsewhere"),
      serveUi: false,
      enableChat: false,
    });
    assert.equal(b.dataDir, join(tempDir, "elsewhere"));
  });

  it("GET /workflows decorates entries with lastRunAt from the run store", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("ran", "v1", { steps: [{ id: "g", type: "log", config: { message: "x" } }] });
    await ws.publishWorkflow("never", "v1", { steps: [{ id: "g", type: "log", config: { message: "x" } }] });
    const strut = await createStrut({
      workspace: ws,
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
    });
    const result = await strut.run("ran", {});
    const res = await strut.app.request("/workflows");
    const list = (await res.json()) as { name: string; lastRunAt?: number }[];
    const byName = Object.fromEntries(list.map((w) => [w.name, w.lastRunAt]));
    assert.equal(byName["ran"], Number(result.runId));
    assert.equal(byName["never"], undefined);
  });

  it("resolves + applies a declared promotion (run output → target param)", async () => {
    const ws = new WorkspaceManager(tempDir);
    // Target workflow whose `system` param we'll promote into.
    await ws.publishWorkflow("target", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "{{ params.system }}" } }],
      params: { system: "old" },
    });
    // Optimizer workflow: emits `{ bestPrompt }` and declares a promote to target.
    await ws.publishWorkflow("opt", "v1", {
      steps: [{ id: "best", type: "emit", config: {} }],
      promotes: [{ from: "bestPrompt", to: "target.system", label: "Sys" }],
    });

    const emit = defineStep({
      type: "emit",
      input: z.object({}),
      output: z.any(),
      async run() {
        return { bestPrompt: "new winner" };
      },
    });

    const strut = await createStrut({
      workspace: ws,
      registry: await createRegistry([emit]),
      store: new FileRunStore(tempDir),
      serveUi: false,
      enableChat: false,
    });

    const result = await strut.run("opt", {});
    assert.equal(result.status, "success");

    // GET promotions resolves the run-output value + the target's current value.
    const pRes = await strut.app.request(`/workflows/opt/runs/${result.runId}/promotions`);
    assert.equal(pRes.status, 200);
    const proms = (await pRes.json()) as any[];
    assert.equal(proms.length, 1);
    assert.equal(proms[0].value, "new winner");
    assert.equal(proms[0].current, "old");
    assert.equal(proms[0].target.workflow, "target");
    assert.equal(proms[0].target.param, "system");
    assert.equal(proms[0].resolved, true);

    // POST promote writes it + publishes a new target version.
    const applyRes = await strut.app.request(`/workflows/opt/runs/${result.runId}/promote`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ to: "target.system" }),
    });
    assert.equal(applyRes.status, 200);
    const applied = (await applyRes.json()) as any;
    assert.equal(applied.version, "v2");
    assert.equal(applied.before, "old");
    assert.equal(applied.after, "new winner");

    const flow = await ws.getWorkflow("target");
    assert.equal(flow.params?.["system"], "new winner");
  });

  it("returns [] promotions when the workflow declares none", async () => {
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("plain", "v1", {
      steps: [{ id: "g", type: "log", config: { message: "hi" } }],
    });
    const strut = await createStrut({
      workspace: ws,
      store: new FileRunStore(tempDir),
      serveUi: false,
      enableChat: false,
    });
    const result = await strut.run("plain", {});
    const pRes = await strut.app.request(`/workflows/plain/runs/${result.runId}/promotions`);
    assert.equal(pRes.status, 200);
    assert.deepEqual(await pRes.json(), []);
  });
});

describe("listen()", () => {
  let tempDir: string;
  beforeEach(async () => {
    tempDir = join(tmpdir(), `strut-listen-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });
  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("binds an OS-picked port on 0, honors the host, and prints a ready line", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
      stt: false,
    });
    const lines: string[] = [];
    const orig = console.log;
    console.log = (...a: unknown[]) => lines.push(a.map(String).join(" "));
    let port: number;
    try {
      port = await strut.listen(0, "127.0.0.1");
    } finally {
      console.log = orig;
    }
    try {
      assert.ok(port > 0, `expected a real port, got ${port}`);
      const ready = lines.map((l) => { try { return JSON.parse(l); } catch { return null; } }).find((j) => j?.event === "ready");
      assert.deepEqual(ready, { event: "ready", port, host: "127.0.0.1" });
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      assert.equal(res.status, 200);
      assert.equal(((await res.json()) as { ok: boolean }).ok, true);
    } finally {
      await strut.close();
    }
    await assert.rejects(fetch(`http://127.0.0.1:${port}/health`), "server should be closed");
  });

  it("puts the API key on the ready line only when STRUT_READY_KEY is set", async () => {
    const strut = await createStrut({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      serveUi: false,
      enableChat: false,
      stt: false,
    });
    const lines: string[] = [];
    const orig = console.log;
    const prevKey = process.env["STRUT_API_KEY"];
    const prevReady = process.env["STRUT_READY_KEY"];
    process.env["STRUT_API_KEY"] = "launch-key";
    process.env["STRUT_READY_KEY"] = "1";
    console.log = (...a: unknown[]) => lines.push(a.map(String).join(" "));
    let port: number;
    try {
      port = await strut.listen(0, "127.0.0.1");
    } finally {
      console.log = orig;
      if (prevKey === undefined) delete process.env["STRUT_API_KEY"]; else process.env["STRUT_API_KEY"] = prevKey;
      if (prevReady === undefined) delete process.env["STRUT_READY_KEY"]; else process.env["STRUT_READY_KEY"] = prevReady;
    }
    try {
      const ready = lines.map((l) => { try { return JSON.parse(l); } catch { return null; } }).find((j) => j?.event === "ready");
      assert.deepEqual(ready, { event: "ready", port, host: "127.0.0.1", key: "launch-key" });
    } finally {
      await strut.close();
    }
  });

  it("rejects when the port is taken instead of crashing the process", async () => {
    const mk = () =>
      createStrut({
        workspace: new WorkspaceManager(tempDir),
        store: new MemoryRunStore(),
        serveUi: false,
        enableChat: false,
        stt: false,
      });
    const a = await mk();
    const port = await a.listen(0, "127.0.0.1");
    const b = await mk();
    try {
      await assert.rejects(b.listen(port, "127.0.0.1"), /EADDRINUSE/);
    } finally {
      await a.close();
      await b.close();
    }
  });
});
