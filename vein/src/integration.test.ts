import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { flow, step, defineStep, type StepRegistry, type RunEvent } from "./core.js";
import { runWorkflow } from "./runner.js";
import { FileRunStore, MemoryRunStore } from "./store.js";
import { readFile } from "node:fs/promises";

// ── Test helpers ───────────────────────────────────────────────────────────

const echoStep = defineStep({
  type: "echo",
  input: z.any(),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});

const valueStep = defineStep({
  type: "value",
  input: z.object({ result: z.any() }),
  output: z.any(),
  async run(cfg) {
    return cfg.result;
  },
});

const failStep = defineStep({
  type: "fail",
  input: z.object({ message: z.string().default("step failed") }),
  output: z.any(),
  async run(cfg) {
    throw new Error(cfg.message);
  },
});

/** Simulates an API call that returns a body with a status field */
const mockApiStep = defineStep({
  type: "mock_api",
  input: z.object({ url: z.string(), responses: z.array(z.any()).optional() }),
  output: z.any(),
  async run(cfg, ctx) {
    // Use a counter in scope to track calls
    const callCount = ((ctx.scope as any).__mockApiCalls ?? 0) + 1;
    (ctx.scope as any).__mockApiCalls = callCount;

    if (cfg.responses && cfg.responses.length >= callCount) {
      return cfg.responses[callCount - 1];
    }
    return { status: 200, body: { url: cfg.url, call: callCount } };
  },
});

function makeRegistry(extra: Record<string, any> = {}): StepRegistry {
  return {
    echo: echoStep,
    value: valueStep,
    fail: failStep,
    mock_api: mockApiStep,
    ...extra,
  } as StepRegistry;
}

// ── Integration: Full workflow patterns ────────────────────────────────────

describe("integration: complete workflow patterns", () => {
  it("deploy workflow pattern: sequential steps with templates", async () => {
    const wf = flow("deploy", {
      input: z.object({ service: z.string(), env: z.string() }),
      steps: [
        step("kick", "echo", {
          url: "/deploy/{{ input.service }}",
          method: "POST",
          env: "{{ input.env }}",
        }),
        step("verify", "echo", {
          url: "/health/{{ input.service }}",
          previousUrl: "{{ kick.url }}",
        }),
        step("done", "value", {
          result: "deployed {{ input.service }} to {{ input.env }}",
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      { service: "api", env: "prod" },
      makeRegistry(),
    );

    assert.equal(result.status, "success");
    assert.equal(result.output, "deployed api to prod");
  });

  it("fan-out/fan-in pattern: DAG branches then merge", async () => {
    const wf = flow("enrich", {
      input: z.object({ id: z.string() }),
      steps: [
        step("profile", "value", {
          result: { name: "User-{{ input.id }}", age: 30 },
        }),
        step("orders", "value", {
          result: { count: 5, total: 250 },
        }, { depends: [] }),
        step("save", "echo", {
          profile: "{{ profile }}",
          orders: "{{ orders }}",
          combined: true,
        }, { depends: ["profile", "orders"] }),
      ],
    });

    const result = await runWorkflow(wf, { id: "42" }, makeRegistry());
    assert.equal(result.status, "success");
    const output = result.output as any;
    assert.deepEqual(output.profile, { name: "User-42", age: 30 });
    assert.deepEqual(output.orders, { count: 5, total: 250 });
    assert.equal(output.combined, true);
  });

  it("conditional branching with downstream use", async () => {
    const wf = flow("conditional-deploy", {
      input: z.object({
        service: z.string(),
        canary: z.boolean(),
      }),
      steps: [
        step("strategy", "if", {
          cond: "{{ input.canary }}",
          then: step("canary", "value", { result: "canary-deploy" }),
          else: step("full", "value", { result: "full-deploy" }),
        }),
        step("execute", "echo", {
          type: "{{ strategy }}",
          service: "{{ input.service }}",
        }),
      ],
    });

    const r1 = await runWorkflow(
      wf,
      { service: "api", canary: true },
      makeRegistry(),
    );
    assert.deepEqual(r1.output, {
      type: "canary-deploy",
      service: "api",
    });

    const r2 = await runWorkflow(
      wf,
      { service: "api", canary: false },
      makeRegistry(),
    );
    assert.deepEqual(r2.output, {
      type: "full-deploy",
      service: "api",
    });
  });

  it("error recovery with onError and retry", async () => {
    let attempts = 0;
    const flakeyDeploy = defineStep({
      type: "flakey_deploy",
      input: z.any(),
      output: z.any(),
      async run() {
        attempts++;
        if (attempts <= 2) throw new Error(`Deploy attempt ${attempts} failed`);
        return { deployed: true, attempt: attempts };
      },
    });

    const wf = flow("resilient-deploy", {
      input: z.object({}),
      steps: [
        step(
          "deploy",
          "flakey_deploy",
          {},
          {
            retry: { max: 3, delayMs: 0 },
            onError: step("fallback", "value", { result: { deployed: false, fallback: true } }),
          },
        ),
        step("report", "echo", {
          deployed: "{{ deploy.deployed }}",
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ flakey_deploy: flakeyDeploy }) as StepRegistry,
    );
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { deployed: true });
  });

  it("subflow composition: parent passes data to child", async () => {
    const processItem = flow("process-item", {
      input: z.object({
        item: z.object({ id: z.string(), value: z.number() }),
      }),
      steps: [
        step("validate", "echo", {
          id: "{{ input.item.id }}",
          valid: "{{ input.item.value > 0 }}",
        }),
        step("transform", "value", {
          result: {
            id: "{{ input.item.id }}",
            processed: true,
            doubled: "{{ input.item.value * 2 }}",
          },
        }),
      ],
    });

    const parent = flow("batch-process", {
      input: z.object({}),
      steps: [
        step("sub1", "subflow", {
          flow: processItem,
          input: { item: { id: "a", value: 10 } },
        }),
        step("sub2", "subflow", {
          flow: processItem,
          input: { item: { id: "b", value: 20 } },
        }),
        step("summary", "echo", {
          item1: "{{ sub1 }}",
          item2: "{{ sub2 }}",
        }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry());
    assert.equal(result.status, "success");
    const output = result.output as any;
    assert.deepEqual(output.item1, {
      id: "a",
      processed: true,
      doubled: 20,
    });
    assert.deepEqual(output.item2, {
      id: "b",
      processed: true,
      doubled: 40,
    });
  });
});

// ── Integration: FileRunStore with runner ──────────────────────────────────

describe("integration: FileRunStore with runner", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-int-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("persists events and summary to disk", async () => {
    const store = new FileRunStore(tempDir);
    const wf = flow("persist-test", {
      input: z.object({ x: z.number() }),
      steps: [
        step("a", "value", { result: "{{ input.x * 2 }}" }),
        step("b", "echo", { doubled: "{{ a }}" }),
      ],
    });

    const result = await runWorkflow(wf, { x: 21 }, makeRegistry(), {
      runId: "persist-1",
      store,
    });

    assert.equal(result.status, "success");

    // Read events.jsonl — now under workflows/<name>/runs/<runId>/
    const eventsRaw = await readFile(
      join(tempDir, "workflows", "persist-test", "runs", "persist-1", "events.jsonl"),
      "utf-8",
    );
    const events = eventsRaw
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));

    // Should have: run.start, step.start(a), step.end(a), step.start(b), step.end(b), run.end
    assert.equal(events.length, 6);
    assert.equal(events[0].type, "run.start");
    assert.equal(events[events.length - 1].type, "run.end");

    // Read run.json
    const summaryRaw = await readFile(
      join(tempDir, "workflows", "persist-test", "runs", "persist-1", "run.json"),
      "utf-8",
    );
    const summary = JSON.parse(summaryRaw);
    assert.equal(summary.runId, "persist-1");
    assert.equal(summary.workflow, "persist-test");
    assert.equal(summary.status, "success");
    assert.deepEqual(summary.output, { doubled: 42 });
  });

  it("persists error runs to disk", async () => {
    const store = new FileRunStore(tempDir);
    const wf = flow("error-persist", {
      input: z.object({}),
      steps: [step("boom", "fail", { message: "test error" })],
    });

    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "err-persist",
      store,
    });

    const summaryRaw = await readFile(
      join(tempDir, "workflows", "error-persist", "runs", "err-persist", "run.json"),
      "utf-8",
    );
    const summary = JSON.parse(summaryRaw);
    assert.equal(summary.status, "error");
    assert.ok(summary.error.message.includes("test error"));
  });
});

// ── Integration: Complex nested workflows ──────────────────────────────────

describe("integration: complex nested workflows", () => {
  it("DAG branches each containing subflows", async () => {
    const worker = flow("worker", {
      input: z.object({ id: z.number() }),
      steps: [
        step("compute", "value", { result: "{{ input.id * 10 }}" }),
      ],
    });

    const wf = flow("dag-subflows", {
      input: z.object({}),
      steps: [
        step("a", "subflow", {
          flow: worker,
          input: { id: 1 },
        }),
        step("b", "subflow", {
          flow: worker,
          input: { id: 2 },
        }, { depends: [] }),
        step("merge", "echo", {
          a: "{{ a }}",
          b: "{{ b }}",
          sum: "{{ a + b }}",
        }, { depends: ["a", "b"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    const output = result.output as any;
    assert.equal(output.a, 10);
    assert.equal(output.b, 20);
    assert.equal(output.sum, 30);
  });

  it("if-then-else branching", async () => {
    const wf = flow("if-branching", {
      input: z.object({ fast: z.boolean() }),
      steps: [
        step("decide", "if", {
          cond: "{{ input.fast }}",
          then: step("quick", "value", { result: "fast-path" }),
          else: step("slow", "value", { result: "slow-path" }),
        }),
      ],
    });

    const fast = await runWorkflow(wf, { fast: true }, makeRegistry());
    assert.equal(fast.output, "fast-path");

    const slow = await runWorkflow(wf, { fast: false }, makeRegistry());
    assert.equal(slow.output, "slow-path");
  });

  it("long chain of 10 sequential steps", async () => {
    const steps = [];
    for (let i = 0; i < 10; i++) {
      if (i === 0) {
        steps.push(step(`s${i}`, "value", { result: 1 }));
      } else {
        steps.push(step(`s${i}`, "value", { result: `{{ s${i - 1} + 1 }}` }));
      }
    }

    const wf = flow("long-chain", {
      input: z.object({}),
      steps,
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, 10);
  });

  it("workflow with all step types combined", async () => {
    const child = flow("child-flow", {
      input: z.object({ val: z.string() }),
      steps: [
        step("echo", "echo", { received: "{{ input.val }}" }),
      ],
    });

    let loopCount = 0;
    const ticker = defineStep({
      type: "ticker",
      input: z.any(),
      output: z.any(),
      async run() {
        loopCount++;
        return { tick: loopCount };
      },
    });

    const wf = flow("kitchen-sink", {
      input: z.object({ mode: z.string() }),
      steps: [
        // Step 1: if
        step("decide", "if", {
          cond: "{{ input.mode === 'full' }}",
          then: step("full", "value", { result: "full-mode" }),
          else: step("lite", "value", { result: "lite-mode" }),
        }),
        // Step 2: DAG parallel branches (both depend on "decide")
        step("left", "value", { result: "left" }, { depends: "decide" }),
        step("right", "value", { result: "right" }, { depends: "decide" }),
        // Step 3: loop (depends on both branches)
        step("poll", "loop", {
          until: "{{ $current.tick >= 3 }}",
          maxIterations: 10,
          body: step("tick", "ticker", {}),
        }, { depends: ["left", "right"] }),
        // Step 4: subflow
        step("sub", "subflow", {
          flow: child,
          input: { val: "{{ decide }}" },
        }),
        // Step 5: echo everything
        step("summary", "echo", {
          mode: "{{ decide }}",
          left: "{{ left }}",
          right: "{{ right }}",
          loopResult: "{{ poll }}",
          subResult: "{{ sub }}",
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      { mode: "full" },
      makeRegistry({ ticker }) as StepRegistry,
    );
    assert.equal(result.status, "success");
    const output = result.output as any;
    assert.equal(output.mode, "full-mode");
    assert.equal(output.left, "left");
    assert.equal(output.right, "right");
    assert.deepEqual(output.loopResult, { tick: 3 });
    assert.deepEqual(output.subResult, { received: "full-mode" });
  });
});

// ── Integration: Event path tracking ───────────────────────────────────────

describe("integration: event path tracking", () => {
  it("tracks correct paths through DAG", async () => {
    const store = new MemoryRunStore();
    const wf = flow("paths", {
      input: z.object({}),
      steps: [
        step("a", "value", { result: 1 }),
        step("b", "value", { result: "B" }),
      ],
    });

    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "path-test",
      store,
    });

    const events = store.getEvents("paths", "path-test");
    const paths = events.map((e) => e.path);

    // Should see paths: paths, paths/a, paths/b
    assert.ok(paths.includes("paths"));
    assert.ok(paths.some((p) => p === "paths/a"));
    assert.ok(paths.some((p) => p === "paths/b"));
  });
});
