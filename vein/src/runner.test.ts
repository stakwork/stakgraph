import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step, defineStep, type StepRegistry, type RunEvent } from "./core.js";
import { runWorkflow } from "./runner.js";
import { MemoryRunStore } from "./store.js";

// ── Test helpers ───────────────────────────────────────────────────────────

/** A simple step that returns its config as output. */
const echoStep = defineStep({
  type: "echo",
  input: z.any(),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});

/** A step that returns a fixed value. */
const valueStep = defineStep({
  type: "value",
  input: z.object({ result: z.any() }),
  output: z.any(),
  async run(cfg) {
    return cfg.result;
  },
});

/** A step that always throws. */
const failStep = defineStep({
  type: "fail",
  input: z.object({ message: z.string().default("step failed") }),
  output: z.any(),
  async run(cfg) {
    throw new Error(cfg.message);
  },
});

/** A step that counts how many times it's been called (stateful). */
function createCounterStep() {
  let count = 0;
  return defineStep({
    type: "counter",
    input: z.any(),
    output: z.number(),
    async run() {
      count++;
      return count;
    },
  });
}

/** A step that fails the first N times, then succeeds. */
function createFlakeyStep(failCount: number) {
  let attempts = 0;
  return defineStep({
    type: "flakey",
    input: z.any(),
    output: z.any(),
    async run(cfg) {
      attempts++;
      if (attempts <= failCount) {
        throw new Error(`Attempt ${attempts} failed`);
      }
      return { attempts, ...(cfg ?? {}) };
    },
  });
}

function makeRegistry(extra: Record<string, any> = {}): StepRegistry {
  return {
    echo: echoStep,
    value: valueStep,
    fail: failStep,
    ...extra,
  } as StepRegistry;
}

function eventTypes(store: MemoryRunStore, workflow: string, runId: string): string[] {
  return store.getEvents(workflow, runId).map((e) => e.type);
}

function eventsOfType(
  store: MemoryRunStore,
  workflow: string,
  runId: string,
  type: string,
): RunEvent[] {
  return store.getEvents(workflow, runId).filter((e) => e.type === type);
}

// ── Basic execution ────────────────────────────────────────────────────────

describe("runWorkflow - basic execution", () => {
  it("runs a single-step workflow", async () => {
    const wf = flow("simple", {
      input: z.object({ msg: z.string() }),
      steps: [step("echo", "echo", { message: "{{ input.msg }}" })],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(wf, { msg: "hello" }, makeRegistry(), {
      runId: "test-1",
      store,
    });

    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { message: "hello" });
    assert.equal(result.runId, "test-1");
  });

  it("runs sequential steps with scope propagation", async () => {
    const wf = flow("sequential", {
      input: z.object({ x: z.number() }),
      steps: [
        step("first", "value", { result: "{{ input.x }}" }),
        step("second", "echo", { prev: "{{ first }}", doubled: "{{ input.x + input.x }}" }),
      ],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(wf, { x: 21 }, makeRegistry(), {
      runId: "test-2",
      store,
    });

    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { prev: 21, doubled: 42 });
  });

  it("output is the last step's output", async () => {
    const wf = flow("multi", {
      input: z.object({}),
      steps: [
        step("a", "value", { result: "first" }),
        step("b", "value", { result: "second" }),
        step("c", "value", { result: "third" }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.output, "third");
  });

  it("empty workflow returns undefined", async () => {
    const wf = flow("empty", { input: z.object({}), steps: [] });
    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, undefined);
  });
});

// ── Input validation ───────────────────────────────────────────────────────

describe("runWorkflow - input validation", () => {
  it("validates input against flow schema", async () => {
    const wf = flow("typed", {
      input: z.object({ name: z.string(), age: z.number() }),
      steps: [step("echo", "echo", { name: "{{ input.name }}" })],
    });

    const result = await runWorkflow(wf, { name: "Alice", age: 30 }, makeRegistry());
    assert.equal(result.status, "success");
  });

  it("returns error on invalid input", async () => {
    const wf = flow("typed", {
      input: z.object({ name: z.string(), age: z.number() }),
      steps: [step("echo", "echo", {})],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(wf, { name: "Alice" }, makeRegistry(), {
      runId: "bad-input",
      store,
    });

    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("Input validation failed"));
  });

  it("emits run.error event on input validation failure", async () => {
    const wf = flow("typed", {
      input: z.object({ required: z.string() }),
      steps: [step("echo", "echo", {})],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "val-err",
      store,
    });

    const types = eventTypes(store, "typed", "val-err");
    assert.ok(types.includes("run.error"));
    assert.ok(!types.includes("run.start"));
  });
});

// ── Event logging ──────────────────────────────────────────────────────────

describe("runWorkflow - event logging", () => {
  it("emits run.start, step.start, step.end, run.end for success", async () => {
    const wf = flow("logged", {
      input: z.object({}),
      steps: [step("a", "value", { result: 1 })],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "log-1",
      store,
    });

    const types = eventTypes(store, "logged", "log-1");
    assert.deepEqual(types, [
      "run.start",
      "step.start",
      "step.end",
      "run.end",
    ]);
  });

  it("emits step events with correct paths", async () => {
    const wf = flow("paths", {
      input: z.object({}),
      steps: [
        step("a", "value", { result: 1 }),
        step("b", "value", { result: 2 }),
      ],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "log-2",
      store,
    });

    const starts = eventsOfType(store, "paths", "log-2", "step.start");
    assert.equal(starts[0]!.path, "paths/a");
    assert.equal(starts[1]!.path, "paths/b");
  });

  it("emits step.error and run.error on failure", async () => {
    const wf = flow("failing", {
      input: z.object({}),
      steps: [step("boom", "fail", { message: "kaboom" })],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(wf, {}, makeRegistry(), {
      runId: "log-3",
      store,
    });

    assert.equal(result.status, "error");
    const types = eventTypes(store, "failing", "log-3");
    assert.ok(types.includes("step.start"));
    assert.ok(types.includes("step.error"));
    assert.ok(types.includes("run.error"));
  });

  it("writes run summary on success", async () => {
    const wf = flow("summary", {
      input: z.object({}),
      steps: [step("a", "value", { result: "done" })],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "sum-1",
      store,
    });

    const summary = store.getSummary("summary", "sum-1");
    assert.ok(summary);
    assert.equal(summary!.status, "success");
    assert.equal(summary!.workflow, "summary");
    assert.equal(summary!.output, "done");
    assert.ok(summary!.durationMs >= 0);
  });

  it("writes run summary on error", async () => {
    const wf = flow("err-summary", {
      input: z.object({}),
      steps: [step("boom", "fail", {})],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "sum-2",
      store,
    });

    const summary = store.getSummary("err-summary", "sum-2");
    assert.ok(summary);
    assert.equal(summary!.status, "error");
    assert.ok(summary!.error?.message);
  });
});

// ── Retry ──────────────────────────────────────────────────────────────────

describe("runWorkflow - retry", () => {
  it("retries on failure and succeeds", async () => {
    const flakey = createFlakeyStep(2); // fails first 2 attempts
    const wf = flow("retry-ok", {
      input: z.object({}),
      steps: [
        step("s", "flakey", {}, { retry: { max: 3, delayMs: 0 } }),
      ],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ flakey }) as StepRegistry,
      { runId: "retry-1", store },
    );

    assert.equal(result.status, "success");
    assert.equal((result.output as any).attempts, 3);

    // Should have retry events
    const retries = eventsOfType(store, "retry-ok", "retry-1", "step.retry");
    assert.equal(retries.length, 2);
  });

  it("fails after exhausting retries", async () => {
    const flakey = createFlakeyStep(5); // fails 5 times, we only retry 2
    const wf = flow("retry-fail", {
      input: z.object({}),
      steps: [
        step("s", "flakey", {}, { retry: { max: 2, delayMs: 0 } }),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ flakey }) as StepRegistry,
    );
    assert.equal(result.status, "error");
  });
});

// ── onError fallback ───────────────────────────────────────────────────────

describe("runWorkflow - onError", () => {
  it("executes fallback step on error", async () => {
    const wf = flow("fallback", {
      input: z.object({}),
      steps: [
        step(
          "risky",
          "fail",
          { message: "oops" },
          { onError: step("recover", "value", { result: "recovered" }) },
        ),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, "recovered");
  });

  it("fallback has access to $error", async () => {
    const wf = flow("fallback-error", {
      input: z.object({}),
      steps: [
        step(
          "risky",
          "fail",
          { message: "something broke" },
          {
            onError: step("recover", "echo", {
              errorMsg: "{{ $error.message }}",
            }),
          },
        ),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal((result.output as any).errorMsg, "something broke");
  });

  it("retry + onError: fallback runs after retries exhausted", async () => {
    const flakey = createFlakeyStep(10); // never succeeds
    const wf = flow("retry-then-fallback", {
      input: z.object({}),
      steps: [
        step(
          "s",
          "flakey",
          {},
          {
            retry: { max: 2, delayMs: 0 },
            onError: step("recover", "value", { result: "fallback" }),
          },
        ),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ flakey }) as StepRegistry,
    );
    assert.equal(result.status, "success");
    assert.equal(result.output, "fallback");
  });
});

// ── Unknown step type ──────────────────────────────────────────────────────

describe("runWorkflow - unknown step type", () => {
  it("fails with unknown step type", async () => {
    const wf = flow("unknown", {
      input: z.object({}),
      steps: [step("s", "nonexistent", {})],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("nonexistent"));
  });
});

// ── Template resolution in runner ──────────────────────────────────────────

describe("runWorkflow - template resolution", () => {
  it("resolves input references", async () => {
    const wf = flow("templates", {
      input: z.object({ url: z.string() }),
      steps: [step("fetch", "echo", { target: "{{ input.url }}" })],
    });

    const result = await runWorkflow(wf, { url: "/api" }, makeRegistry());
    assert.deepEqual(result.output, { target: "/api" });
  });

  it("resolves references to previous step outputs", async () => {
    const wf = flow("chain", {
      input: z.object({}),
      steps: [
        step("first", "value", { result: { data: 42 } }),
        step("second", "echo", { value: "{{ first.data }}" }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.deepEqual(result.output, { value: 42 });
  });

  it("resolves complex expression in template", async () => {
    const wf = flow("expr", {
      input: z.object({ a: z.number(), b: z.number() }),
      steps: [
        step("sum", "echo", { total: "{{ input.a + input.b }}" }),
      ],
    });

    const result = await runWorkflow(wf, { a: 3, b: 7 }, makeRegistry());
    assert.deepEqual(result.output, { total: 10 });
  });

  it("resolves multi-segment templates", async () => {
    const wf = flow("multi-seg", {
      input: z.object({ service: z.string() }),
      steps: [
        step("msg", "echo", {
          text: "deployed {{ input.service }} successfully",
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      { service: "api" },
      makeRegistry(),
    );
    assert.deepEqual(result.output, {
      text: "deployed api successfully",
    });
  });
});
