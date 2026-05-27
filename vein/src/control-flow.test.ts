import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step, defineStep, type StepRegistry, type RunEvent } from "./core.js";
import { runWorkflow } from "./runner.js";
import { MemoryRunStore } from "./store.js";

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

function makeRegistry(extra: Record<string, any> = {}): StepRegistry {
  return { echo: echoStep, value: valueStep, fail: failStep, ...extra } as StepRegistry;
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

// ── if step ────────────────────────────────────────────────────────────────

describe("if step", () => {
  it("executes then branch when condition is true", async () => {
    const wf = flow("if-then", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.flag }}",
          then: step("yes", "value", { result: "then-branch" }),
          else: step("no", "value", { result: "else-branch" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, "then-branch");
  });

  it("executes else branch when condition is false", async () => {
    const wf = flow("if-else", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.flag }}",
          then: step("yes", "value", { result: "then-branch" }),
          else: step("no", "value", { result: "else-branch" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, "else-branch");
  });

  it("returns undefined when condition is false and no else", async () => {
    const wf = flow("if-no-else", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.flag }}",
          then: step("yes", "value", { result: "yes" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, undefined);
  });

  it("evaluates complex condition expression", async () => {
    const wf = flow("if-complex", {
      input: z.object({ x: z.number() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.x > 10 }}",
          then: step("big", "value", { result: "big" }),
          else: step("small", "value", { result: "small" }),
        }),
      ],
    });

    const r1 = await runWorkflow(wf, { x: 15 }, makeRegistry());
    assert.equal(r1.output, "big");

    const r2 = await runWorkflow(wf, { x: 5 }, makeRegistry());
    assert.equal(r2.output, "small");
  });

  it("condition based on previous step output", async () => {
    const wf = flow("if-prev", {
      input: z.object({}),
      steps: [
        step("data", "value", { result: { status: "ready" } }),
        step("check", "if", {
          cond: "{{ data.status === 'ready' }}",
          then: step("go", "value", { result: "proceeding" }),
          else: step("wait", "value", { result: "waiting" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.output, "proceeding");
  });

  it("if result is available to subsequent steps", async () => {
    const wf = flow("if-scope", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.flag }}",
          then: step("yes", "value", { result: "chosen" }),
          else: step("no", "value", { result: "other" }),
        }),
        step("use", "echo", { picked: "{{ check }}" }),
      ],
    });

    const result = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.deepEqual(result.output, { picked: "chosen" });
  });
});

// ── loop step ──────────────────────────────────────────────────────────────

describe("loop step", () => {
  it("loops until condition is met", async () => {
    // Counter step that increments each call
    let count = 0;
    const counter = defineStep({
      type: "counter",
      input: z.any(),
      output: z.any(),
      async run() {
        count++;
        return { value: count };
      },
    });

    const wf = flow("loop-basic", {
      input: z.object({}),
      steps: [
        step("loop", "loop", {
          until: "{{ $current.value >= 3 }}",
          maxIterations: 10,
          body: step("inc", "counter", {}),
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ counter }) as StepRegistry,
    );
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { value: 3 });
  });

  it("fails when maxIterations exceeded", async () => {
    // Always returns false for until
    const noop = defineStep({
      type: "noop",
      input: z.any(),
      output: z.any(),
      async run() {
        return { done: false };
      },
    });

    const wf = flow("loop-max", {
      input: z.object({}),
      steps: [
        step("loop", "loop", {
          until: "{{ $current.done === true }}",
          maxIterations: 3,
          body: step("tick", "noop", {}),
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ noop }) as StepRegistry,
    );
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("maxIterations"));
  });

  it("$current is undefined on first iteration", async () => {
    const outputs: unknown[] = [];
    const capture = defineStep({
      type: "capture",
      input: z.any(),
      output: z.any(),
      async run(cfg) {
        outputs.push(cfg.current);
        return { value: outputs.length };
      },
    });

    const wf = flow("loop-current", {
      input: z.object({}),
      steps: [
        step("loop", "loop", {
          until: "{{ $current.value >= 2 }}",
          maxIterations: 5,
          body: step("cap", "capture", { current: "{{ $current }}" }),
        }),
      ],
    });

    await runWorkflow(
      wf,
      {},
      makeRegistry({ capture }) as StepRegistry,
    );

    // First iteration: $current is undefined
    assert.equal(outputs[0], undefined);
    // Second iteration: $current is the first iteration's output
    assert.deepEqual(outputs[1], { value: 1 });
  });

  it("emits events with iteration path (#0, #1, ...)", async () => {
    let count = 0;
    const counter = defineStep({
      type: "counter",
      input: z.any(),
      output: z.any(),
      async run() {
        count++;
        return { value: count };
      },
    });

    const wf = flow("loop-events", {
      input: z.object({}),
      steps: [
        step("loop", "loop", {
          until: "{{ $current.value >= 2 }}",
          maxIterations: 5,
          body: step("inc", "counter", {}),
        }),
      ],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry({ counter }) as StepRegistry, {
      runId: "loop-ev",
      store,
    });

    const starts = eventsOfType(store, "loop-events", "loop-ev", "step.start");
    const loopStarts = starts.filter((e) => e.path.includes("#"));
    assert.ok(loopStarts.some((e) => e.path.includes("#0")));
    assert.ok(loopStarts.some((e) => e.path.includes("#1")));
  });

  it("loop output is available to subsequent steps", async () => {
    let count = 0;
    const counter = defineStep({
      type: "counter",
      input: z.any(),
      output: z.any(),
      async run() {
        count++;
        return { finalCount: count };
      },
    });

    const wf = flow("loop-scope", {
      input: z.object({}),
      steps: [
        step("loop", "loop", {
          until: "{{ $current.finalCount >= 2 }}",
          maxIterations: 5,
          body: step("inc", "counter", {}),
        }),
        step("use", "echo", { result: "{{ loop.finalCount }}" }),
      ],
    });

    const result = await runWorkflow(
      wf,
      {},
      makeRegistry({ counter }) as StepRegistry,
    );
    assert.deepEqual(result.output, { result: 2 });
  });
});

// ── DAG depends (parallel via shared dependency) ───────────────────────────

describe("DAG depends", () => {
  it("steps with same depends run concurrently", async () => {
    const wf = flow("dag-basic", {
      input: z.object({}),
      steps: [
        step("setup", "value", { result: "ready" }),
        step("left", "value", { result: "left-result" }, { depends: "setup" }),
        step("right", "value", { result: "right-result" }, { depends: "setup" }),
        step("merge", "echo", {
          l: "{{ left }}",
          r: "{{ right }}",
        }, { depends: ["left", "right"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { l: "left-result", r: "right-result" });
  });

  it("parallel branches can access input", async () => {
    const wf = flow("dag-input", {
      input: z.object({ id: z.string() }),
      steps: [
        step("a", "echo", { received: "{{ input.id }}" }),
        step("b", "echo", { also: "{{ input.id }}" }, { depends: [] }),
      ],
    });

    const result = await runWorkflow(wf, { id: "test-id" }, makeRegistry());
    // b is the last step in array, so its output is returned
    assert.deepEqual(result.output, { also: "test-id" });
  });

  it("fan-out outputs are available to downstream merge step", async () => {
    const wf = flow("dag-merge", {
      input: z.object({}),
      steps: [
        step("left", "value", { result: 10 }),
        step("right", "value", { result: 20 }, { depends: [] }),
        step("merge", "echo", {
          total: "{{ left + right }}",
        }, { depends: ["left", "right"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.deepEqual(result.output, { total: 30 });
  });

  it("failure in one parallel branch fails the workflow", async () => {
    const wf = flow("dag-fail", {
      input: z.object({}),
      steps: [
        step("ok", "value", { result: "fine" }),
        step("bad", "fail", { message: "branch failed" }, { depends: [] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("branch failed"));
  });

  it("runs three independent branches", async () => {
    const wf = flow("dag-three", {
      input: z.object({}),
      steps: [
        step("a", "value", { result: "A" }),
        step("b", "value", { result: "B" }, { depends: [] }),
        step("c", "value", { result: "C" }, { depends: [] }),
        step("collect", "echo", {
          a: "{{ a }}",
          b: "{{ b }}",
          c: "{{ c }}",
        }, { depends: ["a", "b", "c"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    const output = result.output as any;
    assert.equal(output.a, "A");
    assert.equal(output.b, "B");
    assert.equal(output.c, "C");
  });

  it("implicit sequential when no depends specified", async () => {
    const wf = flow("dag-implicit", {
      input: z.object({}),
      steps: [
        step("a", "value", { result: 1 }),
        step("b", "value", { result: "{{ a + 1 }}" }),
        step("c", "value", { result: "{{ b + 1 }}" }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, 3);
  });
});

// ── subflow step ───────────────────────────────────────────────────────────

describe("subflow step", () => {
  it("runs a child workflow", async () => {
    const child = flow("child", {
      input: z.object({ msg: z.string() }),
      steps: [step("echo", "echo", { received: "{{ input.msg }}" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { flow: child, input: { msg: "hello" } }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { received: "hello" });
  });

  it("child flow validates its own input", async () => {
    const child = flow("child", {
      input: z.object({ required: z.string() }),
      steps: [step("echo", "echo", {})],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { flow: child, input: {} }),
      ],
    });

    // The child flow's input validation happens inside executeFlow
    // which parses the input — this should propagate as an error
    const result = await runWorkflow(parent, {}, makeRegistry());
    assert.equal(result.status, "error");
  });

  it("child cannot access parent scope", async () => {
    // Child references "parentData" which doesn't exist in its scope
    const child = flow("child", {
      input: z.object({}),
      steps: [step("echo", "echo", { val: "{{ parentData }}" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("parentData", "value", { result: "secret" }),
        step("sub", "subflow", { flow: child, input: {} }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry());
    // Should fail because child can't see "parentData" in its scope
    assert.equal(result.status, "error");
  });

  it("subflow output is available to subsequent steps", async () => {
    const child = flow("child", {
      input: z.object({ x: z.number() }),
      steps: [step("calc", "value", { result: "{{ input.x * 2 }}" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { flow: child, input: { x: 21 } }),
        step("use", "echo", { doubled: "{{ sub }}" }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry());
    assert.deepEqual(result.output, { doubled: 42 });
  });

  it("subflow input can use parent templates", async () => {
    const child = flow("child", {
      input: z.object({ name: z.string() }),
      steps: [step("greet", "echo", { greeting: "hello {{ input.name }}" })],
    });

    const parent = flow("parent", {
      input: z.object({ user: z.string() }),
      steps: [
        step("sub", "subflow", {
          flow: child,
          input: { name: "{{ input.user }}" },
        }),
      ],
    });

    const result = await runWorkflow(parent, { user: "Alice" }, makeRegistry());
    assert.deepEqual(result.output, { greeting: "hello Alice" });
  });
});

// ── Nested control flow ────────────────────────────────────────────────────

describe("nested control flow", () => {
  it("if with parallel independent branches", async () => {
    const wf = flow("nested-if-dag", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", {
          cond: "{{ input.flag }}",
          then: step("yes", "value", { result: "yes" }),
          else: step("no", "value", { result: "no" }),
        }),
        step("use", "echo", { result: "{{ check }}" }),
      ],
    });

    const r1 = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.deepEqual(r1.output, { result: "yes" });

    const r2 = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.deepEqual(r2.output, { result: "no" });
  });

  it("subflow alongside parallel step", async () => {
    const child = flow("child", {
      input: z.object({ val: z.number() }),
      steps: [step("calc", "value", { result: "{{ input.val * 3 }}" })],
    });

    const wf = flow("sub-dag", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", {
          flow: child,
          input: { val: 7 },
        }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.output, 21);
  });

  it("DAG with subflows in parallel branches", async () => {
    const child = flow("child", {
      input: z.object({ val: z.number() }),
      steps: [step("calc", "value", { result: "{{ input.val * 3 }}" })],
    });

    const wf = flow("dag-sub-par", {
      input: z.object({}),
      steps: [
        step("left", "subflow", {
          flow: child,
          input: { val: 7 },
        }),
        step("right", "subflow", {
          flow: child,
          input: { val: 3 },
        }, { depends: [] }),
        step("merge", "echo", {
          l: "{{ left }}",
          r: "{{ right }}",
        }, { depends: ["left", "right"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.deepEqual(result.output, { l: 21, r: 9 });
  });
});
