import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step, defineStep, type StepRegistry, type RunEvent, type Flow } from "./core.js";
import { runWorkflow, type SubflowResolver } from "./runner.js";
import { MemoryRunStore } from "./store.js";
import ifStep from "./steps/core/if.js";
import loopStep from "./steps/core/loop.js";
import foreachStep from "./steps/core/foreach.js";
import subflowStep from "./steps/core/subflow.js";

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
  return {
    echo: echoStep,
    value: valueStep,
    fail: failStep,
    if: ifStep,
    loop: loopStep,
    foreach: foreachStep,
    subflow: subflowStep,
    ...extra,
  } as StepRegistry;
}

/** Build a simple in-memory subflow resolver from a map of workflows. */
function makeResolver(flows: Record<string, Flow>): SubflowResolver {
  return {
    async getWorkflow(name: string) {
      const f = flows[name];
      if (!f) throw new Error(`Workflow "${name}" not found`);
      return f;
    },
    async getWorkflowVersion(name: string, _version: string) {
      const f = flows[name];
      if (!f) throw new Error(`Workflow "${name}" not found`);
      return f;
    },
  };
}

function eventsOfType(
  store: MemoryRunStore,
  workflow: string,
  runId: string,
  type: string,
): RunEvent[] {
  return store.getEvents(workflow, runId).filter((e) => e.type === type);
}

// ── if step (flat gate model) ──────────────────────────────────────────────

describe("if step (gate)", () => {
  it("executes then-branch step when condition is true", async () => {
    const wf = flow("if-then", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("yes", "value", { result: "then-branch" }, { depends: "check", when: true }),
        step("no", "value", { result: "else-branch" }, { depends: "check", when: false }),
      ],
    });

    const result = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.equal(result.status, "success");
    // Last step in array is "no" which was skipped → workflow output is undefined
    assert.equal(result.output, undefined);
  });

  it("executes else-branch step when condition is false", async () => {
    const wf = flow("if-else", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("yes", "value", { result: "then-branch" }, { depends: "check", when: true }),
        step("no", "value", { result: "else-branch" }, { depends: "check", when: false }),
      ],
    });

    const result = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.equal(result.status, "success");
    assert.equal(result.output, "else-branch");
  });

  it("evaluates complex condition expression", async () => {
    const wf = flow("if-complex", {
      input: z.object({ x: z.number() }),
      steps: [
        step("check", "if", { cond: "{{ input.x > 10 }}" }),
        step("big", "value", { result: "big" }, { depends: "check", when: true }),
        step("small", "value", { result: "small" }, { depends: "check", when: false }),
        step("pick", "echo", {
          chosen: "{{ big }}",
          alt: "{{ small }}",
        }, { depends: ["big", "small"] }),
      ],
    });

    const r1 = await runWorkflow(wf, { x: 15 }, makeRegistry());
    assert.deepEqual(r1.output, { chosen: "big", alt: undefined });

    const r2 = await runWorkflow(wf, { x: 5 }, makeRegistry());
    assert.deepEqual(r2.output, { chosen: undefined, alt: "small" });
  });

  it("condition based on previous step output", async () => {
    const wf = flow("if-prev", {
      input: z.object({}),
      steps: [
        step("data", "value", { result: { status: "ready" } }),
        step("check", "if", { cond: "{{ data.status === 'ready' }}" }),
        step("go", "value", { result: "proceeding" }, { depends: "check", when: true }),
        step("wait", "value", { result: "waiting" }, { depends: "check", when: false }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    // "wait" is the last step but was skipped (cond was true) → output undefined
    assert.equal(result.output, undefined);
  });

  it("fan-in: step depending on both branches gets whichever ran", async () => {
    const wf = flow("if-fan-in", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("yes", "value", { result: "chosen-yes" }, { depends: "check", when: true }),
        step("no", "value", { result: "chosen-no" }, { depends: "check", when: false }),
        step("merge", "echo", {
          picked: "{{ yes }}",
          other: "{{ no }}",
        }, { depends: ["yes", "no"] }),
      ],
    });

    const r1 = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.deepEqual(r1.output, { picked: "chosen-yes", other: undefined });

    const r2 = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.deepEqual(r2.output, { picked: undefined, other: "chosen-no" });
  });

  it("branch can contain multiple chained steps (flat)", async () => {
    const wf = flow("if-multi-step", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("y1", "value", { result: "y1-output" }, { depends: "check", when: true }),
        step("y2", "echo", { from: "{{ y1 }}", final: "yes-branch-done" }, { depends: "y1" }),
        step("n1", "value", { result: "n1-output" }, { depends: "check", when: false }),
        step("n2", "echo", { from: "{{ n1 }}", final: "no-branch-done" }, { depends: "n1" }),
      ],
    });

    const r1 = await runWorkflow(wf, { flag: true }, makeRegistry());
    // Both y2 and n2 are leaves; last in array is n2 (skipped) → undefined
    assert.equal(r1.status, "success");
    // But we can verify by adding a fan-in:
    const wf2 = flow("if-multi-fanin", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("y1", "value", { result: "y1" }, { depends: "check", when: true }),
        step("y2", "echo", { v: "{{ y1 }}-y2" }, { depends: "y1" }),
        step("n1", "value", { result: "n1" }, { depends: "check", when: false }),
        step("n2", "echo", { v: "{{ n1 }}-n2" }, { depends: "n1" }),
        step("done", "echo", { y: "{{ y2 }}", n: "{{ n2 }}" }, { depends: ["y2", "n2"] }),
      ],
    });
    const r3 = await runWorkflow(wf2, { flag: true }, makeRegistry());
    assert.deepEqual(r3.output, { y: { v: "y1-y2" }, n: undefined });
    const r4 = await runWorkflow(wf2, { flag: false }, makeRegistry());
    assert.deepEqual(r4.output, { y: undefined, n: { v: "n1-n2" } });
  });

  it("emits step.skipped events for skipped branches", async () => {
    const wf = flow("if-skip-event", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("yes", "value", { result: "y" }, { depends: "check", when: true }),
        step("no", "value", { result: "n" }, { depends: "check", when: false }),
      ],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, { flag: true }, makeRegistry(), { runId: "skip", store });
    const skipped = eventsOfType(store, "if-skip-event", "skip", "step.skipped");
    assert.equal(skipped.length, 1);
    assert.ok(skipped[0]!.path.endsWith("/no"));
  });

  it("transitive skip: step depending only on skipped step is skipped too", async () => {
    const wf = flow("if-transitive", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("y1", "value", { result: "y1" }, { depends: "check", when: true }),
        step("y2", "echo", { v: "{{ y1 }}" }, { depends: "y1" }),
        step("y3", "echo", { v: "{{ y2 }}" }, { depends: "y2" }),
      ],
    });

    const store = new MemoryRunStore();
    const result = await runWorkflow(wf, { flag: false }, makeRegistry(), { runId: "trans", store });
    assert.equal(result.status, "success");
    // All of y1, y2, y3 should be skipped
    const skipped = eventsOfType(store, "if-transitive", "trans", "step.skipped");
    const skippedPaths = skipped.map((e) => e.path);
    assert.ok(skippedPaths.some((p) => p.endsWith("/y1")));
    assert.ok(skippedPaths.some((p) => p.endsWith("/y2")));
    assert.ok(skippedPaths.some((p) => p.endsWith("/y3")));
  });
});

// ── loop step ──────────────────────────────────────────────────────────────

describe("loop step", () => {
  it("loops until condition is met", async () => {
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

    const result = await runWorkflow(wf, {}, makeRegistry({ counter }));
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { value: 3 });
  });

  it("fails when maxIterations exceeded", async () => {
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

    const result = await runWorkflow(wf, {}, makeRegistry({ noop }));
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

    await runWorkflow(wf, {}, makeRegistry({ capture }));

    assert.equal(outputs[0], undefined);
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
    await runWorkflow(wf, {}, makeRegistry({ counter }), {
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

    const result = await runWorkflow(wf, {}, makeRegistry({ counter }));
    assert.deepEqual(result.output, { result: 2 });
  });
});

// ── foreach step ───────────────────────────────────────────────────────────

describe("foreach step", () => {
  it("iterates a list and returns an array of body outputs in order", async () => {
    const wf = flow("foreach-basic", {
      input: z.object({ items: z.array(z.number()) }),
      steps: [
        step("each", "foreach", {
          items: "{{ input.items }}",
          body: step("double", "value", { result: "{{ $current * 2 }}" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, { items: [1, 2, 3, 4] }, makeRegistry());
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, [2, 4, 6, 8]);
  });

  it("exposes $index alongside $current", async () => {
    const wf = flow("foreach-index", {
      input: z.object({ items: z.array(z.string()) }),
      steps: [
        step("each", "foreach", {
          items: "{{ input.items }}",
          body: step("pair", "echo", {
            value: "{{ $current }}",
            index: "{{ $index }}",
          }),
        }),
      ],
    });

    const result = await runWorkflow(
      wf,
      { items: ["a", "b", "c"] },
      makeRegistry(),
    );
    assert.deepEqual(result.output, [
      { value: "a", index: 0 },
      { value: "b", index: 1 },
      { value: "c", index: 2 },
    ]);
  });

  it("returns an empty array when items is empty", async () => {
    const wf = flow("foreach-empty", {
      input: z.object({}),
      steps: [
        step("each", "foreach", {
          items: [],
          body: step("noop", "value", { result: "should not run" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, []);
  });

  it("body is not invoked when items is empty", async () => {
    let calls = 0;
    const counter = defineStep({
      type: "tally",
      input: z.any(),
      output: z.any(),
      async run() {
        calls++;
        return { calls };
      },
    });

    const wf = flow("foreach-no-calls", {
      input: z.object({}),
      steps: [
        step("each", "foreach", {
          items: [],
          body: step("t", "tally", {}),
        }),
      ],
    });

    await runWorkflow(wf, {}, makeRegistry({ tally: counter }));
    assert.equal(calls, 0);
  });

  it("throws when items does not resolve to an array", async () => {
    const wf = flow("foreach-bad-items", {
      input: z.object({ items: z.any() }),
      steps: [
        step("each", "foreach", {
          items: "{{ input.items }}",
          body: step("noop", "value", { result: 1 }),
        }),
      ],
    });

    const result = await runWorkflow(wf, { items: "not-an-array" }, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(
      result.error?.message.includes("to resolve to an array"),
      `expected error about array resolution, got: ${result.error?.message}`,
    );
  });

  it("enforces maxIterations as a safety cap", async () => {
    const wf = flow("foreach-cap", {
      input: z.object({}),
      steps: [
        step("each", "foreach", {
          items: [1, 2, 3, 4, 5],
          maxIterations: 3,
          body: step("v", "value", { result: "{{ $current }}" }),
        }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("maxIterations"));
  });

  it("body output is available to subsequent steps as an array", async () => {
    const wf = flow("foreach-output", {
      input: z.object({}),
      steps: [
        step("nums", "value", { result: [10, 20, 30] }),
        step("each", "foreach", {
          items: "{{ nums }}",
          body: step("plus_one", "value", { result: "{{ $current + 1 }}" }),
        }, { depends: "nums" }),
        step("collect", "echo", { sums: "{{ each }}" }, { depends: "each" }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "success");
    assert.deepEqual(result.output, { sums: [11, 21, 31] });
  });

  it("preserves order even with async body work", async () => {
    // Sequential semantics: items finish in input order regardless of work time.
    const order: number[] = [];
    const slow = defineStep({
      type: "slow",
      input: z.object({ v: z.number(), delay: z.number() }),
      output: z.any(),
      async run(cfg) {
        await new Promise((r) => setTimeout(r, cfg.delay));
        order.push(cfg.v);
        return cfg.v;
      },
    });

    const wf = flow("foreach-order", {
      input: z.object({}),
      steps: [
        step("each", "foreach", {
          items: [1, 2, 3],
          // Item 1 sleeps longest; sequential ordering must still produce [1,2,3].
          body: step(
            "s",
            "slow",
            { v: "{{ $current }}", delay: "{{ $index === 0 ? 15 : 1 }}" },
          ),
        }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry({ slow }));
    assert.deepEqual(result.output, [1, 2, 3]);
    assert.deepEqual(order, [1, 2, 3]);
  });

  it("emits step.start/end events with iteration path (#0, #1, ...)", async () => {
    const wf = flow("foreach-events", {
      input: z.object({}),
      steps: [
        step("each", "foreach", {
          items: ["x", "y"],
          body: step("v", "value", { result: "{{ $current }}" }),
        }),
      ],
    });

    const store = new MemoryRunStore();
    await runWorkflow(wf, {}, makeRegistry(), {
      runId: "fe-ev",
      store,
    });

    const starts = eventsOfType(store, "foreach-events", "fe-ev", "step.start");
    const iter = starts.filter((e) => e.path.includes("#"));
    assert.ok(iter.some((e) => e.path.includes("#0")));
    assert.ok(iter.some((e) => e.path.includes("#1")));
    // Iteration metadata is set on per-item events.
    assert.equal(iter.find((e) => e.path.includes("#0"))?.iteration, 0);
    assert.equal(iter.find((e) => e.path.includes("#1"))?.iteration, 1);
  });

  it("works with a subflow body for per-item composition", async () => {
    const inner = flow("double-inner", {
      input: z.object({ n: z.number() }),
      steps: [step("d", "value", { result: "{{ input.n * 2 }}" })],
    });

    const outer = flow("foreach-subflow", {
      input: z.object({ items: z.array(z.number()) }),
      steps: [
        step("each", "foreach", {
          items: "{{ input.items }}",
          body: step("call", "subflow", {
            workflow: "double-inner",
            input: { n: "{{ $current }}" },
          }),
        }),
      ],
    });

    const resolver = makeResolver({ "double-inner": inner });
    const result = await runWorkflow(outer, { items: [3, 5, 7] }, makeRegistry(), {
      workspace: resolver,
    });

    assert.deepEqual(result.output, [6, 10, 14]);
  });

  it("throws when body is missing", async () => {
    const wf = flow("foreach-no-body", {
      input: z.object({}),
      steps: [
        step("each", "foreach", { items: [1, 2] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("body"));
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

// ── subflow step (workflow reference) ──────────────────────────────────────

describe("subflow step", () => {
  it("runs a child workflow by reference", async () => {
    const child = flow("child", {
      input: z.object({ msg: z.string() }),
      steps: [step("echo", "echo", { received: "{{ input.msg }}" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { workflow: "child", input: { msg: "hello" } }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
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
        step("sub", "subflow", { workflow: "child", input: {} }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
    assert.equal(result.status, "error");
  });

  it("child cannot access parent scope", async () => {
    const child = flow("child", {
      input: z.object({}),
      steps: [step("echo", "echo", { val: "{{ parentData }}" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("parentData", "value", { result: "secret" }),
        step("sub", "subflow", { workflow: "child", input: {} }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
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
        step("sub", "subflow", { workflow: "child", input: { x: 21 } }),
        step("use", "echo", { doubled: "{{ sub }}" }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
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
          workflow: "child",
          input: { name: "{{ input.user }}" },
        }),
      ],
    });

    const result = await runWorkflow(parent, { user: "Alice" }, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
    assert.deepEqual(result.output, { greeting: "hello Alice" });
  });

  it("fails if workspace is not provided", async () => {
    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { workflow: "child", input: {} }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry());
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("workspace"));
  });

  it("fails if referenced workflow doesn't exist", async () => {
    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { workflow: "missing", input: {} }),
      ],
    });

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: makeResolver({}),
    });
    assert.equal(result.status, "error");
    assert.ok(result.error?.message.includes("missing"));
  });

  it("supports versioned workflow reference", async () => {
    const childV1 = flow("child", {
      input: z.object({}),
      steps: [step("result", "value", { result: "v1-output" })],
    });

    const parent = flow("parent", {
      input: z.object({}),
      steps: [
        step("sub", "subflow", { workflow: "child", version: "v1", input: {} }),
      ],
    });

    const resolver: SubflowResolver = {
      async getWorkflow() { throw new Error("should not be called"); },
      async getWorkflowVersion(name: string, version: string) {
        if (name === "child" && version === "v1") return childV1;
        throw new Error(`Not found: ${name}@${version}`);
      },
    };

    const result = await runWorkflow(parent, {}, makeRegistry(), {
      workspace: resolver,
    });
    assert.equal(result.status, "success");
    assert.equal(result.output, "v1-output");
  });
});

// ── Nested control flow ────────────────────────────────────────────────────

describe("nested control flow", () => {
  it("if branches that fan back in", async () => {
    const wf = flow("nested-if-dag", {
      input: z.object({ flag: z.boolean() }),
      steps: [
        step("check", "if", { cond: "{{ input.flag }}" }),
        step("yes", "value", { result: "yes" }, { depends: "check", when: true }),
        step("no", "value", { result: "no" }, { depends: "check", when: false }),
        step("use", "echo", {
          y: "{{ yes }}",
          n: "{{ no }}",
        }, { depends: ["yes", "no"] }),
      ],
    });

    const r1 = await runWorkflow(wf, { flag: true }, makeRegistry());
    assert.deepEqual(r1.output, { y: "yes", n: undefined });

    const r2 = await runWorkflow(wf, { flag: false }, makeRegistry());
    assert.deepEqual(r2.output, { y: undefined, n: "no" });
  });

  it("DAG with subflows in parallel branches", async () => {
    const child = flow("child", {
      input: z.object({ val: z.number() }),
      steps: [step("calc", "value", { result: "{{ input.val * 3 }}" })],
    });

    const wf = flow("dag-sub-par", {
      input: z.object({}),
      steps: [
        step("left", "subflow", { workflow: "child", input: { val: 7 } }),
        step("right", "subflow", { workflow: "child", input: { val: 3 } }, { depends: [] }),
        step("merge", "echo", {
          l: "{{ left }}",
          r: "{{ right }}",
        }, { depends: ["left", "right"] }),
      ],
    });

    const result = await runWorkflow(wf, {}, makeRegistry(), {
      workspace: makeResolver({ child }),
    });
    assert.deepEqual(result.output, { l: 21, r: 9 });
  });
});
