import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm, writeFile, appendFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { flow, step, defineStep, type StepRegistry, type RunEvent } from "./core.js";
import { runWorkflow } from "./runner.js";
import { MemoryRunStore, FileRunStore } from "./store.js";
import { createVein } from "./createVein.js";
import { WorkspaceManager } from "./workspace.js";
import { RunController, CancelledError, isCancelledError } from "./run-control.js";
import { buildJournal, invalidateFrom, readRunStart, transitiveDependents } from "./journal.js";

// ── Test helpers ───────────────────────────────────────────────────────────

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitFor(cond: () => boolean, ms = 2000): Promise<void> {
  const deadline = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > deadline) throw new Error("waitFor timed out");
    await sleep(5);
  }
}

function deferred<T = void>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

/** A step that parks until released from the test, recording starts. */
function createGateStep() {
  const gates = new Map<string, { promise: Promise<void>; resolve: () => void }>();
  const started: string[] = [];
  const entry = (name: string) => {
    let e = gates.get(name);
    if (!e) {
      const d = deferred();
      e = { promise: d.promise, resolve: () => d.resolve() };
      gates.set(name, e);
    }
    return e;
  };
  const stepDef = defineStep({
    type: "gate",
    input: z.object({ name: z.string() }),
    output: z.any(),
    async run(cfg) {
      started.push(cfg.name);
      await entry(cfg.name).promise;
      return cfg.name;
    },
  });
  return {
    stepDef,
    started,
    release: (name: string) => entry(name).resolve(),
    waitForStart: (name: string) => waitFor(() => started.includes(name)),
  };
}

const valueStep = defineStep({
  type: "value",
  input: z.object({ result: z.any() }),
  output: z.any(),
  async run(cfg) {
    return cfg.result;
  },
});

function createCounterStep() {
  let count = 0;
  const stepDef = defineStep({
    type: "counter",
    input: z.any(),
    output: z.number(),
    async run() {
      count++;
      return count;
    },
  });
  return { stepDef, calls: () => count };
}

/** Fails the first `failCount` invocations, then succeeds. */
function createFlakeyStep(failCount: number) {
  let attempts = 0;
  const stepDef = defineStep({
    type: "flakey",
    input: z.any(),
    output: z.any(),
    async run() {
      attempts++;
      if (attempts <= failCount) throw new Error(`Attempt ${attempts} failed`);
      return { attempts };
    },
  });
  return { stepDef, attempts: () => attempts };
}

function reg(extra: Record<string, unknown>): StepRegistry {
  return { value: valueStep, ...extra } as StepRegistry;
}

function types(store: MemoryRunStore, wf: string, runId: string): string[] {
  return store.getEvents(wf, runId).map((e) => e.type);
}

// ── RunController unit tests ───────────────────────────────────────────────

describe("RunController", () => {
  it("checkpoint resolves immediately while running", async () => {
    const c = new RunController("r1", "wf");
    await c.checkpoint(); // must not hang
    assert.equal(c.state, "running");
  });

  it("cancel makes checkpoint throw CancelledError, idempotently", async () => {
    const c = new RunController("r1", "wf");
    c.cancel();
    c.cancel();
    await assert.rejects(() => c.checkpoint(), (e: unknown) => isCancelledError(e));
    assert.equal(c.state, "cancelling");
  });

  it("pause parks checkpoint; resume releases it", async () => {
    const c = new RunController("r1", "wf");
    c.pause();
    let passed = false;
    const p = c.checkpoint().then(() => (passed = true));
    await sleep(20);
    assert.equal(passed, false);
    c.resume();
    await p;
    assert.equal(passed, true);
  });

  it("cancel releases a PARKED checkpoint with CancelledError", async () => {
    const c = new RunController("r1", "wf");
    c.pause();
    const p = c.checkpoint();
    await sleep(10);
    c.cancel();
    await assert.rejects(() => p, (e: unknown) => isCancelledError(e));
  });

  it("effective state inherits the strictest ancestor (subtree control)", async () => {
    const parent = new RunController("p", "wf");
    const child = new RunController("c", "gen", parent);
    const grandchild = new RunController("g", "cand", child);

    parent.pause();
    assert.equal(grandchild.state, "paused"); // nothing busy → quiesced
    let passed = false;
    const p = grandchild.checkpoint().then(() => (passed = true));
    await sleep(10);
    assert.equal(passed, false);

    parent.cancel(); // strictest wins; also wakes the parked waiter
    await assert.rejects(() => p, (e: unknown) => isCancelledError(e));
    assert.equal(child.state, "cancelling");
  });

  it("pausing a child does not pause the parent or a sibling", async () => {
    const parent = new RunController("p", "wf");
    const a = new RunController("a", "genA", parent);
    const b = new RunController("b", "genB", parent);
    a.pause();
    assert.equal(a.state, "paused");
    assert.equal(parent.state, "running");
    await b.checkpoint(); // sibling unaffected
  });

  it("quiesced reflects busy units, including released parked units (forUnit)", async () => {
    const c = new RunController("r1", "wf");
    assert.equal(c.quiesced(), true);
    c.beginUnit();
    c.pause();
    assert.equal(c.quiesced(), false); // a unit is mid-flight
    assert.equal(c.state, "pausing"); // not yet parked

    // A unit-scoped checkpoint releases the unit while parked → quiesced.
    const unit = c.forUnit();
    const p = unit.checkpoint();
    await sleep(10);
    assert.equal(c.quiesced(), true);
    assert.equal(c.state, "paused");

    c.resume();
    await p;
    assert.equal(c.quiesced(), false); // unit re-acquired
    c.endUnit();
    assert.equal(c.quiesced(), true);
  });

  it("quiesced requires every descendant to be parked", () => {
    const parent = new RunController("p", "wf");
    const child = new RunController("c", "gen", parent);
    child.beginUnit();
    parent.pause();
    assert.equal(parent.quiesced(), false);
    child.endUnit();
    assert.equal(parent.quiesced(), true);
  });

  it("detach unlinks a completed child from the parent's quiescence", () => {
    const parent = new RunController("p", "wf");
    const child = new RunController("c", "gen", parent);
    child.beginUnit();
    parent.pause();
    assert.equal(parent.quiesced(), false);
    child.detach();
    assert.equal(parent.quiesced(), true);
  });

  it("resume does not clear cancelling", async () => {
    const c = new RunController("r1", "wf");
    c.cancel();
    c.resume();
    await assert.rejects(() => c.checkpoint(), (e: unknown) => isCancelledError(e));
  });
});

// ── Rung 1: cancel ─────────────────────────────────────────────────────────

describe("cancel", () => {
  it("cancels a run at the next DAG boundary; in-flight unit completes and is journaled", async () => {
    const gate = createGateStep();
    const store = new MemoryRunStore();
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("a", "gate", { name: "a" }),
        step("b", "value", { result: "never" }),
      ],
    });

    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "r1",
      store,
      controller,
    });

    await gate.waitForStart("a");
    controller.cancel();
    gate.release("a"); // the in-flight unit completes...

    const result = await run;
    assert.equal(result.status, "cancelled");

    const evts = types(store, "wf", "r1");
    assert.ok(evts.includes("step.end")); // ...and its output was journaled
    assert.ok(evts.includes("run.cancelled"));
    assert.ok(!evts.includes("run.error"));
    // b never started
    const bEvents = store.getEvents("wf", "r1").filter((e) => e.path === "wf/b");
    assert.equal(bEvents.length, 0);

    const summary = store.getSummary("wf", "r1");
    assert.equal(summary?.status, "cancelled");
    assert.equal(summary?.error, undefined);
  });

  it("fires the onRunEnd teardown hook on cancellation", async () => {
    const gate = createGateStep();
    const controller = new RunController("r1", "wf");
    const disposed: string[] = [];
    const wf = flow("wf", {
      input: z.any(),
      steps: [step("a", "gate", { name: "a" })],
    });
    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "r1",
      controller,
      services: { onRunEnd: async (id: string) => void disposed.push(id) },
    });
    await gate.waitForStart("a");
    controller.cancel();
    gate.release("a");
    await run;
    assert.deepEqual(disposed, ["r1"]);
  });

  it("stops a foreach between iterations", async () => {
    const gate = createGateStep();
    const store = new MemoryRunStore();
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("each", "foreach", {
          items: ["x", "y", "z"],
          body: step("body", "gate", { name: "{{ $current }}" }),
        }),
      ],
    });

    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "r1",
      store,
      controller,
    });

    await gate.waitForStart("x");
    controller.cancel();
    gate.release("x");

    const result = await run;
    assert.equal(result.status, "cancelled");
    assert.deepEqual(gate.started, ["x"]); // y, z never started
    // iteration 0 completed and journaled
    const iter0 = store.getEvents("wf", "r1").find(
      (e) => e.type === "step.end" && e.path === "wf/each#0",
    );
    assert.ok(iter0);
  });

  it("stops retrying at the retry boundary and skips onError (cancel is not the error path)", async () => {
    const flakey = createFlakeyStep(99);
    const fallbackRan = { value: false };
    const fallbackStep = defineStep({
      type: "fallback",
      input: z.any(),
      output: z.any(),
      async run() {
        fallbackRan.value = true;
        return "fallback";
      },
    });
    const store = new MemoryRunStore();
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step(
          "shaky",
          "flakey",
          {},
          {
            retry: { max: 5, delayMs: 30 },
            onError: step("rescue", "fallback", {}),
          },
        ),
      ],
    });

    const run = runWorkflow(wf, {}, reg({ flakey: flakey.stepDef, fallback: fallbackStep }), {
      runId: "r1",
      store,
      controller,
    });

    await waitFor(() => flakey.attempts() >= 1);
    controller.cancel();

    const result = await run;
    assert.equal(result.status, "cancelled");
    assert.equal(flakey.attempts(), 1); // no retry after cancel
    assert.equal(fallbackRan.value, false); // onError never diverted
  });

  it("cancelling a parent controller cancels a nested run attached to it", async () => {
    const gate = createGateStep();
    const parent = new RunController("parent", "outer");
    const child = new RunController("child", "inner", parent);
    const wf = flow("inner", {
      input: z.any(),
      steps: [
        step("a", "gate", { name: "a" }),
        step("b", "value", { result: "never" }),
      ],
    });
    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "child",
      controller: child,
    });
    await gate.waitForStart("a");
    parent.cancel();
    gate.release("a");
    const result = await run;
    assert.equal(result.status, "cancelled");
  });
});

// ── Rung 2: pause / resume (in-memory) ─────────────────────────────────────

describe("pause/resume", () => {
  it("parks between DAG steps, quiesces, and resumes to completion", async () => {
    const gate = createGateStep();
    const store = new MemoryRunStore();
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("a", "gate", { name: "a" }),
        step("b", "value", { result: "done" }),
      ],
    });

    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "r1",
      store,
      controller,
    });

    await gate.waitForStart("a");
    controller.pause();
    assert.equal(controller.quiesced(), false); // a is mid-unit
    gate.release("a");

    await waitFor(() => controller.quiesced());
    assert.equal(controller.state, "paused");
    // a completed, b has not started
    const evts = store.getEvents("wf", "r1");
    assert.ok(evts.some((e) => e.type === "step.end" && e.path === "wf/a"));
    assert.ok(!evts.some((e) => e.path === "wf/b"));

    controller.resume();
    const result = await run;
    assert.equal(result.status, "success");
    assert.equal(result.output, "done");
  });

  it("a step's ctx.control checkpoint parks mid-step and counts as quiesced", async () => {
    const iterations: number[] = [];
    const loopy = defineStep({
      type: "loopy",
      input: z.any(),
      output: z.any(),
      async run(_cfg, ctx) {
        for (let i = 0; i < 4; i++) {
          await ctx.control?.checkpoint();
          iterations.push(i);
        }
        return iterations.length;
      },
    });
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [step("l", "loopy", {})],
    });

    const run = runWorkflow(wf, {}, reg({ loopy }), { runId: "r1", controller });
    await waitFor(() => iterations.length >= 1);
    controller.pause();
    await waitFor(() => controller.quiesced()); // parked INSIDE the step
    const parkedAt = iterations.length;
    await sleep(30);
    assert.equal(iterations.length, parkedAt); // truly parked

    controller.resume();
    const result = await run;
    assert.equal(result.status, "success");
    assert.equal(iterations.length, 4);
  });

  it("pause between loop iterations", async () => {
    const gate = createGateStep();
    const store = new MemoryRunStore();
    const controller = new RunController("r1", "wf");
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("l", "loop", {
          maxIterations: 5,
          until: "{{ $current === 'g1' }}",
          body: step("body", "gate", { name: "g{{ 1 + 0 }}" }),
        }),
      ],
    });
    // Only one iteration needed: until matches after the first.
    const run = runWorkflow(wf, {}, reg({ gate: gate.stepDef }), {
      runId: "r1",
      store,
      controller,
    });
    await gate.waitForStart("g1");
    controller.pause();
    gate.release("g1");
    await waitFor(() => controller.quiesced());
    controller.resume();
    const result = await run;
    assert.equal(result.status, "success");
  });
});

// ── Rung 3: durable resume (journal replay) ────────────────────────────────

describe("journal", () => {
  it("buildJournal maps step.end paths to outputs, last write wins", () => {
    const events = [
      { type: "step.end", path: "wf/a", output: 1 },
      { type: "step.end", path: "wf/b", output: "old" },
      { type: "step.end", path: "wf/b", output: "new" },
      { type: "step.start", path: "wf/c", input: 0 },
    ] as RunEvent[];
    const j = buildJournal(events);
    assert.deepEqual(j, { "wf/a": 1, "wf/b": "new" });
  });

  it("readRunStart recovers input, hash, and params", () => {
    const events = [
      {
        type: "run.start",
        path: "wf",
        input: { x: 1 },
        workflowHash: "abc",
        params: { knob: 2 },
      },
    ] as RunEvent[];
    const rs = readRunStart(events);
    assert.deepEqual(rs?.input, { x: 1 });
    assert.equal(rs?.workflowHash, "abc");
    assert.deepEqual(rs?.params, { knob: 2 });
  });

  it("transitiveDependents follows explicit and implicit deps", () => {
    const steps = [
      step("a", "value", { result: 1 }),
      step("b", "value", { result: 2 }), // implicit: depends a
      step("c", "value", { result: 3 }, { depends: ["a"] }),
      step("d", "value", { result: 4 }, { depends: ["c"] }),
    ];
    assert.deepEqual([...transitiveDependents(steps, "a")].sort(), ["b", "c", "d"]);
    assert.deepEqual([...transitiveDependents(steps, "c")].sort(), ["d"]);
  });

  it("invalidateFrom drops the target, its dependents, and container entries — keeping foreach siblings", async () => {
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("prep", "value", { result: 1 }),
        step("each", "foreach", {
          items: [1, 2, 3],
          body: step("body", "value", { result: "{{ $current }}" }),
        }),
        step("after", "value", { result: "{{ each }}" }),
      ],
    });
    const journal = {
      "wf/prep": 1,
      "wf/each": [1, 2, 3],
      "wf/each#0": 1,
      "wf/each#1": 2,
      "wf/each#2": 3,
      "wf/after": "x",
    };
    const inv = await invalidateFrom(journal, "wf/each#1", wf);
    assert.deepEqual(Object.keys(inv.journal).sort(), ["wf/each#0", "wf/each#2", "wf/prep"]);
    assert.deepEqual(inv.dropped.sort(), ["wf/after", "wf/each", "wf/each#1"]);
  });

  it("invalidateFrom drops LATER iterations of a loop (sequential), keeping earlier ones", async () => {
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("l", "loop", {
          maxIterations: 5,
          until: "{{ $current === 3 }}",
          body: step("body", "value", { result: 1 }),
        }),
      ],
    });
    const journal = {
      "wf/l": 3,
      "wf/l#0": 1,
      "wf/l#1": 2,
      "wf/l#2": 3,
    };
    const inv = await invalidateFrom(journal, "wf/l#1", wf);
    assert.deepEqual(Object.keys(inv.journal), ["wf/l#0"]);
  });

  it("invalidateFrom on a plain top-level step drops it plus downstream only", async () => {
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("a", "value", { result: 1 }),
        step("b", "value", { result: 2 }),
        step("c", "value", { result: 3 }),
      ],
    });
    const journal = { "wf/a": 1, "wf/b": 2, "wf/c": 3 };
    const inv = await invalidateFrom(journal, "wf/b", wf);
    assert.deepEqual(Object.keys(inv.journal), ["wf/a"]);
    assert.deepEqual(inv.dropped.sort(), ["wf/b", "wf/c"]);
  });
});

describe("durable resume", () => {
  it("replays completed steps and re-executes from the first incomplete one", async () => {
    const counter = createCounterStep();
    const flakey = createFlakeyStep(1);
    const store = new MemoryRunStore();
    const wf = flow("wf", {
      input: z.object({ x: z.number() }),
      steps: [
        step("one", "counter", {}),
        step("two", "flakey", {}),
        step("three", "value", { result: "{{ two.attempts }}" }),
      ],
    });
    const registry = reg({ counter: counter.stepDef, flakey: flakey.stepDef });

    // First run: `two` fails → status error. `one` completed and journaled.
    const first = await runWorkflow(wf, { x: 1 }, registry, { runId: "r1", store });
    assert.equal(first.status, "error");
    assert.equal(counter.calls(), 1);

    // Resume: replay the journal, re-execute the failed step + downstream.
    const journal = buildJournal(store.getEvents("wf", "r1"));
    assert.ok("wf/one" in journal);
    const second = await runWorkflow(wf, { x: 1 }, registry, {
      runId: "r1",
      store,
      journal,
      resume: true,
    });
    assert.equal(second.status, "success");
    assert.equal(second.output, 2); // flakey succeeded on its 2nd attempt
    assert.equal(counter.calls(), 1); // `one` was REPLAYED, not re-executed

    const evts = store.getEvents("wf", "r1");
    assert.ok(evts.some((e) => e.type === "run.resumed"));
    assert.ok(evts.some((e) => e.type === "step.replayed" && e.path === "wf/one"));
    // history stays honest: the original failure is still in the log
    assert.ok(evts.some((e) => e.type === "run.error"));
    assert.ok(evts.some((e) => e.type === "run.end"));
    assert.equal(store.getSummary("wf", "r1")?.status, "success"); // superseded
  });

  it("re-runs only the failed foreach iteration; completed ones replay by #i path", async () => {
    let failOnce = true;
    const executed: number[] = [];
    const failAt2 = defineStep({
      type: "failAt2",
      input: z.object({ idx: z.number() }),
      output: z.any(),
      async run(cfg) {
        if (cfg.idx === 2 && failOnce) {
          failOnce = false;
          throw new Error("boom at 2");
        }
        executed.push(cfg.idx);
        return cfg.idx * 10;
      },
    });
    const store = new MemoryRunStore();
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("each", "foreach", {
          items: [0, 1, 2, 3],
          body: step("body", "failAt2", { idx: "{{ $index }}" }),
        }),
      ],
    });
    const registry = reg({ failAt2 });

    const first = await runWorkflow(wf, {}, registry, { runId: "r1", store });
    assert.equal(first.status, "error");
    assert.deepEqual(executed, [0, 1]); // 2 failed, 3 never ran

    const journal = buildJournal(store.getEvents("wf", "r1"));
    const second = await runWorkflow(wf, {}, registry, {
      runId: "r1",
      store,
      journal,
      resume: true,
    });
    assert.equal(second.status, "success");
    assert.deepEqual(second.output, [0, 10, 20, 30]);
    assert.deepEqual(executed, [0, 1, 2, 3]); // 0,1 NOT re-executed

    const replayed = store
      .getEvents("wf", "r1")
      .filter((e) => e.type === "step.replayed")
      .map((e) => e.path);
    assert.deepEqual(replayed.sort(), ["wf/each#0", "wf/each#1"]);
  });

  it("replays completed loop iterations and re-evaluates `until` against replayed $current", async () => {
    let calls = 0;
    let failOnce = true;
    const inc = defineStep({
      type: "inc",
      input: z.any(),
      output: z.number(),
      async run() {
        calls++;
        if (calls === 3 && failOnce) {
          failOnce = false;
          throw new Error("boom at iteration 2");
        }
        return calls;
      },
    });
    const store = new MemoryRunStore();
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("l", "loop", {
          maxIterations: 10,
          until: "{{ $current >= 4 }}",
          body: step("body", "inc", {}),
        }),
      ],
    });
    const registry = reg({ inc });

    const first = await runWorkflow(wf, {}, registry, { runId: "r1", store });
    assert.equal(first.status, "error"); // died on iteration index 2 (call 3)

    const journal = buildJournal(store.getEvents("wf", "r1"));
    const second = await runWorkflow(wf, {}, registry, {
      runId: "r1",
      store,
      journal,
      resume: true,
    });
    assert.equal(second.status, "success");
    assert.equal(second.output, 4);
    // iterations 0 and 1 replayed; 2 and 3 executed live (calls 3 → failed, then 4, 5... )
    const replayed = store
      .getEvents("wf", "r1")
      .filter((e) => e.type === "step.replayed")
      .map((e) => e.path);
    assert.deepEqual(replayed.sort(), ["wf/l#0", "wf/l#1"]);
  });

  it("reconstructs skip/gate logic from replayed outputs", async () => {
    const flakey = createFlakeyStep(1);
    const ran: string[] = [];
    const record = defineStep({
      type: "record",
      input: z.object({ tag: z.string() }),
      output: z.string(),
      async run(cfg) {
        ran.push(cfg.tag);
        return cfg.tag;
      },
    });
    const store = new MemoryRunStore();
    const wf = flow("wf", {
      input: z.any(),
      steps: [
        step("check", "value", { result: false }), // the boolean gate
        step("yes", "record", { tag: "yes" }, { depends: ["check"], when: true }),
        step("no", "record", { tag: "no" }, { depends: ["check"], when: false }),
        step("after", "flakey", {}, { depends: ["yes", "no"] }),
      ],
    });
    const registry = reg({ record, flakey: flakey.stepDef });

    const first = await runWorkflow(wf, {}, registry, { runId: "r1", store });
    assert.equal(first.status, "error");
    assert.deepEqual(ran, ["no"]);

    const journal = buildJournal(store.getEvents("wf", "r1"));
    const second = await runWorkflow(wf, {}, registry, {
      runId: "r1",
      store,
      journal,
      resume: true,
    });
    assert.equal(second.status, "success");
    // gate + `no` replayed; `yes` skipped AGAIN (reconstructed, not run)
    assert.deepEqual(ran, ["no"]);
    const resumedEvents = store.getEvents("wf", "r1");
    const skips = resumedEvents.filter((e) => e.type === "step.skipped" && e.path === "wf/yes");
    assert.equal(skips.length, 2); // one per invocation
  });

  it("hands a step its own synthetic-iteration journal slice as ctx.journal", async () => {
    const seenJournals: Array<Record<string, unknown> | undefined> = [];
    let failOnce = true;
    const gens = defineStep({
      type: "gens",
      input: z.any(),
      output: z.any(),
      async run(_cfg, ctx) {
        seenJournals.push(ctx.journal);
        const done: number[] = [];
        for (let g = 0; g < 3; g++) {
          const key = `${ctx.path}#${g}`;
          if (ctx.journal && key in ctx.journal) {
            done.push(ctx.journal[key] as number);
            continue; // completed generation — skip
          }
          if (g === 2 && failOnce) {
            failOnce = false;
            throw new Error("boom at gen 2");
          }
          await ctx.emit({
            ts: new Date().toISOString(),
            runId: ctx.runId,
            path: key,
            type: "step.end",
            output: g * 100,
            iteration: g,
          });
          done.push(g * 100);
        }
        return done;
      },
    });
    const store = new MemoryRunStore();
    const wf = flow("wf", {
      input: z.any(),
      steps: [step("evolve", "gens", {})],
    });
    const registry = reg({ gens });

    const first = await runWorkflow(wf, {}, registry, { runId: "r1", store });
    assert.equal(first.status, "error");
    assert.equal(seenJournals[0], undefined);

    const journal = buildJournal(store.getEvents("wf", "r1"));
    const second = await runWorkflow(wf, {}, registry, {
      runId: "r1",
      store,
      journal,
      resume: true,
    });
    assert.equal(second.status, "success");
    assert.deepEqual(second.output, [0, 100, 200]);
    // resume handed the step exactly its two completed generations
    assert.deepEqual(Object.keys(seenJournals[1] ?? {}).sort(), ["wf/evolve#0", "wf/evolve#1"]);
  });
});

// ── Crash hardening: torn tail + tail reopening ────────────────────────────

describe("crash hardening", () => {
  it("getRunEvents skips an unparseable trailing line (torn tail)", async () => {
    const dir = join(tmpdir(), `vein-test-${randomUUID()}`);
    const store = new FileRunStore(dir);
    await store.append("wf", "r1", {
      ts: "t",
      runId: "r1",
      path: "wf",
      type: "run.start",
      input: {},
    });
    await store.append("wf", "r1", {
      ts: "t",
      runId: "r1",
      path: "wf/a",
      type: "step.end",
      output: 1,
    });
    // Simulate a SIGKILL mid-append: truncated JSON, no newline.
    await appendFile(
      join(dir, "workflows", "wf", "runs", "r1", "events.jsonl"),
      '{"ts":"t","runId":"r1","path":"wf/b","type":"step.e',
      "utf-8",
    );
    const events = await store.getRunEvents("wf", "r1");
    assert.equal(events.length, 2);
    assert.equal(events[1]!.path, "wf/a");
    await rm(dir, { recursive: true, force: true });
  });

  it("tailEvents scans past run.error when a run.resumed follows (historical reopen)", async () => {
    const dir = join(tmpdir(), `vein-test-${randomUUID()}`);
    const store = new FileRunStore(dir);
    const base = { ts: "t", runId: "r1" };
    const log: RunEvent[] = [
      { ...base, path: "wf", type: "run.start", input: {} },
      { ...base, path: "wf/a", type: "step.end", output: 1 },
      { ...base, path: "wf", type: "run.error", error: { message: "boom" } },
      { ...base, path: "wf", type: "run.resumed" },
      { ...base, path: "wf/a", type: "step.replayed", output: 1 },
      { ...base, path: "wf", type: "run.end", output: "ok" },
    ];
    for (const e of log) await store.append("wf", "r1", e);

    const seen: string[] = [];
    for await (const e of store.tailEvents("wf", "r1", { intervalMs: 10 })) {
      seen.push(e.type);
    }
    assert.deepEqual(seen, [
      "run.start",
      "step.end",
      "run.error",
      "run.resumed",
      "step.replayed",
      "run.end",
    ]);
    await rm(dir, { recursive: true, force: true });
  });

  it("tailEvents still terminates at a plain terminal event with nothing after it", async () => {
    const dir = join(tmpdir(), `vein-test-${randomUUID()}`);
    const store = new FileRunStore(dir);
    await store.append("wf", "r1", {
      ts: "t",
      runId: "r1",
      path: "wf",
      type: "run.cancelled",
    });
    const seen: string[] = [];
    for await (const e of store.tailEvents("wf", "r1", { intervalMs: 10 })) {
      seen.push(e.type);
    }
    assert.deepEqual(seen, ["run.cancelled"]);
    await rm(dir, { recursive: true, force: true });
  });
});

// ── HTTP endpoints: cancel / pause / resume ────────────────────────────────

describe("run control endpoints", () => {
  async function makeServer(steps: Record<string, unknown>) {
    const dir = join(tmpdir(), `vein-test-${randomUUID()}`);
    await mkdir(dir, { recursive: true });
    const workspace = new WorkspaceManager(dir);
    const registry = {
      value: valueStep,
      ...steps,
    } as StepRegistry;
    const vein = await createVein({
      workspace,
      registry,
      serveUi: false,
      enableChat: false,
    });
    const store = vein.store as FileRunStore;
    const cleanup = () => rm(dir, { recursive: true, force: true });
    return { vein, workspace, store, cleanup };
  }

  async function waitForSummary(store: FileRunStore, wf: string, runId: string) {
    const deadline = Date.now() + 3000;
    for (;;) {
      const s = await store.getRunSummary(wf, runId);
      if (s) return s;
      if (Date.now() > deadline) throw new Error("summary never appeared");
      await sleep(10);
    }
  }

  it("POST cancel stops a live run tree; 404 unknown; 409 terminal", async () => {
    const gate = createGateStep();
    const { vein, workspace, store, cleanup } = await makeServer({ gate: gate.stepDef });
    try {
      await workspace.publishWorkflowByContent(
        "cancellable",
        [
          "name: cancellable",
          "steps:",
          "  - id: a",
          "    type: gate",
          "    config: { name: a }",
          "  - id: b",
          "    type: gate",
          "    config: { name: b }",
        ].join("\n"),
      );

      // 404 for an unknown run
      const notFound = await vein.app.request("/workflows/cancellable/runs/9999/cancel", {
        method: "POST",
      });
      assert.equal(notFound.status, 404);

      const launch = await vein.app.request("/workflows/cancellable/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input: {} }),
      });
      assert.equal(launch.status, 202);
      const { runId } = (await launch.json()) as { runId: string };

      await gate.waitForStart("a");
      const cancel = await vein.app.request(`/workflows/cancellable/runs/${runId}/cancel`, {
        method: "POST",
      });
      assert.equal(cancel.status, 202);
      gate.release("a");

      const summary = await waitForSummary(store, "cancellable", runId);
      assert.equal(summary.status, "cancelled");
      assert.ok(!gate.started.includes("b")); // subtree stopped at the boundary

      // 409 once terminal
      const again = await vein.app.request(`/workflows/cancellable/runs/${runId}/cancel`, {
        method: "POST",
      });
      assert.equal(again.status, 409);
    } finally {
      await cleanup();
    }
  });

  it("POST pause parks a run (listing shows paused, log records the gap); resume releases it", async () => {
    const gate = createGateStep();
    const { vein, workspace, store, cleanup } = await makeServer({ gate: gate.stepDef });
    try {
      await workspace.publishWorkflowByContent(
        "pausable",
        [
          "name: pausable",
          "steps:",
          "  - id: a",
          "    type: gate",
          "    config: { name: a }",
          "  - id: b",
          "    type: value",
          "    config: { result: done }",
        ].join("\n"),
      );

      const launch = await vein.app.request("/workflows/pausable/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input: {} }),
      });
      const { runId } = (await launch.json()) as { runId: string };

      await gate.waitForStart("a");
      const pause = await vein.app.request(`/workflows/pausable/runs/${runId}/pause`, {
        method: "POST",
      });
      assert.equal(pause.status, 202);
      gate.release("a");

      // Parked between a and b: no summary, listing reports paused.
      await sleep(50);
      assert.equal(await store.getRunSummary("pausable", runId), null);
      const listing = await vein.app.request("/workflows/pausable/runs");
      const runs = (await listing.json()) as Array<{ runId: string; status: string }>;
      assert.equal(runs.find((r) => r.runId === runId)?.status, "paused");

      const resume = await vein.app.request(`/workflows/pausable/runs/${runId}/resume`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      assert.equal(resume.status, 202);

      const summary = await waitForSummary(store, "pausable", runId);
      assert.equal(summary.status, "success");
      const evts = await store.getRunEvents("pausable", runId);
      assert.ok(evts.some((e) => e.type === "run.paused"));
      assert.ok(evts.some((e) => e.type === "run.resumed"));
    } finally {
      await cleanup();
    }
  });

  it("POST resume durably resumes a failed run (replay + retry) and enforces the hash guard", async () => {
    const counter = createCounterStep();
    const flakey = createFlakeyStep(1);
    const { vein, workspace, store, cleanup } = await makeServer({
      counter: counter.stepDef,
      flakey: flakey.stepDef,
    });
    try {
      await workspace.publishWorkflowByContent(
        "resumable",
        [
          "name: resumable",
          "steps:",
          "  - id: one",
          "    type: counter",
          "  - id: two",
          "    type: flakey",
        ].join("\n"),
      );

      const launch = await vein.app.request("/workflows/resumable/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input: {} }),
      });
      const { runId } = (await launch.json()) as { runId: string };
      const failed = await waitForSummary(store, "resumable", runId);
      assert.equal(failed.status, "error");
      assert.equal(counter.calls(), 1);

      // A successful run refuses resume without `from` — but first, resume the failed one.
      const resume = await vein.app.request(`/workflows/resumable/runs/${runId}/resume`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      assert.equal(resume.status, 202);

      const deadline = Date.now() + 3000;
      let summary = failed;
      while (summary.status !== "success") {
        if (Date.now() > deadline) throw new Error("resume never succeeded");
        await sleep(10);
        summary = (await store.getRunSummary("resumable", runId))!;
      }
      assert.equal(counter.calls(), 1); // `one` replayed, not re-run
      assert.equal(flakey.attempts(), 2); // `two` re-executed and succeeded

      // Now refuse resuming the (successful) run without `from`.
      const refuse = await vein.app.request(`/workflows/resumable/runs/${runId}/resume`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      assert.equal(refuse.status, 400);

      // Hash guard: publish a CHANGED active version → resume with `from` is refused…
      await workspace.publishWorkflowByContent(
        "resumable",
        [
          "name: resumable",
          "steps:",
          "  - id: one",
          "    type: counter",
          "  - id: two",
          "    type: flakey",
          "  - id: three",
          "    type: value",
          "    config: { result: changed }",
        ].join("\n"),
      );
      const mismatch = await vein.app.request(`/workflows/resumable/runs/${runId}/resume`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ from: "resumable/two" }),
      });
      assert.equal(mismatch.status, 409);
    } finally {
      await cleanup();
    }
  });

  it("POST resume with `from` re-runs a completed run from a chosen step", async () => {
    const counter = createCounterStep();
    const tally = createCounterStep();
    const { vein, workspace, store, cleanup } = await makeServer({
      counter: counter.stepDef,
      tally: tally.stepDef,
    });
    try {
      await workspace.publishWorkflowByContent(
        "regrade",
        [
          "name: regrade",
          "steps:",
          "  - id: memo",
          "    type: counter",
          "  - id: grade",
          "    type: tally",
        ].join("\n"),
      );

      const launch = await vein.app.request("/workflows/regrade/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ input: {} }),
      });
      const { runId } = (await launch.json()) as { runId: string };
      const first = await waitForSummary(store, "regrade", runId);
      assert.equal(first.status, "success");

      // "re-grade from grade onward" — costs the grade, not the memo.
      const resume = await vein.app.request(`/workflows/regrade/runs/${runId}/resume`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ from: "regrade/grade" }),
      });
      assert.equal(resume.status, 202);

      await waitFor(() => tally.calls() === 2);
      await waitFor(() => counter.calls() === 1); // memo replayed
      const evts = await store.getRunEvents("regrade", runId);
      assert.ok(evts.some((e) => e.type === "step.replayed" && e.path === "regrade/memo"));
    } finally {
      await cleanup();
    }
  });
});
