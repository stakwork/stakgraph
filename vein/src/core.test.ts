import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step, defineStep } from "./core.js";

// ── step() ─────────────────────────────────────────────────────────────────

describe("step()", () => {
  it("creates a step with id, type, and config", () => {
    const s = step("fetch", "http", { url: "/api" });
    assert.equal(s.id, "fetch");
    assert.equal(s.type, "http");
    assert.deepEqual(s.config, { url: "/api" });
    assert.equal(s.options, undefined);
  });

  it("creates a step with options", () => {
    const s = step("fetch", "http", { url: "/api" }, {
      retry: { max: 3, delayMs: 1000 },
    });
    assert.equal(s.options?.retry?.max, 3);
    assert.equal(s.options?.retry?.delayMs, 1000);
  });

  it("creates a step with onError fallback", () => {
    const fallback = step("alert", "log", { message: "failed" });
    const s = step("fetch", "http", { url: "/api" }, { onError: fallback });
    assert.equal(s.options?.onError?.id, "alert");
    assert.equal(s.options?.onError?.type, "log");
  });

  it("creates a step with both retry and onError", () => {
    const fallback = step("alert", "log", { message: "failed" });
    const s = step("fetch", "http", { url: "/api" }, {
      retry: { max: 2, delayMs: 500 },
      onError: fallback,
    });
    assert.equal(s.options?.retry?.max, 2);
    assert.equal(s.options?.onError?.id, "alert");
  });

  describe("id validation", () => {
    it("accepts valid alphanumeric ids", () => {
      assert.doesNotThrow(() => step("a", "http", {}));
      assert.doesNotThrow(() => step("step1", "http", {}));
      assert.doesNotThrow(() => step("my_step", "http", {}));
      assert.doesNotThrow(() => step("_private", "http", {}));
      assert.doesNotThrow(() => step("CamelCase", "http", {}));
    });

    it("rejects ids starting with a number", () => {
      assert.throws(() => step("1step", "http", {}), /Invalid step id/);
    });

    it("rejects ids with hyphens", () => {
      assert.throws(() => step("my-step", "http", {}), /Invalid step id/);
    });

    it("rejects ids with spaces", () => {
      assert.throws(() => step("my step", "http", {}), /Invalid step id/);
    });

    it("rejects ids with special characters", () => {
      assert.throws(() => step("step!", "http", {}), /Invalid step id/);
      assert.throws(() => step("step.name", "http", {}), /Invalid step id/);
    });

    it("rejects empty id", () => {
      assert.throws(() => step("", "http", {}), /Invalid step id/);
    });
  });
});

// ── flow() ─────────────────────────────────────────────────────────────────

describe("flow()", () => {
  it("creates a flow with name, input, and steps", () => {
    const f = flow("deploy", {
      input: z.object({ service: z.string() }),
      steps: [
        step("kick", "http", { url: "/deploy", method: "POST" }),
        step("done", "log", { message: "deployed" }),
      ],
    });
    assert.equal(f.name, "deploy");
    assert.equal(f.steps.length, 2);
    assert.equal(f.steps[0]!.id, "kick");
    assert.equal(f.steps[1]!.id, "done");
  });

  it("creates a flow with empty input schema", () => {
    const f = flow("simple", {
      input: z.object({}),
      steps: [step("log", "log", { message: "hello" })],
    });
    assert.equal(f.name, "simple");
  });

  it("creates a flow with z.any() input", () => {
    const f = flow("flexible", {
      input: z.any(),
      steps: [step("log", "log", { message: "hello" })],
    });
    assert.equal(f.name, "flexible");
  });

  it("creates a flow with no steps", () => {
    const f = flow("empty", {
      input: z.object({}),
      steps: [],
    });
    assert.equal(f.steps.length, 0);
  });

  it("rejects duplicate step ids", () => {
    assert.throws(
      () =>
        flow("bad", {
          input: z.object({}),
          steps: [
            step("a", "log", { message: "first" }),
            step("a", "log", { message: "second" }),
          ],
        }),
      /Duplicate step id "a"/,
    );
  });

  it("allows same id in different flows", () => {
    const f1 = flow("flow1", {
      input: z.any(),
      steps: [step("a", "log", { message: "f1" })],
    });
    const f2 = flow("flow2", {
      input: z.any(),
      steps: [step("a", "log", { message: "f2" })],
    });
    assert.equal(f1.steps[0]!.id, "a");
    assert.equal(f2.steps[0]!.id, "a");
  });
});

// ── defineStep() ───────────────────────────────────────────────────────────

describe("defineStep()", () => {
  it("returns the step definition as-is", () => {
    const def = defineStep({
      type: "test",
      input: z.object({ x: z.number() }),
      output: z.number(),
      async run(cfg) {
        return cfg.x * 2;
      },
    });
    assert.equal(def.type, "test");
    assert.ok(def.input);
    assert.ok(def.output);
    assert.equal(typeof def.run, "function");
  });

  it("step run function works correctly", async () => {
    const def = defineStep({
      type: "double",
      input: z.object({ value: z.number() }),
      output: z.number(),
      async run(cfg) {
        return cfg.value * 2;
      },
    });
    const result = await def.run({ value: 21 }, {} as any);
    assert.equal(result, 42);
  });

  it("step input schema validates correctly", () => {
    const def = defineStep({
      type: "typed",
      input: z.object({ name: z.string(), age: z.number() }),
      output: z.any(),
      async run(cfg) {
        return cfg;
      },
    });

    // Valid input
    assert.doesNotThrow(() => def.input.parse({ name: "Alice", age: 30 }));

    // Invalid input
    assert.throws(() => def.input.parse({ name: "Alice" }));
    assert.throws(() => def.input.parse({ name: 123, age: 30 }));
  });
});
