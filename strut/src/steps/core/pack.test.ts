import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { flow, step } from "../../core.js";
import { runWorkflow } from "../../runner.js";
import { coreRegistry } from "../registry.js";
import { MemoryRunStore } from "../../store.js";

describe("pack (core)", () => {
  it("returns the resolved config as the step output — the workflow's return value", async () => {
    const wf = flow("pack-test", {
      input: z.object({ repo: z.string() }),
      steps: [
        step("a", "log", { message: "one" }),
        step("b", "log", { message: "two" }, { depends: [] }),
        step("result", "pack", {
          repo: "{{ input.repo }}",
          first: "{{ a }}",
          second: "{{ b }}",
          n: "{{ 1 + 2 }}",
          nested: { ok: true, list: ["{{ input.repo }}", "x"] },
        }, { depends: ["a", "b"] }),
      ],
    });
    const result = await runWorkflow(wf, { repo: "stakgraph" }, coreRegistry(), { store: new MemoryRunStore() });
    assert.equal(result.status, "success", JSON.stringify(result.error));
    assert.deepEqual(result.output, {
      repo: "stakgraph",
      first: "one",
      second: "two",
      n: 3,
      nested: { ok: true, list: ["stakgraph", "x"] },
    });
  });
});
