import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WorkspaceManager } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "./seed-opts.js";

const YAML = "steps:\n  - id: hello\n    type: log\n    config:\n      message: hi\n";
const STEP = [
  'import { z } from "zod";',
  'import { defineStep } from "strut";',
  'export default defineStep({ type: "t/echo", input: z.object({}), output: z.object({}), async run() { return {}; } });',
  "",
].join("\n");

describe("retireWorkflows / retireSteps", () => {
  it("removes what a seeder no longer ships; missing and invalid names never throw", async () => {
    const root = await mkdtemp(join(tmpdir(), "lab-seed-"));
    try {
      const ws = new WorkspaceManager(root);
      await ws.publishWorkflowByContent("old-wf", YAML, undefined, "t", undefined, SEED_OPTS);
      await ws.publishStep("t/echo", STEP, undefined, "t-seed", SEED_OPTS);
      assert.deepEqual((await ws.listWorkflows()).map((w) => w.name), ["old-wf"]);
      assert.deepEqual((await ws.listSteps()).map((s) => s.type), ["t/echo"]);

      await retireWorkflows(ws, ["old-wf", "never-seeded", "bad/name"], "t");
      await retireSteps(ws, ["t/echo", "t/never"], "t");
      assert.deepEqual(await ws.listWorkflows(), []);
      assert.deepEqual(await ws.listSteps(), []);

      // The next boot finds nothing to retire and says nothing.
      await retireWorkflows(ws, ["old-wf"], "t");
      await retireSteps(ws, ["t/echo"], "t");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
