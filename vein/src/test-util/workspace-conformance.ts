import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import type { WorkspaceStore } from "../workspace.js";

/**
 * The `WorkspaceStore` contract as tests — one behavioral suite every
 * backend (file, path-less wrapper, graph) must pass. A backend that needs
 * a different assertion here is a backend that changed the contract.
 */

const STEP_SRC = (type: string, desc: string) => `import { z } from "zod";
import { defineStep } from "vein";
export default defineStep({
  type: ${JSON.stringify(type)},
  description: ${JSON.stringify(desc)},
  input: z.object({}),
  output: z.any(),
  async run() { return ${JSON.stringify(desc)}; },
});
`;

// ── Workspace store ────────────────────────────────────────────────────────

export interface WorkspaceImpl {
  name: string;
  /** Build a fresh, empty store. `dir` is a fresh temp dir the case owns. */
  make: (dir: string) => Promise<WorkspaceStore> | WorkspaceStore;
  /** Reset backend state between cases (graph wipe, …). */
  reset?: () => Promise<void>;
  /** `describe` skip reason (e.g. no live database configured). */
  skip?: string | false;
}

export function workspaceConformance(impl: WorkspaceImpl): void {
  describe(`WorkspaceStore conformance: ${impl.name}`, { skip: impl.skip ?? false }, () => {
    let dir: string;
    let ws: WorkspaceStore;
    beforeEach(async () => {
      dir = join(tmpdir(), `vein-conf-ws-${randomUUID()}`);
      await mkdir(dir, { recursive: true });
      await impl.reset?.();
      ws = await impl.make(dir);
    });
    afterEach(() => rm(dir, { recursive: true, force: true }));

    const steps = [{ id: "a", type: "log", config: { message: "hi" } }];

    it("workflow publish → list → metadata → source → hash round-trip", async () => {
      await ws.publishWorkflow("wf", "v1", { steps }, "first", "exp", "me");
      const list = await ws.listWorkflows();
      assert.deepEqual(
        list.map((w) => [w.name, w.activeVersion, w.versions, w.description, w.category, w.publisher]),
        [["wf", "v1", ["v1"], "first", "exp", "me"]],
      );
      assert.equal("lastRunAt" in list[0]!, false, "runs are the run store's — never listed here");
      const meta = await ws.getWorkflowMetadata("wf");
      assert.equal(meta?.active, "v1");
      assert.equal(meta?.publisher, "me");
      assert.equal(await ws.getWorkflowMetadata("nope"), null);
      const src = await ws.getWorkflowSource("wf", "v1");
      assert.ok(src.includes("type: log"));
      assert.equal(typeof (await ws.getWorkflowHash("wf")), "string");
      assert.equal(await ws.getWorkflowHash("nope"), null);
      assert.equal((await ws.getWorkflow("wf")).steps.length, 1);
      assert.equal((await ws.getWorkflowVersion("wf", "v1")).steps.length, 1);
      await assert.rejects(() => ws.getWorkflow("nope"), /not found/);
    });

    it("versions, active switching, content dedup, category, params", async () => {
      await ws.publishWorkflow("wf", "v1", { steps, params: { greeting: "old" } });
      const first = await ws.publishWorkflowByContent("wf", await ws.getWorkflowSource("wf", "v1"));
      assert.equal(first.changed, false, "same content → no new version");
      assert.equal(first.version, "v1");
      const second = await ws.publishWorkflowByContent(
        "wf",
        (await ws.getWorkflowSource("wf", "v1")).replace("old", "new"),
      );
      assert.equal(second.changed, true);
      assert.notEqual(second.version, "v1");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.active, second.version);
      await ws.setActiveVersion("wf", "v1");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.active, "v1");
      await assert.rejects(() => ws.setActiveVersion("wf", "v99"));
      await ws.setWorkflowCategory("wf", "cat");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.category, "cat");
      const p = await ws.setParam("wf", "greeting", "newer");
      assert.deepEqual([p.before, p.after], ["old", "newer"]);
      assert.equal((await ws.getWorkflow("wf")).params?.["greeting"], "newer");
    });

    it("createWorkflow allocates a fresh name/version and returns it", async () => {
      const a = await ws.createWorkflow("made", { steps });
      const b = await ws.createWorkflow("made", { steps });
      assert.equal(a.name, "made");
      assert.notEqual(b.name, a.name, "a second create under the same name is renamed, not clobbered");
    });

    it("step publish → list → versions → source → active switching → delete", async () => {
      const v1 = await ws.publishStep("my-step", STEP_SRC("my-step", "one"), "one", "svc");
      const again = await ws.publishStep("my-step", STEP_SRC("my-step", "one"), "one", "svc");
      assert.equal(again.changed, false, "same source → no new version");
      const v2 = await ws.publishStep("my-step", STEP_SRC("my-step", "two"), "two", "svc");
      assert.equal(v2.changed, true);
      assert.deepEqual(
        (await ws.listSteps()).map((s) => [s.type, s.description, s.publisher]),
        [["my-step", "two", "svc"]],
      );
      assert.deepEqual(await ws.listSteps({ publisher: "other" }), []);
      const versions = await ws.listStepVersions("my-step");
      assert.equal(versions.active, v2.version);
      assert.deepEqual(new Set(versions.versions), new Set([v1.version, v2.version]));
      assert.ok((await ws.getStepVersionSource("my-step", v1.version)).includes('"one"'));
      await ws.setActiveStepVersion("my-step", v1.version);
      assert.equal((await ws.listStepVersions("my-step")).active, v1.version);
      assert.equal((await ws.getStepSource("my-step"))?.code.includes('"one"'), true);
      assert.equal(await ws.deleteStep("my-step"), true);
      assert.equal(await ws.deleteStep("my-step"), false);
      assert.deepEqual(await ws.listSteps(), []);
    });

    it("deleteStepsByPublisher removes exactly that publisher's steps", async () => {
      await ws.publishStep("a", STEP_SRC("a", "a"), "a", "svc-1");
      await ws.publishStep("ns/b", STEP_SRC("ns/b", "b"), "b", "svc-1");
      await ws.publishStep("c", STEP_SRC("c", "c"), "c", "svc-2");
      assert.deepEqual((await ws.deleteStepsByPublisher("svc-1")).sort(), ["a", "ns/b"]);
      assert.deepEqual((await ws.listSteps()).map((s) => s.type), ["c"]);
    });

    it("getStepSource spans tiers: custom from the store, lib + core from the engine, null otherwise", async () => {
      await ws.publishStep("ns/custom", STEP_SRC("ns/custom", "x"));
      assert.equal((await ws.getStepSource("ns/custom"))?.origin, "custom");
      assert.equal((await ws.getStepSource("log"))?.origin, "core");
      assert.equal((await ws.getStepSource("github/fetch-pr"))?.origin, "lib");
      assert.equal(await ws.getStepSource("no/such/step"), null);
    });

    it("materializeCustomSteps returns a directory holding every active custom step as a file", async () => {
      await ws.publishStep("flat", STEP_SRC("flat", "f"));
      await ws.publishStep("ns/nested", STEP_SRC("ns/nested", "n"));
      const root = await ws.materializeCustomSteps();
      assert.ok((await stat(join(root, "flat.ts"))).isFile());
      assert.ok((await stat(join(root, "ns", "nested.ts"))).isFile());
      assert.equal(await ws.materializeCustomSteps(), root, "idempotent");
    });
  });
}
