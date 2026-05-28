import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFile, rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { WorkspaceManager } from "./workspace.js";

const SAMPLE_YAML = `name: deploy
steps:
  - id: kick
    type: http
    config:
      url: /deploy
      method: POST
  - id: done
    type: log
    config:
      message: deployed
`;

const SAMPLE_STEPS = [
  { id: "kick", type: "http", config: { url: "/deploy", method: "POST" } },
  { id: "done", type: "log", config: { message: "deployed" } },
];

describe("WorkspaceManager", () => {
  let tempDir: string;
  let ws: WorkspaceManager;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ws-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
    ws = new WorkspaceManager(tempDir);
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  // ── publishWorkflow ────────────────────────────────────────────────────

  describe("publishWorkflow", () => {
    it("creates workflow directory and YAML file from steps", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS }, "initial version");

      const content = await readFile(
        join(tempDir, "workflows", "deploy", "v1.yaml"),
        "utf-8",
      );
      assert.ok(content.includes("kick"));
      assert.ok(content.includes("http"));

      const meta = JSON.parse(
        await readFile(join(tempDir, "workflows", "deploy", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.active, "v1");
      assert.ok(meta.versions["v1"]);
      assert.equal(meta.versions["v1"].description, "initial version");
    });

    it("creates workflow from raw YAML string", async () => {
      await ws.publishWorkflow("deploy", "v1", SAMPLE_YAML, "from yaml");

      const content = await readFile(
        join(tempDir, "workflows", "deploy", "v1.yaml"),
        "utf-8",
      );
      assert.equal(content, SAMPLE_YAML);
    });

    it("adds a second version and sets it active", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS }, "first");
      await ws.publishWorkflow("deploy", "v2", { steps: SAMPLE_STEPS }, "second");

      const meta = JSON.parse(
        await readFile(join(tempDir, "workflows", "deploy", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.active, "v2");
      assert.ok(meta.versions["v1"]);
      assert.ok(meta.versions["v2"]);
    });

    it("preserves existing versions when publishing new one", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS }, "first");
      await ws.publishWorkflow("deploy", "v2", { steps: SAMPLE_STEPS }, "second");
      await ws.publishWorkflow("deploy", "v3", { steps: SAMPLE_STEPS }, "third");

      const meta = JSON.parse(
        await readFile(join(tempDir, "workflows", "deploy", "_metadata.json"), "utf-8"),
      );
      assert.equal(Object.keys(meta.versions).length, 3);
    });
  });

  // ── getWorkflow ────────────────────────────────────────────────────────

  describe("getWorkflow", () => {
    it("loads active version and returns Flow with steps", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS });

      const flow = await ws.getWorkflow("deploy");
      assert.equal(flow.name, "deploy");
      assert.equal(flow.steps.length, 2);
      assert.equal(flow.steps[0]!.id, "kick");
      assert.equal(flow.steps[0]!.type, "http");
      assert.equal(flow.steps[1]!.id, "done");
    });

    it("throws for non-existent workflow", async () => {
      await assert.rejects(() => ws.getWorkflow("nope"), /not found/);
    });
  });

  // ── getWorkflowSource ─────────────────────────────────────────────────

  describe("getWorkflowSource", () => {
    it("returns raw YAML", async () => {
      await ws.publishWorkflow("deploy", "v1", SAMPLE_YAML);
      const src = await ws.getWorkflowSource("deploy", "v1");
      assert.equal(src, SAMPLE_YAML);
    });
  });

  // ── setActiveVersion ───────────────────────────────────────────────────

  describe("setActiveVersion", () => {
    it("changes the active version", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS });
      await ws.publishWorkflow("deploy", "v2", { steps: SAMPLE_STEPS });
      await ws.setActiveVersion("deploy", "v1");

      const meta = JSON.parse(
        await readFile(join(tempDir, "workflows", "deploy", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.active, "v1");
    });

    it("throws for non-existent workflow", async () => {
      await assert.rejects(() => ws.setActiveVersion("nope", "v1"), /not found/);
    });

    it("throws for non-existent version", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS });
      await assert.rejects(() => ws.setActiveVersion("deploy", "v99"), /not found/);
    });
  });

  // ── listWorkflows ──────────────────────────────────────────────────────

  describe("listWorkflows", () => {
    it("returns empty array when no workflows", async () => {
      assert.deepEqual(await ws.listWorkflows(), []);
    });

    it("lists published workflows", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS }, "deploy flow");
      await ws.publishWorkflow("notify", "v1", { steps: SAMPLE_STEPS }, "notify flow");

      const list = await ws.listWorkflows();
      assert.equal(list.length, 2);
      assert.deepEqual(list.map((w) => w.name).sort(), ["deploy", "notify"]);
    });

    it("shows active version and all versions", async () => {
      await ws.publishWorkflow("deploy", "v1", { steps: SAMPLE_STEPS }, "first");
      await ws.publishWorkflow("deploy", "v2", { steps: SAMPLE_STEPS }, "second");
      await ws.setActiveVersion("deploy", "v1");

      const list = await ws.listWorkflows();
      const deploy = list.find((w) => w.name === "deploy")!;
      assert.equal(deploy.activeVersion, "v1");
      assert.deepEqual(deploy.versions.sort(), ["v1", "v2"]);
    });

    it("skips directories without _metadata.json", async () => {
      await mkdir(join(tempDir, "workflows", "broken"), { recursive: true });
      assert.equal((await ws.listWorkflows()).length, 0);
    });
  });

  // ── publishStep ────────────────────────────────────────────────────────

  describe("publishStep", () => {
    it("writes the step file under steps/custom/", async () => {
      await ws.publishStep("my-scorer", "export default {};", "A custom scorer");

      const code = await readFile(join(tempDir, "steps", "custom", "my-scorer.ts"), "utf-8");
      assert.equal(code, "export default {};");

      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      assert.ok(meta.steps["my-scorer"]);
      assert.equal(meta.steps["my-scorer"].description, "A custom scorer");
    });

    it("preserves earlier steps when publishing additional ones", async () => {
      await ws.publishStep("a", "// a");
      await ws.publishStep("b", "// b");

      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      assert.ok(meta.steps["a"]);
      assert.ok(meta.steps["b"]);
    });

    it("overwrites an existing step with the same name", async () => {
      await ws.publishStep("dupe", "// v1");
      await ws.publishStep("dupe", "// v2", "second version");

      const code = await readFile(join(tempDir, "steps", "custom", "dupe.ts"), "utf-8");
      assert.equal(code, "// v2");
      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.steps["dupe"].description, "second version");
    });
  });

  // ── listSteps ──────────────────────────────────────────────────────────

  describe("listSteps", () => {
    it("returns empty array when no custom steps exist", async () => {
      assert.deepEqual(await ws.listSteps(), []);
    });

    it("lists custom steps with plain types", async () => {
      await ws.publishStep("my-scorer", "// scorer", "A scorer");

      const list = await ws.listSteps();
      assert.equal(list.length, 1);
      assert.equal(list[0]!.type, "my-scorer");
      assert.equal(list[0]!.description, "A scorer");
    });

    it("lists multiple custom steps", async () => {
      await ws.publishStep("alpha", "// alpha");
      await ws.publishStep("beta", "// beta");

      const list = await ws.listSteps();
      assert.deepEqual(list.map((s) => s.type).sort(), ["alpha", "beta"]);
    });

    it("skips _metadata.json files in listing", async () => {
      await ws.publishStep("my-step", "// step");
      const list = await ws.listSteps();
      assert.equal(list.length, 1);
      assert.equal(list[0]!.type, "my-step");
    });
  });

  // ── path property ──────────────────────────────────────────────────────

  describe("path property", () => {
    it("returns the workspace root path", () => {
      assert.equal(ws.path, tempDir);
    });

    it("uses VEIN_WORKSPACE env var as default", () => {
      const original = process.env["VEIN_WORKSPACE"];
      process.env["VEIN_WORKSPACE"] = "/custom/path";
      const ws2 = new WorkspaceManager();
      assert.equal(ws2.path, "/custom/path");
      if (original !== undefined) {
        process.env["VEIN_WORKSPACE"] = original;
      } else {
        delete process.env["VEIN_WORKSPACE"];
      }
    });
  });
});
