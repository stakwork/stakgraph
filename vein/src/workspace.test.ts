import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFile, rm, mkdir, stat } from "node:fs/promises";
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

  // ── createWorkflow ─────────────────────────────────────────────────────

  describe("createWorkflow", () => {
    it("creates a new workflow at v1 and returns the resolved name", async () => {
      const result = await ws.createWorkflow(
        "deploy",
        { steps: SAMPLE_STEPS },
        "initial",
      );
      assert.deepEqual(result, { name: "deploy", version: "v1" });

      const flow = await ws.getWorkflow("deploy");
      assert.equal(flow.name, "deploy");
      assert.equal(flow.steps.length, 2);
    });

    it("auto-suffixes when the name collides with an existing workflow", async () => {
      await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS });
      const second = await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS });
      assert.equal(second.name, "deploy-2");

      const list = await ws.listWorkflows();
      assert.deepEqual(list.map((w) => w.name).sort(), ["deploy", "deploy-2"]);
    });

    it("keeps incrementing the suffix until a free slot is found", async () => {
      await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS });
      await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS }); // deploy-2
      const third = await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS });
      assert.equal(third.name, "deploy-3");
    });

    it("rewrites the embedded YAML `name:` field to match the resolved name", async () => {
      await ws.createWorkflow("deploy", SAMPLE_YAML);
      const second = await ws.createWorkflow("deploy", SAMPLE_YAML);
      assert.equal(second.name, "deploy-2");

      // The on-disk YAML for the suffixed workflow must reference the new
      // name so `runner.ts` writes runs under `deploy-2/`, not `deploy/`.
      const flow = await ws.getWorkflow("deploy-2");
      assert.equal(flow.name, "deploy-2");
    });

    it("leaves the original workflow untouched on collision", async () => {
      await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS }, "first");
      await ws.createWorkflow("deploy", { steps: SAMPLE_STEPS }, "second");

      const meta = JSON.parse(
        await readFile(
          join(tempDir, "workflows", "deploy", "_metadata.json"),
          "utf-8",
        ),
      );
      assert.equal(meta.versions["v1"].description, "first");
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

    it("round-trips a params block through publish (steps form) and load", async () => {
      await ws.publishWorkflow("knobs", "v1", {
        steps: SAMPLE_STEPS,
        params: { systemPrompt: "be concise", maxFiles: 12 },
      });

      const flow = await ws.getWorkflow("knobs");
      assert.deepEqual(flow.params, { systemPrompt: "be concise", maxFiles: 12 });

      // And it's serialized into the on-disk YAML.
      const src = await ws.getWorkflowSource("knobs", "v1");
      assert.match(src, /params:/);
      assert.match(src, /systemPrompt/);
    });

    it("omits params on the Flow when the workflow declares none", async () => {
      await ws.publishWorkflow("plain", "v1", { steps: SAMPLE_STEPS });
      const flow = await ws.getWorkflow("plain");
      assert.equal(flow.params, undefined);
    });

    it("resolves param-to-param references at load (shared value factored into one param)", async () => {
      await ws.publishWorkflow("podcfg", "v1", {
        steps: SAMPLE_STEPS,
        params: {
          podDomain: "workspaces.sphinx.chat",
          finalAnswer: "Public URLs use https://$POD_ID.{{ params.podDomain }} — and keep {{ input.keep }} alone.",
          dataset: [{ expected: "host: {{ params.podDomain }}" }],
        },
      });
      const flow = await ws.getWorkflow("podcfg");
      const p = flow.params as Record<string, any>;
      // params.* refs are substituted (deeply, incl. inside arrays)...
      assert.equal(p.finalAnswer, "Public URLs use https://$POD_ID.workspaces.sphinx.chat — and keep {{ input.keep }} alone.");
      assert.equal(p.dataset[0].expected, "host: workspaces.sphinx.chat");
      // ...while non-params templates (input/step refs) are left for run time.
      assert.match(p.finalAnswer, /\{\{ input\.keep \}\}/);
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
      const info = meta.steps["my-scorer"];
      assert.ok(info);
      assert.equal(info.versions[info.active].description, "A custom scorer");
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

    it("publishes a new active version when the same name is updated", async () => {
      await ws.publishStep("dupe", "// v1");
      await ws.publishStep("dupe", "// v2", "second version");

      // Flat active file holds the latest content (what the registry loads).
      const code = await readFile(join(tempDir, "steps", "custom", "dupe.ts"), "utf-8");
      assert.equal(code, "// v2");
      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      const info = meta.steps["dupe"];
      // Both versions retained; active points at the latest.
      assert.equal(Object.keys(info.versions).length, 2);
      assert.equal(info.versions[info.active].description, "second version");
    });

    it("writes nested step names under subdirectories", async () => {
      await ws.publishStep(
        "gitree/save-feature",
        "// save-feature",
        "Persist a Feature",
      );

      const code = await readFile(
        join(tempDir, "steps", "custom", "gitree", "save-feature.ts"),
        "utf-8",
      );
      assert.equal(code, "// save-feature");

      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      // Full slash-name is the metadata key
      const info = meta.steps["gitree/save-feature"];
      assert.ok(info);
      assert.equal(info.versions[info.active].description, "Persist a Feature");
    });

    it("records publisher in metadata when provided", async () => {
      await ws.publishStep("gitree/get-pr", "// get-pr", undefined, "mcp-gitree");

      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.steps["gitree/get-pr"].publisher, "mcp-gitree");
    });

    it("writes helper files with a leading underscore", async () => {
      await ws.publishStep("gitree/_shared", "export const x = 1;");

      const code = await readFile(
        join(tempDir, "steps", "custom", "gitree", "_shared.ts"),
        "utf-8",
      );
      assert.equal(code, "export const x = 1;");
    });

    it("rejects path traversal attempts", async () => {
      await assert.rejects(
        () => ws.publishStep("../escape", "// nope"),
        /Invalid step name/,
      );
      await assert.rejects(
        () => ws.publishStep("gitree/../escape", "// nope"),
        /Invalid step name/,
      );
    });

    it("rejects absolute paths and empty segments", async () => {
      await assert.rejects(
        () => ws.publishStep("/abs", "// nope"),
        /Invalid step name/,
      );
      await assert.rejects(
        () => ws.publishStep("gitree//double", "// nope"),
        /Invalid step name/,
      );
      await assert.rejects(
        () => ws.publishStep("trailing/", "// nope"),
        /Invalid step name/,
      );
    });

    it("rejects names with invalid characters", async () => {
      await assert.rejects(
        () => ws.publishStep("has spaces", "// nope"),
        /Invalid step name/,
      );
      await assert.rejects(
        () => ws.publishStep("9starts-with-digit", "// nope"),
        /Invalid step name/,
      );
    });
  });

  // ── step versioning ──────────────────────────────────────────────────────

  describe("step versioning", () => {
    it("is idempotent: republishing identical content does not change", async () => {
      const first = await ws.publishStep("idem", "// same");
      const second = await ws.publishStep("idem", "// same");
      assert.equal(first.changed, true);
      assert.equal(second.changed, false);
      assert.equal(first.version, second.version);

      const { versions } = await ws.listStepVersions("idem");
      assert.equal(versions.length, 1);
    });

    it("uses sequential version labels (v1, v2, …)", async () => {
      const a = await ws.publishStep("seq", "// content A");
      const b = await ws.publishStep("seq", "// content B");
      assert.equal(a.version, "v1");
      assert.equal(b.version, "v2");
    });

    it("re-activates an older version (no new version) when content matches", async () => {
      await ws.publishStep("react", "// one"); // v1
      await ws.publishStep("react", "// two"); // v2
      const back = await ws.publishStep("react", "// one"); // matches v1
      assert.equal(back.version, "v1");
      assert.equal(back.changed, true);
      const { active, versions } = await ws.listStepVersions("react");
      assert.equal(active, "v1");
      assert.equal(versions.length, 2);
    });

    it("archives every version under steps/_history/<name>/", async () => {
      const v1 = await ws.publishStep("arch", "// one");
      const v2 = await ws.publishStep("arch", "// two");

      const a = await ws.getStepVersionSource("arch", v1.version);
      const b = await ws.getStepVersionSource("arch", v2.version);
      assert.equal(a, "// one");
      assert.equal(b, "// two");
    });

    it("setActiveStepVersion checks out an older version into the flat file", async () => {
      const v1 = await ws.publishStep("roll", "// original");
      await ws.publishStep("roll", "// updated");

      let code = await readFile(join(tempDir, "steps", "custom", "roll.ts"), "utf-8");
      assert.equal(code, "// updated");

      await ws.setActiveStepVersion("roll", v1.version);
      code = await readFile(join(tempDir, "steps", "custom", "roll.ts"), "utf-8");
      assert.equal(code, "// original");

      const { active } = await ws.listStepVersions("roll");
      assert.equal(active, v1.version);
    });

    it("setActiveStepVersion throws on an unknown version", async () => {
      await ws.publishStep("known", "// a");
      await assert.rejects(
        () => ws.setActiveStepVersion("known", "c-doesnotexist"),
        /not found/,
      );
    });

    it("deleteStep removes the version archive too", async () => {
      const v1 = await ws.publishStep("gone", "// bye");
      assert.equal(await ws.deleteStep("gone"), true);
      await assert.rejects(
        () => ws.getStepVersionSource("gone", v1.version),
        /ENOENT/,
      );
    });
  });

  // ── workflow content-hash publishing ───────────────────────────────────────

  describe("publishWorkflowByContent", () => {
    const wf = (n: number) =>
      `name: hashwf\nsteps:\n  - id: s${n}\n    type: log\n    config:\n      message: "v${n}"\n`;

    it("is idempotent for identical content and re-activates known versions", async () => {
      const a = await ws.publishWorkflowByContent("hashwf", wf(1));
      const b = await ws.publishWorkflowByContent("hashwf", wf(1));
      assert.equal(a.changed, true);
      assert.equal(b.changed, false);
      assert.equal(a.version, b.version);

      // New content → new active version; both retained.
      const c = await ws.publishWorkflowByContent("hashwf", wf(2));
      assert.equal(c.changed, true);
      assert.notEqual(c.version, a.version);

      // Re-publishing the older content flips active back without a new version.
      const d = await ws.publishWorkflowByContent("hashwf", wf(1));
      assert.equal(d.version, a.version);
      const list = await ws.listWorkflows();
      const entry = list.find((w) => w.name === "hashwf")!;
      assert.equal(entry.activeVersion, a.version);
      assert.equal(entry.versions.length, 2);
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

    it("lists nested steps with their slash-names", async () => {
      await ws.publishStep("flat", "// flat");
      await ws.publishStep("gitree/save-feature", "// sf");
      await ws.publishStep("gitree/get-pr", "// gp");

      const list = await ws.listSteps();
      assert.deepEqual(
        list.map((s) => s.type).sort(),
        ["flat", "gitree/get-pr", "gitree/save-feature"],
      );
    });

    it("omits helper files (leading underscore)", async () => {
      await ws.publishStep("gitree/save-feature", "// sf");
      await ws.publishStep("gitree/_shared", "// helper");

      const list = await ws.listSteps();
      assert.deepEqual(list.map((s) => s.type), ["gitree/save-feature"]);
    });

    it("surfaces publisher in the listing", async () => {
      await ws.publishStep("anon", "// anon");
      await ws.publishStep("owned", "// owned", undefined, "mcp-gitree");

      const list = await ws.listSteps();
      const byName = Object.fromEntries(list.map((s) => [s.type, s]));
      assert.equal(byName["anon"]!.publisher, undefined);
      assert.equal(byName["owned"]!.publisher, "mcp-gitree");
    });

    it("filters by publisher when requested", async () => {
      await ws.publishStep("a", "// a", undefined, "svc-1");
      await ws.publishStep("b", "// b", undefined, "svc-2");
      await ws.publishStep("c", "// c"); // no publisher

      const onlySvc1 = await ws.listSteps({ publisher: "svc-1" });
      assert.deepEqual(onlySvc1.map((s) => s.type), ["a"]);
    });
  });

  // ── deleteStep / deleteStepsByPublisher ────────────────────────────────

  describe("deleteStep", () => {
    it("removes the file and metadata entry", async () => {
      await ws.publishStep("doomed", "// bye");
      const ok = await ws.deleteStep("doomed");
      assert.equal(ok, true);

      const list = await ws.listSteps();
      assert.equal(list.length, 0);

      const meta = JSON.parse(
        await readFile(join(tempDir, "steps", "custom", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.steps["doomed"], undefined);
    });

    it("returns false when the step doesn't exist", async () => {
      const ok = await ws.deleteStep("nope");
      assert.equal(ok, false);
    });

    it("removes nested step files and prunes empty parent directories", async () => {
      await ws.publishStep("gitree/save-feature", "// sf");
      assert.equal(await ws.deleteStep("gitree/save-feature"), true);

      // Parent `gitree/` dir should be gone since it's now empty
      await assert.rejects(
        () => stat(join(tempDir, "steps", "custom", "gitree")),
      );
    });

    it("does not prune a parent that still contains siblings", async () => {
      await ws.publishStep("gitree/save-feature", "// sf");
      await ws.publishStep("gitree/get-pr", "// gp");

      await ws.deleteStep("gitree/save-feature");

      // Parent dir still holds get-pr.ts
      const remaining = await readFile(
        join(tempDir, "steps", "custom", "gitree", "get-pr.ts"),
        "utf-8",
      );
      assert.equal(remaining, "// gp");
    });
  });

  describe("deleteStepsByPublisher", () => {
    it("removes all steps owned by a publisher and returns their names", async () => {
      await ws.publishStep("gitree/save-feature", "// sf", undefined, "mcp");
      await ws.publishStep("gitree/get-pr", "// gp", undefined, "mcp");
      await ws.publishStep("other", "// o", undefined, "different-svc");
      await ws.publishStep("anonymous", "// a"); // no publisher

      const deleted = await ws.deleteStepsByPublisher("mcp");
      assert.deepEqual(deleted.sort(), [
        "gitree/get-pr",
        "gitree/save-feature",
      ]);

      const remaining = await ws.listSteps();
      assert.deepEqual(
        remaining.map((s) => s.type).sort(),
        ["anonymous", "other"],
      );
    });

    it("returns an empty array when no steps match", async () => {
      await ws.publishStep("anon", "// a");
      const deleted = await ws.deleteStepsByPublisher("nobody");
      assert.deepEqual(deleted, []);
    });

    it("is a no-op when the workspace has no custom steps at all", async () => {
      const deleted = await ws.deleteStepsByPublisher("anyone");
      assert.deepEqual(deleted, []);
    });
  });

  // ── path property ──────────────────────────────────────────────────────

  describe("promotes + setParam", () => {
    const PROMOTE_YAML = `name: opt
steps:
  - id: run
    type: log
    config:
      message: "{{ params.prompt }}"
promotes:
  - from: bestPrompt
    to: target.system
    label: System prompt
params:
  prompt: hello
`;

    it("loads the promotes block on the flow", async () => {
      await ws.publishWorkflow("opt", "v1", PROMOTE_YAML);
      const flow = await ws.getWorkflow("opt");
      assert.deepEqual(flow.promotes, [
        { from: "bestPrompt", to: "target.system", label: "System prompt" },
      ]);
    });

    it("round-trips promotes through a steps/params publish", async () => {
      await ws.publishWorkflow("opt", "v1", {
        steps: [{ id: "run", type: "log", config: { message: "x" } }],
        params: { prompt: "hello" },
        promotes: [{ from: "bestPrompt", to: "target.system" }],
      });
      const flow = await ws.getWorkflow("opt");
      assert.deepEqual(flow.promotes, [{ from: "bestPrompt", to: "target.system" }]);
    });

    it("setParam overwrites one param + publishes the next version", async () => {
      await ws.publishWorkflow("target", "v1", {
        steps: [{ id: "run", type: "log", config: { message: "{{ params.system }}" } }],
        params: { system: "old prompt", other: "keep" },
      });
      const res = await ws.setParam("target", "system", "new prompt");
      assert.equal(res.version, "v2");
      assert.equal(res.before, "old prompt");
      assert.equal(res.after, "new prompt");

      const flow = await ws.getWorkflow("target");
      assert.equal(flow.params?.["system"], "new prompt");
      // Other params survive.
      assert.equal(flow.params?.["other"], "keep");
      // New version is active.
      const meta = JSON.parse(
        await readFile(join(tempDir, "workflows", "target", "_metadata.json"), "utf-8"),
      );
      assert.equal(meta.active, "v2");
    });

    it("setParam preserves the promotes block across versions", async () => {
      await ws.publishWorkflow("target", "v1", PROMOTE_YAML.replace("name: opt", "name: target"));
      await ws.setParam("target", "prompt", "world");
      const flow = await ws.getWorkflow("target");
      assert.equal(flow.params?.["prompt"], "world");
      assert.deepEqual(flow.promotes, [
        { from: "bestPrompt", to: "target.system", label: "System prompt" },
      ]);
    });

    it("setParam throws on an unknown workflow", async () => {
      await assert.rejects(() => ws.setParam("nope", "x", 1), /not found/);
    });
  });

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
