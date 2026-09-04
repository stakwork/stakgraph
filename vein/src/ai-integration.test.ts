import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { defineStep } from "./core.js";
import { createRegistry } from "./steps/registry.js";
import { WorkspaceManager } from "./workspace.js";
import { MemoryRunStore, FileRunStore } from "./store.js";
import { MemorySecretStore } from "./secret-store.js";
import { lsSteps, searchSteps, readStepSource } from "./ai/stepHelpers.js";
import { buildSystem } from "./ai/prompts.js";
import { buildTools } from "./ai/tools.js";
import { zodToFields } from "./ai/schemaHelpers.js";

/**
 * End-to-end verification that the AI workflow-builder tools can see
 * steps that were registered in code via `createRegistry([...])`.
 * Without this wiring, in-code consumers' steps would be invisible to
 * the chat agent even though execution worked fine.
 */

describe("AI tools see in-code registered steps", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-integration-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  async function setup() {
    const myThing = defineStep({
      type: "do-thing",
      description: "Does a custom thing in-memory",
      input: z.object({ x: z.number() }),
      output: z.string(),
      async run(cfg) {
        return `did ${cfg.x}`;
      },
    });

    const gitreeSave = defineStep({
      type: "gitree/store-feature",
      description: "Stores a feature description in gitree",
      input: z.object({ feature: z.string() }),
      output: z.any(),
      async run(cfg) {
        return { saved: cfg.feature };
      },
    });

    const registry = await createRegistry([myThing, gitreeSave]);

    return {
      myThing,
      gitreeSave,
      deps: {
        workspace: new WorkspaceManager(tempDir),
        registry,
        store: new MemoryRunStore(),
        getRegistry: async () => registry,
      },
    };
  }

  it("list_steps('steps/custom') includes in-code registry entries", async () => {
    const { deps } = await setup();
    const result = await lsSteps("steps/custom", deps);
    const entries = (result as { entries?: string[] }).entries ?? [];

    // Should include both the flat and the namespaced step.
    const matchThing = entries.find((e) => e.startsWith("do-thing"));
    const matchGitree = entries.find((e) =>
      e.startsWith("gitree/store-feature"),
    );

    assert.ok(matchThing, `expected "do-thing" in custom listing, got: ${JSON.stringify(entries)}`);
    assert.ok(matchGitree, `expected "gitree/store-feature" in custom listing, got: ${JSON.stringify(entries)}`);
    // Descriptions are included.
    assert.ok(matchThing.includes("Does a custom thing"));
    assert.ok(matchGitree.includes("Stores a feature"));
  });

  it("search_steps finds in-code steps by name and description", async () => {
    const { deps } = await setup();

    const byName = await searchSteps("thing", deps);
    const names = byName.matches.map((m) => m.type);
    assert.ok(names.includes("do-thing"), `expected do-thing in matches: ${JSON.stringify(names)}`);

    const byDesc = await searchSteps("gitree feature", deps);
    const names2 = byDesc.matches.map((m) => m.type);
    assert.ok(
      names2.includes("gitree/store-feature"),
      `expected gitree/store-feature in matches: ${JSON.stringify(names2)}`,
    );
  });

  it("get_step (registry lookup + zodToFields) returns schema for in-code steps", async () => {
    const { deps } = await setup();
    const def = deps.registry["do-thing"];
    assert.ok(def, "step should be in registry");

    const fields = zodToFields(def.input);
    assert.deepEqual(fields, [
      { name: "x", kind: "number", required: true, default: undefined },
    ]);
    assert.equal(def.description, "Does a custom thing in-memory");

    // Source is undefined for in-code steps (no file on disk) — this is
    // expected and OK; the model gets the schema and description.
    const source = await readStepSource("do-thing", deps);
    assert.equal(source, undefined);
  });

  it("get_step works for namespaced in-code steps", async () => {
    const { deps } = await setup();
    const def = deps.registry["gitree/store-feature"];
    assert.ok(def, "namespaced step should be in registry");

    const fields = zodToFields(def.input);
    assert.deepEqual(fields, [
      { name: "feature", kind: "string", required: true, default: undefined },
    ]);
  });

  it("the system prompt's 'Available steps' tree lists in-code steps under custom/", async () => {
    const { deps } = await setup();
    const system = await buildSystem(deps);
    const tree = system.slice(system.indexOf("Available steps:"));

    // Tree groups: steps/ → core/ + lib/ + custom/. Our two should appear
    // under custom/.
    assert.ok(tree.includes("custom/"), "tree should have a custom/ section");
    assert.ok(
      tree.includes("do-thing"),
      `expected 'do-thing' in the seeded tree:\n${tree}`,
    );
    assert.ok(
      tree.includes("gitree/store-feature"),
      `expected 'gitree/store-feature' in the seeded tree:\n${tree}`,
    );
  });

  it("does NOT duplicate steps that are both on disk AND in the registry", async () => {
    // If someone publishes a step to the workspace AND has the same
    // name in their in-code registry (unusual but possible), it should
    // appear once.
    const ws = new WorkspaceManager(tempDir);
    await ws.publishStep(
      "shared",
      `import { z } from "zod";
       import { defineStep } from "vein";
       export default defineStep({
         type: "shared",
         input: z.object({}),
         output: z.any(),
         async run() { return null; },
       });`,
      "on-disk version",
    );

    const inCode = defineStep({
      type: "shared",
      description: "in-code version",
      input: z.object({}),
      output: z.any(),
      async run() {
        return null;
      },
    });

    const registry = await createRegistry([inCode]);
    const deps = {
      workspace: ws,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
    };

    const result = await lsSteps("steps/custom", deps);
    const entries = (result as { entries?: string[] }).entries ?? [];
    const matches = entries.filter((e) => e.startsWith("shared"));
    assert.equal(
      matches.length,
      1,
      `expected exactly one 'shared' entry, got: ${JSON.stringify(matches)}`,
    );
  });
});

describe("AI create_step / edit_step tools", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-steps-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  function makeDeps(opts: { publishingEnabled?: boolean } = {}) {
    const ws = new WorkspaceManager(tempDir);
    return {
      workspace: ws,
      registry: {} as any,
      store: new MemoryRunStore(),
      // Static getRegistry — the temp workspace lives outside the project tree
      // so a published .ts can't resolve `vein` to actually load. These tests
      // verify publish/version semantics, not registry loading.
      getRegistry: async () => ({} as any),
      ...opts,
    };
  }

  const code = (n: number) =>
    `import { z, defineStep } from "vein";\nexport default defineStep({ type: "my/step", input: z.object({}), output: z.any(), async run(){ return ${n}; } });\n`;

  it("create_step publishes a new step at v1; edit_step bumps to v2", async () => {
    const deps = makeDeps();
    const tools = buildTools(deps) as any;

    const created = await tools.create_step.execute({ name: "my/step", code: code(1) });
    assert.equal(created.ok, true);
    assert.equal(created.version, "v1");

    const edited = await tools.edit_step.execute({ type: "my/step", code: code(2) });
    assert.equal(edited.ok, true);
    assert.equal(edited.version, "v2");
    assert.equal(edited.changed, true);

    const { active, versions } = await deps.workspace.listStepVersions("my/step");
    assert.equal(active, "v2");
    assert.deepEqual(versions, ["v1", "v2"]);
  });

  it("set_active_version rolls a custom step back and refreshes the registry", async () => {
    const deps = makeDeps();
    let refreshed = 0;
    deps.getRegistry = async () => { refreshed++; return {} as any; };
    const tools = buildTools(deps) as any;
    await tools.create_step.execute({ name: "my/step", code: code(1) });
    await tools.edit_step.execute({ type: "my/step", code: code(2) });
    const before = refreshed;

    const res = await tools.set_active_version.execute({ kind: "step", name: "my/step", version: "v1" });
    assert.deepEqual(res, { ok: true, kind: "step", name: "my/step", active: "v1" });
    assert.equal(refreshed, before + 1);
    const { active, versions } = await deps.workspace.listStepVersions("my/step");
    assert.equal(active, "v1");
    assert.deepEqual(versions, ["v1", "v2"]);

    const bad = await tools.set_active_version.execute({ kind: "step", name: "my/step", version: "v7" });
    assert.match(bad.error, /Version "v7" not found/);
  });

  it("set_active_version refuses step rollback when publishing is disabled", async () => {
    const tools = buildTools(makeDeps({ publishingEnabled: false })) as any;
    const res = await tools.set_active_version.execute({ kind: "step", name: "my/step", version: "v1" });
    assert.match(res.error, /disabled/);
  });

  it("create_step rejects an existing step name", async () => {
    const deps = makeDeps();
    const tools = buildTools(deps) as any;
    await tools.create_step.execute({ name: "my/step", code: code(1) });
    const again = await tools.create_step.execute({ name: "my/step", code: code(2) });
    assert.ok(again.error && /already exists/.test(again.error));
  });

  it("edit_step rejects a step that does not exist", async () => {
    const deps = makeDeps();
    const tools = buildTools(deps) as any;
    const res = await tools.edit_step.execute({ type: "nope/missing", code: code(1) });
    assert.ok(res.error && /not found/.test(res.error));
  });

  it("edit_step is a no-op (changed:false) for identical content", async () => {
    const deps = makeDeps();
    const tools = buildTools(deps) as any;
    await tools.create_step.execute({ name: "my/step", code: code(1) });
    const same = await tools.edit_step.execute({ type: "my/step", code: code(1) });
    assert.equal(same.changed, false);
    assert.equal(same.version, "v1");
  });

  it("both tools refuse when publishing is disabled", async () => {
    const deps = makeDeps({ publishingEnabled: false });
    const tools = buildTools(deps) as any;
    const c = await tools.create_step.execute({ name: "my/step", code: code(1) });
    assert.ok(c.error && /disabled/.test(c.error));
    const e = await tools.edit_step.execute({ type: "my/step", code: code(1) });
    assert.ok(e.error && /disabled/.test(e.error));
  });
});

describe("AI list_workflows / get_workflow tools", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-wf-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  function makeDeps() {
    const ws = new WorkspaceManager(tempDir);
    return {
      ws,
      deps: {
        workspace: ws,
        registry: {} as any,
        store: new MemoryRunStore(),
        getRegistry: async () => ({} as any),
      },
    };
  }

  const wfYaml = (name: string) =>
    `name: ${name}\nsteps:\n  - id: hello\n    type: log\n    config:\n      message: hi\n`;

  it("list_workflows returns published workflows with versions", async () => {
    const { ws, deps } = makeDeps();
    await ws.createWorkflow("alpha", wfYaml("alpha"), "first one");
    await ws.createWorkflow("beta", wfYaml("beta"));
    const tools = buildTools(deps) as any;

    const { workflows } = await tools.list_workflows.execute({});
    const names = workflows.map((w: any) => w.name).sort();
    assert.deepEqual(names, ["alpha", "beta"]);
    const alpha = workflows.find((w: any) => w.name === "alpha");
    assert.equal(alpha.activeVersion, "v1");
    assert.deepEqual(alpha.versions, ["v1"]);
    assert.equal(alpha.description, "first one");
  });

  it("get_workflow returns the active version's YAML + metadata", async () => {
    const { ws, deps } = makeDeps();
    await ws.createWorkflow("alpha", wfYaml("alpha"), "first one");
    const tools = buildTools(deps) as any;

    const res = await tools.get_workflow.execute({ name: "alpha" });
    assert.equal(res.name, "alpha");
    assert.equal(res.version, "v1");
    assert.equal(res.activeVersion, "v1");
    assert.ok(res.yaml.includes("type: log"));
  });

  it("create_workflow / edit_workflow refuse invalid YAML with a readable error; warnings ride along", async () => {
    const { ws, deps } = makeDeps();
    const echo = defineStep({ type: "echo", input: z.object({ message: z.string() }), output: z.any(), async run(c) { return c; } });
    const registry = await createRegistry([echo]);
    const tools = buildTools({ ...deps, registry, getRegistry: async () => registry }) as any;

    // Cycle + unknown type → refused, nothing written.
    const bad = await tools.create_workflow.execute({
      name: "broken",
      yaml: "name: broken\nsteps:\n  - id: a\n    type: echo\n    config: { message: x }\n    depends: b\n  - id: b\n    type: nope\n    config: {}\n    depends: a\n",
    });
    assert.match(bad.error, /^Not published: the workflow YAML has 2 validation errors \(nothing was written; no version was created\)/);
    assert.match(bad.error, /steps\[1\]\.type: Unknown step type "nope"/);
    assert.match(bad.error, /Dependency cycle/);
    assert.match(bad.error, /validate_workflow re-checks without publishing/);
    assert.equal(bad.validation.ok, false);
    assert.deepEqual(await ws.listWorkflows(), []);

    // Missing `name:` in YAML is fine — the tool's name is stamped in.
    // An unknown config field is a warning: published, warning returned.
    const ok = await tools.create_workflow.execute({
      name: "fine",
      yaml: "steps:\n  - id: a\n    type: echo\n    config: { message: x, extra: 1 }\n",
    });
    assert.equal(ok.ok, true);
    assert.equal(ok.version, "v1");
    assert.equal(ok.warnings.length, 1);
    assert.match(ok.warnings[0].message, /Unknown config field "extra"/);

    // edit_workflow: same gate; version stays v1 on refusal.
    const edit = await tools.edit_workflow.execute({
      name: "fine",
      yaml: "name: fine\nsteps:\n  - id: a\n    type: echo\n    config: { message: \"{{ nope.x }}\" }\n",
    });
    assert.match(edit.error, /^Not published: .*1 validation error /);
    assert.match(edit.error, /unknown root "nope"/);
    assert.equal((await tools.get_workflow.execute({ name: "fine" })).activeVersion, "v1");

    const clean = await tools.edit_workflow.execute({
      name: "fine",
      yaml: "name: fine\nsteps:\n  - id: a\n    type: echo\n    config: { message: y }\n",
    });
    assert.equal(clean.ok, true);
    assert.equal(clean.version, "v2");
    assert.equal("warnings" in clean, false);
  });

  it("set_active_version rolls a workflow back without publishing", async () => {
    const { ws, deps } = makeDeps();
    await ws.createWorkflow("alpha", wfYaml("alpha"));
    await ws.publishWorkflowByContent("alpha", wfYaml("alpha") + "  - id: more\n    type: log\n    config:\n      message: v2\n");
    const tools = buildTools(deps) as any;
    assert.equal((await tools.get_workflow.execute({ name: "alpha" })).activeVersion, "v2");

    const res = await tools.set_active_version.execute({ kind: "workflow", name: "alpha", version: "v1" });
    assert.deepEqual(res, { ok: true, kind: "workflow", name: "alpha", active: "v1" });
    const after = await tools.get_workflow.execute({ name: "alpha" });
    assert.equal(after.activeVersion, "v1");
    assert.deepEqual(after.versions, ["v1", "v2"]); // history kept
    assert.ok(!after.yaml.includes("message: v2"));

    const bad = await tools.set_active_version.execute({ kind: "workflow", name: "alpha", version: "v9" });
    assert.match(bad.error, /Version "v9" not found/);
    const missing = await tools.set_active_version.execute({ kind: "workflow", name: "nope", version: "v1" });
    assert.match(missing.error, /not found/);
  });

  it("get_workflow errors on unknown workflow and unknown version", async () => {
    const { ws, deps } = makeDeps();
    await ws.createWorkflow("alpha", wfYaml("alpha"));
    const tools = buildTools(deps) as any;

    const missing = await tools.get_workflow.execute({ name: "nope" });
    assert.ok(missing.error && /not found/.test(missing.error));

    const badVer = await tools.get_workflow.execute({ name: "alpha", version: "v9" });
    assert.ok(badVer.error && /v9/.test(badVer.error));
  });
});

describe("AI list_runs / get_run tools", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-runs-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("list_runs + get_run surface run history and events from a FileRunStore", async () => {
    const echo = defineStep({
      type: "echo",
      input: z.object({ msg: z.string() }),
      output: z.any(),
      async run(cfg: any) {
        return { echoed: cfg.msg };
      },
    });
    const registry = await createRegistry([echo]);
    const ws = new WorkspaceManager(tempDir);
    const store = new FileRunStore(tempDir);
    const deps = {
      workspace: ws,
      registry,
      store,
      getRegistry: async () => registry,
    };

    await ws.createWorkflow(
      "greeter",
      `name: greeter\nsteps:\n  - id: say\n    type: echo\n    config:\n      msg: hello\n`,
    );
    const tools = buildTools(deps) as any;

    const run = await tools.run_workflow.execute({ name: "greeter", input: {} });
    assert.equal(run.status, "success");

    const { runs } = await tools.list_runs.execute({ name: "greeter" });
    assert.equal(runs.length, 1);
    assert.equal(runs[0].runId, run.runId);
    assert.equal(runs[0].status, "success");

    // Slimmed events by default (no payloads).
    const slim = await tools.get_run.execute({ name: "greeter", runId: run.runId });
    assert.equal(slim.summary.status, "success");
    assert.ok(slim.events.length > 0);
    assert.ok(slim.events.every((e: any) => !("input" in e) && !("output" in e)));

    // Full events include payloads.
    const full = await tools.get_run.execute({
      name: "greeter",
      runId: run.runId,
      fullEvents: true,
    });
    assert.ok(full.events.some((e: any) => e.output !== undefined));
  });

  it("get_run errors for an unknown run id", async () => {
    const registry = await createRegistry([]);
    const deps = {
      workspace: new WorkspaceManager(tempDir),
      registry,
      store: new FileRunStore(tempDir),
      getRegistry: async () => registry,
    };
    const tools = buildTools(deps) as any;
    const res = await tools.get_run.execute({ name: "ghost", runId: "123" });
    assert.ok(res.error && /not found/.test(res.error));
  });

  it("run-history tools read from a MemoryRunStore like any other", async () => {
    const registry = await createRegistry([]);
    const store = new MemoryRunStore();
    await store.append("x", "1", { ts: new Date().toISOString(), runId: "1", path: "x", type: "run.start" });
    const deps = {
      workspace: new WorkspaceManager(tempDir),
      registry,
      store,
      getRegistry: async () => registry,
    };
    const tools = buildTools(deps) as any;
    const list = await tools.list_runs.execute({ name: "x", limit: 20 });
    assert.deepEqual(list.runs.map((r: { runId: string }) => r.runId), ["1"]);
    const get = await tools.get_run.execute({ name: "x", runId: "1", fullEvents: false });
    assert.equal(get.runId, "1");
    assert.equal(get.events.length, 1);
    const missing = await tools.get_run.execute({ name: "x", runId: "2", fullEvents: false });
    assert.ok(missing.error && /not found/.test(missing.error));
  });
});

// ── run_workflow coerces a stringified input ────────────────────────────────

describe("AI run_workflow tool: stringified input coercion", () => {
  let tempDir: string;
  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-runwf-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });
  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("parses an input passed as a JSON string so {{ input.* }} resolves", async () => {
    const echo = defineStep({
      type: "echo",
      input: z.any(),
      output: z.any(),
      async run(cfg) {
        return cfg;
      },
    });
    const registry = await createRegistry([echo]);
    const ws = new WorkspaceManager(tempDir);
    await ws.publishWorkflow("echo-wf", "v1", {
      steps: [
        { id: "a", type: "echo", config: { owner: "{{ input.owner }}", pull_number: "{{ input.pull_number }}" } },
      ],
    });

    const deps = {
      workspace: ws,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
    };
    const tools = buildTools(deps) as any;

    // The model passes input as a JSON STRING (the exact bug from the logs).
    const res = await tools.run_workflow.execute({
      name: "echo-wf",
      input: '{ "owner": "vercel", "pull_number": 1234 }',
    });

    assert.equal(res.status, "success");
    assert.equal(res.output.owner, "vercel");
    assert.equal(res.output.pull_number, 1234);
    assert.equal(typeof res.output.pull_number, "number");
  });
});

// ── list_secrets tool ────────────────────────────────────────────────────────

describe("AI list_secrets tool", () => {
  function makeDeps(secrets?: MemorySecretStore) {
    const registry = {} as any;
    return {
      workspace: {} as any,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
      secrets,
    };
  }

  it("returns secret NAMES + updatedAt, never values", async () => {
    const store = new MemorySecretStore();
    await store.set("GITHUB_TOKEN", "ghp_superSecret");
    await store.set("GOOGLE_SERVICE_ACCOUNT_JSON", '{"private_key":"xyz"}');

    const tools = buildTools(makeDeps(store)) as any;
    const res = await tools.list_secrets.execute({});

    assert.deepEqual(
      res.secrets.map((s: { name: string }) => s.name).sort(),
      ["GITHUB_TOKEN", "GOOGLE_SERVICE_ACCOUNT_JSON"],
    );
    // The agent must never see values.
    const blob = JSON.stringify(res);
    assert.ok(!blob.includes("ghp_superSecret"));
    assert.ok(!blob.includes("private_key"));
    assert.ok(res.secrets.every((s: { updatedAt?: string }) => typeof s.updatedAt === "string"));
  });

  it("degrades gracefully when no secret store is wired", async () => {
    const tools = buildTools(makeDeps(undefined)) as any;
    const res = await tools.list_secrets.execute({});
    assert.ok(res.error && /not available/.test(res.error));
  });
});

describe("run_workflow dispatch mode (auto-detach)", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-ai-detach-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  async function setup(sleepMs: number) {
    const sleeper = defineStep({
      type: "sleeper",
      description: "sleeps then returns",
      input: z.object({ ms: z.number().default(0) }),
      output: z.any(),
      async run(cfg) {
        await new Promise((r) => setTimeout(r, cfg.ms));
        return { slept: cfg.ms };
      },
    });
    const registry = await createRegistry([sleeper]);
    const workspace = new WorkspaceManager(tempDir);
    await workspace.createWorkflow("nap", {
      steps: [{ id: "s", type: "sleeper", config: { ms: sleepMs } }],
    });
    return { workspace, registry };
  }

  it("fast runs return synchronously, exactly as without the detach seam", async () => {
    const { workspace, registry } = await setup(0);
    const detached: unknown[] = [];
    const tools = buildTools({
      workspace,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
      detach: { waitMs: 2000, onDetach: (info: unknown) => detached.push(info) },
    } as any) as any;

    const res = await tools.run_workflow.execute({ name: "nap", input: {} });
    assert.equal(res.status, "success");
    assert.equal(typeof res.runId, "string");
    assert.deepEqual(res.output, { slept: 0 });
    assert.equal(detached.length, 0, "fast run must not detach");
  });

  it("a run outliving the wait window returns a detached stub and hands the promise to onDetach", async () => {
    const { workspace, registry } = await setup(300);
    const detached: any[] = [];
    const tools = buildTools({
      workspace,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
      detach: { waitMs: 40, onDetach: (info: any) => detached.push(info) },
    } as any) as any;

    const stub = await tools.run_workflow.execute({ name: "nap", input: {} });
    assert.equal(stub.status, "running");
    assert.equal(stub.detached, true);
    assert.equal(stub.workflow, "nap");
    assert.equal(typeof stub.runId, "string");
    assert.ok(/run-notification/.test(stub.note), "stub teaches the wake contract");

    assert.equal(detached.length, 1);
    assert.equal(detached[0].workflow, "nap");
    assert.equal(detached[0].runId, stub.runId);
    assert.equal(typeof detached[0].startedAt, "number");

    // The handed-off promise settles with the real result, same runId.
    const result = await detached[0].promise;
    assert.equal(result.status, "success");
    assert.equal(result.runId, stub.runId);
    assert.deepEqual(result.output, { slept: 300 });
  });

  it("without the detach seam, even slow runs are awaited to completion", async () => {
    const { workspace, registry } = await setup(150);
    const tools = buildTools({
      workspace,
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
    } as any) as any;

    const res = await tools.run_workflow.execute({ name: "nap", input: {} });
    assert.equal(res.status, "success");
    assert.deepEqual(res.output, { slept: 150 });
  });
});

describe("run control tools", () => {
  function baseDeps() {
    return {
      workspace: new WorkspaceManager("/nonexistent-run-control-test"),
      registry: {} as any,
      store: new MemoryRunStore(),
      getRegistry: async () => ({} as any),
    };
  }

  it("are absent unless the host wires controlRun", () => {
    const tools = buildTools(baseDeps());
    for (const t of ["cancel_run", "pause_run", "resume_run"]) assert.ok(!(t in tools), t);
  });

  it("relay (workflow, runId, action) to controlRun and return its result verbatim", async () => {
    const calls: unknown[] = [];
    const controlRun = async (workflow: string, runId: string, action: "cancel" | "pause" | "resume") => {
      calls.push([workflow, runId, action]);
      return action === "resume"
        ? ({ ok: false, error: "Run already terminal (success)" } as const)
        : ({ ok: true, runId, state: action === "cancel" ? "cancelling" : "pausing" } as const);
    };
    const tools = buildTools({ ...baseDeps(), controlRun }) as any;
    assert.deepEqual(await tools.cancel_run.execute({ name: "wf", runId: "r1" }), { ok: true, runId: "r1", state: "cancelling" });
    assert.deepEqual(await tools.pause_run.execute({ name: "wf", runId: "r2" }), { ok: true, runId: "r2", state: "pausing" });
    assert.deepEqual(await tools.resume_run.execute({ name: "wf", runId: "r3" }), { ok: false, error: "Run already terminal (success)" });
    assert.deepEqual(calls, [["wf", "r1", "cancel"], ["wf", "r2", "pause"], ["wf", "r3", "resume"]]);
  });
});
