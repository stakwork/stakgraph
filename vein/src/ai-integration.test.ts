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

  it("run-history tools degrade gracefully on a non-readable store", async () => {
    const registry = await createRegistry([]);
    const deps = {
      workspace: new WorkspaceManager(tempDir),
      registry,
      store: new MemoryRunStore(),
      getRegistry: async () => registry,
    };
    const tools = buildTools(deps) as any;
    const list = await tools.list_runs.execute({ name: "x" });
    assert.ok(list.error && /unavailable/.test(list.error));
    const get = await tools.get_run.execute({ name: "x", runId: "1" });
    assert.ok(get.error && /unavailable/.test(get.error));
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
