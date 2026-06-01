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
import { MemoryRunStore } from "./store.js";
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
