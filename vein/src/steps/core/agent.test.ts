import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { coreRegistry } from "../registry.js";
import { defineStep, type StepContext, type StepRegistry } from "../../core.js";
import agent, { repoTree, textEdit, buildRegistryTools } from "./agent.js";

// These tests are OFFLINE: they exercise registration, the input schema, and the
// config-validation guards in run() that fire BEFORE any model call. The actual
// generation loop needs a provider key + network and is covered by lab smokes.

describe("core agent step", () => {
  it("is registered in the core registry", () => {
    const reg = coreRegistry();
    assert.ok(reg["agent"], "agent should be a core step");
    assert.equal(reg["agent"]!.type, "agent");
  });

  it("applies input defaults (maxSteps, toolFilter)", () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "you are a tester",
      prompt: "do the thing",
    });
    assert.equal(cfg.maxSteps, 40);
    assert.deepEqual(cfg.toolFilter, []);
  });

  it("rejects setting BOTH schema and finalAnswer", async () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "s",
      prompt: "p",
      finalAnswer: "return the answer",
      schema: { type: "object", properties: {} },
    });
    await assert.rejects(() => (agent.run as any)(cfg, {}), /EITHER `schema`.*OR `finalAnswer`/);
  });

  it("rejects an unknown provider", async () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "s",
      prompt: "p",
      provider: "not-a-provider",
    });
    await assert.rejects(() => (agent.run as any)(cfg, {}), /Unknown LLM provider/);
  });
});

describe("agentTools (buildRegistryTools — tools are steps)", () => {
  // A fake `tool()` factory: identity, so we can inspect the produced tool def
  // (description/inputSchema/execute) without importing the AI SDK.
  const fakeTool = (def: any) => def;

  const echoStep = defineStep({
    type: "demo/echo",
    description: "Echo the message back.",
    input: z.object({ msg: z.string() }),
    output: z.string(),
    async run(cfg) {
      return `got:${cfg.msg}`;
    },
  });
  const registry = { "demo/echo": echoStep } as StepRegistry;

  it("builds a tool per registry step, sanitizing the slash in the tool name", () => {
    const tools = buildRegistryTools(["demo/echo"], registry, undefined, fakeTool);
    assert.ok(tools["demo_echo"], "slash sanitized to underscore");
    assert.equal((tools["demo_echo"] as any).description, "Echo the message back.");
    assert.equal((tools["demo_echo"] as any).inputSchema, echoStep.input);
  });

  it("skips unknown step types (no throw)", () => {
    const tools = buildRegistryTools(["nope/missing", "demo/echo"], registry, undefined, fakeTool);
    assert.deepEqual(Object.keys(tools), ["demo_echo"]);
  });

  it("returns {} with no names or no registry", () => {
    assert.deepEqual(buildRegistryTools([], registry, undefined, fakeTool), {});
    assert.deepEqual(buildRegistryTools(["demo/echo"], undefined, undefined, fakeTool), {});
  });

  it("executes the step and emits nested step.start/step.end events", async () => {
    const events: any[] = [];
    const ctx = {
      runId: "r1",
      path: "wf/diagnose",
      scope: {},
      input: undefined,
      emit: async (e: any) => { events.push(e); },
      services: undefined,
      registry,
    } as unknown as StepContext;

    const tools = buildRegistryTools(["demo/echo"], registry, ctx, fakeTool);
    const out = await (tools["demo_echo"] as any).execute({ msg: "hi" });

    assert.equal(out, "got:hi");
    assert.equal(events.length, 2);
    assert.equal(events[0].type, "step.start");
    assert.equal(events[0].path, "wf/diagnose/001-demo_echo");
    assert.equal(events[0].stepType, "demo/echo");
    assert.deepEqual(events[0].input, { msg: "hi" });
    assert.equal(events[1].type, "step.end");
    assert.equal(events[1].path, "wf/diagnose/001-demo_echo");
    assert.equal(events[1].output, "got:hi");
  });

  it("returns an Error string (not throw) on invalid tool input", async () => {
    const tools = buildRegistryTools(["demo/echo"], registry, undefined, fakeTool);
    const out = await (tools["demo_echo"] as any).execute({ wrong: 1 });
    assert.match(String(out), /Error: invalid input for "demo\/echo"/);
  });
});

describe("repo_overview adaptive tree (repoTree)", () => {
  it("always shows every root entry and collapses noise dirs", () => {
    const files = [
      "package.json",
      "src/index.ts",
      "src/lib/a.ts",
      "prisma/migrations/0001_init/migration.sql",
      "prisma/migrations/0002_next/migration.sql",
      "node_modules/foo/index.js",
      "dist/bundle.js",
    ];
    const { text } = repoTree(files, { maxLines: 1000, maxDepth: 8 });
    // root entries present
    assert.ok(text.includes("package.json"), "root file shown");
    assert.ok(text.includes("src/"), "root dir shown");
    // noise dirs are shown but NOT expanded
    assert.ok(text.includes("migrations/"), "migrations dir shown");
    assert.ok(!text.includes("migration.sql"), "migration files collapsed away");
    assert.ok(text.includes("node_modules/") && !text.includes("foo"), "node_modules collapsed");
    assert.ok(text.includes("dist/") && !text.includes("bundle.js"), "dist collapsed");
  });

  it("deepens while under the line budget and steps back when it busts", () => {
    // 50 top-level dirs, each with a nested file: depth 1 = 50 lines, depth 2 = 100.
    const files: string[] = [];
    for (let i = 0; i < 50; i++) files.push(`dir${i}/sub/file${i}.ts`);

    const tight = repoTree(files, { maxLines: 60, maxDepth: 8 });
    assert.equal(tight.depth, 1, "depth-2 (100 lines) busts a 60-line budget → stay at depth 1");

    const roomy = repoTree(files, { maxLines: 1000, maxDepth: 8 });
    assert.ok(roomy.depth >= 2, "with room, it deepens past the root");
  });

  it("respects the hard depth cap", () => {
    const files = ["a/b/c/d/e/f/g/h/i/j/deep.ts"];
    const { depth } = repoTree(files, { maxLines: 10000, maxDepth: 3 });
    assert.equal(depth, 3, "never deeper than maxDepth even with budget to spare");
  });
});

describe("textEdit (str_replace_based_edit_tool handler)", () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "vein-textedit-"));
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it("views a file with 1-indexed line numbers", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo\nthree");
    const out = textEdit({ command: "view", path: "a.txt" }, cwd);
    assert.equal(out, "1: one\n2: two\n3: three");
  });

  it("views a line range", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo\nthree\nfour");
    assert.equal(textEdit({ command: "view", path: "a.txt", view_range: [2, 3] }, cwd), "2: two\n3: three");
    assert.equal(textEdit({ command: "view", path: "a.txt", view_range: [3, -1] }, cwd), "3: three\n4: four");
  });

  it("lists a directory on view", () => {
    writeFileSync(join(cwd, "z.txt"), "");
    writeFileSync(join(cwd, "a.txt"), "");
    const out = textEdit({ command: "view", path: "." }, cwd);
    assert.equal(out, "a.txt\nz.txt");
  });

  it("creates a new file (including nested dirs)", () => {
    const out = textEdit({ command: "create", path: "sub/dir/new.txt", file_text: "hi" }, cwd);
    assert.match(out, /Successfully created/);
    assert.equal(readFileSync(join(cwd, "sub/dir/new.txt"), "utf-8"), "hi");
  });

  it("str_replace replaces exactly one match", () => {
    writeFileSync(join(cwd, "a.txt"), "foo bar baz");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "bar", new_str: "QUX" }, cwd);
    assert.match(out, /Successfully replaced/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "foo QUX baz");
  });

  it("str_replace refuses zero matches", () => {
    writeFileSync(join(cwd, "a.txt"), "foo");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "nope", new_str: "x" }, cwd);
    assert.match(out, /No match found/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "foo");
  });

  it("str_replace refuses multiple matches", () => {
    writeFileSync(join(cwd, "a.txt"), "x x x");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "x", new_str: "y" }, cwd);
    assert.match(out, /Found 3 matches/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "x x x");
  });

  it("inserts text after a line (0 = top of file)", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo");
    textEdit({ command: "insert", path: "a.txt", insert_line: 0, insert_text: "ZERO" }, cwd);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "ZERO\none\ntwo");
  });

  it("refuses paths that escape the working dir", () => {
    writeFileSync(join(cwd, "a.txt"), "secret");
    const out = textEdit({ command: "view", path: "../../../etc/passwd" }, cwd);
    assert.match(out, /escapes the working directory/);
  });

  it("returns File not found for missing files", () => {
    assert.match(textEdit({ command: "view", path: "nope.txt" }, cwd), /File not found/);
    assert.ok(!existsSync(join(cwd, "nope.txt")));
  });
});
