import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { coreRegistry } from "../registry.js";
import agent, { repoTree } from "./agent.js";

// These tests are OFFLINE: they exercise registration, the input schema, and the
// config-validation guards in run() that fire BEFORE any model call. The actual
// generation loop needs a provider key + network and is covered by lab smokes.

describe("core agent step", () => {
  it("is registered in the core registry", () => {
    const reg = coreRegistry();
    assert.ok(reg["agent"], "agent should be a core step");
    assert.equal(reg["agent"]!.type, "agent");
  });

  it("applies input defaults (fileLines, maxSteps, toolFilter)", () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "you are a tester",
      prompt: "do the thing",
    });
    assert.equal(cfg.fileLines, 100);
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
