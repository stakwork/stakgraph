import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { buildRegistry, CORE_STEP_TYPES } from "./registry.js";
import { WorkspaceManager } from "../workspace.js";

describe("buildRegistry", () => {
  let tempDir: string;
  let ws: WorkspaceManager;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-reg-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
    ws = new WorkspaceManager(tempDir);
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("returns a registry and a parallel sources map", async () => {
    const { registry, sources } = await buildRegistry(tempDir);

    assert.ok(typeof registry === "object" && registry !== null);
    assert.ok(typeof sources === "object" && sources !== null);
  });

  it("tags every core step with source: core", async () => {
    const { sources } = await buildRegistry(tempDir);

    for (const type of CORE_STEP_TYPES) {
      assert.equal(
        sources[type],
        "core",
        `core step "${type}" should have source: core`,
      );
    }
  });

  it("tags lib steps (shipped under src/steps/lib) with source: lib", async () => {
    const { registry, sources } = await buildRegistry(tempDir);

    // Shipped lib steps live under src/steps/lib (e.g. github/fetch-pr,
    // gdrive/export-file). If lib steps change, this test should adapt —
    // but at minimum we expect anything not in CORE_STEP_TYPES to be either
    // "lib" or "custom", and there should be at least one "lib" entry to
    // validate the tagging works.
    const libEntries = Object.entries(sources).filter(([, s]) => s === "lib");
    assert.ok(
      libEntries.length > 0,
      "expected at least one lib step to be registered",
    );

    for (const [type] of libEntries) {
      assert.ok(
        type in registry,
        `lib-tagged step "${type}" must exist in registry`,
      );
      assert.ok(
        !CORE_STEP_TYPES.includes(type),
        `lib-tagged step "${type}" should not also be a core step`,
      );
    }
  });

  // Minimal step source that satisfies the registry's duck-typed check
  // (`"type" in def && "run" in def`) without any external imports. This
  // keeps the tests independent of how Node resolves modules from the
  // temp dir.
  const minimalStep = (type: string) =>
    `export default {\n` +
    `  type: ${JSON.stringify(type)},\n` +
    `  input: { _def: { typeName: 'ZodObject', shape: () => ({}) } },\n` +
    `  output: { _def: { typeName: 'ZodAny' } },\n` +
    `  async run() { return {}; },\n` +
    `};\n`;

  it("tags user-published custom steps with source: custom", async () => {
    await ws.publishStep("my-custom", minimalStep("my-custom"));

    const { registry, sources } = await buildRegistry(tempDir);

    assert.ok(registry["my-custom"], "custom step should be in registry");
    assert.equal(sources["my-custom"], "custom");
  });

  it("tags nested custom steps with slash names as source: custom", async () => {
    // Critical regression check: the old heuristic mislabeled any step
    // with a "/" in its name as "lib". With source-at-load-time tracking,
    // a service can publish `gitree/save-feature` and it gets tagged
    // correctly as custom.
    await ws.publishStep(
      "gitree/save-feature",
      minimalStep("gitree/save-feature"),
    );

    const { registry, sources } = await buildRegistry(tempDir);

    assert.ok(
      registry["gitree/save-feature"],
      "nested custom step should be in registry",
    );
    assert.equal(
      sources["gitree/save-feature"],
      "custom",
      "nested custom steps must be tagged custom even though the name contains a slash",
    );
  });

  it("does not register helper files (_-prefixed)", async () => {
    await ws.publishStep("gitree/_shared", "export const HELPER = 1;");

    const { registry, sources } = await buildRegistry(tempDir);
    assert.equal(registry["gitree/_shared"], undefined);
    assert.equal(sources["gitree/_shared"], undefined);
  });
});
