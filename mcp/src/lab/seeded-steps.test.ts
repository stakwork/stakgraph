import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WorkspaceManager, buildRegistry } from "strut";
import { seedEvalSteps } from "./eval/seed.js";
import { seedConceptSteps } from "./concepts/seed.js";
import { seedGitseeSteps } from "./gitsee/seed.js";
import { seedJarvisSteps } from "./jarvis/seed.js";
import { seedSheetsSteps } from "./sheets/seed.js";
import { seedHarveySteps } from "./harvey/seed.js";
import { seedGaiaSteps } from "./gaia/seed.js";
import { seedArtifactSteps } from "./artifacts/seed.js";
import { seedWfbenchSteps } from "./wfbench/seed.js";

// The mcp dir, like the real lab-workspace: a seeded step's bare package
// imports (`ai`, `aieo`, `js-yaml`, …) resolve via mcp/node_modules from there.
const MCP_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

describe("seeded lab steps", () => {
  it("every seeded step loads from the workspace it was materialized into", async () => {
    const root = await mkdtemp(join(MCP_DIR, ".lab-seeded-steps-"));
    try {
      const ws = new WorkspaceManager(root);
      for (const seed of [
        seedEvalSteps,
        seedConceptSteps,
        seedGitseeSteps,
        seedJarvisSteps,
        seedSheetsSteps,
        seedHarveySteps,
        seedGaiaSteps,
        seedArtifactSteps,
        seedWfbenchSteps,
      ]) {
        await seed(ws);
      }
      const seeded = (await ws.listSteps()).map((s) => s.type);
      const { registry } = await buildRegistry(await ws.materializeCustomSteps());

      // A step that imports mcp source by relative path (e.g. `../../cost.js`)
      // is skipped by discovery with only a warning.
      assert.deepEqual(seeded.filter((t) => !registry[t]), []);
      for (const t of ["eval/reflect", "gitsee/boot-and-exercise", "gitsee/score-setup", "gitsee/verify-setup"]) {
        assert.ok(registry[t], `registry missing ${t}`);
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
