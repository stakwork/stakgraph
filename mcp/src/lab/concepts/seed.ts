import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Workflow templates shipped with concepts. Seeded into the workspace on
 * first boot; thereafter they live in the workspace and are edited /
 * versioned through the vein UI (the workspace is the source of truth).
 */
const SEED_WORKFLOWS = [
  "process-change",
  "process-repo-chronological",
  "bootstrap-then-process",
];

/**
 * Publish the bundled workflow YAML templates into the workspace if they
 * are not already present. Idempotent — existing workflows are left
 * untouched so UI edits are never clobbered.
 */
export async function seedConceptWorkflows(
  workspace: WorkspaceManager,
): Promise<void> {
  const dir = join(dirname(fileURLToPath(import.meta.url)), "workflows");
  const existing = new Set((await workspace.listWorkflows()).map((w) => w.name));

  for (const name of SEED_WORKFLOWS) {
    if (existing.has(name)) continue;
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      await workspace.createWorkflow(name, yaml);
      console.log(`[concepts] seeded workflow: ${name}`);
    } catch (err) {
      console.warn(
        `[concepts] could not seed workflow "${name}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
