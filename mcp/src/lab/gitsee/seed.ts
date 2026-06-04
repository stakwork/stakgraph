import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Workflow + step templates for the `gitsee` experiment (self-contained port of
 * gitsee "services" mode). Reconciled into the workspace by content hash on boot
 * — see concepts/seed.ts for the reconciliation contract (unchanged → no-op,
 * edited → new active version, prior versions archived).
 *
 * Unlike concepts (whose steps reach runtime objects via `ctx.services`), these
 * steps are FULLY self-contained: they import only `vein`, the third-party AI
 * SDK, and Node builtins. So there's no services bag to merge here.
 */

const SEED_WORKFLOWS = [
  "gitsee-explore-services",
  // Eval/optimize stack (wires the generic eval/* steps with the gitsee
  // rubric / task / dataset), mirroring the concepts-* workflows.
  "gitsee-eval", // harness: produce setup files → score
  "gitsee-eval-score", // eval/score + setup-facts rubric
  "gitsee-eval-reflect", // eval/reflect + setup task/guidance
  "gitsee-optimize", // eval/optimize loop, wired to the above
];

const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "clone-workspace.ts", type: "gitsee/clone-workspace" },
  { file: "explore-services.ts", type: "gitsee/explore-services" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedGitseeWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml);
      if (changed) console.log(`[gitsee] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(
        `[gitsee] could not seed workflow "${name}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}

export async function seedGitseeSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "gitsee-seed");
      if (changed) console.log(`[gitsee] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[gitsee] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
