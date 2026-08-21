import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Harvey LAB verification steps — THIN plumbing only. The actual grader is
 * the in-code `harvey` service (`service.ts`), which subprocess-runs the real
 * eval from the harvey-labs checkout; editing these seeded steps can only
 * break plumbing, never change how grading works. That split is deliberate:
 * the benchmark verifier must stay outside the agent-editable surface.
 *
 * - `harvey/get-task` — instructions + documents, rubric stripped. Safe for
 *   producing agents.
 * - `harvey/evaluate` — stage this run's artifact deliverables + run the real
 *   eval. Grant ONLY to harness workflows, never to the producing agent.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "get-task.ts", type: "harvey/get-task" },
  { file: "evaluate.ts", type: "harvey/evaluate" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

const SEED_WORKFLOWS = ["harvey-run"];

export async function seedHarveyWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, "harvey-seed");
      if (changed) console.log(`[harvey] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[harvey] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}

export async function seedHarveySteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "harvey-seed");
      if (changed) console.log(`[harvey] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[harvey] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
