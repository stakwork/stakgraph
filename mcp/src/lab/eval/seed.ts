import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Generic eval machinery (scorer step + scoring workflow), seeded into the
 * vein workspace like the concepts experiment. Reconciled by content hash on
 * boot, so edits publish a new active version (see concepts/seed.ts for the
 * reconciliation contract).
 *
 * Not concepts-specific: `eval/score` compares any `actual` vs `expected`
 * markdown using a `rubric`. The default `eval-score` workflow ships a
 * concept-quality rubric in its `params`, overridable per run.
 */

const SEED_WORKFLOWS = ["eval-score"];

const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "score.ts", type: "eval/score" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedEvalWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml);
      if (changed) console.log(`[eval] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(
        `[eval] could not seed workflow "${name}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}

export async function seedEvalSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "eval-seed");
      if (changed) console.log(`[eval] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[eval] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
