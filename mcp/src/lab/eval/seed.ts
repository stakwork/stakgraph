import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * GENERIC, domain-agnostic eval primitives (STEPS only), seeded into the vein
 * workspace. Reconciled by content hash on boot, so edits publish a new active
 * version (see concepts/seed.ts for the reconciliation contract).
 *
 * These steps are reusable across ALL experiments — they carry no domain
 * config. An experiment supplies its own eval WORKFLOWS that wire these steps
 * with its rubric / task / dataset (e.g. the concepts experiment ships
 * `concepts-eval*` in concepts/workflows, seeded by concepts/seed.ts):
 *   - `eval/score`    — match produced vs expected by a `rubric`, recall-weighted.
 *   - `eval/reflect`  — propose a better prompt from AGGREGATED results.
 *   - `eval/optimize` — eval → keep best → reflect loop (a detached job).
 */

const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "score.ts", type: "eval/score" },
  { file: "reflect.ts", type: "eval/reflect" },
  { file: "optimize.ts", type: "eval/optimize" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

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
