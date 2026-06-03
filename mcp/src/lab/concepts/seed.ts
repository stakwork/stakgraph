import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Workflow + step templates shipped with the concepts experiment. They are
 * the canonical, committed source of truth in this repo; on boot they're
 * reconciled into the workspace by **content hash**:
 *
 *   - unchanged template  → same version id → no-op (no churn)
 *   - edited template      → new version id  → published + activated, while
 *                            prior versions are retained for rollback
 *
 * This is what makes "edit here, commit, redeploy/reseed → propagates to
 * every vein instance" work, without clobbering local experiment history.
 * (See vein `publishWorkflowByContent` / `publishStep`.)
 */

const SEED_WORKFLOWS = [
  "process-change",
  "process-repo-chronological",
  "bootstrap-then-process",
  // Concept-specific eval workflows: wire the generic eval/* steps with the
  // concept rubric / task / dataset. (The generic steps are seeded by eval/seed.)
  "concepts-eval", // harness: bootstrap-only produce → score
  "concepts-eval-score", // eval/score + concept rubric
  "concepts-eval-reflect", // eval/reflect + concept task/guidance
  "concepts-optimize", // eval/optimize loop, wired to the above
];

/**
 * Step templates: the on-disk source file (under `steps/`) and the step
 * **type** it publishes as. Filenames don't always match the type
 * (e.g. `concept-decide.ts` → `concepts/decide`), so the mapping is explicit.
 * The publish name IS the type, so it lands at `custom/<type>.ts` and the
 * registry discovers it under that exact type.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "resolve-checkpoint.ts", type: "concepts/resolve-checkpoint" },
  { file: "fetch-changes.ts", type: "concepts/fetch-changes" },
  { file: "prioritize-changes.ts", type: "concepts/prioritize-changes" },
  { file: "is-new-repo.ts", type: "concepts/is-new-repo" },
  { file: "clone-repo.ts", type: "concepts/clone-repo" },
  { file: "bootstrap-explore.ts", type: "concepts/bootstrap-explore" },
  { file: "fetch-content.ts", type: "concepts/fetch-content" },
  { file: "concept-decide.ts", type: "concepts/decide" },
  { file: "apply-decision.ts", type: "concepts/apply-decision" },
  { file: "collect-results.ts", type: "concepts/collect-results" },
  { file: "summarize-concept.ts", type: "concepts/summarize" },
  { file: "link-files.ts", type: "concepts/link-files" },
  { file: "collect-for-eval.ts", type: "concepts/collect-for-eval" },
  { file: "reset-repo.ts", type: "concepts/reset-repo" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

/**
 * Reconcile the bundled workflow YAML templates into the workspace by
 * content hash. Idempotent; edits publish a new active version.
 */
export async function seedConceptWorkflows(
  workspace: WorkspaceManager,
): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(
        name,
        yaml,
      );
      if (changed) console.log(`[concepts] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(
        `[concepts] could not seed workflow "${name}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}

/**
 * Reconcile the bundled self-contained step templates into the workspace's
 * `custom/` tier by content hash, so the registry discovers them from disk
 * (and they become editable/versioned through the vein API + UI). Idempotent;
 * edits publish a new active version while prior versions are archived.
 */
export async function seedConceptSteps(
  workspace: WorkspaceManager,
): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(
        type,
        code,
        undefined,
        "concepts-seed",
      );
      if (changed) console.log(`[concepts] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[concepts] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
