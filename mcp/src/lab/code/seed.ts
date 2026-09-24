import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS } from "../seed-opts.js";

/**
 * code — code changes as strut workflows (strut `plans/code-change.md`).
 * Phase 1 is `code-change-propose`: hive's `propose_code_change` preview,
 * run by strut's own agent in an isolated checkout and captured as one
 * unified diff for hive to show and land. NO custom steps: `git/checkout`,
 * `agent`, `git/diff` and `pack` are all strut's, so this seeder ships YAML
 * only. Seeded UNSTAMPED (no publisher → not "ai"), content-hash reconciled
 * (SEED_OPTS): a changed committed copy wins at boot, an unchanged one leaves
 * a workspace-side edit active.
 *
 * Phase 2 adds `code-change-land` (git/apply → git/push → github/create-pr)
 * once those lib steps exist in strut.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "code-change-propose",
    description:
      "Propose a code change: check the repo out (git/checkout), let the agent make the change in an isolated working copy, capture it as one unified diff (git/diff) — nothing is pushed. Hive's propose_code_change preview. Input: { repo, prompt }. Output: { diff, diffSha256, filesChanged, files, baseBranch, baseSha, summary, cost }.",
  },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedCodeWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "code", undefined, SEED_OPTS);
      if (changed) console.log(`[code] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[code] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}
