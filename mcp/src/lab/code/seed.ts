import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import type { ConceptSet } from "../concept-seed.js";
import { SEED_OPTS, retireWorkflows } from "../seed-opts.js";

/**
 * code — code changes as strut workflows (strut `plans/code-change.md`).
 * `code-change-propose` (phase 1) is hive's `propose_code_change` preview:
 * strut's own agent makes the change in an isolated checkout and git/diff
 * captures it as one unified diff for hive to show. `code-change-land`
 * (phase 2) is the approval: git/apply lands exactly the approved bytes,
 * git/push commits and pushes them as the user, github/create-pr opens the
 * pull request — no model between the diff the user saw and the PR.
 * `code-change-pr` is both in ONE run, for a caller that reviews on the
 * pull request itself — the `job` agent (strut plans/jobs.md §4): checkout →
 * agent → git/diff → git/push → github/create-pr; run again with `branch`
 * to revise the same PR. How a job agent learns to run it is graph DATA,
 * not prompt text: the `Code Change` Concept (`concepts/Code Change.md`, a
 * child of the job agent's `Job` page — CODE_CONCEPTS, seeded
 * by concept-seed.ts) names the workflow, its input, the brief, the
 * artifact, and when a sandbox is warranted instead. NO custom steps:
 * every step is strut's, so this seeder ships YAML (and one Concept) only.
 * Seeded UNSTAMPED (no publisher → not "ai"), content-hash reconciled
 * (SEED_OPTS): a changed committed copy wins at boot, an unchanged one leaves
 * a workspace-side edit active.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "code-change-propose",
    description:
      "Propose a code change: check the repo out (git/checkout), let the agent make the change in an isolated working copy, capture it as one unified diff (git/diff) — nothing is pushed. Hive's propose_code_change preview. Input: { repo, prompt }. Output: { diff, diffSha256, filesChanged, files, baseBranch, baseSha, summary, cost }.",
  },
  {
    name: "code-change-land",
    description:
      "Land an approved code change: check the repo out at baseBranch (git/checkout), apply exactly the approved diff (git/apply, sha256-checked), commit and push it to a new branch as the user (git/push), open the pull request (github/create-pr). Hive's approval of a code-change-propose result — no agent step, deterministic: a base that moved fails with patch_conflict: instead of being fixed by a model. Input: { repo, baseBranch, diff, diffSha256, branch, title, body }. Output: { url, number, branch, base, headSha, filesChanged, files, diffSha256 }. Errors: patch_conflict: | push_rejected: | no_push_permission: | pr_create_failed: prefixes on error.message.",
  },
  {
    name: "code-change-pr",
    description:
      "A code change as ONE run, task to pull request: check the repo out (git/checkout — at `branch` on a follow-up), let the agent make the change in the working copy, stage and scan it (git/diff), commit and push it as the user (git/push), open the pull request (github/create-pr, idempotent by head). The PR is the proposal: run again with the same `branch` to revise it. Input: { repo, prompt, title, branch?, base?, body? }. Output: { url, number, repo (owner/name), branch, base, headSha, created, filesChanged, files, summary, cost }. Errors: no change → git/push's plain nothing-staged error; push_rejected: | no_push_permission: | pr_create_failed: prefixes as code-change-land.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
/** The `Code Change` kind page under `Job` (plans/jobs.md §5: how the job agent is routed to code-change-pr). */
export const CODE_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/code/concepts/" };

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
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "code");
}
