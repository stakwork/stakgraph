import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import type { ConceptSet } from "../concept-seed.js";
import { SEED_OPTS, retireSteps } from "../seed-opts.js";

/**
 * pods — hive sandboxes as strut steps (strut `plans/jobs.md` §5–6). A pod
 * is a container hive's pool manager hands out: the workspace's
 * repositories, a dev server and an IDE a person can open, and staklink, a
 * control server that runs a coding agent (goose), resets repositories,
 * diffs, pushes and tests. Every staklink call is one `pod/*` step, and the
 * same steps are the job agent's tools (`agentTools: ["pod/*"]`): it claims
 * a pod for the job, hands the task to the agent in it, reads the diff,
 * pushes the pull request and releases the pod when the job is done. The
 * `Pod` Concept (`concepts/Pod.md`, under `Code Change`) tells it when and
 * how; WHICH kind of change takes a pod is graph data, not prompt text.
 *
 * The pod outlives a run: `pod/claim` on a run with a job registers a HOLD
 * on the job (`ctx.services.jobs`) naming `pod/release` as the way to let it
 * go, which strut runs when the job is deleted or idle past the TTL. So the
 * agent in the pod keeps its session (its memory lives on the pod's disk)
 * and the dev server stays up between turns for the person to watch.
 *
 * The pod's password is a credential: `pod/claim` returns it sealed (under
 * HIVE_API_KEY), every other step takes `sealed` and opens it in-process —
 * it never reaches an output, the run log or an agent's context. Secrets:
 * HIVE_URL + HIVE_API_KEY (the deployment's; an org-scoped key — the hub
 * claims for every workspace of its org), GITHUB_TOKEN (the run's, for
 * fetches and pushes); the agent in the pod calls the model through this
 * run's LLM gateway grant, never a key of its own.
 *
 * The steps are self-contained source: they import only `strut` and
 * `./_shared.js`. Content-hash reconciled (SEED_OPTS): a changed committed
 * copy wins at boot, an unchanged one leaves a workspace-side edit active.
 */

/** One file per step under steps/, `<name>.ts` → `pod/<name>`. */
export const SEED_STEPS = [
  // the helper the registry skips and every step imports
  "_shared",
  // the pool: claim (registers the hold on a job) / release (drops it)
  "claim",
  "release",
  // the repositories: reset to a branch
  "latest",
  // the agent in the pod: start now and look later, or wait
  "agent-start",
  "agent-status",
  "agent",
  // the change: the working tree, the whole branch, the push
  "diff",
  "branch-diff",
  "push",
  // the pod's own test commands (`run-tests`, not `test`: the graph keys a
  // step by its name with every separator stripped, so `pod/test` would
  // collide with a custom `pod_test` a workspace may already hold)
  "run-tests",
] as const;

// Step types this seeder USED to publish (see retireSteps).
const RETIRED_STEPS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
export const STEPS_DIR = join(HERE, "steps");
/** The `Pod` page under `Code Change`: when a change takes a sandbox, and how. */
export const POD_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/pods/concepts/" };

export async function seedPodSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `pod/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "pods-seed", SEED_OPTS);
      if (changed) console.log(`[pods] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[pods] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "pods");
}
