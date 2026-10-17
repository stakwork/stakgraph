import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { ConceptSet, WorkspaceStore } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "../seed-opts.js";

/**
 * pod-setup — a workspace's first pod setup, drafted from its repositories
 * (strut plans/onboarding.md): the `identify-services` workflow, its code steps
 * (`pod-setup/repo-facts`, `pod-setup/pages`, `pod-setup/finish`), and the `Pod Setup`
 * Concept pages it loads — the contract every pod's files follow, under
 * `Pod Config`, with a page per platform and per local service. The
 * onboarding job runs the workflow (the `Onboarding` page under `Job`).
 *
 * The steps are `pod-setup/*`, not `pod/*`: the job agent is granted `pod/*`
 * as a glob, and these are the workflow's, not its tools. Self-contained
 * source, content-hash reconciled (SEED_OPTS), like every lab seed.
 */

const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "identify-services",
    description:
      "Draft a workspace's pod setup from its repositories: check them out, scan them in code (package files and managers, scripts, env vars), load the Pod Setup pages by name (the root, the Local Services catalog, then the pages the profile picks — enums built from their children, so a missing page fails the run and a new one is offered with no change here), profile the workspace and write its services and compose services with two read-only agents, then make them into hive's pool config shape in code (`frontend` named, every cwd from a repository and a directory that exists, no `app` compose service, volumes declared). Input: { repos: [URL, …] }. Output: { services, containerFiles, notes, profile, pages, cost } — services + containerFiles are pod/config-set's input.",
  },
];
const RETIRED_WORKFLOWS: string[] = [];

/** One file per step under steps/, `<name>.ts` → `pod-setup/<name>`. */
export const SEED_STEPS = ["repo-facts", "pages", "finish"] as const;
const RETIRED_STEPS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
const CATEGORY = "pods";
export const STEPS_DIR = join(HERE, "steps");
/** `Pod Setup` (under `Pod Config`) and the pages under it. */
export const POD_SETUP_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/pod-setup/concepts/" };

export async function seedPodSetupSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `pod-setup/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "pod-setup-seed", SEED_OPTS);
      if (changed) console.log(`[pod-setup] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[pod-setup] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "pod-setup");
}

export async function seedPodSetupWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, CATEGORY, undefined, SEED_OPTS);
      if (changed) console.log(`[pod-setup] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[pod-setup] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "pod-setup");
}
