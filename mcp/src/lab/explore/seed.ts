import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "../seed-opts.js";

/**
 * explore — the explorer as a strut workflow (strut `plans/federation.md`
 * §2.2, the first dispatch-through use case). Another strut names this
 * swarm as a peer and runs `explore` here — `strut/run-workflow { peer,
 * workflow: "explore", input: { prompt, repos? } }` — to ask this swarm's
 * knowledge graph, and the repositories it names, a question and get back
 * what is relevant with what the answer rests on. Locally it is an
 * ordinary workflow (the Run button, the builder's run_workflow).
 *
 * The SAME skeleton and launch as `job` (../job; strut plans/jobs.md §2):
 * `job/dir` → `checkouts` (one `git/checkout` per `input.repos`, INTO the
 * run's directory — kept under a job, a fresh copy in a one-shot's own
 * dir) → `agent` (cwd = that directory; `session` only when the launch
 * names one — a job turn runs this as a child while its own agent holds
 * the job's session) → `pack`. What differs is the agent: read-only over the graph's READ
 * steps (never the `graph/*` glob — it would hand out the write steps) and
 * the read-only FILE tools (`params.builtins`; never `bash` or the web —
 * the caller may be another strut, and it must not get a shell here).
 * One step is the lab's own, `explore/repositories` (steps/repositories.ts):
 * which repositories the graph holds parsed, beside the ones checked out —
 * repo_agent's prependRepoInfo — so the prompt says, per repository, whether
 * to search the graph or read the files. The rest are strut's. `params` are the
 * evolvable surface (model, step budget, tools, built-ins, system). Seeded
 * UNSTAMPED, content-hash reconciled (SEED_OPTS): a changed committed copy
 * wins at boot, an unchanged one leaves a workspace-side edit active.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "explore",
    description:
      "Explore this swarm's knowledge graph, and the repositories named on the launch, for a question and return what is relevant: a read-only agent searches, reads, expands and walks the graph with its read tools (graph/graph-search, graph-get, graph-neighbors, graph/walk, the ontology), reads the repositories checked out into its directory with its file tools — told per repository whether the graph holds it parsed (explore/repositories) — and answers from what it read. The same launch as `job`: { prompt, repos? (repository URLs, one subdirectory each), session? (cold unless given, under a job too) } and `job` on the launch. Output: { answer (markdown, from what was read alone — or that nothing here bears on it), sources: [{ name, why, ref_id?, node_type?, path? }], confidence: high | medium | low, cost }. The workflow another strut asks this one to run (strut/run-workflow { peer, workflow: \"explore\", input: { prompt, repos? } }); params.tools / params.builtins / params.system are the evolvable surface.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

/** One file per step under steps/, `<name>.ts` → `explore/<name>`. */
export const SEED_STEPS = ["repositories"] as const;
const RETIRED_STEPS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
const CATEGORY = "explore";
export const STEPS_DIR = join(HERE, "steps");

export async function seedExploreSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `explore/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "explore-seed", SEED_OPTS);
      if (changed) console.log(`[explore] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[explore] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "explore");
}

export async function seedExploreWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, CATEGORY, undefined, SEED_OPTS);
      if (changed) console.log(`[explore] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[explore] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "explore");
}
