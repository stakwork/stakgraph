import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { PublishByContentOptions, WorkspaceStore } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "../seed-opts.js";
import type { ConceptSet } from "../concept-seed.js";

/**
 * janitor — two seeded engine workflows, three steps of our own, and a
 * `Janitor` Concept tree of mandates.
 *
 * A janitor is an automated agent that cleans something up. Its mandate —
 * what "dirty" means, what to flag, what to propose — lives in the knowledge
 * graph as a `Concept` under `Janitor`, one child per janitor, so ONE seeded
 * engine workflow reads its mandates from whatever graph it runs on, and a
 * workspace grows its own janitors by adding Concepts (hive's Learn page,
 * Jamie, strut's graph/* steps). `Janitor` itself hangs under `Workflow
 * Builder` (../builder): a janitor is one of the things the builder builds,
 * and the builder learns the convention by reading `Janitor`'s docs.
 *
 * `concepts/Janitor.md` is seeded by ../concept-seed.ts. No mandate ships:
 * what counts as dirty is each workspace's to say.
 *
 * The ENGINE is `workflows/graph-janitor.yaml`, seeded like the code
 * workflows (YAML only, unstamped, SEED_OPTS): `input.concept` names a
 * child of `Janitor` (the mandate), `input.start` the Concept whose
 * subtree to sweep — so a janitor IS an automation on it, one per mandate.
 * The agent stays read-only and proposes; `input.apply` (default false)
 * lets code write what it validated: `janitor/plan` (steps/plan.ts) turns
 * the findings' typed actions into a plan against the LIVE graph (never
 * Janitor, a mandate or a git-seeded Concept, never a rename, never a value
 * that moved on), strut's graph/edit-node + graph/move-node write it, and
 * `janitor/readback` (steps/readback.ts) reads every write back and keeps
 * the undo log, `applied.json`. `graph-janitor-revert.yaml` replays that
 * log backwards through `janitor/revert-plan` (steps/revert-plan.ts) and
 * the same apply branch. Automations are not seeded: each workspace
 * schedules its own.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
export const JANITOR_CONCEPTS: ConceptSet = {
  dir: join(HERE, "concepts"),
  prefix: "lab/janitor/concepts/",
  // A mandate about eval rubrics: one workspace's business, shipped to all of them by mistake.
  retired: ["Overfit Concept Janitor.md"],
};
/** A mandate for the tests and the smokes to plant. NOT seeded at boot (and not in the build: no `concepts` dir). */
export const JANITOR_FIXTURES: ConceptSet = { dir: join(HERE, "fixtures"), prefix: "lab/janitor/fixtures/" };
const WORKFLOWS_DIR = join(HERE, "workflows");
export const STEPS_DIR = join(HERE, "steps");
const CATEGORY = "graph-maintenance";

const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "graph-janitor",
    description:
      "Knowledge-graph janitor: a read-only agent sweeps the Concept subtree under input.start following the mandate in the docs of input.concept (a child of the Janitor Concept) and proposes cleanups as findings with typed actions; code validates them into a plan (never the Janitor node, a mandate, a git-seeded Concept or a rename; never a value that moved on since the agent read it) and, with input.apply=true, writes it, reads every write back and keeps an undo log for graph-janitor-revert. apply=false (default) is a dry run. One automation per janitor. Output: { mandate, start, start_ref_id, apply, visited, findings, summary, plan: { edits, moves, skipped, manual, counts }, applied, applied_json, report_json, report_md, report_plan }; errors not_found: | not_a_janitor: | apply_failed:.",
  },
  {
    name: "graph-janitor-revert",
    description:
      "Undo what a graph-janitor run applied: reads that run's undo log (applied.json: every change with the value or parent it replaced) and writes the inverse back — only where the node still holds what the janitor wrote; a node someone changed since is skipped (moved_on). input: { run_id, changes? (ids like e1, m2 to undo some of them) }. The revert is logged too, so it can be reverted. Output: { revert_of, source, plan, applied, applied_json, report_plan, report_md }; errors no_applied_log: | apply_failed:.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

/** One file per step under steps/, `<name>.ts` → `janitor/<name>`. */
export const SEED_STEPS = ["plan", "readback", "revert-plan"] as const;
const RETIRED_STEPS: string[] = [];

export async function seedJanitorSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `janitor/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "janitor-seed", SEED_OPTS);
      if (changed) console.log(`[janitor] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[janitor] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "janitor");
}

/** `opts` is the boot seeder's SEED_OPTS; a live test passes `{ ...SEED_OPTS, reactivateKnown: true }`
 *  so the committed YAML is the ACTIVE version on a shared, persistent test graph whatever ran there before. */
export async function seedJanitorWorkflows(workspace: WorkspaceStore, opts: PublishByContentOptions = SEED_OPTS): Promise<void> {
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(WORKFLOWS_DIR, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, CATEGORY, undefined, opts);
      if (changed) console.log(`[janitor] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[janitor] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "janitor");
}
