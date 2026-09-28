import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireWorkflows } from "../seed-opts.js";
import type { ConceptSet } from "../concept-seed.js";

/**
 * janitor — one seeded engine workflow + a `Janitor` Concept tree of mandates.
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
 * The stock tree is `concepts/<Name>.md`, seeded by ../concept-seed.ts.
 *
 * The ENGINE is `workflows/graph-janitor.yaml`, seeded like the code
 * workflows (YAML only, every step strut's or artifacts/dir, unstamped,
 * SEED_OPTS): `input.concept` names a child of `Janitor` (the mandate),
 * `input.start` the Concept whose subtree to sweep — so a janitor IS an
 * automation on it, one per mandate. Automations are not seeded: each
 * workspace schedules its own.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
export const JANITOR_CONCEPTS: ConceptSet = { dir: join(HERE, "concepts"), prefix: "lab/janitor/concepts/" };
const WORKFLOWS_DIR = join(HERE, "workflows");
const CATEGORY = "graph-maintenance";

const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "graph-janitor",
    description:
      "Knowledge-graph janitor: a read-only agent sweeps the Concept subtree under input.start following the mandate in the docs of input.concept (a child of the Janitor Concept) and proposes cleanups it never applies. One automation per janitor. Output: { mandate, start, start_ref_id, visited, findings, summary, report_json, report_md }; errors not_found: | not_a_janitor:.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

export async function seedJanitorWorkflows(workspace: WorkspaceStore): Promise<void> {
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(WORKFLOWS_DIR, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, CATEGORY, undefined, SEED_OPTS);
      if (changed) console.log(`[janitor] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[janitor] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "janitor");
}
