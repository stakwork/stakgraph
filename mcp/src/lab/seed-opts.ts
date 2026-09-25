import type { PublishByContentOptions, WorkspaceStore } from "strut";

/**
 * How every lab seeder reconciles its committed templates into the workspace
 * at boot. Content-hash keyed: a template whose hash the workspace has never
 * seen publishes the next version and activates it (a real code change wins);
 * a template whose hash is already a known version is a NO-OP even when that
 * version isn't active — so an edit made through the strut UI/API (or by the
 * authoring agent) stays active across restarts until the committed template
 * itself changes. Porting the winner back into the committed template is
 * still the only way to propagate it to other instances.
 */
export const SEED_OPTS: PublishByContentOptions = { reactivateKnown: false };

/**
 * Retire steps and workflows a seeder no longer ships. Seeding is ADDITIVE —
 * a step dropped from a `SEED_STEPS` list, or a workflow dropped from
 * `SEED_WORKFLOWS` (its YAML deleted from git), stays live in every existing
 * workspace (the graph workspace is persistent, and the file one keeps its
 * materialized files): an author agent keeps discovering and using the step,
 * and the workflow stays listed, runnable and schedulable. Each seeder keeps
 * a `RETIRED_STEPS` / `RETIRED_WORKFLOWS` list of the names it used to
 * publish and calls these at the end of its step / workflow seeding. Both
 * deletes are soft on the graph store (restorable by a later publish under
 * the same name; a workflow's schedules, owner, cap and category are cleared
 * first so it comes back clean, and its runs on disk are left alone) and an
 * unlink / `rm -rf` on the file store (where a workflow's runs live under it
 * and go with it — what `DELETE /workflows/:name` removes too). Missing
 * names are a silent no-op, so a fresh workspace pays nothing.
 */
export function retireSteps(workspace: WorkspaceStore, types: readonly string[], tag: string): Promise<void> {
  return retire(types, tag, "step", (type) => workspace.deleteStep(type));
}

export function retireWorkflows(workspace: WorkspaceStore, names: readonly string[], tag: string): Promise<void> {
  return retire(names, tag, "workflow", (name) => workspace.deleteWorkflow(name));
}

async function retire(names: readonly string[], tag: string, kind: string, del: (name: string) => Promise<boolean>): Promise<void> {
  for (const name of names) {
    try {
      if (await del(name)) console.log(`[${tag}] retired ${kind}: ${name}`);
    } catch (err) {
      console.warn(`[${tag}] could not retire ${kind} "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}
