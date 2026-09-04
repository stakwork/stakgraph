import type { PublishByContentOptions, WorkspaceStore } from "vein";

/**
 * How every lab seeder reconciles its committed templates into the workspace
 * at boot. Content-hash keyed: a template whose hash the workspace has never
 * seen publishes the next version and activates it (a real code change wins);
 * a template whose hash is already a known version is a NO-OP even when that
 * version isn't active — so an edit made through the vein UI/API (or by the
 * authoring agent) stays active across restarts until the committed template
 * itself changes. Porting the winner back into the committed template is
 * still the only way to propagate it to other instances.
 */
export const SEED_OPTS: PublishByContentOptions = { reactivateKnown: false };

/**
 * Retire steps a seeder no longer ships. Seeding is ADDITIVE — a step dropped
 * from a `SEED_STEPS` list stays live in every existing workspace (the graph
 * workspace is persistent, and the file one keeps its materialized file), so
 * an author agent keeps discovering and using it. Each seeder keeps a
 * `RETIRED_STEPS` list of the types it used to publish and calls this at the
 * end of its step seeding; `deleteStep` is a soft delete on the graph store
 * (restorable by a later publish under the same name) and an unlink on the
 * file store. Missing types are a silent no-op, so a fresh workspace pays
 * nothing.
 */
export async function retireSteps(workspace: WorkspaceStore, types: readonly string[], tag: string): Promise<void> {
  for (const type of types) {
    try {
      if (await workspace.deleteStep(type)) console.log(`[${tag}] retired step: ${type}`);
    } catch (err) {
      console.warn(`[${tag}] could not retire step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
}
