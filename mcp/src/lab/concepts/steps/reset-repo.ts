import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Delete all Concepts for a repo so the next run starts fresh. Needed for
 * iterative evals: the `is-new-repo` gate only bootstraps when no concepts
 * exist, so without a reset a second eval run would skip bootstrap. Bootstrap
 * overwrites the checkpoint on its own, so clearing concepts is sufficient.
 *
 * Output: { repoId, deleted }.
 */
export default defineStep({
  type: "concepts/reset-repo",
  description:
    "Delete all Concepts for owner/repo so a re-run bootstraps fresh (for iterative evals). Output: { repoId, deleted }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const repoId = `${cfg.owner}/${cfg.repo}`;
    const all = await storage.getAllConcepts(repoId);
    for (const c of all) {
      await storage.deleteConcept(c.id, repoId);
    }
    return { repoId, deleted: all.length };
  },
});
