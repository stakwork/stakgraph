import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Link concepts to File nodes in the graph (MODIFIES edges, weighted by
 * importance) based on the files touched by their PRs/commits.
 * Output: { linkResult }.
 */
export default defineStep({
  type: "concepts/link-files",
  description: "Create MODIFIES edges from concepts to File nodes. Optional conceptId to scope to one. Output: { linkResult }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    conceptId: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const repoId = `${cfg.owner}/${cfg.repo}`;
    const linkResult = await storage.linkConceptsToFiles(cfg.conceptId, repoId);
    return { linkResult };
  },
});
