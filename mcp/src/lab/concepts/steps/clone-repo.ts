import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Clone (or update) the repo locally so bootstrap / doc generation can
 * explore the codebase. Output: { repoPath }.
 */
export default defineStep({
  type: "concepts/clone-repo",
  description: "Clone or update a GitHub repo locally. Output: { repoPath }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    token: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { clone } = ctx.services as ConceptServices;
    const repoPath = await clone(cfg.owner, cfg.repo, cfg.token);
    return { repoPath };
  },
});
