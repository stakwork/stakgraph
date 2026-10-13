import { z, defineStep } from "strut";
import type { ConceptServices } from "../services.js";

/**
 * Clone (or update) the repo locally so bootstrap / doc generation can
 * explore the codebase. Output: { repoPath }.
 *
 * The GitHub token is the run's `GITHUB_TOKEN` secret — the principal's
 * actor secret, else the deployment's — never a step input: a run's input
 * and every step's config are written to the run log.
 */
export default defineStep({
  type: "concepts/clone-repo",
  description: "Clone or update a GitHub repo locally, with the run's GITHUB_TOKEN secret. Output: { repoPath }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { clone, secrets } = ctx.services as ConceptServices;
    const repoPath = await clone(cfg.owner, cfg.repo, await secrets?.get("GITHUB_TOKEN"));
    return { repoPath };
  },
});
