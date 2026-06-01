import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Gate step: returns `true` when the repo has no concepts yet. Used to
 * conditionally run bootstrap (clone + explore) only for new repos —
 * vein gates a `when: true` step on a boolean dependency's output.
 */
export default defineStep({
  type: "concepts/is-new-repo",
  description: "Boolean gate: true if the repo has no concepts yet (drives the bootstrap branch). Output: boolean.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.boolean(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const concepts = await storage.getAllConcepts(`${cfg.owner}/${cfg.repo}`);
    return concepts.length === 0;
  },
});
