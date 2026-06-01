import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Seed initial concepts for a brand-new repo by agentically exploring the
 * local clone, then set the checkpoint to a recent lookback window.
 * Output: { concepts, usage }.
 */
export default defineStep({
  type: "concepts/bootstrap-explore",
  description: "Seed initial concepts by exploring the cloned repo (for new repos). Config: lookbackDays (how many days back to set the checkpoint, default 10). Output: { concepts, usage }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    repoPath: z.string(),
    lookbackDays: z.number().int().positive().default(10),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { bootstrap } = ctx.services as ConceptServices;
    const { concepts, usage } = await bootstrap(
      cfg.owner,
      cfg.repo,
      cfg.repoPath,
      cfg.lookbackDays,
    );
    return { concepts, usage };
  },
});
