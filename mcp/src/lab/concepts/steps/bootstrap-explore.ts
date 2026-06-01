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
    const { storage } = ctx.services as ConceptServices;
    // Lazy import: keeps the agentic/repo deps out of the registry's
    // static import graph (they connect to Neo4j at load).
    const { bootstrapConcepts } = await import("../bootstrap.js");
    const result = await bootstrapConcepts(
      cfg.owner,
      cfg.repo,
      cfg.repoPath,
      storage,
      undefined,
      cfg.lookbackDays,
    );
    return { concepts: result.features, usage: result.usage };
  },
});
