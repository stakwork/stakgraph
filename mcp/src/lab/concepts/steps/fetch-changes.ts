import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import { fetchAllChanges } from "../pipeline.js";

/**
 * Fetch all merged PRs + standalone commits since the checkpoint,
 * sorted chronologically (oldest first). Output: { changes, count }.
 */
export default defineStep({
  type: "concepts/fetch-changes",
  description: "Fetch merged PRs and standalone commits from GitHub since the checkpoint. Output: { changes, count }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    checkpoint: z.any().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { octokit } = ctx.services as ConceptServices;
    const changes = await fetchAllChanges(
      octokit,
      cfg.owner,
      cfg.repo,
      cfg.checkpoint ?? null,
    );
    return { changes, count: changes.length };
  },
});
