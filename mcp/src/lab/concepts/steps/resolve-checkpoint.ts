import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Resolve the chronological checkpoint for a repo (migrating legacy
 * per-PR/per-commit checkpoints if needed). The checkpoint marks where
 * the last run left off so processing is incremental + resumable.
 *
 * Self-contained: the checkpoint algorithm lives inline so it can be
 * rewritten/experimented on. The only external dep is `ctx.services.storage`.
 * Output: { checkpoint, repo }.
 */
export default defineStep({
  type: "concepts/resolve-checkpoint",
  description: "Load the chronological checkpoint for owner/repo (or migrate a legacy one). Output: { checkpoint, repo }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const repoId = `${cfg.owner}/${cfg.repo}`;

    let checkpoint = await storage.getChronologicalCheckpoint(repoId);

    // Migrate a legacy per-PR/per-commit checkpoint into the chronological form.
    if (!checkpoint) {
      const lastPR = await storage.getLastProcessedPR(repoId);
      const lastCommit = await storage.getLastProcessedCommit(repoId);
      if (lastPR !== 0 || lastCommit) {
        let latestDate: Date | null = null;
        if (lastPR > 0) {
          const pr = await storage.getPR(lastPR, repoId);
          if (pr) latestDate = pr.mergedAt;
        }
        if (lastCommit) {
          const commit = await storage.getCommit(lastCommit, repoId);
          if (commit && (!latestDate || commit.committedAt > latestDate)) {
            latestDate = commit.committedAt;
          }
        }
        if (latestDate) {
          checkpoint = {
            lastProcessedTimestamp: latestDate.toISOString(),
            processedAtTimestamp: [],
          };
          await storage.setChronologicalCheckpoint(repoId, checkpoint);
        }
      }
    }

    return { checkpoint: checkpoint ?? null, repo: repoId };
  },
});
