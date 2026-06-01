import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import { resolveCheckpoint } from "../pipeline.js";

/**
 * Resolve the chronological checkpoint for a repo (migrating legacy
 * per-PR/per-commit checkpoints if needed). The checkpoint marks where
 * the last run left off so processing is incremental + resumable.
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
    const checkpoint = await resolveCheckpoint(storage, repoId);
    return { checkpoint, repo: repoId };
  },
});
