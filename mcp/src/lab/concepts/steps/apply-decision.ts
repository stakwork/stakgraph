import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";
import { applyDecision, updateCheckpoint } from "../pipeline.js";

/**
 * Persist the change record, apply the LLM decision to concepts
 * (add/create/update + themes), and advance the checkpoint.
 * Output: { modifiedConceptIds, usage }.
 */
export default defineStep({
  type: "concepts/apply-decision",
  description: "Apply an LLM decision to concepts and advance the checkpoint. Output: { modifiedConceptIds, usage }.",
  input: z.object({
    change: z.any(),
    decision: z.any(),
    owner: z.string(),
    repo: z.string(),
    repoPath: z.string().optional(),
    usage: z.any().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage, octokit } = ctx.services as ConceptServices;
    const change = cfg.change;

    const modifiedConceptIds = await applyDecision(
      storage,
      octokit,
      cfg.owner,
      cfg.repo,
      change,
      cfg.decision,
      cfg.usage,
      cfg.repoPath,
    );

    await updateCheckpoint(
      storage,
      `${cfg.owner}/${cfg.repo}`,
      new Date(change.date),
      String(change.id),
    );

    return { modifiedConceptIds, usage: cfg.usage };
  },
});
