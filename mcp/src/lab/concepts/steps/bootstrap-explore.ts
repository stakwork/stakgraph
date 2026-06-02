import { z, defineStep } from "vein";
import type { ConceptServices } from "../services.js";

/**
 * Seed initial concepts for a brand-new repo by agentically exploring the
 * local clone, then set the checkpoint to a recent lookback window.
 * Output: { concepts, usage }.
 *
 * Pure mechanism: the exploration prompt — `systemPrompt` and `promptTemplate`
 * — is NOT baked in here. It's the experiment surface and lives in the calling
 * workflow's `params` block (see `workflows/bootstrap-then-process.yaml`),
 * passed in via config. `promptTemplate` uses `{slot}` placeholders
 * ({owner}, {repo}, {size}, {fileCount}, {min}, {max}) filled at runtime.
 * Sweep prompt variants by overriding `params`; promote a winner by editing
 * the workflow's `params` default.
 */
export default defineStep({
  type: "concepts/bootstrap-explore",
  description: "Seed initial concepts by exploring the cloned repo (for new repos). Required config: systemPrompt, promptTemplate — supplied by the calling workflow's `params` block (the prompt experiment surface). Optional: lookbackDays (checkpoint lookback, default 10). Output: { concepts, usage }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    repoPath: z.string(),
    systemPrompt: z.string(),
    promptTemplate: z.string(),
    lookbackDays: z.number().int().positive().default(10),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { bootstrap } = ctx.services as ConceptServices;
    const { concepts, usage } = await bootstrap(
      cfg.owner,
      cfg.repo,
      cfg.repoPath,
      { system: cfg.systemPrompt, template: cfg.promptTemplate },
      cfg.lookbackDays,
    );
    return { concepts, usage };
  },
});
