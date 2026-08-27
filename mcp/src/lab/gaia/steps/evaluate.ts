import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/**
 * HARNESS-ONLY plumbing over ctx.services.gaia.score. Never expose this
 * step (or any gaia/*) as an agentTool to a producing agent — it would let
 * the agent see grading/scoring internals or self-grade.
 */
export default defineStep({
  type: "gaia/evaluate",
  description:
    "Score answer pairs against the GAIA gold set via ctx.services.gaia.score(pairs). HARNESS-ONLY — never grant to a producing agent's agentTools. Config: pairs [{ taskId, answer }]. Output: { accuracy, correct, total, byLevel, results, benchmarkRev, scorerSha256 }.",
  input: z.object({
    pairs: z
      .array(
        z.object({
          taskId: z.string(),
          answer: z.string(),
        }),
      )
      .min(1),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<VeinCapabilities & { gaia?: any }>;
    const gaia = c.services?.gaia;
    if (!gaia) throw new Error("gaia capability unavailable in this deployment");
    return await gaia.score(cfg.pairs);
  },
});
