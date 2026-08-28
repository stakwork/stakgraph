import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";

/**
 * HARNESS-ONLY plumbing over ctx.services.gaia.score. Never expose this
 * step (or any gaia/*) as an agentTool to a producing agent — it would let
 * the agent see grading/scoring internals or self-grade.
 *
 * Two modes:
 *  - `pairs`: score explicit { taskId, answer } pairs (gaia-run/gaia-batch).
 *  - `fromRun`: score ONE candidate run's reported answer (gaia-candidate-
 *    run). The whole run result is passed in and unpacked HERE, in code —
 *    the template evaluator does not short-circuit, so YAML can never
 *    safely deep-access `run.output.answer` on a failed run (EVOLVE_SPEC
 *    §5.3.5). A failed run or missing/non-string answer scores as "" — a
 *    certainly-wrong answer, an honest zero, never an aborted batch.
 *    Output adds { taskId, answer, isCorrect, level, question } convenience
 *    fields (`isCorrect` — the report's `correct` is a COUNT; `question`
 *    comes from gaia.getTask, which strips the gold, and is null rather
 *    than fatal when the lookup fails — grading never dies on metadata).
 */
export default defineStep({
  type: "gaia/evaluate",
  description:
    "Score answers against the GAIA gold set via ctx.services.gaia.score(pairs). HARNESS-ONLY — never grant to a producing agent's agentTools. Config: pairs [{ taskId, answer }] OR fromRun { taskId, run } (a candidate RunResult — the answer is unpacked in code, '' when the run failed). Output: { accuracy, correct, total, byLevel, results, benchmarkRev, scorerSha256 }; fromRun adds { taskId, answer, isCorrect, level, question }.",
  input: z.object({
    pairs: z
      .array(
        z.object({
          taskId: z.string(),
          answer: z.string(),
        }),
      )
      .min(1)
      .optional(),
    fromRun: z
      .object({
        taskId: z.string(),
        run: z.any().describe("the candidate run's RunResult ({ runId, status, output?, error? })"),
      })
      .optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<VeinCapabilities & { gaia?: any }>;
    const gaia = c.services?.gaia;
    if (!gaia) throw new Error("gaia capability unavailable in this deployment");
    if (!cfg.pairs === !cfg.fromRun) {
      throw new Error("gaia/evaluate: provide exactly one of `pairs` or `fromRun`");
    }

    if (cfg.pairs) return await gaia.score(cfg.pairs);

    const { taskId, run } = cfg.fromRun!;
    const output = (run && typeof run === "object" ? (run as any).output : undefined) ?? {};
    const answer = typeof output.answer === "string" ? output.answer : "";
    const report = await gaia.score([{ taskId, answer }]);
    const first = report.results?.[0];
    let question: string | null = null;
    try {
      question = (await gaia.getTask(taskId))?.question ?? null;
    } catch {
      // metadata only (for the digest) — never fail a grade over it
    }
    return {
      ...report,
      taskId,
      answer,
      isCorrect: first?.correct === true,
      level: first?.level ?? null,
      question,
    };
  },
});
