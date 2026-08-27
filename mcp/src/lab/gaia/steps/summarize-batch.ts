import { z, defineStep } from "vein";

/**
 * Combine per-task produce results (taskId, question, level, answer, cost,
 * steps) with the ONE gaia/evaluate score call's results (matched correctness
 * per taskId) into the batch's final report shape.
 */
export default defineStep({
  type: "gaia/summarize-batch",
  description:
    "Merge per-task produce results with a gaia/evaluate score response into a batch report. Config: produced (array of {taskId, question, level, answer, cost, steps, hasFile}), score (gaia/evaluate output). Output: { accuracy, correct, total, byLevel, totalCost, totalSteps, perTask: [{taskId, level, question, answer, correct, cost, steps}] }.",
  input: z.object({
    produced: z.array(z.any()),
    score: z.any(),
  }),
  output: z.any(),
  async run(cfg) {
    const resultsByTask = new Map<string, any>();
    for (const r of cfg.score?.results ?? []) resultsByTask.set(r.taskId, r);

    let totalCost = 0;
    let totalSteps = 0;
    const perTask = cfg.produced.map((p: any) => {
      const graded = resultsByTask.get(p.taskId);
      const cost = typeof p.cost === "number" ? p.cost : 0;
      const steps = typeof p.steps === "number" ? p.steps : 0;
      totalCost += cost;
      totalSteps += steps;
      return {
        taskId: p.taskId,
        level: p.level,
        question: p.question,
        answer: p.answer,
        correct: graded ? graded.correct : null,
        cost: p.cost,
        steps: p.steps,
      };
    });

    return {
      accuracy: cfg.score?.accuracy ?? null,
      correct: cfg.score?.correct ?? null,
      total: cfg.score?.total ?? null,
      byLevel: cfg.score?.byLevel ?? null,
      totalCost,
      totalSteps,
      benchmarkRev: cfg.score?.benchmarkRev,
      perTask,
    };
  },
});
