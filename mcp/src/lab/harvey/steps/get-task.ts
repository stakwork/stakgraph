import { z, defineStep } from "vein";
import type { HarveyServices } from "../service.js";

/**
 * Thin plumbing over the in-code Harvey LAB service (`ctx.services.harvey`) —
 * the eval logic itself is NOT in this file and cannot be edited here.
 *
 * Returns the task's instructions + input-document listing WITHOUT the
 * grading rubric (the service strips `criteria`; a producing agent must never
 * see how it will be graded). Safe to grant to producing agents.
 */
export default defineStep({
  type: "harvey/get-task",
  description:
    "Load a Harvey LAB (Legal Agent Benchmark) task: title, instructions, expected deliverable " +
    "filenames, and the input documents directory + file listing. Does NOT include the grading " +
    "rubric. Task ids look like 'corporate-ma/review-data-room-red-flag-review' " +
    "(practice-area/task-slug, sometimes with a /scenario-NN suffix).",
  input: z.object({
    task: z.string().describe("Task id, e.g. 'corporate-ma/review-data-room-red-flag-review'."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const harvey = (ctx.services as { harvey?: HarveyServices } | undefined)?.harvey;
    if (!harvey) throw new Error("harvey service unavailable — is this the lab vein?");
    return harvey.getTask(cfg.task);
  },
});
