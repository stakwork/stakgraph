import { z, defineStep, type StepContext, type VeinCapabilities } from "vein";
import type { HarveyServices } from "../service.js";

/**
 * Thin plumbing over the in-code Harvey LAB service (`ctx.services.harvey`) —
 * the grader itself is NOT in this file and cannot be edited here. It runs
 * the REAL benchmark eval (uv subprocess in the pinned, clean harvey-labs
 * checkout) against deliverables this run wrote to its artifacts dir.
 *
 * GUARDRAIL: grant this step only to the eval/harness workflow — NEVER to the
 * producing agent's agentTools (an agent that can query its own grader
 * mid-task trains against the rubric and contaminates the benchmark).
 */
export default defineStep({
  type: "harvey/evaluate",
  description:
    "Grade this run's deliverables against a Harvey LAB task rubric by running the REAL " +
    "benchmark eval (LLM judge, all-pass scoring) from the pinned harvey-labs checkout. " +
    "Reads deliverables from this run's artifacts directory (subdir `from`, default 'output') — " +
    "produce files there first (e.g. an agent step with cwd at the artifacts dir). " +
    "Returns the benchmark's scores (score, all_pass, criteria_results, summary, …) plus " +
    "benchmarkRev (the exact benchmark commit) and reportPath. " +
    "Refuses to run if the benchmark checkout has local modifications.",
  input: z.object({
    task: z.string().describe("Task id, e.g. 'corporate-ma/review-data-room-red-flag-review'."),
    from: z
      .string()
      .optional()
      .default("output")
      .describe("Subdirectory of this run's artifacts dir holding the deliverables (default 'output')."),
    metrics: z
      .record(z.string(), z.any())
      .optional()
      .describe("Optional producer metrics folded into the report: input_tokens, output_tokens, wall_clock_seconds, …"),
    judgeModel: z
      .string()
      .optional()
      .describe("Single-judge model override (harness default: claude-sonnet-4-6). Ignored when dual=true."),
    dual: z
      .boolean()
      .optional()
      .default(false)
      .describe("Official-style dual judging (Sonnet + GPT judges; needs OPENAI_API_KEY too)."),
    parallel: z.number().int().positive().optional().describe("Concurrent judge calls (harness default 6)."),
    timeoutMs: z.number().int().positive().optional().describe("Eval subprocess timeout (default 30 min)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const services = ctx.services as ({ harvey?: HarveyServices } & VeinCapabilities) | undefined;
    const harvey = services?.harvey;
    if (!harvey) throw new Error("harvey service unavailable — is this the lab vein?");
    const artifacts = services?.artifacts;
    if (!artifacts) throw new Error("artifacts capability unavailable");
    const typedCtx = ctx as StepContext<VeinCapabilities>;
    const dir = await artifacts.dir(typedCtx.runId);
    const sub = (cfg.from ?? "output").replace(/^\/+|\/+$/g, "");
    return harvey.evaluate({
      task: cfg.task,
      sourceDir: sub ? `${dir}/${sub}` : dir,
      runId: `vein-${typedCtx.runId}`,
      metrics: cfg.metrics,
      judgeModel: cfg.judgeModel,
      dual: cfg.dual,
      parallel: cfg.parallel,
      timeoutMs: cfg.timeoutMs,
    });
  },
});
