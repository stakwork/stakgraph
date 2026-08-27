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
    dir: z
      .string()
      .optional()
      .describe(
        "ABSOLUTE path of the deliverables directory to stage, overriding the runId-derived " +
          "`<this run's artifacts dir>/<from>`. Use when grading a produce run that ran under its " +
          "OWN runId — e.g. an agent-authored candidate via meta/run-workflow, whose output " +
          "reports `outputDir` (see harvey-candidate-run).",
      ),
    fromRun: z
      .any()
      .optional()
      .describe(
        "A meta/run-workflow RESULT object ({ runId, status, output?, error? }) whose output " +
          "reports `outputDir` — pass the WHOLE object (e.g. `{{ run }}`) and this step extracts " +
          "the staging dir and produce metrics (cost/steps/usage) in code. Exists because workflow " +
          "template expressions cannot safely deep-access a possibly-undefined `output` (the " +
          "evaluator does not short-circuit). Takes precedence over `from`; `dir` wins over both.",
      ),
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
    const typedCtx = ctx as StepContext<VeinCapabilities>;
    // A produce run's result (meta/run-workflow shape). Extracted in code —
    // template expressions can't guard deep access on a failed run's missing
    // `output`.
    const fromRun = cfg.fromRun as
      | { output?: { outputDir?: unknown; cost?: unknown; steps?: unknown; usage?: { inputTokens?: unknown; outputTokens?: unknown } } }
      | undefined;
    const produceOut = fromRun && typeof fromRun === "object" ? fromRun.output : undefined;
    let sourceDir: string;
    if (cfg.dir) {
      sourceDir = cfg.dir;
    } else if (typeof produceOut?.outputDir === "string" && produceOut.outputDir) {
      sourceDir = produceOut.outputDir;
    } else if (fromRun) {
      // The produce run failed before reporting an outputDir — there is
      // nothing to stage. Fail loudly (the harness's grade onError records
      // the honest zero) instead of accidentally staging this run's dir.
      throw new Error("harvey/evaluate: fromRun has no output.outputDir (the produce run failed)");
    } else {
      const artifacts = services?.artifacts;
      if (!artifacts) throw new Error("artifacts capability unavailable");
      const dir = await artifacts.dir(typedCtx.runId);
      const sub = (cfg.from ?? "output").replace(/^\/+|\/+$/g, "");
      sourceDir = sub ? `${dir}/${sub}` : dir;
    }
    // Fold the produce run's own metrics in (explicit cfg.metrics wins).
    const runMetrics: Record<string, unknown> = produceOut
      ? {
          ...(produceOut.usage && typeof produceOut.usage === "object"
            ? { input_tokens: produceOut.usage.inputTokens, output_tokens: produceOut.usage.outputTokens }
            : {}),
          produce_cost: produceOut.cost,
          produce_steps: produceOut.steps,
        }
      : {};
    const metrics =
      cfg.metrics || Object.keys(runMetrics).length ? { ...runMetrics, ...cfg.metrics } : undefined;
    // Include the step PATH in the benchmark run id: subflows share their
    // parent's runId, so a batch harness (harvey-evolve) grades many times
    // under one vein runId — a bare `vein-<runId>` would re-stage every grade
    // into the same results/<id>/output/ dir, leaving the previous task's
    // deliverables mixed in (cp adds, it doesn't clean). The path is unique
    // per step invocation (foreach iterations include `#<i>`); the service
    // sanitizes the separators.
    return harvey.evaluate({
      task: cfg.task,
      sourceDir,
      runId: `vein-${typedCtx.runId}-${typedCtx.path}`,
      metrics,
      judgeModel: cfg.judgeModel,
      dual: cfg.dual,
      parallel: cfg.parallel,
      timeoutMs: cfg.timeoutMs,
    });
  },
});
