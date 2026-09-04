import { z, defineStep } from "vein";

/**
 * 58313's resolve_webhook_payload (4-way, priority order) as one pure step
 * — and the workflow's own result is built from the SAME object, so what
 * the run returns is what was POSTed (58313's set_output diverged on error
 * paths). Byte-compatible with what Hive parses (RunnerScoreSchema):
 *
 *   success: { task_slug, task_title, n_passed, n_total, all_pass,
 *              pass_rate, judge_model, criteria_results }
 *   failure: { task_slug, task_title, harness_error: true, error_type, error }
 *            error_type ∈ no_workflow_produced | input_keys_mismatch |
 *              launch_refused | not_launched | no_materials_produced |
 *              judge_failed
 *            (no score fields — Hive must never read a fake 0/N)
 *
 * Priority: launch/exec harness error → judge error → success → no materials.
 * A produced workflow that FAILED at runtime is not a harness error — it is
 * still judged (partial credit), exactly as 58313 classifies it.
 */
export default defineStep({
  type: "wfbench/webhook-body",
  description:
    "Resolve the single Hive callback body from the harness branches: launch gate (check-input-keys) → run classification → materials → judge scores. Success: { task_slug, task_title, n_passed, n_total, all_pass, pass_rate, judge_model, criteria_results }; failure: { task_slug, task_title, harness_error: true, error_type, error }.",
  input: z.object({
    task_slug: z.string(),
    task_title: z.string(),
    judge_model: z.string().optional(),
    keys: z.any().optional().describe("wfbench/check-input-keys output."),
    cls: z.any().optional().describe("wfbench/classify-run output."),
    mats: z.any().optional().describe("wfbench/build-materials output."),
    scores: z.any().optional().describe("eval/aggregate-scores output (absent when the judge never ran)."),
  }),
  output: z.any(),
  async run(cfg) {
    const base = { task_slug: cfg.task_slug, task_title: cfg.task_title };
    const fail = (error_type: string, error: unknown) => ({
      ...base,
      harness_error: true,
      error_type,
      error: typeof error === "string" && error ? error : error_type,
    });
    const keys = cfg.keys && typeof cfg.keys === "object" ? cfg.keys : {};
    const cls = cfg.cls && typeof cfg.cls === "object" ? cfg.cls : {};
    const mats = cfg.mats && typeof cfg.mats === "object" ? cfg.mats : {};
    const scores = cfg.scores && typeof cfg.scores === "object" ? cfg.scores : null;

    if (keys.keys_match === false) return fail(keys.error_type ?? "input_keys_mismatch", keys.error);
    if (cls.launch_ok === false) return fail(cls.error_type ?? "not_launched", cls.error);
    const n_materials = typeof mats.n_materials === "number" ? mats.n_materials : 0;
    if (n_materials === 0) return fail("no_materials_produced", "the produced artifact resolved to no judge materials");
    if (!scores || scores.error || typeof scores.all_pass !== "boolean" || !Array.isArray(scores.criteria_results)) {
      return fail("judge_failed", scores?.error ?? "the judge produced no valid score");
    }
    return {
      ...base,
      n_passed: scores.n_passed,
      n_total: scores.n_total,
      all_pass: scores.all_pass,
      pass_rate: scores.pass_rate,
      judge_model: cfg.judge_model ?? scores.judge_model ?? null,
      criteria_results: scores.criteria_results,
    };
  },
});
