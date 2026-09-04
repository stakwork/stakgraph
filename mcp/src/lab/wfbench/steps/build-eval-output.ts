import { z, defineStep, type StepContext } from "vein";

/**
 * 58312 "Record Eval Trigger Output and Criterion Results" as a batch-
 * triplet payload (the write is graph/create-batch-triplet). Ids and edges
 * follow 58312 / 58115 exactly, on the ontology's declared relationships:
 *
 *   EvalTrigger(<slug>-<runId>) -HAS_OUTPUT-> EvalTriggerOutput(<slug>-<runId>)
 *   EvalTriggerOutput -HAS_CRITERION_RESULT-> CriterionResult (one per criterion)
 *   EvalRequirement(<slug>-<criterion_id>) -HAS_CRITERION_RESULT-> CriterionResult
 *
 * EvalTriggerOutput carries 58312's fields: result, verdict, score,
 * max_score, n_passed, n_total, judge_model (its `name` is not declared on
 * the ontology and is omitted). 58312's CriterionResult -HAS_CAUSE->
 * Workflow_version edge is NOT written: the ontology has no such
 * relationship and vein's produced workflow is not a Workflow_version node.
 * Ids derive from the runId, so a retried write merges instead of
 * duplicating. `criterionSlots` index each criterion's first
 * HAS_CRITERION_RESULT triplet for eval/criterion-refs.
 */
export default defineStep({
  type: "wfbench/build-eval-output",
  description:
    "Build the create-batch-triplet payload recording a judged run per 58312: EvalTrigger -HAS_OUTPUT-> EvalTriggerOutput { id: <slug>-<runId>, result, verdict, score, max_score, n_passed, n_total, judge_model } and, per criterion, EvalTriggerOutput/EvalRequirement -HAS_CRITERION_RESULT-> CriterionResult { id: <slug>-<runId>-<criterion_id>, criterion_id, title, verdict, reasoning }. Output: { scored, triplets, criterionSlots, trigger_id, output_id }.",
  input: z.object({
    task_slug: z.string().min(1),
    scores: z.any().optional().describe("eval/aggregate-scores output (absent/errored → scored=false, no triplets)."),
    trigger_ref_id: z.any().optional().describe("The EvalTrigger's ref_id (preferred); falls back to an inline { id } source."),
    trigger_id: z.string().optional().describe("The EvalTrigger id (default <slug>-<runId>)."),
    judge_model: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const runId = (ctx as StepContext)?.runId || "no-run";
    const slug = cfg.task_slug;
    const trigger_id = cfg.trigger_id || `${slug}-${runId}`;
    const output_id = `${slug}-${runId}`;
    const s = cfg.scores && typeof cfg.scores === "object" && !cfg.scores.error ? (cfg.scores as Record<string, any>) : null;
    if (!s || typeof s.all_pass !== "boolean" || !Array.isArray(s.criteria_results)) {
      return { scored: false, triplets: [], criterionSlots: [], trigger_id, output_id };
    }
    const outputNodeData = {
      id: output_id,
      result: s.all_pass ? "pass" : "fail",
      verdict: s.all_pass ? "pass" : "fail",
      score: typeof s.score === "number" ? s.score : 0,
      max_score: typeof s.max_score === "number" ? s.max_score : 0,
      n_passed: typeof s.n_passed === "number" ? s.n_passed : 0,
      n_total: typeof s.n_total === "number" ? s.n_total : 0,
      ...(cfg.judge_model || s.judge_model ? { judge_model: String(cfg.judge_model || s.judge_model) } : {}),
    };
    const triggerSide =
      typeof cfg.trigger_ref_id === "string" && cfg.trigger_ref_id
        ? { source_ref_id: cfg.trigger_ref_id }
        : { source_type: "EvalTrigger", source_data: { id: trigger_id } };

    const criterionSlots: Array<{ criterion_id: string; index: number }> = [];
    const triplets: Array<Record<string, any>> = [
      { ...triggerSide, target_type: "EvalTriggerOutput", target_data: outputNodeData, edge_type: "HAS_OUTPUT" },
    ];
    for (const c of s.criteria_results as Array<Record<string, any>>) {
      const cid = String(c?.criterion_id ?? c?.id ?? "");
      if (!cid) continue;
      const critData = {
        id: `${slug}-${runId}-${cid}`,
        criterion_id: cid,
        title: typeof c?.title === "string" ? c.title : "",
        verdict: c?.verdict === "pass" ? "pass" : "fail",
        reasoning: typeof c?.reasoning === "string" ? c.reasoning : "",
      };
      criterionSlots.push({ criterion_id: cid, index: triplets.length });
      triplets.push({
        source_type: "EvalTriggerOutput",
        source_data: outputNodeData,
        target_type: "CriterionResult",
        target_data: critData,
        edge_type: "HAS_CRITERION_RESULT",
      });
      triplets.push({
        source_type: "EvalRequirement",
        source_data: { id: `${slug}-${cid}` },
        target_type: "CriterionResult",
        target_data: critData,
        edge_type: "HAS_CRITERION_RESULT",
      });
    }
    return { scored: true, triplets, criterionSlots, trigger_id, output_id };
  },
});
