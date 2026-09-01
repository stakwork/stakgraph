import { z, defineStep, type StepContext } from "vein";

/**
 * Build the batch-triplet payload that persists one scored attempt as the
 * unified eval chain (the record_eval_chain equivalent), matching the jarvis
 * ontology exactly (schema_library.py):
 *
 *   EvalSet -HAS_TRIGGER-> EvalTrigger -HAS_OUTPUT-> EvalTriggerOutput
 *     -HAS_CRITERION_RESULT-> CriterionResult (one per judged criterion)
 *   EvalRequirement -HAS_TRIGGER-> EvalTrigger (one per judged criterion)
 *
 * Pure payload construction — the write itself is jarvis/create-batch-triplet
 * (its inline-node dedupe collapses the repeated EvalTriggerOutput side).
 * Ids are derived from ctx.runId, so a retried write MERGES instead of
 * duplicating the chain. Dispute annotations (flagged / llm_flag_reason /
 * contested) ride in on the CriterionResult nodes when the annotated
 * criteria_results are passed.
 */
export default defineStep({
  type: "harvey/build-eval-chain",
  description:
    "Construct the create-batch-triplet payload persisting a scored attempt: EvalSet→EvalTrigger→" +
    "EvalTriggerOutput→CriterionResult(+EvalRequirement links), ids derived from the runId (idempotent " +
    "rewrites). Output: { triplets, trigger_id, output_id }.",
  input: z.object({
    evalsetId: z.string().describe("EvalSet id (the task namespace slug)."),
    task: z.string().describe("The harvey task id (recorded as the trigger's workflow_input)."),
    scores: z.any().describe("harvey/aggregate-scores output (criteria_results may carry dispute annotations)."),
    criteria_results: z
      .array(z.any())
      .optional()
      .describe("Override for scores.criteria_results — pass harvey/merge-disputes' ANNOTATED list."),
    judge_model: z.string().optional(),
    workflow: z.string().default("harvey-deliver").describe("Recorded as EvalTrigger.workflow_id/agent."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const runId = (ctx as StepContext)?.runId || "no-run";
    const trigger_id = `trigger-${runId}`;
    const output_id = `output-${runId}`;
    const s = (cfg.scores ?? {}) as Record<string, any>;
    const criteria: Array<Record<string, any>> = Array.isArray(cfg.criteria_results)
      ? cfg.criteria_results
      : Array.isArray(s.criteria_results)
        ? s.criteria_results
        : [];

    // ONE node_data object for EVERY EvalTriggerOutput side. The batch step's
    // inline-node dedup cache keys on (type + full node_data) — when the
    // criterion triplets carried a bare { id } while the spine carried the
    // full attributes, the cache missed and the bare re-create failed jarvis
    // schema validation ("Missing required attribute 'result'"), dropping all
    // 60 CriterionResult writes in the first live run. Identical objects →
    // one resolve, reused everywhere.
    const outputNodeData = {
      id: output_id,
      result: s.all_pass ? "pass" : "fail",
      verdict: s.all_pass ? "pass" : "fail",
      score: typeof s.score === "number" ? s.score : 0,
      max_score: typeof s.max_score === "number" ? s.max_score : 0,
      n_total: typeof s.n_total === "number" ? s.n_total : 0,
      n_passed: typeof s.n_passed === "number" ? s.n_passed : 0,
      ...(cfg.judge_model ? { judge_model: cfg.judge_model } : {}),
    };

    // Position of each criterion's HAS_CRITERION_RESULT triplet in the array
    // below — harvey/criterion-refs zips these with the batch write's results
    // to recover the created CriterionResult ref_ids.
    const criterionSlots: Array<{ criterion_id: string; index: number }> = [];
    const triplets: Array<Record<string, any>> = [
      {
        source_type: "EvalSet",
        source_data: { id: cfg.evalsetId },
        target_type: "EvalTrigger",
        target_data: {
          id: trigger_id,
          agent: cfg.workflow,
          environment: "vein-lab",
          source: "vein",
          workflow_id: cfg.workflow,
          workflow_input: JSON.stringify({ task: cfg.task }),
          run_count: 1,
        },
        edge_type: "HAS_TRIGGER",
      },
      {
        source_type: "EvalTrigger",
        source_data: { id: trigger_id },
        target_type: "EvalTriggerOutput",
        target_data: outputNodeData,
        edge_type: "HAS_OUTPUT",
      },
    ];

    for (const c of criteria) {
      const critId = String(c?.criterion_id ?? c?.id ?? "");
      if (!critId) continue;
      criterionSlots.push({ criterion_id: critId, index: triplets.length });
      triplets.push({
        source_type: "EvalTriggerOutput",
        source_data: outputNodeData,
        target_type: "CriterionResult",
        target_data: {
          id: `crit-${runId}-${critId}`,
          criterion_id: critId,
          title: typeof c?.title === "string" ? c.title : "",
          verdict: c?.verdict === "pass" ? "pass" : "fail",
          reasoning: typeof c?.reasoning === "string" ? c.reasoning : "",
          ...(c?.flagged === true ? { flagged: true, llm_flag_reason: String(c?.llm_flag_reason ?? "") } : {}),
          ...(c?.contested === true ? { contested: true } : {}),
          ...(typeof c?.document_excerpt === "string" && c.document_excerpt
            ? { document_excerpt: c.document_excerpt }
            : {}),
        },
        edge_type: "HAS_CRITERION_RESULT",
      });
      triplets.push({
        source_type: "EvalRequirement",
        // Production id convention: "<task_slug>-<criterion_id>".
        source_data: { id: `${cfg.evalsetId}-${critId}` },
        target_type: "EvalTrigger",
        target_data: { id: trigger_id },
        edge_type: "HAS_TRIGGER",
      });
    }

    return { triplets, criterionSlots, trigger_id, output_id };
  },
});
