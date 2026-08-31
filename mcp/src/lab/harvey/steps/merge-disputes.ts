import { z, defineStep } from "vein";

/**
 * The merge_dispute_flags equivalent: LEFT-JOIN dispute audit results onto
 * the FAILED criteria results only (a pass verdict is never annotated).
 * Dispute entries arrive in the same order as `failed` (the dispute foreach
 * iterates the failed list), so identity comes from the zip — the disputed
 * criterion's echoed `id` is cross-checked when present but never trusted
 * over position.
 *
 * The dispute contract (HARVEY_LAB_JUDGE_DISPUTE_PROMPT's classification):
 *   flagged    — the FAIL verdict looks wrong (judge_error) or the criterion
 *                is defective (criterion_validity)
 *   contested  — criterion_validity specifically: the criterion definition is
 *                bad, durably excluded from future scoring
 *   flag_basis / llm_flag_reason / document_excerpt — the audit narrative
 *
 * FAIL-SOFT by design: a broken/missing/short dispute array annotates
 * nothing — the original verdicts always survive untouched. Disputes never
 * flip a verdict.
 *
 * Outputs, beyond the annotated criteria_results:
 *   annotations             — per disputed criterion, joined with its
 *                             CriterionResult ref_id (via `criterionRefs`) for
 *                             the graph write-back foreach
 *   contested_requirements  — EvalRequirement ref_ids to mark contested=true
 */
export default defineStep({
  type: "harvey/merge-disputes",
  description:
    "Left-join dispute audit results (flagged / flag_basis / contested / llm_flag_reason / " +
    "document_excerpt) onto the FAILED criteria results, zip-by-order. Fail-soft: any shape surprise " +
    "annotates nothing. Output: { criteria_results, annotations, flagged_count, contested_count, " +
    "contested_requirements }.",
  input: z.object({
    criteria_results: z.array(z.any()).describe("Full criteria_results from harvey/aggregate-scores."),
    failed: z.array(z.any()).default([]).describe("The failed subset, in the order disputes ran."),
    disputes: z
      .any()
      .optional()
      .describe("Dispute foreach output — one agent schema-mode result per failed criterion, same order."),
    criterionRefs: z
      .any()
      .optional()
      .describe("[{ criterion_id, ref_id }] for the persisted CriterionResult nodes (harvey/criterion-refs output)."),
    requirements: z
      .any()
      .optional()
      .describe("EvalRequirement nodes (graph-get-batched output) — used to resolve contested ref_ids."),
    evalsetId: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    type Annotation = {
      criterion_id: string;
      ref_id?: string;
      flagged: boolean;
      contested: boolean;
      flag_basis?: string;
      llm_flag_reason: string;
      document_excerpt?: string;
    };
    const annotations = new Map<string, Annotation>();
    try {
      const disputes = Array.isArray(cfg.disputes) ? cfg.disputes : [];
      const refs = new Map<string, string>();
      if (Array.isArray(cfg.criterionRefs)) {
        for (const r of cfg.criterionRefs) {
          const cid = (r as Record<string, any>)?.criterion_id;
          const ref = (r as Record<string, any>)?.ref_id;
          if (typeof cid === "string" && typeof ref === "string") refs.set(cid, ref);
        }
      }
      cfg.failed.forEach((f, i) => {
        const id = String((f as Record<string, any>)?.id ?? "");
        const d = disputes[i] as Record<string, any> | null | undefined;
        const obj = d && typeof d === "object" ? (d.object as Record<string, any> | undefined) : undefined;
        if (!id || !obj) return;
        annotations.set(id, {
          criterion_id: id,
          ...(refs.has(id) ? { ref_id: refs.get(id) } : {}),
          flagged: obj.flagged === true,
          contested: obj.contested === true,
          ...(typeof obj.flag_basis === "string" ? { flag_basis: obj.flag_basis } : {}),
          llm_flag_reason:
            typeof obj.llm_flag_reason === "string" ? obj.llm_flag_reason : typeof obj.reason === "string" ? obj.reason : "",
          ...(typeof obj.document_excerpt === "string" ? { document_excerpt: obj.document_excerpt } : {}),
        });
      });
    } catch {
      annotations.clear(); // fail-soft: original verdicts survive
    }

    const criteria_results = cfg.criteria_results.map((c) => {
      const id = String((c as Record<string, any>)?.id ?? "");
      const a = annotations.get(id);
      if (!a || (c as Record<string, any>)?.verdict !== "fail") return c;
      const { criterion_id: _cid, ref_id: _ref, ...fields } = a;
      return { ...(c as Record<string, any>), ...fields };
    });

    // Resolve the EvalRequirement ref_ids for durably-contested criteria so
    // the workflow can write contested=true back onto the requirement nodes.
    // Requirement ids follow the production convention "<evalsetId>-<criterionId>".
    const contestedIds = [...annotations.values()].filter((a) => a.contested).map((a) => a.criterion_id);
    const contested_requirements: Array<{ criterion_id: string; ref_id: string }> = [];
    try {
      const nodes = Array.isArray(cfg.requirements) ? cfg.requirements : [];
      for (const id of contestedIds) {
        const wanted = cfg.evalsetId ? `${cfg.evalsetId}-${id}` : id;
        for (const n of nodes) {
          const props = (n as Record<string, any>)?.properties;
          const refId = (n as Record<string, any>)?.ref_id;
          if (props && typeof refId === "string" && (props.id === wanted || props.id === id)) {
            contested_requirements.push({ criterion_id: id, ref_id: refId });
            break;
          }
        }
      }
    } catch {
      contested_requirements.length = 0;
    }

    return {
      criteria_results,
      // Only annotations that changed something AND can be written somewhere.
      annotations: [...annotations.values()].filter((a) => a.ref_id && (a.flagged || a.contested)),
      flagged_count: [...annotations.values()].filter((a) => a.flagged).length,
      contested_count: contestedIds.length,
      contested_requirements,
    };
  },
});
