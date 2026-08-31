import { z, defineStep } from "vein";

/**
 * The merge_dispute_flags equivalent: LEFT-JOIN dispute annotations onto the
 * FAILED criteria results only (a pass verdict is never annotated). Dispute
 * entries arrive in the same order as `failed` (the dispute foreach iterates
 * the failed list), so identity comes from the zip.
 *
 * FAIL-SOFT by design: a broken/missing/short dispute array annotates
 * nothing — the original verdicts always survive untouched. Disputes never
 * flip a verdict; they only add:
 *   - flagged + llm_flag_reason  — "this FAIL verdict looks wrong" (per-run)
 *   - contested                  — "this criterion definition is bad" (durable;
 *                                  also surfaced as contested_requirements for
 *                                  the EvalRequirement write-back)
 */
export default defineStep({
  type: "harvey/merge-disputes",
  description:
    "Left-join dispute annotations (flagged / llm_flag_reason / contested) onto the FAILED criteria " +
    "results, zip-by-order. Fail-soft: any shape surprise annotates nothing. Output: { criteria_results, " +
    "flagged_count, contested_count, contested_requirements }.",
  input: z.object({
    criteria_results: z.array(z.any()).describe("Full criteria_results from harvey/aggregate-scores."),
    failed: z.array(z.any()).default([]).describe("The failed subset, in the order disputes ran."),
    disputes: z
      .any()
      .optional()
      .describe("Dispute foreach output — one agent schema-mode result per failed criterion, same order."),
    requirements: z
      .any()
      .optional()
      .describe("EvalRequirement nodes (graph-get-batched output) — used to resolve contested ref_ids."),
    evalsetId: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const annotations = new Map<string, { flagged: boolean; llm_flag_reason: string; contested: boolean }>();
    try {
      const disputes = Array.isArray(cfg.disputes) ? cfg.disputes : [];
      cfg.failed.forEach((f, i) => {
        const id = String((f as Record<string, any>)?.id ?? "");
        const d = disputes[i] as Record<string, any> | null | undefined;
        const obj = d && typeof d === "object" ? (d.object as Record<string, any> | undefined) : undefined;
        if (!id || !obj) return;
        annotations.set(id, {
          flagged: obj.flagged === true,
          llm_flag_reason: typeof obj.reason === "string" ? obj.reason : "",
          contested: obj.contested === true,
        });
      });
    } catch {
      annotations.clear(); // fail-soft: original verdicts survive
    }

    const criteria_results = cfg.criteria_results.map((c) => {
      const id = String((c as Record<string, any>)?.id ?? "");
      const a = annotations.get(id);
      if (!a || (c as Record<string, any>)?.verdict !== "fail") return c;
      return { ...(c as Record<string, any>), flagged: a.flagged, llm_flag_reason: a.llm_flag_reason, contested: a.contested };
    });

    // Resolve the EvalRequirement ref_ids for durably-contested criteria so
    // the workflow can write contested=true back onto the requirement nodes.
    const contestedIds = [...annotations.entries()].filter(([, a]) => a.contested).map(([id]) => id);
    const contested_requirements: Array<{ criterion_id: string; ref_id: string }> = [];
    try {
      const nodes = Array.isArray(cfg.requirements) ? cfg.requirements : [];
      for (const id of contestedIds) {
        const wanted = cfg.evalsetId ? `${cfg.evalsetId}/${id}` : id;
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
      flagged_count: [...annotations.values()].filter((a) => a.flagged).length,
      contested_count: contestedIds.length,
      contested_requirements,
    };
  },
});
