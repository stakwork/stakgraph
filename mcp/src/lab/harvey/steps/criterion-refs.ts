import { z, defineStep } from "vein";

/**
 * Zip harvey/build-eval-chain's criterion slots with graph/create-batch-
 * triplet's per-triplet results to recover each persisted CriterionResult
 * node's ref_id: slots name the triplet-array INDEX of each criterion's
 * HAS_CRITERION_RESULT write, and the batch step returns results in input
 * order with the created target's ref_id.
 *
 * The refs feed the dispute agents ("## Criterion Result Refs" — Cause
 * triplets hang off them) and the annotation write-back foreach.
 *
 * FAIL-SOFT: a failed/absent record write yields [] — dispute and write-back
 * then simply run without graph anchoring, never blocking scoring.
 */
export default defineStep({
  type: "harvey/criterion-refs",
  description:
    "Recover persisted CriterionResult ref_ids: zip build-eval-chain's criterionSlots with " +
    "create-batch-triplet's results. Fail-soft ([] on any shape surprise). Output: " +
    "[{ criterion_id, ref_id }].",
  input: z.object({
    slots: z.any().optional().describe("harvey/build-eval-chain's criterionSlots: [{ criterion_id, index }]."),
    record: z.any().optional().describe("graph/create-batch-triplet's output ({ results: [...] })."),
  }),
  output: z.any(),
  async run(cfg) {
    const out: Array<{ criterion_id: string; ref_id: string }> = [];
    try {
      const slots = Array.isArray(cfg.slots) ? cfg.slots : [];
      const results = Array.isArray((cfg.record as Record<string, any>)?.results)
        ? ((cfg.record as Record<string, any>).results as Array<Record<string, any>>)
        : [];
      for (const s of slots) {
        const cid = (s as Record<string, any>)?.criterion_id;
        const idx = (s as Record<string, any>)?.index;
        if (typeof cid !== "string" || typeof idx !== "number") continue;
        const ref = results[idx]?.target_ref_id;
        if (typeof ref === "string" && ref && !results[idx]?.error) out.push({ criterion_id: cid, ref_id: ref });
      }
    } catch {
      out.length = 0;
    }
    return out;
  },
});
