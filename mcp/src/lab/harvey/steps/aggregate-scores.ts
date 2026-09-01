import { z, defineStep } from "vein";

/**
 * Fold the per-criterion judge verdicts into the run's scores object (the
 * format_results / scores_json equivalent, shaped like the harvey-labs
 * harness: score = number of passes, all-pass gating).
 *
 * `results` is the judge foreach's output array, IN THE SAME ORDER as
 * `rubric` (vein's foreach preserves input order), so criterion identity
 * comes from the zip — the judge LLM never has to echo ids back. Each entry
 * is a core-agent schema-mode output ({ object: { verdict, reasoning },
 * cost, usage }); a null/malformed entry (judge blew up, onError fallback)
 * counts as an honest FAIL with the error recorded, never as a pass.
 */
export default defineStep({
  type: "harvey/aggregate-scores",
  description:
    "Zip rubric criteria with their judge verdicts (same order) into scores_json: { score, max_score, " +
    "n_passed, n_total, pass_rate, all_pass, judge_model, criteria_results, failed, judgeCost }. " +
    "Null/malformed judge entries count as fails.",
  input: z.object({
    rubric: z.array(z.any()).describe("The judged criteria, in the order they were judged."),
    results: z.array(z.any()).describe("Judge foreach output — one agent schema-mode result per criterion, same order."),
    judge_model: z.string().optional(),
    dropped: z.array(z.any()).default([]).describe("Criterion ids excluded as contested (recorded, not scored)."),
  }),
  output: z.any(),
  async run(cfg) {
    if (cfg.results.length !== cfg.rubric.length) {
      // A length mismatch means the zip is unsafe — misattributed verdicts
      // are worse than a loud failure.
      throw new Error(
        `harvey/aggregate-scores: ${cfg.results.length} results for ${cfg.rubric.length} criteria — refusing to zip`,
      );
    }
    const criteria_results = cfg.rubric.map((c, i) => {
      const crit = c as Record<string, any>;
      const r = cfg.results[i] as Record<string, any> | null | undefined;
      const obj = r && typeof r === "object" ? (r.object as Record<string, any> | undefined) : undefined;
      const verdict = obj?.verdict === "pass" ? "pass" : "fail";
      const reasoning =
        typeof obj?.reasoning === "string" && obj.reasoning
          ? obj.reasoning
          : r && typeof r === "object" && typeof r.error === "string"
            ? `judge error: ${r.error}`
            : obj
              ? "(no reasoning returned)"
              : "judge produced no verdict";
      return {
        id: String(crit?.id ?? `criterion-${i + 1}`),
        criterion_id: String(crit?.id ?? `criterion-${i + 1}`),
        title: crit?.title ?? "",
        deliverables: Array.isArray(crit?.deliverables) ? crit.deliverables : [],
        verdict,
        reasoning,
      };
    });

    const n_total = criteria_results.length;
    const n_passed = criteria_results.filter((c) => c.verdict === "pass").length;
    const failed = criteria_results
      .filter((c) => c.verdict === "fail")
      .map((c, _, __) => ({ ...c, match_criteria: (cfg.rubric.find((r: any) => String(r?.id) === c.id) as any)?.match_criteria ?? "" }));

    const judgeCost = cfg.results.reduce(
      (sum, r) => sum + (typeof (r as any)?.cost === "number" ? (r as any).cost : 0),
      0,
    );

    return {
      score: n_passed,
      max_score: n_total,
      n_passed,
      n_total,
      pass_rate: n_total > 0 ? n_passed / n_total : 0,
      all_pass: n_total > 0 && n_passed === n_total,
      judge_model: cfg.judge_model ?? null,
      criteria_results,
      failed,
      dropped: cfg.dropped,
      judgeCost,
    };
  },
});
