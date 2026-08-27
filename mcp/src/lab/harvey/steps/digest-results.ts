import { z, defineStep } from "vein";

/**
 * Aggregate a batch of graded Harvey results into the compact digest the
 * evolve loop's PROPOSE beat consumes (EVOLVE_SPEC §2: a proposer must see
 * the aggregate across the dataset, never one example, or it overfits).
 *
 * RUBRIC DISCIPLINE: what leaves this step is the VERDICT channel — per-
 * criterion pass/fail plus a truncated excerpt of the judge's reasoning for
 * FAILED criteria only. That is the channel EVOLVE_SPEC §6/§7 knowingly
 * accepts (hill-climbing against verdicts is the train-set problem, answered
 * by a train/val split — scores on the tuned tasks are TRAIN scores). The
 * rubric itself is never read here: `harvey/get-task` strips it, and this
 * step only sees what the benchmark's own score report emits.
 *
 * Input entries are harvey-run / harvey-candidate-run outputs (score,
 * all_pass, criteria_results, plus optional error fields from onError
 * fallbacks). Field access is defensive — the score report is the
 * benchmark's own JSON, not a shape we control.
 */

interface AnyRec {
  [k: string]: unknown;
}

function truncate(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? t.slice(0, max) + " […]" : t;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v : undefined;
}

/** An error field may be a string or an { message } object (a RunResult's
 *  error) — normalize both. */
function errStr(v: unknown): string | undefined {
  if (typeof v === "string") return str(v);
  if (v && typeof v === "object") return str((v as AnyRec)["message"]);
  return undefined;
}

function num(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

function criterionFailed(c: AnyRec): boolean {
  const verdict = str(c["verdict"]);
  if (verdict) return verdict.toLowerCase() !== "pass";
  const passed = c["passed"] ?? c["pass"] ?? c["met"];
  if (typeof passed === "boolean") return !passed;
  return false; // unknown shape — don't invent failures
}

function criterionNote(c: AnyRec, maxChars: number): string {
  const label = str(c["id"]) ?? str(c["title"]) ?? str(c["criterion"]) ?? str(c["name"]) ?? "criterion";
  const reason = str(c["reasoning"]) ?? str(c["rationale"]) ?? str(c["reason"]);
  return truncate(reason ? `${label}: ${reason}` : label, maxChars);
}

export default defineStep({
  type: "harvey/digest-results",
  description:
    "Aggregate an array of graded Harvey results (harvey-run / harvey-candidate-run outputs) into a compact digest: per-task score, all_pass, failed-criteria excerpts (capped), errors, plus mean score / all-pass count and a preformatted `text` block for an LLM prompt. Config: results (array), maxCriteria? (failed-criteria excerpts per task, default 6), maxChars? (per excerpt, default 240). Output: { n, meanScore, allPassCount, results, text }.",
  input: z.object({
    results: z.array(z.any()).describe("graded results, one per task (harvey-run / harvey-candidate-run outputs)"),
    maxCriteria: z
      .number()
      .int()
      .min(0)
      .default(6)
      .describe("max failed-criteria excerpts to keep per task (overflow is counted, not silently dropped)"),
    maxChars: z.number().int().positive().default(240).describe("max characters per failed-criterion excerpt"),
  }),
  output: z.any(),
  async run(cfg) {
    const entries = (cfg.results as AnyRec[]).map((raw, i) => {
      const r = (raw ?? {}) as AnyRec;
      const task = str(r["task"]) ?? `result ${i + 1}`;
      const score = typeof r["score"] === "number" ? (r["score"] as number) : 0;
      const allPass = r["all_pass"] === true;
      const crit = Array.isArray(r["criteria_results"]) ? (r["criteria_results"] as AnyRec[]) : [];
      const failed = crit.filter(criterionFailed);
      const error = errStr(r["error"]) ?? errStr(r["gradeError"]) ?? errStr(r["produceError"]);
      const runOut = ((r["runResult"] as AnyRec | undefined)?.["output"] ?? {}) as AnyRec;
      const cost = num(r["produceCost"]) ?? num(r["cost"]) ?? num(runOut["cost"]);
      return {
        task,
        score,
        all_pass: allPass,
        nCriteria: crit.length,
        nFailed: failed.length,
        failed: failed.slice(0, cfg.maxCriteria).map((c) => criterionNote(c, cfg.maxChars)),
        failedOmitted: Math.max(0, failed.length - cfg.maxCriteria),
        ...(cost != null ? { cost } : {}),
        ...(error ? { error: truncate(error, cfg.maxChars) } : {}),
      };
    });

    const n = entries.length;
    const meanScore = n ? Math.round((entries.reduce((s, e) => s + e.score, 0) / n) * 1000) / 1000 : 0;
    const allPassCount = entries.filter((e) => e.all_pass).length;

    // Preformatted for an LLM prompt ({{ digest.text }}) — objects
    // interpolated into template strings would arrive as raw JSON.
    const lines: string[] = [`${n} task(s) — mean score ${meanScore}, all-pass ${allPassCount}/${n}`];
    for (const e of entries) {
      lines.push(
        `- ${e.task}  score ${e.score}  all_pass=${e.all_pass}` +
          (e.nCriteria ? `  (${e.nFailed}/${e.nCriteria} criteria failed)` : ""),
      );
      for (const f of e.failed) lines.push(`    ✗ ${f}`);
      if (e.failedOmitted > 0) lines.push(`    (+${e.failedOmitted} more failed criteria not shown)`);
      if (e.error) lines.push(`    ERROR: ${e.error}`);
    }

    return { n, meanScore, allPassCount, results: entries, text: lines.join("\n") };
  },
});
