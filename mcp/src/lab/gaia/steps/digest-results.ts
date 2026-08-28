import { z, defineStep } from "vein";

/**
 * Aggregate graded GAIA results into the compact digest the evolve loop's
 * PROPOSE beat consumes (EVOLVE_SPEC §2: a proposer must see the aggregate
 * across the dataset, never one example, or it overfits).
 *
 * GOLD DISCIPLINE: what leaves this step is the VERDICT channel only —
 * correct/wrong per task, plus the candidate's OWN answer (its output, not
 * the gold) and the question text (task-visible to every producer anyway).
 * The gold never appears in any input here: gaia/evaluate results carry
 * only { taskId, level, correct }, and gaia.getTask strips `Final answer`.
 *
 * FITNESS: plain accuracy, emitted as `fitness` (the field eval/evolve-loop
 * reads). Binary per task is fine here — unlike harvey's all-pass score,
 * the task SET supplies the gradient (each flip moves accuracy by 1/n).
 *
 * MISS ROUTING (EVOLVE_SPEC §8's taxonomy, the cheap code-only version):
 * each miss is tagged from mechanical signals — produce ERROR (harness/
 * tooling blew up), EMPTY answer (the agent gave up or crashed into the
 * fallback), or plain WRONG (formatting or substance — the answer excerpt
 * is there so the author can tell which). The summary line counts them so
 * an author sees at a glance which layer owns the misses.
 *
 * Input entries are gaia-run / gaia-candidate-run outputs. Field access is
 * defensive: gaia-run reports `correct` as a COUNT with a per-task
 * `results` array (its score call), gaia-candidate-run as a boolean.
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
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/** Normalize the three correctness shapes that reach this step:
 *  boolean (gaia-candidate-run), a single-entry score-results array
 *  (gaia-run carries its score call's `results`), or a 0/1 count with
 *  total 1 (gaia-run's `correct`). Unreadable → false, never true. */
function correctOf(r: AnyRec): boolean {
  if (typeof r["correct"] === "boolean") return r["correct"] as boolean;
  const arr = Array.isArray(r["results"]) ? (r["results"] as AnyRec[]) : [];
  if (arr.length === 1 && typeof arr[0]?.["correct"] === "boolean") return arr[0]["correct"] as boolean;
  if (typeof r["correct"] === "number" && num(r["total"]) === 1) return r["correct"] === 1;
  return false;
}

export default defineStep({
  type: "gaia/digest-results",
  description:
    "Aggregate an array of graded GAIA results (gaia-run / gaia-candidate-run outputs) into a compact digest: accuracy as `fitness` (what eval/evolve-loop reads), per-task correct/wrong with the produced answer excerpt, miss tags (wrong-answer / empty-answer / produce-error), question excerpts for misses, and a preformatted `text` block for an LLM prompt. Gold never enters or leaves this step. Config: results (array), maxAnswerChars? (default 160), maxQuestionChars? (default 240). Output: { n, correctCount, accuracy, fitness, byLevel, results, text }.",
  input: z.object({
    results: z.array(z.any()).describe("graded results, one per task (gaia-run / gaia-candidate-run outputs)"),
    maxAnswerChars: z.number().int().positive().default(160).describe("max characters of the produced answer per task"),
    maxQuestionChars: z
      .number()
      .int()
      .positive()
      .default(240)
      .describe("max characters of the question excerpt shown for missed tasks"),
  }),
  output: z.any(),
  async run(cfg) {
    const entries = (cfg.results as AnyRec[]).map((raw, i) => {
      const r = (raw ?? {}) as AnyRec;
      const taskId = str(r["taskId"]) ?? `result ${i + 1}`;
      const level = num(r["level"]) ?? null;
      const correct = correctOf(r);
      const answer = typeof r["answer"] === "string" ? (r["answer"] as string) : "";
      const error = errStr(r["error"]) ?? errStr(r["produceError"]) ?? errStr(r["gradeError"]);
      // gaia-candidate-run carries cost/steps only inside the candidate's
      // run result (YAML cannot deep-access a failed run's output — the
      // template evaluator does not short-circuit), so unpack here in code.
      const runOut = ((r["runResult"] as AnyRec | undefined)?.["output"] ?? {}) as AnyRec;
      const cost = num(r["cost"]) ?? num(runOut["cost"]);
      const steps = num(r["steps"]) ?? num(runOut["steps"]);
      // Route each miss to the layer that owns it (§8): a produce error is
      // harness/tooling, an empty answer is a give-up (the onError fallback
      // or a bailed agent), a non-empty wrong answer is formatting or
      // substance — the excerpt lets the author tell which.
      const miss = correct ? null : error ? "produce-error" : answer.trim() === "" ? "empty-answer" : "wrong-answer";
      return {
        taskId,
        level,
        correct,
        answer: truncate(answer, cfg.maxAnswerChars),
        ...(miss ? { miss } : {}),
        ...(correct ? {} : { question: truncate(str(r["question"]) ?? "", cfg.maxQuestionChars) }),
        ...(cost != null ? { cost } : {}),
        ...(steps != null ? { steps } : {}),
        ...(error ? { error: truncate(error, 240) } : {}),
      };
    });

    const n = entries.length;
    const correctCount = entries.filter((e) => e.correct).length;
    const accuracy = n ? Math.round((correctCount / n) * 1000) / 1000 : 0;
    const byLevel: Record<string, { correct: number; total: number }> = {};
    for (const e of entries) {
      const key = e.level == null ? "?" : String(e.level);
      byLevel[key] ??= { correct: 0, total: 0 };
      byLevel[key].total += 1;
      if (e.correct) byLevel[key].correct += 1;
    }
    const missCounts: Record<string, number> = {};
    for (const e of entries) {
      if (e.miss) missCounts[e.miss] = (missCounts[e.miss] ?? 0) + 1;
    }

    // Preformatted for an LLM prompt ({{ digest.text }}) — objects
    // interpolated into template strings would arrive as raw JSON.
    const missSummary = Object.entries(missCounts)
      .map(([k, v]) => `${v} ${k}`)
      .join(", ");
    const lines: string[] = [
      `${n} task(s) — accuracy ${accuracy} (${correctCount}/${n} correct)` +
        (missSummary ? `; misses: ${missSummary}` : ""),
    ];
    for (const e of entries) {
      const lvl = e.level == null ? "" : ` (L${e.level})`;
      const meta = [e.steps != null ? `steps ${e.steps}` : "", e.cost != null ? `$${e.cost}` : ""]
        .filter(Boolean)
        .join(", ");
      if (e.correct) {
        lines.push(`- ${e.taskId}${lvl} ✓ correct${meta ? `  (${meta})` : ""}`);
        continue;
      }
      lines.push(
        `- ${e.taskId}${lvl} ✗ ${e.miss?.toUpperCase()}${meta ? `  (${meta})` : ""}` +
          (e.miss === "wrong-answer" ? ` — answered: "${e.answer}"` : ""),
      );
      if (e.question) lines.push(`    question: ${e.question}`);
      if (e.error) lines.push(`    ERROR: ${e.error}`);
    }

    return { n, correctCount, accuracy, fitness: accuracy, byLevel, results: entries, text: lines.join("\n") };
  },
});
