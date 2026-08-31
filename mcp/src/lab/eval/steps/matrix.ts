import { z, defineStep } from "vein";

/**
 * The task×version MATRIX — the evolve harness's cross-measurement memory
 * (plans/evolve-scoreboard-and-task-matrix.md, Phase 1). Where a digest
 * summarizes ONE measurement of one version, this step folds EVERY
 * measurement of every version into the per-task view that single digests
 * structurally cannot show:
 *
 *  - BANDS: floor (correct in every measurement — regression ballast, not
 *    signal), movable (flips between measurements — where ALL the fitness
 *    dynamic range lives), ceiling (never correct — unreachable by the
 *    approaches measured so far).
 *  - EMPIRICAL NOISE FLOOR: same-version re-measurements are re-runs of
 *    identical YAML, so their fitness deltas and per-task flips measure
 *    produce-sampling noise directly. `noise.maxAbsFitnessDelta` is the
 *    margin a challenger must clear before a comparison means anything —
 *    an observed number, not a hand-set param.
 *  - BIAS vs VARIANCE: a never-correct task whose non-empty wrong answers
 *    are byte-identical across ≥3 measurements is a BIAS failure — the
 *    approach is systematically wrong (broken data path, wrong method) and
 *    provably immune to redundancy/reconciliation and to prompt nudging.
 *    Distinct wrong answers are VARIANCE — sampling scatter, where
 *    redundancy helps. The two demand opposite fixes; one measurement can
 *    never tell them apart.
 *
 * GOLD DISCIPLINE (EVOLVE_SPEC §6): verdicts, the candidates' own answers,
 * and produce/tool errors only — the gold never enters or leaves.
 *
 * Input entries are gaia-run / gaia-candidate-run style graded results (the
 * same shapes gaia/digest-results normalizes; the correctness/answer/error
 * normalizers are duplicated here because seeded steps are self-contained
 * files). Measurements are ordered oldest → newest; a version may appear in
 * any number of measurements (baseline k-samples, incumbent re-measurements,
 * a challenger's first sample).
 */

interface AnyRec {
  [k: string]: unknown;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v : undefined;
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/** An error field may be a string or an { message } object. */
function errStr(v: unknown): string | undefined {
  if (typeof v === "string") return str(v);
  if (v && typeof v === "object") return str((v as AnyRec)["message"]);
  return undefined;
}

/** Same three correctness shapes gaia/digest-results accepts: boolean
 *  (candidate-run), single-entry score-results array (gaia-run), 0/1 count
 *  with total 1 (gaia-run). Unreadable → false, never true. */
function correctOf(r: AnyRec): boolean {
  if (typeof r["correct"] === "boolean") return r["correct"] as boolean;
  const arr = Array.isArray(r["results"]) ? (r["results"] as AnyRec[]) : [];
  if (arr.length === 1 && typeof arr[0]?.["correct"] === "boolean") return arr[0]["correct"] as boolean;
  if (typeof r["correct"] === "number" && num(r["total"]) === 1) return r["correct"] === 1;
  return false;
}

function truncate(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? t.slice(0, max) + " […]" : t;
}

function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

/** One task's observation in one measurement. */
interface Obs {
  version: string;
  measurement: number;
  correct: boolean;
  answer: string;
  error?: string;
}

export default defineStep({
  type: "eval/matrix",
  description:
    "Fold MULTIPLE graded measurements (each: one version run over the task set) into the task×version matrix: per-task bands (floor / movable / ceiling), per-version fitness samples, an EMPIRICAL noise floor from same-version re-measurements (fitness deltas + task flips on identical YAML), and bias-vs-variance tags for never-correct tasks (byte-identical wrong answer across ≥3 measurements = bias; distinct wrong answers = variance). Gold never enters or leaves. Two input modes (exactly one): `measurements` (array of { version, results }, oldest → newest) for mixed versions, or `version` + `samples` (array of results-arrays) for k re-measurements of ONE version — the baseline-capture form a workflow's nested foreach produces, since template expressions cannot construct object arrays. When exactly one version was measured, the output adds top-level `fitness` (the MAX sample — the conservative bar a challenger must beat), so the matrix object plugs straight into eval/evolve-loop's `baseline`. Results are gaia-run / gaia-candidate-run style graded outputs. Config: measurements? | (version? + samples?), maxAnswerChars? (default 120), maxQuestionChars? (default 200). Output: { tasks, versions, bands, noise, text, fitness? }.",
  input: z.object({
    measurements: z
      .array(
        z.object({
          version: z.string().describe("the workflow version this measurement graded (baseline runs use the base workflow name)"),
          results: z.array(z.any()).describe("graded results, one per task"),
        }),
      )
      .min(1)
      .optional()
      .describe("all measurements so far, oldest → newest (mixed-version mode)"),
    version: z.string().optional().describe("single-version mode: the version every entry in `samples` measured"),
    samples: z
      .array(z.array(z.any()))
      .min(1)
      .optional()
      .describe("single-version mode: k re-measurements of `version`, each an array of graded results (a nested foreach's output)"),
    maxAnswerChars: z.number().int().positive().default(120),
    maxQuestionChars: z.number().int().positive().default(200),
  }),
  output: z.any(),
  async run(rawCfg) {
    const modeA = rawCfg.measurements != null;
    const modeB = rawCfg.version != null && rawCfg.samples != null;
    if (modeA === modeB) {
      throw new Error(
        "eval/matrix takes EITHER `measurements` OR `version` + `samples` — exactly one mode",
      );
    }
    const cfg = {
      ...rawCfg,
      measurements:
        rawCfg.measurements ??
        rawCfg.samples!.map((results) => ({ version: rawCfg.version!, results })),
    };
    // ── fold every measurement into per-task observations ────────────────
    const byTask = new Map<string, { level: number | null; question?: string; obs: Obs[] }>();
    const versionOrder: string[] = [];
    const byVersion = new Map<string, { fitness: number[]; vectors: Map<string, boolean>[] }>();

    cfg.measurements.forEach((m, mi) => {
      if (!byVersion.has(m.version)) {
        byVersion.set(m.version, { fitness: [], vectors: [] });
        versionOrder.push(m.version);
      }
      const vec = new Map<string, boolean>();
      let correctCount = 0;
      const entries = m.results as AnyRec[];
      for (const raw of entries) {
        const r = (raw ?? {}) as AnyRec;
        const taskId = str(r["taskId"]);
        if (!taskId) continue;
        const correct = correctOf(r);
        if (correct) correctCount++;
        vec.set(taskId, correct);
        const rec = byTask.get(taskId) ?? { level: num(r["level"]) ?? null, obs: [] };
        rec.level ??= num(r["level"]) ?? null;
        // Task-visible text only (every producer sees the question) — never
        // the gold. Kept so briefings can show WHAT a stuck task asks.
        rec.question ??= str(r["question"]);
        rec.obs.push({
          version: m.version,
          measurement: mi,
          correct,
          answer: typeof r["answer"] === "string" ? (r["answer"] as string) : "",
          error: errStr(r["error"]) ?? errStr(r["produceError"]) ?? errStr(r["gradeError"]),
        });
        byTask.set(taskId, rec);
      }
      const v = byVersion.get(m.version)!;
      v.fitness.push(entries.length ? round3(correctCount / entries.length) : 0);
      v.vectors.push(vec);
    });

    // ── per-task rows: band, flips, bias-vs-variance ─────────────────────
    const tasks = [...byTask.entries()].map(([taskId, { level, question, obs }]) => {
      const n = obs.length;
      const solved = obs.filter((o) => o.correct).length;
      const band = solved === n ? "floor" : solved === 0 ? "ceiling" : "movable";
      // Flips in measurement order — how unstable this task is overall.
      let flips = 0;
      for (let i = 1; i < obs.length; i++) if (obs[i]!.correct !== obs[i - 1]!.correct) flips++;
      const wrong = obs.filter((o) => !o.correct);
      const wrongAnswers = [...new Set(wrong.map((o) => o.answer.trim()).filter(Boolean))];
      const emptyCount = wrong.filter((o) => !o.answer.trim()).length;
      const errors = [...new Set(wrong.map((o) => o.error).filter((e): e is string => Boolean(e)))];
      // BIAS: never correct, one distinct non-empty wrong answer, seen ≥3×.
      const bias =
        band === "ceiling" && wrongAnswers.length === 1 && wrong.length - emptyCount >= 3;
      return {
        taskId,
        level,
        band,
        n,
        solved,
        flips,
        ...(band !== "floor" && question ? { question: truncate(question, cfg.maxQuestionChars) } : {}),
        ...(wrongAnswers.length
          ? { wrongAnswers: wrongAnswers.slice(0, 5).map((a) => truncate(a, cfg.maxAnswerChars)) }
          : {}),
        ...(emptyCount ? { emptyCount } : {}),
        ...(bias ? { bias: true, repeatedAnswer: truncate(wrongAnswers[0]!, cfg.maxAnswerChars) } : {}),
        ...(errors.length ? { errors: errors.slice(0, 3).map((e) => truncate(e, 200)) } : {}),
      };
    });
    const bandOrder = { floor: 0, movable: 1, ceiling: 2 } as const;
    tasks.sort(
      (a, b) =>
        bandOrder[a.band as keyof typeof bandOrder] - bandOrder[b.band as keyof typeof bandOrder] ||
        b.solved / b.n - a.solved / a.n ||
        a.taskId.localeCompare(b.taskId),
    );

    // ── per-version rows ─────────────────────────────────────────────────
    const versions = versionOrder.map((version) => {
      const v = byVersion.get(version)!;
      const mean = v.fitness.reduce((s, f) => s + f, 0) / v.fitness.length;
      return {
        version,
        n: v.fitness.length,
        fitness: v.fitness,
        meanFitness: round3(mean),
        minFitness: Math.min(...v.fitness),
        maxFitness: Math.max(...v.fitness),
      };
    });

    // ── empirical noise floor: same-version re-measurement pairs ─────────
    // Identical YAML re-run — every fitness delta and task flip between such
    // a pair is pure produce-sampling noise, measured for free.
    let pairs = 0;
    let maxAbsFitnessDelta = 0;
    let sumAbsFitnessDelta = 0;
    let maxTaskFlips = 0;
    for (const v of byVersion.values()) {
      for (let i = 0; i < v.fitness.length; i++) {
        for (let j = i + 1; j < v.fitness.length; j++) {
          pairs++;
          const d = Math.abs(v.fitness[i]! - v.fitness[j]!);
          maxAbsFitnessDelta = Math.max(maxAbsFitnessDelta, d);
          sumAbsFitnessDelta += d;
          let flips = 0;
          for (const [taskId, ci] of v.vectors[i]!) {
            const cj = v.vectors[j]!.get(taskId);
            if (cj !== undefined && cj !== ci) flips++;
          }
          maxTaskFlips = Math.max(maxTaskFlips, flips);
        }
      }
    }
    const noise = {
      sameVersionPairs: pairs,
      ...(pairs
        ? {
            maxAbsFitnessDelta: round3(maxAbsFitnessDelta),
            meanAbsFitnessDelta: round3(sumAbsFitnessDelta / pairs),
            maxTaskFlips,
          }
        : {}),
      // The margin a fitness comparison must clear to be signal. With no
      // same-version pairs there is NO measured floor — "unknown" must read
      // as "re-measure something", never as zero.
      floorKnown: pairs > 0,
      ...(pairs ? { suggestedMargin: round3(maxAbsFitnessDelta) } : {}),
    };

    const bands = {
      floor: tasks.filter((t) => t.band === "floor").length,
      movable: tasks.filter((t) => t.band === "movable").length,
      ceiling: tasks.filter((t) => t.band === "ceiling").length,
    };

    // ── text rendering for briefings ─────────────────────────────────────
    const lines: string[] = [];
    lines.push(
      `TASK×VERSION MATRIX — ${tasks.length} task(s) × ${cfg.measurements.length} measurement(s) of ${versions.length} version(s).`,
    );
    lines.push(
      `Bands: ${bands.floor} floor (correct in every measurement — not signal), ` +
        `${bands.movable} movable (the fitness dynamic range), ` +
        `${bands.ceiling} ceiling (never correct by any measured approach).`,
    );
    if (noise.floorKnown) {
      lines.push(
        `Measured noise floor (from ${pairs} same-version re-measurement pair(s)): identical YAML ` +
          `re-runs differed by up to ${noise.maxAbsFitnessDelta} fitness (${maxTaskFlips} task flip(s)). ` +
          `Fitness deltas within ±${noise.suggestedMargin} are NOISE — treat them as ties.`,
      );
    } else {
      lines.push(
        `Noise floor UNKNOWN — no version has been measured twice yet. Until one is, no fitness ` +
          `comparison here is trustworthy; re-measure the incumbent before believing any delta.`,
      );
    }
    lines.push("");
    lines.push("VERSIONS (oldest → newest):");
    for (const v of versions) {
      lines.push(
        `- ${v.version}: ${v.n === 1 ? `fitness ${v.fitness[0]}` : `fitness ${v.fitness.join(" / ")} (mean ${v.meanFitness})`} over ${v.n} measurement(s)`,
      );
    }
    const movable = tasks.filter((t) => t.band === "movable");
    if (movable.length) {
      lines.push("");
      lines.push("MOVABLE tasks (all fitness movement lives here; high flip counts are sampling noise, not approach signal):");
      for (const t of movable) {
        lines.push(
          `- ${t.taskId}${t.level != null ? ` (L${t.level})` : ""}: correct ${t.solved}/${t.n}, ${t.flips} flip(s)` +
            (t.wrongAnswers?.length ? ` — wrong answers seen: ${t.wrongAnswers.map((a) => `"${a}"`).join(", ")}` : ""),
        );
        if (t.question) lines.push(`    question: ${t.question}`);
      }
    }
    const ceiling = tasks.filter((t) => t.band === "ceiling");
    if (ceiling.length) {
      lines.push("");
      lines.push("CEILING tasks (0 correct across every measurement):");
      for (const t of ceiling) {
        lines.push(
          `- ${t.taskId}${t.level != null ? ` (L${t.level})` : ""}: ` +
            (t.bias
              ? `BIAS — answered "${t.repeatedAnswer}" identically in every non-empty measurement. The approach is ` +
                `systematically wrong (bad data path or method); redundancy, reconciliation, and prompt nudges cannot fix it — root-cause it.`
              : t.wrongAnswers?.length
                ? `VARIANCE — distinct wrong answers across measurements: ${t.wrongAnswers.map((a) => `"${a}"`).join(", ")}`
                : `every measurement returned an empty answer`) +
            (t.errors?.length ? ` [errors: ${t.errors.join(" | ")}]` : ""),
        );
        if (t.question) lines.push(`    question: ${t.question}`);
      }
    }

    // Single-version matrices (the k-sample baseline capture) also report a
    // top-level `fitness` — the MAX sample, i.e. the baseline's best observed
    // draw, the conservative bar a challenger must beat by more than the
    // measured margin — so this object plugs directly into eval/evolve-loop's
    // `baseline` (which reads `.fitness` and `.text`).
    const single = versions.length === 1 ? { fitness: versions[0]!.maxFitness } : {};

    return { tasks, versions, bands, noise, ...single, text: lines.join("\n") };
  },
});
