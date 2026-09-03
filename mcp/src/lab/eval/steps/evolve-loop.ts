import { z, defineStep, type RunEvent } from "vein";

/**
 * The GENERIC loop over workflow versions (EVOLVE_SPEC §5.3.3 — promoted
 * from the harvey instance): run a one-generation workflow up to
 * maxGenerations times, briefing each generation with the EVIDENCE so far
 * and letting the author decide what to build on.
 *
 * The briefing is deliberately just evidence, laid out once:
 *   - the TASK LIST (id, level, question) — the only place questions appear;
 *   - a task × version GRID of every version graded in this run, baseline
 *     first (✓ / ✗ / ∅ empty / ! error, or the per-task score where the
 *     domain grades on a gradient), with the fitness row underneath;
 *   - per attempt: version, fitness, the author's approach summary, and
 *     its MISSES only (task → wrong answer / failed criteria / error).
 *
 * There is NO best-so-far anchor, noise margin, or exploit/explore
 * directive any more. One run of one version wobbles by a task or two on
 * sampling luck alone, so every one of those mechanisms was a decision made
 * on noise — and the author, which can read any version with
 * meta/get-workflow, is better placed to weigh the grid than the loop is.
 *
 * Domain-agnostic on purpose: a domain plugs in via
 *   - `genWorkflow`: its one-generation workflow (author → run candidate
 *     over tasks → digest), invoked with
 *     { tasks, mission, candidateName, generation, briefing } and returning
 *     { version?, summary?, changes?, missingSecrets?, authorCost?, digest,
 *       noop? }
 *   - the digest's FITNESS: `digest.fitness` (falling back to
 *     `digest.meanPassRate`, the harvey digest's field), a number in [0,1];
 *   - the digest's `results`: one entry per task, read through
 *     normalizeVerdicts (gaia: taskId/correct/answer/question; harvey:
 *     task/passRate/failed[]) — the grid and miss lines are built from it;
 *   - `fitnessName`: how briefings name the fitness ("accuracy",
 *     "pass-rate", …).
 *
 * Runs generations through `services.optimizer` (vein.run — same capability
 * eval/optimize uses), each as its own persisted run linked from this
 * step's per-generation progress events. Stops on: stopFitness reached,
 * generations exhausted, a cost/time cap, or two consecutive generation-run
 * failures (a broken harness should not burn ten generations of budget).
 *
 * The report names EVERY version that reached the top fitness (ties are
 * ties — one noisy run is no reason to hide one of them); `bestVersion` is
 * the oldest of them, kept for callers that want a single field.
 *
 * TRAIN-SET caveat rides on the output: every generation tunes against the
 * same tasks; the final fitness is a train score (EVOLVE_SPEC §7).
 */

interface AnyRec {
  [k: string]: unknown;
}

interface RunResultLike {
  runId: string;
  status: string;
  output?: unknown;
  error?: { message?: string };
}

interface Optimizer {
  run(
    name: string,
    input: unknown,
    opts?: { paramOverrides?: Record<string, Record<string, unknown>>; parentRunId?: string },
  ): Promise<RunResultLike>;
}

function excerpt(s: unknown, max: number): string {
  if (typeof s !== "string") return "";
  const t = s.replace(/[ \t]+/g, " ").trim();
  return t.length > max ? t.slice(0, max) + " […]" : t;
}

function oneLine(s: unknown, max: number): string {
  if (typeof s !== "string") return "";
  const t = s.replace(/\s+/g, " ").trim();
  return t.length > max ? t.slice(0, max) + " […]" : t;
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v : undefined;
}

/**
 * Authors occasionally end schema mode on a bare text turn, echoing filler
 * ("placeholder", "") into `summary`. Junk there poisons every later
 * briefing and the report the human reads. Replace it with an honest marker
 * instead of passing it through. (The version echo has its own fallback in
 * the gen workflows; this is the summary-channel counterpart.)
 */
const NO_SUMMARY =
  "(no usable approach summary reported by this generation's author — read this version's YAML diff to see what it changed)";
function usableSummary(v: unknown): string | undefined {
  if (typeof v !== "string") return undefined;
  const t = v.trim();
  if (t.length < 8) return undefined;
  if (/^(placeholder|todo|tbd|n\/?a|none|null|summary|unknown)[.!]?$/i.test(t)) return undefined;
  return t;
}

/** How much of an author's approach summary the next generation sees. It
 *  is the only record of what a generation tried, so it is roomy. */
const SUMMARY_CHARS = 3000;
const QUESTION_CHARS = 300;
const ANSWER_CHARS = 200;
const FAILED_CHARS = 600;

/**
 * One task's verdict inside one measurement, normalized across the gaia
 * digest (taskId, correct boolean, answer, question) and the harvey digest
 * (task, passRate, all_pass, failed[]). Unknown shapes yield no verdict —
 * the grid shows "?" rather than inventing one.
 */
export interface TaskVerdict {
  taskId: string;
  level?: number | null;
  question?: string;
  /** binary verdict, when the domain has one */
  correct?: boolean;
  /** graded score in [0,1], when the domain grades on a gradient */
  score?: number;
  answer?: string;
  error?: string;
  failed?: string[];
}

export function normalizeVerdicts(digest: unknown): TaskVerdict[] {
  const d = (digest ?? {}) as AnyRec;
  const arr = Array.isArray(d["results"]) ? (d["results"] as unknown[]) : [];
  const out: TaskVerdict[] = [];
  for (const raw of arr) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as AnyRec;
    const taskId = str(r["taskId"]) ?? str(r["task"]) ?? str(r["label"]);
    if (!taskId) continue;
    const correct =
      typeof r["correct"] === "boolean"
        ? (r["correct"] as boolean)
        : typeof r["all_pass"] === "boolean"
          ? (r["all_pass"] as boolean)
          : undefined;
    // harvey's `score` is the binary all-pass — passRate is the gradient.
    const score = num(r["passRate"]);
    const err = r["error"];
    const failed = Array.isArray(r["failed"])
      ? (r["failed"] as unknown[]).filter((f): f is string => typeof f === "string" && f.trim() !== "")
      : [];
    out.push({
      taskId,
      ...(typeof r["level"] === "number" ? { level: r["level"] as number } : {}),
      ...(str(r["question"]) ? { question: r["question"] as string } : {}),
      ...(correct != null ? { correct } : {}),
      ...(score != null ? { score } : {}),
      ...(typeof r["answer"] === "string" ? { answer: r["answer"] as string } : {}),
      ...(typeof err === "string" && err.trim()
        ? { error: err }
        : err && typeof err === "object" && str((err as AnyRec)["message"])
          ? { error: (err as AnyRec)["message"] as string }
          : {}),
      ...(failed.length ? { failed } : {}),
    });
  }
  return out;
}

/** A task counts as a MISS when it was not fully solved. */
function isMiss(v: TaskVerdict): boolean {
  if (v.correct != null) return !v.correct;
  if (v.score != null) return v.score < 1;
  return false;
}

/** The grid cell for one verdict. */
function cell(v: TaskVerdict | undefined): string {
  if (!v) return "·";
  if (v.score != null && v.correct !== true) return String(v.score);
  if (v.correct === true) return "✓";
  if (v.correct === false) return v.error ? "!" : (v.answer ?? "").trim() === "" ? "∅" : "✗";
  return "?";
}

/** One graded version — the baseline or a completed generation. */
interface Measured {
  label: string;
  fitness: number;
  verdicts: TaskVerdict[];
}

/** One completed generation, as both output record and briefing material. */
interface GenEntry {
  gen: number;
  genRunId: string;
  version?: string;
  fitness: number;
  allPassCount?: number;
  summary?: string;
  changes?: unknown;
  missingSecrets?: unknown;
  /** per-task verdicts (normalized) — the grid column + miss lines */
  verdicts?: TaskVerdict[];
  authorCost?: number;
  produceCost?: number;
  error?: string;
  /** The author published nothing — nothing was graded, so there is no
   *  fitness datapoint here (see the gen workflows' `published` gate). */
  noop?: boolean;
}

function missLine(v: TaskVerdict): string {
  const parts: string[] = [];
  if (v.error) parts.push(`ERROR: ${oneLine(v.error, ANSWER_CHARS)}`);
  else if (v.correct === false && v.answer != null) {
    // Only domains with an answer channel (gaia) get the answer phrase; a
    // harvey entry has criteria, not an answer.
    parts.push(v.answer.trim() === "" ? "EMPTY answer" : `answered "${oneLine(v.answer, ANSWER_CHARS)}"`);
  }
  if (v.score != null && v.correct !== true) parts.push(`score ${v.score}`);
  if (v.failed?.length) parts.push(`failed: ${oneLine(v.failed.join(" | "), FAILED_CHARS)}`);
  return `${v.taskId} → ${parts.join("; ") || "not solved"}`;
}

function renderGrid(taskIds: string[], columns: Measured[]): string[] {
  const idW = Math.min(12, Math.max(4, ...taskIds.map((t) => t.length)));
  const short = (t: string) => (t.length > idW ? t.slice(0, idW) : t.padEnd(idW));
  const colW = Math.max(4, ...columns.map((c) => c.label.length));
  const pad = (s: string) => s.padStart(colW);
  const lines: string[] = [];
  lines.push(`${"task".padEnd(idW)} | ${columns.map((c) => pad(c.label)).join(" | ")}`);
  lines.push(`${"-".repeat(idW)}-+-${columns.map(() => "-".repeat(colW)).join("-+-")}`);
  for (const t of taskIds) {
    const cells = columns.map((c) => pad(cell(c.verdicts.find((v) => v.taskId === t))));
    lines.push(`${short(t)} | ${cells.join(" | ")}`);
  }
  lines.push(`${"fitness".padEnd(idW)} | ${columns.map((c) => pad(String(c.fitness))).join(" | ")}`);
  return lines;
}

export function composeBriefing(args: {
  baseWorkflow: string;
  candidateName: string;
  fitnessName: string;
  tasks: string[];
  baseline: Measured;
  generations: GenEntry[];
}): string {
  const { baseline, generations, fitnessName, candidateName } = args;
  const lines: string[] = [];

  // Every version graded in this run, oldest → newest, baseline first.
  const columns: Measured[] = [baseline];
  for (const g of generations) {
    if (g.error || g.noop || !g.verdicts) continue;
    columns.push({ label: g.version ?? `gen${g.gen}`, fitness: g.fitness, verdicts: g.verdicts });
  }

  // Row order: the configured task list, then anything the digests know
  // that the list does not. Task metadata (level, question) comes from the
  // first measurement that carries it.
  const taskIds = [...args.tasks];
  const meta = new Map<string, { level?: number | null; question?: string }>();
  for (const c of columns) {
    for (const v of c.verdicts) {
      if (!taskIds.includes(v.taskId)) taskIds.push(v.taskId);
      const m = meta.get(v.taskId) ?? {};
      if (m.level == null && v.level != null) m.level = v.level;
      if (!m.question && v.question) m.question = v.question;
      meta.set(v.taskId, m);
    }
  }

  lines.push(
    `BASELINE — the seeded produce workflow "${args.baseWorkflow}" was run once over the task set: mean ${fitnessName} ${baseline.fitness}.`,
  );
  const baseMisses = baseline.verdicts.filter(isMiss);
  if (baseMisses.length) {
    lines.push("  misses:");
    for (const v of baseMisses) lines.push(`    - ${missLine(v)}`);
  }
  lines.push("");

  lines.push("TASKS:");
  for (const t of taskIds) {
    const m = meta.get(t);
    const lvl = m?.level != null ? ` (L${m.level})` : "";
    lines.push(`- ${t}${lvl}${m?.question ? `: ${oneLine(m.question, QUESTION_CHARS)}` : ""}`);
  }
  lines.push("");

  lines.push(
    `RESULTS GRID — every version graded in this run, oldest → newest (${fitnessName} in the last row). ` +
      "✓ solved, ✗ wrong, ∅ empty answer, ! produce error, · not run; a number is that task's graded score.",
  );
  lines.push(...renderGrid(taskIds, columns));
  lines.push("");

  lines.push("ATTEMPTS in this evolution run (oldest → newest):");
  if (!generations.length) {
    lines.push("  (none — this is the first attempt)");
  }
  for (const g of generations) {
    if (g.error) {
      lines.push(`- attempt ${g.gen}: FAILED to complete (${excerpt(g.error, 200)})`);
      continue;
    }
    // A no-op attempt has no score — saying "fitness 0" here would read as
    // a catastrophic approach rather than an author that never shipped.
    if (g.noop) {
      lines.push(
        `- attempt ${g.gen}: NO CANDIDATE PUBLISHED — its author finished without publishing a new ` +
          `version, so nothing was graded. Do not read this as evidence about any approach.`,
      );
      continue;
    }
    lines.push(`- attempt ${g.gen} → ${candidateName}@${g.version ?? "?"}: ${fitnessName} ${g.fitness}`);
    if (g.summary) lines.push(`  approach: ${excerpt(g.summary, SUMMARY_CHARS)}`);
    const misses = (g.verdicts ?? []).filter(isMiss);
    if (misses.length) {
      lines.push("  misses:");
      for (const v of misses) lines.push(`    - ${missLine(v)}`);
    } else if (g.verdicts?.length) {
      lines.push("  misses: none");
    }
  }
  lines.push("");

  lines.push(
    `Every version above is readable with meta/get-workflow ("${candidateName}" + the exact version; the base ` +
      `workflow is "${args.baseWorkflow}"). Choose your own starting point from this evidence: build on whichever ` +
      "version the grid shows solving the most, or take a different route when the same misses keep recurring " +
      "across versions. One run of the same YAML can flip a task or two by sampling luck alone — read patterns " +
      "across columns, not single cells.",
  );

  return lines.join("\n");
}

export default defineStep({
  type: "eval/evolve-loop",
  description:
    "GENERIC loop over candidate produce workflow versions: repeatedly run a domain's one-generation workflow (author → run candidate over tasks → digest), briefing each generation with the evidence so far — the task list, a task×version grid of every version graded (baseline first), and each attempt's approach summary + misses — and letting the author choose what to build on. Fitness is the generation digest's `fitness` (fallback `meanPassRate`); per-task verdicts come from the digest's `results`. Requires services.optimizer. Config: tasks, mission, baseline (a digest object with fitness/meanPassRate + results), candidateName, baseWorkflow, genWorkflow, fitnessName? (default 'pass-rate'), maxGenerations? (default 5, max 20), stopFitness? (default 1), genParams? (paramOverrides for the generation workflow), maxCost?, maxMinutes?. Output: { candidate, baselineFitness, topFitness, topVersions, bestGen, bestVersion, bestFitness, improved, generations, totalKnownCost, stopReason }.",
  input: z.object({
    tasks: z.array(z.string()).min(1).describe("task ids the loop optimizes against (the TRAIN set)"),
    mission: z.string().describe("the standing gap statement handed to every generation's author"),
    baseline: z
      .any()
      .describe("the baseline digest object ({ fitness | meanPassRate, results, … }) — the grid's first column"),
    candidateName: z.string().describe("the candidate workflow name every generation publishes to (versioned lineage)"),
    baseWorkflow: z.string().describe("the seeded produce workflow the baseline ran"),
    genWorkflow: z.string().describe("the domain's one-generation workflow the loop runs"),
    fitnessName: z
      .string()
      .default("pass-rate")
      .describe("how briefings name the fitness number ('pass-rate', 'accuracy', …)"),
    maxGenerations: z.number().int().min(1).max(20).default(5),
    stopFitness: z.number().min(0).max(1).default(1).describe("stop early once a generation's fitness reaches this"),
    genParams: z
      .record(z.string(), z.any())
      .optional()
      .describe("param overrides for the generation workflow (e.g. { authorModel, authorMaxSteps }) — applied via paramOverrides keyed by genWorkflow"),
    // Generation COUNT is a poor budget: each generation costs whatever the
    // architecture the authors evolved costs, and authors reliably evolve
    // toward more expensive shapes. These caps bound the run in the units a
    // human actually budgets in. Both are checked BETWEEN generations, so
    // the cap is a floor on when the loop stops, never a mid-generation kill.
    maxCost: z
      .number()
      .positive()
      .nullish()
      .describe("stop before starting a generation once totalKnownCost (author + produce) reaches this many dollars — omit for no cost cap"),
    maxMinutes: z
      .number()
      .positive()
      .nullish()
      .describe("stop before starting a generation once this many minutes of wall-clock have elapsed in the loop — omit for no time cap"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const opt = (ctx.services as { optimizer?: Optimizer } | undefined)?.optimizer;
    if (!opt) {
      throw new Error("eval/evolve-loop requires a `services.optimizer` capability (injected in createLabVein).");
    }

    const baselineDigest = (cfg.baseline ?? {}) as AnyRec;
    const baselineFitness = num(baselineDigest["fitness"]) ?? num(baselineDigest["meanPassRate"]) ?? 0;
    const baseline: Measured = { label: "base", fitness: baselineFitness, verdicts: normalizeVerdicts(baselineDigest) };

    const loopStart = Date.now();
    let consecutiveFailures = 0;
    let totalKnownCost = 0;
    let stopReason = "maxGenerations exhausted";
    const generations: GenEntry[] = [];

    // Per-generation progress rows (same pattern as eval/optimize): synthetic
    // `<path>#<gen>` events so each try is visible — and linkable — mid-run.
    const emitGen = (gen: number, e: Partial<RunEvent> & { type: RunEvent["type"] }) =>
      ctx.emit({
        ts: new Date().toISOString(),
        runId: ctx.runId,
        path: `${ctx.path}#${gen}`,
        stepType: "eval/evolve-loop",
        iteration: gen,
        ...e,
      } as RunEvent);

    for (let gen = 0; gen < cfg.maxGenerations; gen++) {
      // Cooperative boundary between generations (RUN_CONTROL_SPEC §2.1
      // code-step opt-in): pause parks here; cancel stops the loop here.
      await ctx.control?.checkpoint();

      // Budget gates, checked before spending the next generation. Deliberately
      // NOT applied on the journal-replay path below: a resumed run must reach
      // the same state it left, and replay spends nothing.
      const elapsedMin = (Date.now() - loopStart) / 60000;
      if (cfg.maxCost != null && totalKnownCost >= cfg.maxCost) {
        stopReason = `maxCost $${cfg.maxCost} reached (spent $${Math.round(totalKnownCost * 100) / 100}) after ${gen} generation(s)`;
        break;
      }
      if (cfg.maxMinutes != null && elapsedMin >= cfg.maxMinutes) {
        stopReason = `maxMinutes ${cfg.maxMinutes} reached (elapsed ${Math.round(elapsedMin)}m) after ${gen} generation(s)`;
        break;
      }

      // Durable resume (§5, iterative code steps): a generation whose
      // synthetic `#gen` step.end is journaled replays — its run is NOT
      // re-launched. State is rebuilt from the journaled output so the loop
      // continues where it left off.
      const journaled = ctx.journal?.[`${ctx.path}#${gen}`] as AnyRec | undefined;
      if (journaled) {
        const noop = journaled["noop"] === true;
        const fitness = num(journaled["fitness"]) ?? num(journaled["passRate"]) ?? 0;
        const entry: GenEntry = {
          gen,
          genRunId: String((journaled["runs"] as AnyRec[] | undefined)?.[0]?.["runId"] ?? ""),
          version: typeof journaled["version"] === "string" ? (journaled["version"] as string) : undefined,
          fitness,
          summary: usableSummary(journaled["summary"]) ?? NO_SUMMARY,
          ...(noop ? { noop: true } : { verdicts: normalizeVerdicts({ results: journaled["verdicts"] }) }),
        };
        generations.push(entry);
        consecutiveFailures = 0;
        totalKnownCost += num(journaled["knownCost"]) ?? 0;
        await emitGen(gen, { type: "step.replayed", output: journaled });
        if (!noop && fitness >= cfg.stopFitness) {
          stopReason = `stopFitness ${cfg.stopFitness} reached`;
          break;
        }
        continue;
      }

      const briefing = composeBriefing({
        baseWorkflow: cfg.baseWorkflow,
        candidateName: cfg.candidateName,
        fitnessName: cfg.fitnessName,
        tasks: cfg.tasks,
        baseline,
        generations,
      });
      const genStart = Date.now();
      await emitGen(gen, {
        type: "step.start",
        input: { gen, briefing: excerpt(briefing, 4000) },
      });

      const run = await opt.run(
        cfg.genWorkflow,
        { tasks: cfg.tasks, mission: cfg.mission, candidateName: cfg.candidateName, generation: gen, briefing },
        {
          // Tree linkage: cancelling/pausing THIS run reaches the generation
          // run (and its candidate runs) — RUN_CONTROL_SPEC §2.2.
          parentRunId: ctx.runId,
          ...(cfg.genParams && Object.keys(cfg.genParams).length
            ? { paramOverrides: { [cfg.genWorkflow]: cfg.genParams } }
            : {}),
        },
      );

      if (run.status !== "success") {
        const message = run.error?.message ?? "unknown";
        generations.push({ gen, genRunId: run.runId, fitness: 0, error: message });
        await emitGen(gen, {
          type: "step.error",
          durationMs: Date.now() - genStart,
          error: { message: `gen ${gen}: ${message}` },
        });
        if (++consecutiveFailures >= 2) {
          stopReason = "two consecutive generation failures";
          break;
        }
        continue;
      }
      consecutiveFailures = 0;

      const out = (run.output ?? {}) as AnyRec;

      // NO-OP generation: the gen workflow's `published` gate found that this
      // generation's author shipped no new version, so it skipped grading
      // rather than re-running an already-scored version over the whole task
      // set. Record the wasted author budget and move on — never write a
      // fitness of 0, which would libel an approach that was never tried.
      if (out["noop"] === true) {
        const authorOnly = num(out["authorCost"]) ?? 0;
        totalKnownCost += authorOnly;
        generations.push({
          gen,
          genRunId: run.runId,
          fitness: 0,
          noop: true,
          authorCost: authorOnly,
          summary: usableSummary(out["summary"]) ?? NO_SUMMARY,
        });
        await emitGen(gen, {
          type: "step.end",
          durationMs: Date.now() - genStart,
          output: {
            gen,
            noop: true,
            note: "author published no new candidate version — grading skipped, no fitness recorded",
            knownCost: Math.round(authorOnly * 10000) / 10000,
            runs: [{ label: `generation ${gen} (no-op)`, workflow: cfg.genWorkflow, runId: run.runId }],
          },
        });
        continue;
      }

      const digest = (out["digest"] ?? {}) as AnyRec;
      const fitness = num(digest["fitness"]) ?? num(digest["meanPassRate"]) ?? 0;
      const digestResults = Array.isArray(digest["results"]) ? (digest["results"] as AnyRec[]) : [];
      const authorCost = num(out["authorCost"]) ?? 0;
      const produceCost = digestResults.reduce((s, r) => s + (num(r["cost"]) ?? 0), 0);
      totalKnownCost += authorCost + produceCost;

      const entry: GenEntry = {
        gen,
        genRunId: run.runId,
        version: typeof out["version"] === "string" ? (out["version"] as string) : undefined,
        fitness,
        allPassCount: num(digest["allPassCount"]),
        summary: usableSummary(out["summary"]) ?? NO_SUMMARY,
        changes: out["changes"],
        missingSecrets: out["missingSecrets"],
        verdicts: normalizeVerdicts(digest),
        authorCost,
        produceCost: Math.round(produceCost * 10000) / 10000,
      };
      generations.push(entry);

      await emitGen(gen, {
        type: "step.end",
        durationMs: Date.now() - genStart,
        output: {
          gen,
          version: entry.version,
          fitness,
          knownCost: Math.round((authorCost + produceCost) * 10000) / 10000,
          runs: [{ label: `generation ${gen}`, workflow: cfg.genWorkflow, runId: run.runId }],
          // Carried so a durable resume can rebuild later generations'
          // briefings (grid column + approach summary) from the journal.
          ...(entry.summary ? { summary: excerpt(entry.summary, SUMMARY_CHARS) } : {}),
          verdicts: entry.verdicts,
        },
      });

      if (fitness >= cfg.stopFitness) {
        stopReason = `stopFitness ${cfg.stopFitness} reached`;
        break;
      }
    }

    // The top: every graded generation that reached the highest fitness,
    // oldest first. Ties stay ties. The baseline is not a candidate version,
    // so it never appears here; `improved` says whether anything beat it.
    const graded = generations.filter((g) => !g.error && !g.noop);
    const topFitness = graded.length ? Math.max(...graded.map((g) => g.fitness)) : baselineFitness;
    const top = graded.filter((g) => g.fitness === topFitness);
    const topVersions = top.map((g) => ({ gen: g.gen, version: g.version, fitness: g.fitness }));
    const oldest = top[0];

    return {
      candidate: cfg.candidateName,
      baselineFitness,
      topFitness,
      topVersions,
      bestGen: oldest?.gen ?? -1,
      bestVersion: oldest?.version,
      bestFitness: topFitness,
      improved: graded.length > 0 && topFitness > baselineFitness,
      generations,
      totalKnownCost: Math.round(totalKnownCost * 10000) / 10000,
      stopReason,
      note: "TRAIN scores — every generation tuned against the same tasks (EVOLVE_SPEC §7). Validate the top version(s) on held-out tasks before promoting. Each version was graded ONCE; a one-task difference is within sampling noise. totalKnownCost = author + produce costs; grading cost (where the benchmark bills it) is additional.",
    };
  },
});
