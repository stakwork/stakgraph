import { z, defineStep, type RunEvent } from "vein";

/**
 * The GENERIC hill-climb over workflow versions (EVOLVE_SPEC §5.3.3 — the
 * generalization §9.5 called for; promoted from the harvey instance): run a
 * one-generation workflow up to maxGenerations times, feeding each
 * generation a BRIEFING composed from the baseline digest plus every
 * previous attempt's version, fitness, approach summary, and failure
 * digest — anchored to the best-so-far (never the latest, which may have
 * regressed; the same guarantee eval/optimize makes for prompts).
 *
 * Domain-agnostic on purpose: the loop knows nothing about rubrics or
 * scorers. A domain plugs in via
 *   - `genWorkflow`: its one-generation workflow (author → run candidate
 *     over tasks → digest), invoked with
 *     { tasks, mission, candidateName, generation, briefing } and returning
 *     { version?, summary?, changes?, missingSecrets?, authorCost?, digest }
 *   - the digest's FITNESS: the loop reads `digest.fitness` (falling back
 *     to `digest.meanPassRate`, the harvey digest's field) — a number in
 *     [0,1] that MUST have a gradient (harvey: criteria pass-rate, since
 *     binary all-pass has none; gaia: plain accuracy — binary per task is
 *     fine there because the set supplies the gradient).
 *   - `fitnessName`: how briefings name that number ("pass-rate",
 *     "accuracy", …) so authors read the right thing.
 *
 * EXPLOIT vs EXPLORE: while attempts keep beating the best, the directive
 * says refine the best version. After `exploreAfter` consecutive
 * non-improving attempts, it flips: try a GENUINELY DIFFERENT approach,
 * with the already-tried approaches listed so "different" is checkable.
 *
 * Noise: an improvement must clear `improveMargin` to count; smaller
 * deltas are ties. What the margin answers is per-domain — judge noise for
 * LLM-judged benchmarks (harvey: 0.02 ≈ one criterion at n=50), produce-
 * sampling noise for deterministic scorers (gaia: 0, any task flip counts,
 * but the same caveat rides: validate on held-out tasks).
 *
 * Runs generations through `services.optimizer` (vein.run — same capability
 * eval/optimize uses), each as its own persisted run linked from this
 * step's per-generation progress events. Stops on: stopFitness reached,
 * generations exhausted, or two consecutive generation-run failures (a
 * broken harness should not burn ten generations of budget).
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

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/**
 * Authors occasionally end schema mode on a bare text turn, echoing filler
 * ("placeholder", "") into `summary`. Junk there poisons every later
 * briefing — the EXPLORE directive's "pick an approach that is none of the
 * above" is only checkable against real summaries — and the report the
 * human reads. Replace it with an honest marker instead of passing it
 * through. (The version echo has its own fallback in the gen workflows;
 * this is the summary-channel counterpart.)
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

/**
 * A generation only becomes the new best if it BEAT the bar by more than the
 * noise margin AND it is a version this run has not already scored.
 *
 * The second half matters because fitness is resampled: re-running an
 * already-graded version can land above its own recorded score by produce-
 * sampling luck alone. The gen workflows' `published` gate stops the common
 * cause (an author that ships nothing, so the version fallback resolves to
 * the previous generation's publish), but a *deliberate* republish of
 * identical YAML under a new version string is indistinguishable from here —
 * this is the backstop for the case the gate cannot see, and it keeps the
 * reported best pinned to the run where that version was first measured.
 */
function isNewBest(
  version: string | undefined,
  fitness: number,
  best: { fitness: number },
  margin: number,
  scored: Map<string, number>,
): boolean {
  if (!(fitness > best.fitness + margin)) return false;
  return version == null || !scored.has(version);
}

function indent(s: string, pad: string): string {
  return s
    .split("\n")
    .map((l) => pad + l)
    .join("\n");
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
  digestText?: string;
  authorCost?: number;
  produceCost?: number;
  explore: boolean;
  error?: string;
  /** The author published nothing — nothing was graded, so there is no
   *  fitness datapoint here (see the gen workflows' `published` gate). */
  noop?: boolean;
}

export function composeBriefing(args: {
  baseWorkflow: string;
  candidateName: string;
  fitnessName: string;
  baselineText: string;
  baselineFitness: number;
  generations: GenEntry[];
  best: { gen: number; version?: string; fitness: number; digestText: string };
  explore: boolean;
  sinceImprove: number;
  improveMargin: number;
}): string {
  const { generations, best, fitnessName } = args;
  const lines: string[] = [];

  lines.push(
    `BASELINE — the seeded produce workflow "${args.baseWorkflow}" was run on every task and graded (mean ${fitnessName} ${args.baselineFitness}):`,
  );
  lines.push(indent(args.baselineText, "  "));
  lines.push("");

  lines.push("PREVIOUS ATTEMPTS in this evolution run (oldest → newest):");
  if (!generations.length) {
    lines.push("  (none — this is the first attempt)");
  } else {
    for (const g of generations) {
      if (g.error) {
        lines.push(`- attempt ${g.gen}: FAILED to complete (${excerpt(g.error, 200)})`);
        continue;
      }
      // A no-op attempt has no score to compare — saying "mean accuracy 0"
      // here would read as a catastrophic approach rather than an author
      // that never shipped, and would push later generations to explore
      // away from a strategy that was never actually tried.
      if (g.noop) {
        lines.push(
          `- attempt ${g.gen}: NO CANDIDATE PUBLISHED — its author finished without publishing a new ` +
            `version, so nothing was graded. Do not read this as evidence about any approach.`,
        );
        continue;
      }
      const delta = Math.round((g.fitness - args.baselineFitness) * 1000) / 1000;
      lines.push(
        `- attempt ${g.gen} → published ${args.candidateName}@${g.version ?? "?"}: mean ${fitnessName} ${g.fitness} (${delta >= 0 ? "+" : ""}${delta} vs baseline)`,
      );
      // The approach summary is the ONLY channel telling the EXPLORE
      // directive what has already been tried — keep it roomy enough that
      // "pick an approach that is none of the above" stays checkable.
      if (g.summary) lines.push(`  approach: ${excerpt(g.summary, 1200)}`);
      if (g.digestText) lines.push(indent(excerpt(g.digestText, 700), "    "));
    }
  }
  lines.push("");

  if (best.gen < 0) {
    lines.push(
      `BEST SO FAR: the baseline itself (mean ${fitnessName} ${best.fitness}) — no attempt has beaten it yet.`,
    );
  } else {
    lines.push(
      `BEST SO FAR: attempt ${best.gen} → ${args.candidateName}@${best.version} (mean ${fitnessName} ${best.fitness}). Its result digest:`,
    );
    lines.push(indent(excerpt(best.digestText, 900), "  "));
  }
  lines.push("");

  if (args.explore) {
    lines.push(
      `DIRECTIVE — EXPLORE. The last ${args.sinceImprove} attempt(s) did NOT beat the best. ` +
        `Do NOT keep refining the same approach. Choose a GENUINELY DIFFERENT strategy this ` +
        `generation: a different structure (e.g. sub-agent split vs single agent, a fresh-eyes ` +
        `verification pass vs a research phase), a different method, a different allocation of the ` +
        `step budget. Every approach summarized above has been tried — pick one that is none of ` +
        `them. A regression from a distinct attempt is worth more than a marginal tweak that ` +
        `changes nothing.`,
    );
  } else if (best.gen < 0) {
    lines.push(
      `DIRECTIVE — EXPLOIT. Start from the base workflow "${args.baseWorkflow}" (read its YAML with ` +
        `meta/get-workflow) and fix exactly what the baseline failures above show.`,
    );
  } else {
    lines.push(
      `DIRECTIVE — EXPLOIT. Improve FROM the best attempt: read ${args.candidateName}@${best.version} ` +
        `(meta/get-workflow with that EXACT version — the active version may be a worse later ` +
        `attempt) and refine it: keep what worked, fix exactly what its failures show.`,
    );
  }
  lines.push(
    `Deltas within ±${args.improveMargin} are noise — treat them as ties, not as signal to chase.`,
  );

  return lines.join("\n");
}

export default defineStep({
  type: "eval/evolve-loop",
  description:
    "GENERIC hill-climb of candidate produce workflows over generations: repeatedly run a domain's one-generation workflow (author → run candidate over tasks → digest), composing each generation's briefing from the baseline digest + all previous attempts, anchored to the best-so-far, flipping to an explore directive after `exploreAfter` non-improving attempts. Fitness is the generation digest's `fitness` (fallback `meanPassRate`). Requires services.optimizer. Config: tasks, mission, baseline (a digest object with fitness/meanPassRate + text), candidateName, baseWorkflow, genWorkflow, fitnessName? (default 'pass-rate'), maxGenerations? (default 5, max 20), stopFitness? (default 1), improveMargin? (default 0.02), exploreAfter? (default 2), genParams? (paramOverrides for the generation workflow). Output: { candidate, baselineFitness, bestGen, bestVersion, bestFitness, improved, generations, totalKnownCost, stopReason }.",
  input: z.object({
    tasks: z.array(z.string()).min(1).describe("task ids the loop optimizes against (the TRAIN set)"),
    mission: z.string().describe("the standing gap statement handed to every generation's author"),
    baseline: z
      .any()
      .describe("the baseline digest object ({ fitness | meanPassRate, text, … }) — generation 0's anchor"),
    candidateName: z.string().describe("the candidate workflow name every generation publishes to (versioned lineage)"),
    baseWorkflow: z.string().describe("the seeded produce workflow the baseline ran"),
    genWorkflow: z.string().describe("the domain's one-generation workflow the loop runs"),
    fitnessName: z
      .string()
      .default("pass-rate")
      .describe("how briefings name the fitness number ('pass-rate', 'accuracy', …)"),
    maxGenerations: z.number().int().min(1).max(20).default(5),
    stopFitness: z.number().min(0).max(1).default(1).describe("stop early once a generation's fitness reaches this"),
    improveMargin: z
      .number()
      .min(0)
      .default(0.02)
      .describe("an attempt must beat the best by MORE than this to count as an improvement (noise floor — judge noise for LLM-judged domains, produce-sampling noise for deterministic scorers)"),
    exploreAfter: z
      .number()
      .int()
      .min(1)
      .default(2)
      .describe("consecutive non-improving attempts before the directive flips from exploit to explore"),
    genParams: z
      .record(z.any())
      .optional()
      .describe("param overrides for the generation workflow (e.g. { authorModel, authorMaxSteps }) — applied via paramOverrides keyed by genWorkflow"),
    // Generation COUNT is a poor budget: each generation costs whatever the
    // architecture the authors evolved costs, and authors reliably evolve
    // toward more expensive shapes (redundant attempts, reconcilers, extra
    // verification passes). A 10-generation run that started at ~1h/gen can
    // finish at ~2.5h/gen. These caps bound the run in the units a human
    // actually budgets in. Both are checked BETWEEN generations, so the cap
    // is a floor on when the loop stops, never a mid-generation kill.
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

    const baseline = (cfg.baseline ?? {}) as AnyRec;
    const baselineFitness = num(baseline["fitness"]) ?? num(baseline["meanPassRate"]) ?? 0;
    const baselineText =
      typeof baseline["text"] === "string" && baseline["text"]
        ? (baseline["text"] as string)
        : excerpt(JSON.stringify(baseline), 800);

    let best = { gen: -1, version: undefined as string | undefined, fitness: baselineFitness, digestText: baselineText };
    // version → the fitness it was FIRST measured at, so a later re-score of
    // the same version cannot be promoted as an improvement (see isNewBest).
    const scored = new Map<string, number>();
    const loopStart = Date.now();
    let sinceImprove = 0;
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
      // re-launched. State (best / sinceImprove / stop logic) is rebuilt
      // from the journaled output so the loop continues where it left off.
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
          digestText: typeof journaled["digestText"] === "string" ? (journaled["digestText"] as string) : undefined,
          explore: journaled["directive"] === "explore",
          ...(noop ? { noop: true } : {}),
        };
        generations.push(entry);
        consecutiveFailures = 0;
        totalKnownCost += num(journaled["knownCost"]) ?? 0;
        if (!noop && isNewBest(entry.version, fitness, best, cfg.improveMargin, scored)) {
          best = { gen, version: entry.version, fitness, digestText: entry.digestText ?? "" };
          sinceImprove = 0;
        } else {
          sinceImprove++;
        }
        if (!noop && entry.version && !scored.has(entry.version)) scored.set(entry.version, fitness);
        await emitGen(gen, { type: "step.replayed", output: journaled });
        if (!noop && fitness >= cfg.stopFitness) {
          stopReason = `stopFitness ${cfg.stopFitness} reached`;
          break;
        }
        continue;
      }

      const explore = sinceImprove >= cfg.exploreAfter;
      const briefing = composeBriefing({
        baseWorkflow: cfg.baseWorkflow,
        candidateName: cfg.candidateName,
        fitnessName: cfg.fitnessName,
        baselineText,
        baselineFitness,
        generations,
        best,
        explore,
        sinceImprove,
        improveMargin: cfg.improveMargin,
      });
      const genStart = Date.now();
      await emitGen(gen, {
        type: "step.start",
        input: { gen, directive: explore ? "explore" : "exploit", bestFitness: best.fitness, briefing: excerpt(briefing, 2000) },
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
        generations.push({ gen, genRunId: run.runId, fitness: 0, explore, error: message });
        await emitGen(gen, {
          type: "step.error",
          durationMs: Date.now() - genStart,
          error: { message: `gen ${gen}: ${message}` },
        });
        sinceImprove++;
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
      // set. Record the wasted author budget, leave `best` alone, and let the
      // non-improvement push the directive toward explore — but never write a
      // fitness of 0, which would libel an approach that was never tried.
      if (out["noop"] === true) {
        const authorOnly = num(out["authorCost"]) ?? 0;
        totalKnownCost += authorOnly;
        generations.push({
          gen,
          genRunId: run.runId,
          fitness: 0,
          noop: true,
          explore,
          authorCost: authorOnly,
          summary: usableSummary(out["summary"]) ?? NO_SUMMARY,
        });
        sinceImprove++;
        await emitGen(gen, {
          type: "step.end",
          durationMs: Date.now() - genStart,
          output: {
            gen,
            directive: explore ? "explore" : "exploit",
            noop: true,
            note: "author published no new candidate version — grading skipped, no fitness recorded",
            bestFitness: best.fitness,
            bestGen: best.gen,
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
        digestText: typeof digest["text"] === "string" ? (digest["text"] as string) : undefined,
        authorCost,
        produceCost: Math.round(produceCost * 10000) / 10000,
        explore,
      };
      generations.push(entry);

      const rescored = entry.version != null && scored.has(entry.version);
      if (isNewBest(entry.version, fitness, best, cfg.improveMargin, scored)) {
        best = { gen, version: entry.version, fitness, digestText: entry.digestText ?? "" };
        sinceImprove = 0;
      } else {
        sinceImprove++;
      }
      if (entry.version && !rescored) scored.set(entry.version, fitness);

      await emitGen(gen, {
        type: "step.end",
        durationMs: Date.now() - genStart,
        output: {
          gen,
          directive: explore ? "explore" : "exploit",
          version: entry.version,
          fitness,
          bestFitness: best.fitness,
          bestGen: best.gen,
          // A version this run already scored — its fitness here is a
          // resample, not a hill-climb step, and cannot become the best.
          ...(rescored ? { rescoredVersion: true } : {}),
          knownCost: Math.round((authorCost + produceCost) * 10000) / 10000,
          runs: [{ label: `generation ${gen}`, workflow: cfg.genWorkflow, runId: run.runId }],
          // Carried so a durable resume can rebuild later generations'
          // briefings (approach summaries + best digest) from the journal.
          ...(entry.summary ? { summary: excerpt(entry.summary, 1200) } : {}),
          ...(entry.digestText ? { digestText: excerpt(entry.digestText, 900) } : {}),
        },
      });

      if (fitness >= cfg.stopFitness) {
        stopReason = `stopFitness ${cfg.stopFitness} reached`;
        break;
      }
    }

    return {
      candidate: cfg.candidateName,
      baselineFitness,
      bestGen: best.gen,
      bestVersion: best.version,
      bestFitness: best.fitness,
      improved: best.gen >= 0,
      generations,
      totalKnownCost: Math.round(totalKnownCost * 10000) / 10000,
      stopReason,
      note: "TRAIN scores — every generation tuned against the same tasks (EVOLVE_SPEC §7). Validate the best version on held-out tasks before promoting. totalKnownCost = author + produce costs; grading cost (where the benchmark bills it) is additional.",
    };
  },
});
