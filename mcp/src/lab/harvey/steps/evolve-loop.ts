import { z, defineStep, type RunEvent } from "vein";

/**
 * The HILL-CLIMB over workflow versions (EVOLVE_SPEC §5.3.3, the harvey
 * instance): run `harvey-evolve-gen` up to maxGenerations times, feeding
 * each generation a BRIEFING composed from the baseline digest plus every
 * previous attempt's version, pass-rate, approach summary, and failure
 * digest — anchored to the best-so-far (never the latest, which may have
 * regressed; the same guarantee eval/optimize makes for prompts).
 *
 * EXPLOIT vs EXPLORE: while attempts keep beating the best, the directive
 * says refine the best version. After `exploreAfter` consecutive
 * non-improving attempts, it flips: try a GENUINELY DIFFERENT approach,
 * with the already-tried approaches listed so "different" is checkable.
 *
 * Judge noise: an improvement must clear `improveMargin` (default 0.02 —
 * one criterion on a 50-criterion task) to count; smaller deltas are ties.
 *
 * Runs generations through `services.optimizer` (vein.run — same capability
 * eval/optimize uses), each as its own persisted run linked from this
 * step's per-generation progress events. Stops on: stopPassRate reached,
 * generations exhausted, or two consecutive generation-run failures (a
 * broken harness should not burn ten generations of budget).
 *
 * TRAIN-SET caveat rides on the output: every generation tunes against the
 * same tasks; the final pass-rate is a train score (EVOLVE_SPEC §7).
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
  passRate: number;
  allPassCount?: number;
  summary?: string;
  changes?: unknown;
  missingSecrets?: unknown;
  digestText?: string;
  authorCost?: number;
  produceCost?: number;
  explore: boolean;
  error?: string;
}

export function composeBriefing(args: {
  baseWorkflow: string;
  candidateName: string;
  baselineText: string;
  baselinePassRate: number;
  generations: GenEntry[];
  best: { gen: number; version?: string; passRate: number; digestText: string };
  explore: boolean;
  sinceImprove: number;
  improveMargin: number;
}): string {
  const { generations, best } = args;
  const lines: string[] = [];

  lines.push(
    `BASELINE — the seeded produce workflow "${args.baseWorkflow}" was run on every task and graded (mean pass-rate ${args.baselinePassRate}):`,
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
      const delta = Math.round((g.passRate - args.baselinePassRate) * 1000) / 1000;
      lines.push(
        `- attempt ${g.gen} → published ${args.candidateName}@${g.version ?? "?"}: mean pass-rate ${g.passRate} (${delta >= 0 ? "+" : ""}${delta} vs baseline)`,
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
    lines.push(`BEST SO FAR: the baseline itself (mean pass-rate ${best.passRate}) — no attempt has beaten it yet.`);
  } else {
    lines.push(
      `BEST SO FAR: attempt ${best.gen} → ${args.candidateName}@${best.version} (mean pass-rate ${best.passRate}). Its result digest:`,
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
        `attempt) and refine it: keep what worked, fix exactly what its failed criteria show.`,
    );
  }
  lines.push(
    `Deltas within ±${args.improveMargin} are judge noise — treat them as ties, not as signal to chase.`,
  );

  return lines.join("\n");
}

export default defineStep({
  type: "harvey/evolve-loop",
  description:
    "Hill-climb candidate produce workflows over generations: repeatedly run a generation workflow (default harvey-evolve-gen: author → run candidate over tasks → digest), composing each generation's briefing from the baseline digest + all previous attempts, anchored to the best-so-far, flipping to an explore directive after `exploreAfter` non-improving attempts. Requires services.optimizer. Config: tasks, mission, baseline (a harvey/digest-results object), candidateName, baseWorkflow?, genWorkflow?, maxGenerations? (default 5, max 20), stopPassRate? (default 1), improveMargin? (default 0.02), exploreAfter? (default 2), genParams? (paramOverrides for the generation workflow). Output: { candidate, baselinePassRate, bestGen, bestVersion, bestPassRate, improved, generations, totalKnownCost, stopReason }.",
  input: z.object({
    tasks: z.array(z.string()).min(1).describe("Harvey task ids the loop optimizes against (the TRAIN set)"),
    mission: z.string().describe("the standing gap statement handed to every generation's author"),
    baseline: z
      .any()
      .describe("the baseline harvey/digest-results object ({ meanPassRate, text, … }) — generation 0's anchor"),
    candidateName: z.string().describe("the candidate workflow name every generation publishes to (versioned lineage)"),
    baseWorkflow: z.string().default("harvey-produce").describe("the seeded produce workflow the baseline ran"),
    genWorkflow: z.string().default("harvey-evolve-gen").describe("the one-generation workflow the loop runs"),
    maxGenerations: z.number().int().min(1).max(20).default(5),
    stopPassRate: z.number().min(0).max(1).default(1).describe("stop early once a generation's mean pass-rate reaches this"),
    improveMargin: z
      .number()
      .min(0)
      .default(0.02)
      .describe("an attempt must beat the best by MORE than this to count as an improvement (judge noise floor; 0.02 ≈ one criterion on a 50-criterion task)"),
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
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const opt = (ctx.services as { optimizer?: Optimizer } | undefined)?.optimizer;
    if (!opt) {
      throw new Error("harvey/evolve-loop requires a `services.optimizer` capability (injected in createLabVein).");
    }

    const baseline = (cfg.baseline ?? {}) as AnyRec;
    const baselinePassRate = num(baseline["meanPassRate"]) ?? 0;
    const baselineText =
      typeof baseline["text"] === "string" && baseline["text"]
        ? (baseline["text"] as string)
        : excerpt(JSON.stringify(baseline), 800);

    let best = { gen: -1, version: undefined as string | undefined, passRate: baselinePassRate, digestText: baselineText };
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
        stepType: "harvey/evolve-loop",
        iteration: gen,
        ...e,
      } as RunEvent);

    for (let gen = 0; gen < cfg.maxGenerations; gen++) {
      // Cooperative boundary between generations (RUN_CONTROL_SPEC §2.1
      // code-step opt-in): pause parks here; cancel stops the loop here.
      await ctx.control?.checkpoint();

      // Durable resume (§5, iterative code steps): a generation whose
      // synthetic `#gen` step.end is journaled replays — its run is NOT
      // re-launched. State (best / sinceImprove / stop logic) is rebuilt
      // from the journaled output so the loop continues where it left off.
      const journaled = ctx.journal?.[`${ctx.path}#${gen}`] as AnyRec | undefined;
      if (journaled) {
        const passRate = num(journaled["passRate"]) ?? 0;
        const entry: GenEntry = {
          gen,
          genRunId: String((journaled["runs"] as AnyRec[] | undefined)?.[0]?.["runId"] ?? ""),
          version: typeof journaled["version"] === "string" ? (journaled["version"] as string) : undefined,
          passRate,
          summary: typeof journaled["summary"] === "string" ? (journaled["summary"] as string) : undefined,
          digestText: typeof journaled["digestText"] === "string" ? (journaled["digestText"] as string) : undefined,
          explore: journaled["directive"] === "explore",
        };
        generations.push(entry);
        consecutiveFailures = 0;
        totalKnownCost += num(journaled["knownCost"]) ?? 0;
        if (passRate > best.passRate + cfg.improveMargin) {
          best = { gen, version: entry.version, passRate, digestText: entry.digestText ?? "" };
          sinceImprove = 0;
        } else {
          sinceImprove++;
        }
        await emitGen(gen, { type: "step.replayed", output: journaled });
        if (passRate >= cfg.stopPassRate) {
          stopReason = `stopPassRate ${cfg.stopPassRate} reached`;
          break;
        }
        continue;
      }

      const explore = sinceImprove >= cfg.exploreAfter;
      const briefing = composeBriefing({
        baseWorkflow: cfg.baseWorkflow,
        candidateName: cfg.candidateName,
        baselineText,
        baselinePassRate,
        generations,
        best,
        explore,
        sinceImprove,
        improveMargin: cfg.improveMargin,
      });
      const genStart = Date.now();
      await emitGen(gen, {
        type: "step.start",
        input: { gen, directive: explore ? "explore" : "exploit", bestPassRate: best.passRate, briefing: excerpt(briefing, 2000) },
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
        generations.push({ gen, genRunId: run.runId, passRate: 0, explore, error: message });
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
      const digest = (out["digest"] ?? {}) as AnyRec;
      const passRate = num(digest["meanPassRate"]) ?? 0;
      const digestResults = Array.isArray(digest["results"]) ? (digest["results"] as AnyRec[]) : [];
      const authorCost = num(out["authorCost"]) ?? 0;
      const produceCost = digestResults.reduce((s, r) => s + (num(r["cost"]) ?? 0), 0);
      totalKnownCost += authorCost + produceCost;

      const entry: GenEntry = {
        gen,
        genRunId: run.runId,
        version: typeof out["version"] === "string" ? (out["version"] as string) : undefined,
        passRate,
        allPassCount: num(digest["allPassCount"]),
        summary: typeof out["summary"] === "string" ? (out["summary"] as string) : undefined,
        changes: out["changes"],
        missingSecrets: out["missingSecrets"],
        digestText: typeof digest["text"] === "string" ? (digest["text"] as string) : undefined,
        authorCost,
        produceCost: Math.round(produceCost * 10000) / 10000,
        explore,
      };
      generations.push(entry);

      if (passRate > best.passRate + cfg.improveMargin) {
        best = { gen, version: entry.version, passRate, digestText: entry.digestText ?? "" };
        sinceImprove = 0;
      } else {
        sinceImprove++;
      }

      await emitGen(gen, {
        type: "step.end",
        durationMs: Date.now() - genStart,
        output: {
          gen,
          directive: explore ? "explore" : "exploit",
          version: entry.version,
          passRate,
          bestPassRate: best.passRate,
          bestGen: best.gen,
          knownCost: Math.round((authorCost + produceCost) * 10000) / 10000,
          runs: [{ label: `generation ${gen}`, workflow: cfg.genWorkflow, runId: run.runId }],
          // Carried so a durable resume can rebuild later generations'
          // briefings (approach summaries + best digest) from the journal.
          ...(entry.summary ? { summary: excerpt(entry.summary, 1200) } : {}),
          ...(entry.digestText ? { digestText: excerpt(entry.digestText, 900) } : {}),
        },
      });

      if (passRate >= cfg.stopPassRate) {
        stopReason = `stopPassRate ${cfg.stopPassRate} reached`;
        break;
      }
    }

    return {
      candidate: cfg.candidateName,
      baselinePassRate,
      bestGen: best.gen,
      bestVersion: best.version,
      bestPassRate: best.passRate,
      improved: best.gen >= 0,
      generations,
      totalKnownCost: Math.round(totalKnownCost * 10000) / 10000,
      stopReason,
      note: "TRAIN scores — every generation tuned against the same tasks (EVOLVE_SPEC §7). Validate the best version on held-out tasks before promoting. totalKnownCost = author + produce costs; judge cost is not surfaced by the benchmark and is additional.",
    };
  },
});
