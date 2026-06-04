import { z, defineStep, type RunEvent } from "vein";

/**
 * GENERIC self-improving LOOP (EVAL_SPEC §7/§11.4). Runs the optimize cycle as a
 * single step so the whole thing is one detached run (a "background job", §8):
 *
 *   eval  → score the current candidate prompt on the dataset
 *   keep  → track the best-scoring candidate so far
 *   stop? → score ≥ threshold, or out of generations
 *   reflect → propose the next candidate (generalizing) and repeat
 *
 * Domain-agnostic: it tunes some `promptParam` on a `targetWorkflow` by running
 * an `evalWorkflow` (whose output has { score, missing, spurious, insight }) and
 * a `reflectWorkflow` (which proposes the next prompt). The concepts experiment
 * wires these in `concepts-optimize`. "Store the params to optimize" = the
 * returned `generations` array (persisted in this run's run.json). Promote a
 * winner by writing `bestPrompt` into the target's `promptParam` default and
 * publishing a new version (a separate, explicit action).
 *
 * A leaf step has no runner, so the host injects a tiny `services.optimizer`
 * capability (a closure over `vein.run` + `workspace.getWorkflow`) — see
 * `createLabVein`. Each eval/reflect is its own `vein.run`, because the
 * candidate prompt varies per generation via `paramOverrides` (run-global).
 *
 * MULTI-EXAMPLE (the real overfitting fix, §11.2): a generation evals the
 * candidate over a DATASET of inputs (`evalInputs`, e.g. 5 repos), AVERAGES the
 * per-example scores into the generation score, and passes the full per-example
 * array to reflect (which already aggregates — it optimizes for the common
 * failure modes, not one example's quirks). Each example's own gold set rides on
 * its dataset entry (e.g. `{ owner, repo, expected }`) — the eval workflow reads
 * it from `input`. (A single example is just a 1-entry dataset.)
 */

interface EvalOutput {
  score?: number;
  missing?: string[];
  spurious?: string[];
  insight?: string;
}

interface RunResultLike {
  runId: string;
  status: string;
  output?: unknown;
  error?: { message?: string };
}

/** A pointer to a child run, surfaced on progress events so the UI can link
 *  "open this generation's eval/reflect run" (and drill into its nested logs). */
interface RunRef {
  label: string;
  workflow: string;
  runId: string;
}

interface Optimizer {
  run(
    name: string,
    input: unknown,
    opts?: { paramOverrides?: Record<string, Record<string, unknown>> },
  ): Promise<RunResultLike>;
  getParams(name: string): Promise<Record<string, unknown>>;
}

/** Derive a human label for a dataset entry (for the reflect digest + run refs):
 *  explicit `label`, else `owner/repo`, else its position. */
function labelFor(datum: unknown, i: number): string {
  const d = (datum ?? {}) as Record<string, unknown>;
  if (typeof d["label"] === "string" && d["label"].trim()) return d["label"];
  if (typeof d["owner"] === "string" && typeof d["repo"] === "string") return `${d["owner"]}/${d["repo"]}`;
  return `example ${i + 1}`;
}

export default defineStep({
  type: "eval/optimize",
  description:
    "Generic self-improving loop: eval the candidate prompt over a DATASET → average the score → keep best → reflect a better one → repeat until mean score ≥ threshold or generations exhausted. Requires a `services.optimizer` capability. Config: evalWorkflow, evalInputs (array, the dataset; ≥1 entry), targetWorkflow, promptParam, reflectWorkflow, maxGenerations, threshold, prompt? (starting candidate, defaults to target's current value). Output: { bestScore, bestGen, bestPrompt, firstScore, generations }.",
  input: z.object({
    evalWorkflow: z.string().describe("workflow that scores a candidate; output must have { score, missing, spurious, insight }"),
    evalInputs: z.array(z.any()).min(1).describe("the dataset: inputs evaluated each generation and averaged (e.g. [{ owner, repo, expected }, ...]). Each entry's gold rides on the entry."),
    targetWorkflow: z.string().describe("workflow whose param is being tuned"),
    promptParam: z.string().describe("the param on targetWorkflow to tune (overridden per generation via paramOverrides)"),
    reflectWorkflow: z.string().describe("workflow that proposes the next prompt; output must have { prompt }"),
    maxGenerations: z.number().int().positive().default(5),
    threshold: z.number().min(0).max(1).default(0.9),
    prompt: z.string().optional().describe("starting candidate; defaults to targetWorkflow's current promptParam value"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const opt = (ctx.services as { optimizer?: Optimizer })?.optimizer;
    if (!opt) {
      throw new Error(
        "eval/optimize requires a `services.optimizer` capability — inject it in createLabVein (run + getParams over vein).",
      );
    }

    let candidate =
      cfg.prompt ?? ((await opt.getParams(cfg.targetWorkflow))[cfg.promptParam] as string | undefined);
    if (typeof candidate !== "string" || !candidate.trim()) {
      throw new Error(
        `No starting prompt: pass config.prompt, or ensure ${cfg.targetWorkflow}.${cfg.promptParam} has a default.`,
      );
    }

    // The dataset. A generation evals the candidate over EVERY entry and
    // averages — that mean is the generation's score (the overfitting fix).
    const dataset: unknown[] = cfg.evalInputs;

    // One per-example score for a generation.
    interface ExampleResult {
      label: string;
      score: number;
      missing: string[];
      spurious: string[];
      insight?: string;
      evalRunId: string;
    }

    const generations: Array<{
      gen: number;
      score: number; // mean across the dataset
      prompt: string;
      results: ExampleResult[];
    }> = [];
    let best = { gen: -1, prompt: candidate, score: -1 };

    // Per-generation progress is emitted under a synthetic `<path>#<gen>` path
    // (the loop runs inside one step, so generations aren't real runner events).
    // These show up live as rows in the Events panel and persist to the log, so
    // you can watch the individual tries — and reattach to them — mid-run.
    const emitGen = (gen: number, e: Partial<RunEvent> & { type: RunEvent["type"] }) =>
      ctx.emit({
        ts: new Date().toISOString(),
        runId: ctx.runId,
        path: `${ctx.path}#${gen}`,
        stepType: "eval/optimize",
        iteration: gen,
        ...e,
      });

    // Why the current candidate looks the way it does (the prior reflect's
    // rationale + a link to that reflect run). Surfaced in each generation's
    // start so you can read the prompt's evolution try-by-try and open the run
    // that proposed it. Undefined for gen 0 (the seed prompt).
    let rationale: string | undefined;
    let fromReflect: RunRef | undefined;

    for (let gen = 0; gen < cfg.maxGenerations; gen++) {
      const genStart = Date.now();
      await emitGen(gen, {
        type: "step.start",
        input: { gen, prompt: candidate, rationale, ...(fromReflect ? { runs: [fromReflect] } : {}) },
      });

      try {
        // ── eval the current candidate over the WHOLE dataset ─────────────
        // Each example is its own run (so its own gold + logs); the candidate
        // prompt is injected into all of them via the same paramOverrides.
        const paramOverrides = { [cfg.targetWorkflow]: { [cfg.promptParam]: candidate } };
        const evalRuns = await Promise.all(
          dataset.map((datum, i) =>
            opt.run(cfg.evalWorkflow, datum ?? {}, { paramOverrides }).then((run) => {
              if (run.status !== "success") {
                throw new Error(`eval run for "${labelFor(datum, i)}" failed: ${run.error?.message ?? "unknown"}`);
              }
              const out = (run.output ?? {}) as EvalOutput;
              const result: ExampleResult = {
                label: labelFor(datum, i),
                score: out.score ?? 0,
                missing: out.missing ?? [],
                spurious: out.spurious ?? [],
                insight: out.insight,
                evalRunId: run.runId,
              };
              return result;
            }),
          ),
        );

        // The generation score = MEAN across the dataset (the overfitting fix).
        const score = evalRuns.reduce((s, r) => s + r.score, 0) / evalRuns.length;
        // `evalRunId` per example links to its full eval run (which nests the
        // bootstrap-then-process subflow — i.e. the bootstrap logs).
        generations.push({ gen, score, prompt: candidate, results: evalRuns });

        if (score > best.score) best = { gen, prompt: candidate, score };

        // One run ref per example so the UI can open each generation's evals.
        const runs: RunRef[] = evalRuns.map((r) => ({
          label: `eval: ${r.label}`,
          workflow: cfg.evalWorkflow,
          runId: r.evalRunId,
        }));
        await emitGen(gen, {
          type: "step.end",
          durationMs: Date.now() - genStart,
          output: {
            gen,
            score,
            bestScore: best.score,
            bestGen: best.gen,
            results: evalRuns.map((r) => ({ label: r.label, score: r.score, missing: r.missing, spurious: r.spurious })),
            runs,
          },
        });

        // "until it's satisfied": stop once good enough, or on the last generation.
        if (score >= cfg.threshold || gen === cfg.maxGenerations - 1) break;

        // ── reflect → next candidate (generalizing ACROSS the dataset) ─────
        // Pass the full per-example array so reflect optimizes for the COMMON
        // failure modes, not one example's quirks.
        const reflectRun = await opt.run(cfg.reflectWorkflow, {
          prompt: candidate,
          results: evalRuns.map((r) => ({
            label: r.label,
            score: r.score,
            missing: r.missing,
            spurious: r.spurious,
            insight: r.insight,
          })),
        });
        if (reflectRun.status !== "success") {
          throw new Error(`reflect run failed: ${reflectRun.error?.message ?? "unknown"}`);
        }
        const proposal = reflectRun.output as { prompt?: string; rationale?: string } | undefined;
        if (typeof proposal?.prompt !== "string" || !proposal.prompt.trim()) {
          throw new Error(`reflect returned no prompt`);
        }
        candidate = proposal.prompt;
        rationale = proposal.rationale;
        fromReflect = { label: "reflect run", workflow: cfg.reflectWorkflow, runId: reflectRun.runId };
      } catch (err) {
        // Log the failure on THIS generation's row before bubbling up, so a
        // failed try is visible in the panel (not just the outer step error).
        const message = err instanceof Error ? err.message : String(err);
        await emitGen(gen, {
          type: "step.error",
          durationMs: Date.now() - genStart,
          error: { message: `gen ${gen}: ${message}` },
        });
        throw err;
      }
    }

    return {
      bestScore: best.score,
      bestGen: best.gen,
      bestPrompt: best.prompt,
      firstScore: generations[0]?.score ?? null,
      target: cfg.targetWorkflow,
      promptParam: cfg.promptParam,
      generations,
    };
  },
});
