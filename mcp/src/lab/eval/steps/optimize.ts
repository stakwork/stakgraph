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
 * Single-example for now: `results` passed to reflect has one entry. Multi-
 * example (the real overfitting fix, §11.2) = loop the eval over a dataset of
 * `evalInput`s and pass the aggregate array — reflect already accepts it.
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

export default defineStep({
  type: "eval/optimize",
  description:
    "Generic self-improving loop: eval the candidate prompt → keep best → reflect a better one → repeat until score ≥ threshold or generations exhausted. Requires a `services.optimizer` capability. Config: evalWorkflow, evalInput, targetWorkflow, promptParam, reflectWorkflow, maxGenerations, threshold, prompt? (starting candidate, defaults to target's current value), label?. Output: { bestScore, bestGen, bestPrompt, firstScore, generations }.",
  input: z.object({
    evalWorkflow: z.string().describe("workflow that scores a candidate; output must have { score, missing, spurious, insight }"),
    evalInput: z.any().describe("input passed to evalWorkflow each generation (e.g. { owner, repo })"),
    targetWorkflow: z.string().describe("workflow whose param is being tuned"),
    promptParam: z.string().describe("the param on targetWorkflow to tune (overridden per generation via paramOverrides)"),
    reflectWorkflow: z.string().describe("workflow that proposes the next prompt; output must have { prompt }"),
    maxGenerations: z.number().int().positive().default(5),
    threshold: z.number().min(0).max(1).default(0.9),
    prompt: z.string().optional().describe("starting candidate; defaults to targetWorkflow's current promptParam value"),
    label: z.string().optional().describe("label for this example in the reflect digest"),
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

    const generations: Array<{
      gen: number;
      score: number;
      prompt: string;
      missing: string[];
      spurious: string[];
      evalRunId: string;
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
        // ── eval the current candidate ────────────────────────────────────
        const evalRun = await opt.run(cfg.evalWorkflow, cfg.evalInput ?? {}, {
          paramOverrides: { [cfg.targetWorkflow]: { [cfg.promptParam]: candidate } },
        });
        if (evalRun.status !== "success") {
          throw new Error(`eval run failed: ${evalRun.error?.message ?? "unknown"}`);
        }
        const out = (evalRun.output ?? {}) as EvalOutput;
        const score = out.score ?? 0;
        const missing = out.missing ?? [];
        const spurious = out.spurious ?? [];
        // `evalRunId` links this generation to its full eval run (which nests
        // the bootstrap-then-process subflow — i.e. the bootstrap logs).
        generations.push({ gen, score, prompt: candidate, missing, spurious, evalRunId: evalRun.runId });

        if (score > best.score) best = { gen, prompt: candidate, score };

        const evalRef: RunRef = { label: "eval run", workflow: cfg.evalWorkflow, runId: evalRun.runId };
        await emitGen(gen, {
          type: "step.end",
          durationMs: Date.now() - genStart,
          output: { gen, score, bestScore: best.score, bestGen: best.gen, missing, spurious, runs: [evalRef] },
        });

        // "until it's satisfied": stop once good enough, or on the last generation.
        if (score >= cfg.threshold || gen === cfg.maxGenerations - 1) break;

        // ── reflect → next candidate (generalizing across the dataset) ─────
        const reflectRun = await opt.run(cfg.reflectWorkflow, {
          prompt: candidate,
          results: [{ label: cfg.label, score, missing, spurious, insight: out.insight }],
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
