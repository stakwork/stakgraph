import { z } from "zod";

// ── Types ──────────────────────────────────────────────────────────────────

/** A step definition registered in the step registry.
 *
 * `TServices` is the shape of the capabilities bag injected at run time
 * via `runWorkflow({ services })`. Consumers define this as an interface
 * in their own app and pass concrete implementations per environment
 * (e.g. Neo4j in prod, in-memory in tests). Steps that don't touch
 * services leave it as the default `unknown` and ignore `ctx.services`.
 * An adapter that wants typed `http`/`secrets` can opt in by annotating
 * `defineStep<"t", In, Out, VeinCapabilities>(…)` (import the type from
 * "vein") — kept opt-in so it doesn't collide with consumer `ctx.services`
 * casts (the standard server always provides `http`/`secrets` regardless).
 */
export interface StepDef<
  TType extends string = string,
  TInput extends z.ZodTypeAny = z.ZodTypeAny,
  TOutput extends z.ZodTypeAny = z.ZodTypeAny,
  TServices = unknown,
> {
  type: TType;
  description?: string;
  input: TInput;
  output: TOutput;
  /** Optional source code for this step, surfaced by `GET /steps/:type/source`
   *  and the UI's step viewer. Set by consumers that inject in-code steps via
   *  `createRegistry([...])` (which have no discoverable on-disk file) so their
   *  real implementation can still be inspected. */
  source?: string;
  run: (
    cfg: z.infer<TInput>,
    ctx: StepContext<TServices>,
  ) => Promise<z.infer<TOutput>>;
}

/** Context passed to every step's `run` function.
 *
 * `services` is the consumer-defined capabilities bag — typically a
 * record of typed interfaces (graph store, file system, LLM client, …)
 * whose concrete implementations are swapped per environment.
 */
export interface StepContext<TServices = unknown> {
  runId: string;
  path: string;
  scope: Record<string, unknown>;
  input: unknown;
  emit: (event: RunEvent) => Promise<void>;
  services: TServices;
}

/** Error handling options for a step. */
export interface StepOptions {
  retry?: { max: number; delayMs: number };
  onError?: Step;
}

/** A step instance in a workflow. */
export interface Step {
  id: string;
  type: string;
  config: Record<string, unknown>;
  depends?: string | string[];
  /** Gate condition: only run this step when the gate step it depends on
   *  evaluated to this value. `true`/`false` match the boolean result of
   *  an `if` gate. Omit to always run (no gating). */
  when?: boolean;
  options?: StepOptions;
}

/**
 * A declared mapping from a value in THIS workflow's run output to a `param`
 * default on a (possibly different) target workflow — the "promote a winner"
 * surface (Tier 2). After a run, the UI resolves each spec against the run's
 * output and offers a one-click "promote": write the resolved value into the
 * target's `params[param]` default and publish a new version. Fully generic —
 * nothing keys off a specific output name; the optimize loop just declares
 * `from: bestPrompt`, `to: <targetWorkflow>.<promptParam>`.
 *
 * Inert by itself: declaring a promote NEVER writes anything. Promotion is a
 * human-reviewed action (see the diff, click Apply); only then is the target's
 * param overwritten + a new version published.
 */
export interface PromoteSpec {
  /** Dotted path into this workflow's run output (e.g. `bestPrompt`,
   *  `best.prompt`, `results[0].prompt`). */
  from: string;
  /** Destination as `"<workflow>.<param>"` — the param default to overwrite.
   *  Split on the FIRST dot (workflow names contain no dots). */
  to: string;
  /** Optional human label for the UI (defaults to the `to` string). */
  label?: string;
}

/** A workflow (flow) definition. */
export interface Flow {
  name: string;
  input: z.ZodTypeAny;
  steps: Step[];
  /** Tunable default knobs (prompts, thresholds, sample sizes, …) exposed
   *  to step configs via `{{ params.* }}`. Distinct from `input`: `input`
   *  is the per-run subject (validated, no defaults); `params` are the
   *  experiment surface (all defaults, sparsely overridden per run via
   *  `RunOptions.params`). Override precedence: run override > these
   *  defaults. Omit for workflows with no knobs. */
  params?: Record<string, unknown>;
  /** Declared "promote a run output → a target param default" mappings.
   *  Resolved against a run's output by the UI to offer one-click promotion
   *  of a winning value (e.g. an optimize loop's `bestPrompt`). */
  promotes?: PromoteSpec[];
}

/** Run event types for the JSONL log. */
export type RunEventType =
  | "step.start"
  | "step.end"
  | "step.error"
  | "step.retry"
  | "step.skipped"
  | "run.start"
  | "run.end"
  | "run.error";

/** A single event in the run log. */
export interface RunEvent {
  ts: string;
  runId: string;
  path: string;
  type: RunEventType;
  stepType?: string;
  input?: unknown;
  output?: unknown;
  error?: { message: string; stack?: string };
  durationMs?: number;
  iteration?: number;
}

/** Result of running a workflow. */
export interface RunResult {
  runId: string;
  status: "success" | "error";
  output?: unknown;
  error?: { message: string; stack?: string };
}

/** Run summary written to run.json. */
export interface RunSummary {
  runId: string;
  workflow: string;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  status: "success" | "error";
  input: unknown;
  output?: unknown;
  error?: { message: string; stack?: string };
}

/** A step definition with erased generics, for use in the registry.
 *
 * The runtime is intentionally untyped over services — a registry can
 * hold steps from different consumers expecting different service
 * shapes, and the runner just hands each step whatever was passed in
 * via `RunOptions.services`. Type safety lives at `defineStep` time.
 */
export interface AnyStepDef {
  type: string;
  description?: string;
  input: z.ZodTypeAny;
  output: z.ZodTypeAny;
  /** Optional source code for in-code steps (see `StepDef.source`). */
  source?: string;
  run: (cfg: any, ctx: StepContext<any>) => Promise<any>;
}

/** Step registry — maps step type names to their definitions. */
export type StepRegistry = Record<string, AnyStepDef>;

// ── Builder functions ──────────────────────────────────────────────────────

/**
 * Define a new step type. Used in step definition files.
 *
 * ```ts
 * export default defineStep({
 *   type: "http",
 *   input: z.object({ url: z.string() }),
 *   output: z.any(),
 *   async run(cfg, ctx) { ... },
 * });
 * ```
 */
export function defineStep<
  TType extends string,
  TInput extends z.ZodTypeAny,
  TOutput extends z.ZodTypeAny,
  TServices = unknown,
>(
  def: StepDef<TType, TInput, TOutput, TServices>,
): StepDef<TType, TInput, TOutput, TServices> {
  return def;
}

/**
 * Create a step instance for use in a workflow's `steps` array.
 *
 * ```ts
 * step("check", "http", { url: "{{ input.url }}" })
 * step("check", "http", { url: "/health" }, { retry: { max: 3, delayMs: 1000 } })
 * ```
 */
export function step(
  id: string,
  type: string,
  config: Record<string, unknown>,
  options?: StepOptions & { depends?: string | string[]; when?: boolean },
): Step {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(id)) {
    throw new Error(
      `Invalid step id "${id}": must match [a-zA-Z_][a-zA-Z0-9_]*`,
    );
  }
  const { depends, when, ...opts } = options ?? {};
  const hasOpts = Object.keys(opts).length > 0;
  return {
    id,
    type,
    config,
    ...(depends != null ? { depends } : {}),
    ...(when != null ? { when } : {}),
    ...(hasOpts ? { options: opts } : {}),
  };
}

/**
 * Define a workflow.
 *
 * ```ts
 * export default flow("deploy", {
 *   input: z.object({ service: z.string() }),
 *   steps: [
 *     step("kick", "http", { url: "/deploy", method: "POST" }),
 *     step("done", "log", { message: "deployed {{ input.service }}" }),
 *   ],
 * });
 * ```
 */
export function flow(
  name: string,
  opts: { input: z.ZodTypeAny; steps: Step[]; params?: Record<string, unknown> },
): Flow {
  // Validate step id uniqueness within this flow
  const ids = new Set<string>();
  for (const s of opts.steps) {
    if (ids.has(s.id)) {
      throw new Error(
        `Duplicate step id "${s.id}" in flow "${name}"`,
      );
    }
    ids.add(s.id);
  }
  return {
    name,
    input: opts.input,
    steps: opts.steps,
    ...(opts.params != null ? { params: opts.params } : {}),
  };
}
