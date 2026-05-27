import { z } from "zod";

// ── Types ──────────────────────────────────────────────────────────────────

/** A step definition registered in the step registry. */
export interface StepDef<
  TType extends string = string,
  TInput extends z.ZodTypeAny = z.ZodTypeAny,
  TOutput extends z.ZodTypeAny = z.ZodTypeAny,
> {
  type: TType;
  input: TInput;
  output: TOutput;
  run: (cfg: z.infer<TInput>, ctx: StepContext) => Promise<z.infer<TOutput>>;
}

/** Context passed to every step's `run` function. */
export interface StepContext {
  runId: string;
  path: string;
  scope: Record<string, unknown>;
  input: unknown;
  emit: (event: RunEvent) => Promise<void>;
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
  options?: StepOptions;
}

/** A workflow (flow) definition. */
export interface Flow {
  name: string;
  input: z.ZodTypeAny;
  steps: Step[];
}

/** Run event types for the JSONL log. */
export type RunEventType =
  | "step.start"
  | "step.end"
  | "step.error"
  | "step.retry"
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

/** A step definition with erased generics, for use in the registry. */
export interface AnyStepDef {
  type: string;
  input: z.ZodTypeAny;
  output: z.ZodTypeAny;
  run: (cfg: any, ctx: StepContext) => Promise<any>;
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
>(def: StepDef<TType, TInput, TOutput>): StepDef<TType, TInput, TOutput> {
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
  options?: StepOptions,
): Step {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(id)) {
    throw new Error(
      `Invalid step id "${id}": must match [a-zA-Z_][a-zA-Z0-9_]*`,
    );
  }
  return { id, type, config, ...(options ? { options } : {}) };
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
  opts: { input: z.ZodTypeAny; steps: Step[] },
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
  return { name, input: opts.input, steps: opts.steps };
}
