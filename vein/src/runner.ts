import type {
  Flow,
  Step,
  StepContext,
  StepRegistry,
  RunEvent,
  RunResult,
  RunSummary,
} from "./core.js";
import { resolveConfig } from "./expr.js";
import type { RunStore } from "./store.js";
import { MemoryRunStore, generateRunId } from "./store.js";

// ── Runner ─────────────────────────────────────────────────────────────────

export interface RunOptions {
  runId?: string;
  store?: RunStore;
  /** Called for every event as it happens (for SSE streaming). */
  onEvent?: (event: RunEvent) => void | Promise<void>;
}

export async function runWorkflow(
  workflow: Flow,
  input: unknown,
  registry: StepRegistry,
  opts?: RunOptions,
): Promise<RunResult> {
  const runId = opts?.runId ?? generateRunId();
  const store = opts?.store ?? new MemoryRunStore();
  const wfName = workflow.name;
  const startedAt = new Date().toISOString();

  const onEvent = opts?.onEvent;
  const emit = async (event: Partial<RunEvent> & { type: RunEvent["type"] }) => {
    const full: RunEvent = {
      ts: new Date().toISOString(),
      runId,
      path: event.path ?? wfName,
      ...event,
    };
    await store.append(wfName, runId, full);
    await onEvent?.(full);
  };

  // Validate input
  let parsedInput: unknown;
  try {
    parsedInput = workflow.input.parse(input);
  } catch (err) {
    const error = {
      message: `Input validation failed: ${err instanceof Error ? err.message : String(err)}`,
      stack: err instanceof Error ? err.stack : undefined,
    };
    await emit({ type: "run.error", path: wfName, error });
    const finishedAt = new Date().toISOString();
    await store.finalize(wfName, runId, {
      runId,
      workflow: wfName,
      startedAt,
      finishedAt,
      durationMs: Date.parse(finishedAt) - Date.parse(startedAt),
      status: "error",
      input,
      error,
    });
    return { runId, status: "error", error };
  }

  await emit({ type: "run.start", path: wfName, input: parsedInput });

  try {
    const output = await executeFlow(
      workflow,
      parsedInput,
      registry,
      runId,
      wfName,
      emit,
    );

    const finishedAt = new Date().toISOString();
    await emit({ type: "run.end", path: wfName, output });
    await store.finalize(wfName, runId, {
      runId,
      workflow: wfName,
      startedAt,
      finishedAt,
      durationMs: Date.parse(finishedAt) - Date.parse(startedAt),
      status: "success",
      input: parsedInput,
      output,
    });
    return { runId, status: "success", output };
  } catch (err) {
    const error = {
      message: err instanceof Error ? err.message : String(err),
      stack: err instanceof Error ? err.stack : undefined,
    };
    const finishedAt = new Date().toISOString();
    await emit({ type: "run.error", path: wfName, error });
    await store.finalize(wfName, runId, {
      runId,
      workflow: wfName,
      startedAt,
      finishedAt,
      durationMs: Date.parse(finishedAt) - Date.parse(startedAt),
      status: "error",
      input: parsedInput,
      error,
    });
    return { runId, status: "error", error };
  }
}

// ── Internal: execute a flow's steps as a DAG ─────────────────────────────

type EmitFn = (event: Partial<RunEvent> & { type: RunEvent["type"] }) => Promise<void>;

/**
 * Get the dependency list for a step. If `depends` is set, use it.
 * Otherwise, the step implicitly depends on the previous step in the array
 * (sequential by default).
 */
function getDeps(step: Step, index: number, steps: Step[]): string[] {
  if (step.depends != null) {
    return Array.isArray(step.depends) ? step.depends : [step.depends];
  }
  // Implicit: depends on previous step (if any)
  if (index > 0) return [steps[index - 1]!.id];
  return [];
}

async function executeFlow(
  workflow: Flow,
  input: unknown,
  registry: StepRegistry,
  runId: string,
  basePath: string,
  emit: EmitFn,
): Promise<unknown> {
  const scope: Record<string, unknown> = { input };
  const steps = workflow.steps;

  if (steps.length === 0) return undefined;

  // Build dependency graph
  const depMap = new Map<string, string[]>();
  const stepById = new Map<string, Step>();
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]!;
    stepById.set(s.id, s);
    depMap.set(s.id, getDeps(s, i, steps));
  }

  // Find which steps depend on each step (reverse map)
  const dependents = new Map<string, Set<string>>();
  for (const s of steps) dependents.set(s.id, new Set());
  for (const [id, deps] of depMap) {
    for (const dep of deps) {
      dependents.get(dep)?.add(id);
    }
  }

  // Track completion
  const completed = new Set<string>();
  const pending = new Map<string, { resolve: () => void; promise: Promise<void> }>();

  // Create a promise for each step that resolves when it completes
  for (const s of steps) {
    let resolve!: () => void;
    const promise = new Promise<void>((r) => { resolve = r; });
    pending.set(s.id, { resolve, promise });
  }

  // Execute a single step once its deps are met
  async function runStep(s: Step): Promise<void> {
    // Wait for all dependencies
    const deps = depMap.get(s.id) ?? [];
    await Promise.all(deps.map((d) => pending.get(d)!.promise));

    const stepPath = `${basePath}/${s.id}`;
    const output = await executeStep(s, scope, registry, runId, stepPath, emit);
    scope[s.id] = output;
    completed.add(s.id);
    pending.get(s.id)!.resolve();
  }

  // Launch all steps — each waits for its own deps internally
  await Promise.all(steps.map((s) => runStep(s)));

  // Return last step's output (by array order)
  return scope[steps[steps.length - 1]!.id];
}

// ── Internal: execute a single step with retry/onError ─────────────────────

// Control flow steps that manage their own template resolution.
// These should NOT have their config pre-resolved by executeStep.
const SELF_RESOLVING_STEPS = new Set(["if", "loop", "subflow"]);

async function executeStep(
  step: Step,
  scope: Record<string, unknown>,
  registry: StepRegistry,
  runId: string,
  path: string,
  emit: EmitFn,
): Promise<unknown> {
  const maxRetries = step.options?.retry?.max ?? 0;
  const retryDelay = step.options?.retry?.delayMs ?? 0;

  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        await emit({
          type: "step.retry",
          path,
          stepType: step.type,
          iteration: attempt,
        });
        await sleep(retryDelay);
      }

      const startTime = Date.now();

      // Control flow steps handle their own config resolution
      // (e.g. loop needs to re-resolve `until` each iteration,
      //  and may reference $current which doesn't exist yet).
      const resolvedConfig = SELF_RESOLVING_STEPS.has(step.type)
        ? step.config
        : (resolveConfig(step.config, scope) as Record<string, unknown>);

      await emit({
        type: "step.start",
        path,
        stepType: step.type,
        input: SELF_RESOLVING_STEPS.has(step.type) ? undefined : resolvedConfig,
      });

      // Execute based on step type
      const output = await dispatchStep(
        step,
        resolvedConfig,
        scope,
        registry,
        runId,
        path,
        emit,
      );

      const durationMs = Date.now() - startTime;

      await emit({
        type: "step.end",
        path,
        stepType: step.type,
        output,
        durationMs,
      });

      return output;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));

      if (attempt === maxRetries) {
        // All retries exhausted
        if (step.options?.onError) {
          // Run fallback step
          const errorScope = {
            ...scope,
            $error: { message: lastError.message, stack: lastError.stack },
          };
          try {
            const fallbackOutput = await executeStep(
              step.options.onError,
              errorScope,
              registry,
              runId,
              `${path}/onError`,
              emit,
            );
            return fallbackOutput;
          } catch (fallbackErr) {
            // Fallback itself failed
            await emit({
              type: "step.error",
              path,
              stepType: step.type,
              error: {
                message: lastError.message,
                stack: lastError.stack,
              },
            });
            throw fallbackErr;
          }
        }

        await emit({
          type: "step.error",
          path,
          stepType: step.type,
          error: {
            message: lastError.message,
            stack: lastError.stack,
          },
        });
        throw lastError;
      }
    }
  }

  // Should not reach here
  throw lastError ?? new Error("Unknown error");
}

// ── Internal: dispatch to the correct step handler ─────────────────────────

async function dispatchStep(
  step: Step,
  resolvedConfig: Record<string, unknown>,
  scope: Record<string, unknown>,
  registry: StepRegistry,
  runId: string,
  path: string,
  emit: EmitFn,
): Promise<unknown> {
  // Handle core control flow steps specially
  switch (step.type) {
    case "if":
      return executeIf(resolvedConfig, scope, registry, runId, path, emit);

    case "loop":
      return executeLoop(step, resolvedConfig, scope, registry, runId, path, emit);

    case "subflow":
      return executeSubflow(resolvedConfig, scope, registry, runId, path, emit);

    default: {
      // Look up in registry
      const def = registry[step.type];
      if (!def) {
        throw new Error(`Unknown step type: "${step.type}"`);
      }

      // Validate config against step's input schema.
      // Default to {} when no config is provided in the YAML.
      const validConfig = def.input.parse(resolvedConfig ?? {});

      const ctx: StepContext = {
        runId,
        path,
        scope,
        input: scope["input"],
        emit,
      };

      return def.run(validConfig, ctx);
    }
  }
}

// ── Control flow implementations ───────────────────────────────────────────

async function executeIf(
  config: Record<string, unknown>,
  scope: Record<string, unknown>,
  registry: StepRegistry,
  runId: string,
  path: string,
  emit: EmitFn,
): Promise<unknown> {
  // Resolve the condition (but not then/else which are Step objects)
  const cond = resolveConfig(config["cond"], scope);
  const thenStep = config["then"] as Step;
  const elseStep = config["else"] as Step;

  const branch = cond ? thenStep : elseStep;
  if (!branch) return undefined;

  const branchName = cond ? "then" : "else";
  return executeStep(branch, scope, registry, runId, `${path}/${branchName}`, emit);
}

async function executeLoop(
  step: Step,
  _config: Record<string, unknown>,
  scope: Record<string, unknown>,
  registry: StepRegistry,
  runId: string,
  path: string,
  emit: EmitFn,
): Promise<unknown> {
  // Resolve scalar config values (but not `until` or `body` which need per-iteration resolution)
  const maxIterations = resolveConfig(step.config["maxIterations"], scope) as number;
  const delayMs = (resolveConfig(step.config["delayMs"], scope) as number) ?? 0;
  const untilExpr = step.config["until"] as string; // raw template, re-evaluated each iteration
  // Use the raw (unresolved) body step — we re-resolve its config each iteration
  const rawBody = step.config["body"] as Step;

  let current: unknown = undefined;

  for (let i = 0; i < maxIterations; i++) {
    // Make $current available in scope for template resolution
    const iterScope = { ...scope, $current: current };

    // Resolve the body step's config with this iteration's scope
    const resolvedBodyConfig = resolveConfig(rawBody.config, iterScope) as Record<string, unknown>;

    await emit({
      type: "step.start",
      path: `${path}#${i}`,
      stepType: rawBody.type,
      iteration: i,
    });

    const startTime = Date.now();
    current = await dispatchStep(
      rawBody,
      resolvedBodyConfig,
      iterScope,
      registry,
      runId,
      `${path}#${i}`,
      emit,
    );

    await emit({
      type: "step.end",
      path: `${path}#${i}`,
      stepType: rawBody.type,
      output: current,
      durationMs: Date.now() - startTime,
      iteration: i,
    });

    if (delayMs > 0 && i < maxIterations - 1) {
      await sleep(delayMs);
    }

    // Evaluate the `until` condition with updated $current
    const untilScope = { ...scope, $current: current };
    const done = resolveConfig(untilExpr, untilScope);
    if (done) return current;
  }

  throw new Error(
    `Loop "${step.id}" exceeded maxIterations (${maxIterations}) without until becoming true`,
  );
}

async function executeSubflow(
  config: Record<string, unknown>,
  scope: Record<string, unknown>,
  registry: StepRegistry,
  runId: string,
  path: string,
  emit: EmitFn,
): Promise<unknown> {
  const childFlow = config["flow"] as Flow;
  // Resolve the input (may contain templates referencing parent scope)
  const childInput = resolveConfig(config["input"], scope);

  if (!childFlow || !childFlow.name || !childFlow.steps) {
    throw new Error("subflow step requires a 'flow' config with a Flow value");
  }

  // Validate child flow input against its schema
  const validatedInput = childFlow.input.parse(childInput);

  return executeFlow(childFlow, validatedInput, registry, runId, path, emit);
}

// ── Utilities ──────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
