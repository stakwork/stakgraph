import { z } from "zod";
import type { Flow, StepRegistry, RunEvent, RunResult } from "./core.js";
import { runWorkflow, type SubflowResolver } from "./runner.js";
import { MemoryRunStore } from "./store.js";
import {
  withCassette,
  loadCassette,
  saveCassette,
  type CassetteMode,
} from "./cassette.js";

/**
 * Run a SINGLE step in isolation — the tight inner loop for authoring adapters.
 *
 * Unlike a workflow run (detached, launch+tail), this is synchronous: it wraps
 * the step in an ad-hoc one-step flow, runs it to completion against an
 * in-memory store, and returns the output + events directly — so the chat agent
 * (or a developer) can author → run → fix without wiring anything into a
 * workflow.
 *
 * With `cassette`, external service calls go through the record/replay wrapper:
 *   - `record` — run live, capture every `ctx.services` call to the cassette
 *     file (secrets scrubbed), so the next iteration can…
 *   - `replay` — …serve those calls from the file: offline, deterministic, no
 *     rate limits, no cost, no side effects.
 */
export interface RunStepOptions {
  /** The step's config (same shape as a workflow step's `config`). Templates
   *  like `{{ input.* }}` / `{{ params.* }}` are resolved. */
  config?: Record<string, unknown>;
  /** Workflow input, referenced in config via `{{ input.* }}`. */
  input?: unknown;
  /** Params knobs, referenced via `{{ params.* }}`. */
  params?: Record<string, unknown>;
  /** Record/replay external service calls against a cassette file. */
  cassette?: { mode: CassetteMode; path: string };
  /** Subflow resolver — only needed if the step itself is a `subflow`. */
  workspace?: SubflowResolver;
}

export interface RunStepResult {
  status: "success" | "error";
  output?: unknown;
  error?: { message: string; stack?: string };
  /** Every event the step emitted (start/end/error, plus nested for containers). */
  events: RunEvent[];
  /** Number of recorded service calls (present when a cassette was used). */
  recorded?: number;
}

export async function runSingleStep(
  type: string,
  registry: StepRegistry,
  services: unknown,
  opts: RunStepOptions = {},
): Promise<RunStepResult> {
  if (!registry[type]) {
    return {
      status: "error",
      error: { message: `Step type "${type}" not found` },
      events: [],
    };
  }

  const flow: Flow = {
    name: "__run_step__",
    input: z.any(),
    steps: [{ id: "step", type, config: opts.config ?? {} }],
    ...(opts.params != null ? { params: opts.params } : {}),
  };

  const cassette = opts.cassette ? await loadCassette(opts.cassette.path) : null;
  const runServices =
    opts.cassette && cassette
      ? withCassette(services as Record<string, unknown>, {
          mode: opts.cassette.mode,
          cassette,
        })
      : services;

  const events: RunEvent[] = [];
  const result: RunResult = await runWorkflow(flow, opts.input ?? {}, registry, {
    store: new MemoryRunStore(),
    services: runServices,
    workspace: opts.workspace,
    onEvent: (e) => {
      events.push(e);
    },
  });

  // Persist newly-captured calls only when recording.
  if (opts.cassette?.mode === "record" && cassette) {
    await saveCassette(opts.cassette.path, cassette);
  }

  return {
    status: result.status,
    output: result.output,
    error: result.error,
    events,
    ...(cassette ? { recorded: cassette.entries.length } : {}),
  };
}

/** Default on-disk location for a step's cassette, under the workspace. */
export function cassettePath(workspacePath: string, name: string): string {
  // `name` may contain slashes (namespaced step types) — they become subdirs.
  return `${workspacePath}/steps/_cassettes/${name}.json`;
}
