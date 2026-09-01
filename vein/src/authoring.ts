import { join } from "node:path";
import type { RunEvent, RunResult, RunSummary, StepRegistry } from "./core.js";
import type { WorkspaceManager } from "./workspace.js";
import type { RunStore } from "./store.js";
import { generateRunId } from "./store.js";
import { runWorkflow } from "./runner.js";
import { runSingleStep, cassettePath, type RunStepResult } from "./run-step.js";
import { stepLoadError } from "./steps/registry.js";
import type { CassetteMode } from "./cassette.js";
import type { SecretInfo } from "./secret-store.js";
import { lsSteps, searchSteps, readStepSource } from "./ai/stepHelpers.js";
import { zodToFields } from "./ai/schemaHelpers.js";

/**
 * The AUTHORING core — the workspace's author/test/inspect operations, shared
 * by two consumers:
 *
 *   - the chat builder's AI tools (`ai/tools.ts`) — the human-supervised
 *     authoring surface; and
 *   - the `meta/*` lib steps — the same operations as REGISTRY STEPS, so an
 *     in-workflow agent (`agentTools: ["meta/*"]`) can author, test, and
 *     inspect candidate workflows from inside a run (EVOLVE_SPEC §5).
 *
 * The exported helpers are the shared MECHANISM (conflict checks, strict
 * load-verification, run-history reads). `buildAuthoringCapability` bakes in
 * the meta-surface POLICY on top (EVOLVE_SPEC §6): everything the capability
 * publishes is stamped `publisher: "ai"`, and its publish / run /
 * run-history operations are CLOSED over that stamped set — it refuses to
 * touch, run, or read runs of workflows it didn't publish. A harness
 * workflow's run log records what its grader steps were handed, so
 * run-history reads on unstamped workflows fail closed.
 */

/** The provenance stamp for AI-authored artifacts (steps AND workflows). */
export const AI_PUBLISHER = "ai";

// ── Run-history reads (shared mechanism) ───────────────────────────────────

/** Drop bulky input/output payloads from an event so a run's event list stays
 *  token-cheap; the caller can re-fetch a specific run's full events if needed. */
export function slimEvent(e: RunEvent) {
  return {
    type: e.type,
    path: e.path,
    ...(e.stepType ? { stepType: e.stepType } : {}),
    ...(e.durationMs != null ? { durationMs: e.durationMs } : {}),
    ...(e.iteration != null ? { iteration: e.iteration } : {}),
    ...(e.error ? { error: e.error } : {}),
  };
}

/** List a workflow's recent runs (newest first) as slim summaries. */
export async function listRunSummaries(
  store: Pick<RunStore, "listRuns" | "getRunSummary" | "getRunEvents">,
  name: string,
  limit: number,
) {
  const ids = (await store.listRuns(name)).slice(0, limit);
  return Promise.all(
    ids.map(async (runId) => {
      const s = await store.getRunSummary(name, runId);
      return {
        runId,
        status: s?.status,
        startedAt: s?.startedAt,
        durationMs: s?.durationMs,
        ...(s?.error ? { error: s.error } : {}),
      };
    }),
  );
}

/** Read one run's summary + events (slimmed unless `fullEvents`). */
export async function readRun(
  store: Pick<RunStore, "listRuns" | "getRunSummary" | "getRunEvents">,
  name: string,
  runId: string,
  fullEvents: boolean,
) {
  const [summary, rawEvents] = await Promise.all([
    store.getRunSummary(name, runId),
    store.getRunEvents(name, runId),
  ]);
  if (!summary && rawEvents.length === 0) {
    return { error: `Run "${runId}" not found for workflow "${name}".` };
  }
  return {
    workflow: name,
    runId,
    summary,
    events: fullEvents ? rawEvents : rawEvents.map(slimEvent),
  };
}

// ── Run search (shared mechanism) ──────────────────────────────────────────

export interface RunSearchOptions {
  /** Explicit run ids to search (e.g. one eval batch). Default: newest `runLimit` runs. */
  runIds?: string[];
  /** How many recent runs to scan when `runIds` is absent. Default 20. */
  runLimit?: number;
  /** Cap on returned match entries; scanning stops once reached. Default 50. */
  maxMatches?: number;
  /** Case-insensitive matching. Default true (signature hunting favors recall). */
  ignoreCase?: boolean;
}

/** One matching event from a run search. */
export interface RunSearchMatch {
  runId: string;
  path: string;
  type: string;
  stepType?: string;
  /** Matches within this one event (the snippet shows the first). */
  count: number;
  /** The matched text with surrounding context from the event's JSON. */
  snippet: string;
}

const SNIPPET_BEFORE = 80;
const SNIPPET_AFTER = 160;

/**
 * Grep across a workflow's run event logs — the cross-run question the
 * per-run `readRun` can't answer without N calls and N payloads ("which runs
 * hit `ModuleNotFoundError`, and how often?" — EVOLVE_SPEC §4.2 capture).
 * Each event is matched as its JSON line (the same shape events.jsonl holds),
 * so input/output/error payloads are all searchable; matches come back as
 * (runId, event path, snippet) tuples plus a per-run frequency summary.
 * Scanning stops at `maxMatches` (`truncated: true`) — narrow the pattern or
 * the run window rather than raising the cap.
 */
export async function searchRunEvents(
  store: Pick<RunStore, "listRuns" | "getRunSummary" | "getRunEvents">,
  name: string,
  pattern: string,
  opts: RunSearchOptions = {},
) {
  const { runLimit = 20, maxMatches = 50, ignoreCase = true } = opts;
  let re: RegExp;
  try {
    re = new RegExp(pattern, ignoreCase ? "gi" : "g");
  } catch (err) {
    return { error: `Invalid pattern: ${err instanceof Error ? err.message : String(err)}` };
  }

  const runIds = opts.runIds?.length
    ? opts.runIds
    : (await store.listRuns(name)).slice(0, runLimit);

  const matches: RunSearchMatch[] = [];
  const perRun: { runId: string; matchingEvents: number }[] = [];
  let runsScanned = 0;
  let truncated = false;

  for (const runId of runIds) {
    if (truncated) break;
    const events = await store.getRunEvents(name, runId);
    runsScanned++;
    let matchingEvents = 0;
    for (const e of events) {
      const line = JSON.stringify(e);
      re.lastIndex = 0;
      const first = re.exec(line);
      if (!first) continue;
      matchingEvents++;
      let count = 1;
      while (re.exec(line) !== null) count++;
      matches.push({
        runId,
        path: e.path,
        type: e.type,
        ...(e.stepType ? { stepType: e.stepType } : {}),
        count,
        snippet: line.slice(
          Math.max(0, first.index - SNIPPET_BEFORE),
          first.index + first[0].length + SNIPPET_AFTER,
        ),
      });
      if (matches.length >= maxMatches) {
        truncated = true;
        break;
      }
    }
    if (matchingEvents > 0) perRun.push({ runId, matchingEvents });
  }

  return {
    workflow: name,
    pattern,
    runsScanned,
    runsWithMatches: perRun,
    matches,
    ...(truncated
      ? { truncated: true, note: "Match cap reached — narrow the pattern or run window." }
      : {}),
  };
}

// ── Step publishing (shared mechanism) ─────────────────────────────────────

/** LLMs sometimes pass an object-valued arg as a JSON *string* (e.g.
 *  run_workflow's `input`). The template engine then sees a string, so
 *  `{{ input.owner }}` resolves to undefined and every field fails validation.
 *  Defensively parse a JSON string back into the object/array it represents;
 *  leave anything else untouched. */
export function coerceJsonArg(v: unknown): unknown {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!(t.startsWith("{") || t.startsWith("["))) return v;
  try {
    return JSON.parse(t);
  } catch {
    return v;
  }
}

/** What both publish paths need. The chat builder's `AiDeps` satisfies it
 *  structurally; the authoring capability builds its own. `getRegistry` must
 *  return a FRESH registry (re-scanned from the workspace). */
export interface StepPublishDeps {
  workspace: WorkspaceManager;
  getRegistry(): Promise<StepRegistry>;
  publishingEnabled?: boolean;
}

export interface StepPublishResult {
  ok?: true;
  error?: string;
  type?: string;
  version?: string;
  changed?: boolean;
  /** Whether the published source actually loaded into the registry. */
  loaded?: boolean;
  /** The import/shape error when `loaded` is false (§5.3.4: a broken step
   *  otherwise fails silently — `loadStepFile` warns and returns null, so the
   *  step simply doesn't exist). */
  loadError?: string;
}

/** Verify a just-published step actually loads; surface the error if not. */
async function verifyLoaded(
  deps: StepPublishDeps,
  name: string,
): Promise<{ loaded: boolean; loadError?: string }> {
  const fresh = await deps.getRegistry();
  if (fresh[name]) return { loaded: true };
  const err = await stepLoadError(
    join(deps.workspace.path, "steps", "custom", `${name}.ts`),
  );
  return {
    loaded: false,
    loadError:
      err ??
      "step did not appear in the registry — check the source imports only 'vein' and has a valid defineStep default export",
  };
}

/** Author a NEW custom step (the chat `create_step` / `meta/create-step`
 *  mechanism): refuses existing names and built-in collisions, publishes as
 *  v1, then load-verifies the source and hands any import error back. */
export async function publishNewStep(
  deps: StepPublishDeps,
  name: string,
  code: string,
  description?: string,
  publisher?: string,
): Promise<StepPublishResult> {
  if (deps.publishingEnabled === false) {
    return { error: "Step publishing is disabled (the registry was injected at construction)." };
  }
  const customs = await deps.workspace.listSteps();
  if (customs.some((s) => s.type === name)) {
    return { error: `Step "${name}" already exists. Use edit_step to publish a new version.` };
  }
  if ((await deps.getRegistry())[name]) {
    return { error: `"${name}" conflicts with a built-in (core/lib) step. Choose another name.` };
  }
  let result;
  try {
    result = await deps.workspace.publishStep(name, code, description, publisher);
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
  return { ok: true, type: name, version: result.version, ...(await verifyLoaded(deps, name)) };
}

/** Publish a NEW VERSION of an existing custom step (the chat `edit_step` /
 *  `meta/edit-step` mechanism). Pass `requirePublisher` to enforce the meta
 *  ownership rule: only steps stamped with that publisher may be edited. */
export async function publishStepVersion(
  deps: StepPublishDeps,
  type: string,
  code: string,
  description?: string,
  opts: { requirePublisher?: string } = {},
): Promise<StepPublishResult> {
  if (deps.publishingEnabled === false) {
    return { error: "Step publishing is disabled (the registry was injected at construction)." };
  }
  const customs = await deps.workspace.listSteps();
  const existing = customs.find((s) => s.type === type);
  if (!existing) {
    return {
      error: (await deps.getRegistry())[type]
        ? `"${type}" is a built-in step and can't be edited. Use create_step with a new name.`
        : `Step "${type}" not found. Use create_step to author a new step.`,
    };
  }
  if (opts.requirePublisher && existing.publisher !== opts.requirePublisher) {
    return {
      error:
        `Step "${type}" was not published by "${opts.requirePublisher}" (publisher: ` +
        `${existing.publisher ?? "none"}) — the meta surface only edits steps it authored. ` +
        `Author a new step under a different name instead.`,
    };
  }
  let result;
  try {
    result = await deps.workspace.publishStep(type, code, description);
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
  return {
    ok: true,
    type,
    version: result.version,
    changed: result.changed,
    ...(await verifyLoaded(deps, type)),
  };
}

// ── The capability ─────────────────────────────────────────────────────────

export interface RunStepArgs {
  config?: Record<string, unknown>;
  input?: unknown;
  params?: Record<string, unknown>;
  cassette?: CassetteMode;
  cassetteName?: string;
}

/**
 * The authoring capability injected as `services.authoring` — what the
 * `meta/*` lib steps are thin plumbing over. Auto-provided by `createVein`
 * (like `http` / `secrets` / `artifacts`); embedders can inject their own.
 */
export interface AuthoringCapability {
  listSteps(path?: string): Promise<unknown>;
  searchSteps(query: string): Promise<unknown>;
  getStep(type: string): Promise<unknown>;
  createStep(name: string, code: string, description?: string): Promise<StepPublishResult>;
  editStep(type: string, code: string, description?: string): Promise<StepPublishResult>;
  runStep(type: string, args?: RunStepArgs): Promise<RunStepResult | { error: string }>;
  listWorkflows(): Promise<unknown>;
  getWorkflow(name: string, version?: string): Promise<unknown>;
  publishWorkflow(
    name: string,
    yaml: string,
    description?: string,
    category?: string,
  ): Promise<unknown>;
  runWorkflow(
    name: string,
    input?: unknown,
    params?: Record<string, unknown>,
    version?: string,
    /** `parentRunId` = the calling step's `ctx.runId`, linking the nested
     *  run's controller under the launching run's (subtree control). */
    opts?: { parentRunId?: string },
  ): Promise<RunResult | { error: string }>;
  listRuns(name: string, limit?: number): Promise<unknown>;
  getRun(name: string, runId: string, fullEvents?: boolean): Promise<unknown>;
  searchRuns(name: string, pattern: string, opts?: RunSearchOptions): Promise<unknown>;
  listSecrets(): Promise<unknown>;
}

export interface AuthoringDeps extends StepPublishDeps {
  store: RunStore;
  /** Capabilities bag threaded into `runWorkflow` / `runStep` so authored
   *  steps reach `ctx.services` (http, secrets, and any consumer services). */
  services?: unknown;
  /** Read-only view of the deployment's secret store (NAMES only — never
   *  values). Optional: `listSecrets` degrades gracefully when absent. */
  secrets?: { list(): Promise<SecretInfo[]> };
  /** Register a nested run as in-flight, creating its RunController —
   *  attached to the launching run's controller when `parentRunId` is given,
   *  so cancelling/pausing the parent reaches this run (RUN_CONTROL_SPEC
   *  §2.2 tree linkage). Also drives the runs listing ("running" vs
   *  "stale"). Optional: embedders without a live server need not care. */
  trackRun?: (
    workflow: string,
    runId: string,
    parentRunId?: string,
  ) => { controller?: import("./run-control.js").RunController; untrack: () => void };
}

export function buildAuthoringCapability(deps: AuthoringDeps): AuthoringCapability {
  const { workspace, store } = deps;

  const explorerDeps = async () => ({ workspace, registry: await deps.getRegistry() });

  const findWorkflow = async (name: string) =>
    (await workspace.listWorkflows()).find((w) => w.name === name);

  /** The ownership gate: the meta surface acts only on workflows it
   *  published (EVOLVE_SPEC §6). Returns an error message, or null when the
   *  workflow exists and is stamped. */
  const notOwned = async (name: string, verb: string): Promise<string | null> => {
    const entry = await findWorkflow(name);
    if (!entry) return `Workflow "${name}" not found`;
    if (entry.publisher !== AI_PUBLISHER) {
      return (
        `Workflow "${name}" is not agent-authored (publisher: ${entry.publisher ?? "none"}) — ` +
        `the meta surface only ${verb} workflows it published. ` +
        `Author a candidate with meta/publish-workflow under a new name instead.`
      );
    }
    return null;
  };

  return {
    async listSteps(path = "steps") {
      return lsSteps(path, await explorerDeps());
    },

    async searchSteps(query) {
      return searchSteps(query, await explorerDeps());
    },

    async getStep(type) {
      const d = await explorerDeps();
      const def = d.registry[type];
      if (!def) return { error: `Step type "${type}" not found` };
      return {
        type,
        description: def.description,
        fields: zodToFields(def.input),
        source: await readStepSource(type, d),
      };
    },

    async createStep(name, code, description) {
      const result = await publishNewStep(deps, name, code, description, AI_PUBLISHER);
      // For the in-workflow author a broken publish is a FAILURE, not a
      // warning — §5.3.4: hand the import error back loudly.
      if (result.ok && result.loaded === false) {
        return {
          ...result,
          ok: undefined,
          error: `Published as ${result.version} but the step failed to load: ${result.loadError}`,
        };
      }
      return result;
    },

    async editStep(type, code, description) {
      const result = await publishStepVersion(deps, type, code, description, {
        requirePublisher: AI_PUBLISHER,
      });
      if (result.ok && result.loaded === false) {
        return {
          ...result,
          ok: undefined,
          error: `Published as ${result.version} but the step failed to load: ${result.loadError}. Prior versions are retained — fix and edit again, or roll back.`,
        };
      }
      return result;
    },

    async runStep(type, args = {}) {
      // FRESH registry: a step authored earlier in this same run must be
      // runnable here — the run's own `ctx.registry` is a start-of-run
      // snapshot and would not contain it (EVOLVE_SPEC §5.3.1).
      const registry = await deps.getRegistry();
      if (!registry[type]) return { error: `Step type "${type}" not found` };
      return runSingleStep(type, registry, deps.services, {
        config: coerceJsonArg(args.config) as Record<string, unknown> | undefined,
        input: coerceJsonArg(args.input),
        params: coerceJsonArg(args.params) as Record<string, unknown> | undefined,
        workspace,
        ...(args.cassette
          ? {
              cassette: {
                mode: args.cassette,
                path: cassettePath(workspace.path, args.cassetteName ?? type),
              },
            }
          : {}),
      });
    },

    async listWorkflows() {
      return { workflows: await workspace.listWorkflows() };
    },

    async getWorkflow(name, version) {
      const entry = await findWorkflow(name);
      if (!entry) return { error: `Workflow "${name}" not found` };
      const resolved = version ?? entry.activeVersion;
      let yaml;
      try {
        yaml = await workspace.getWorkflowSource(name, resolved);
      } catch {
        return {
          error: `Version "${resolved}" not found for "${name}". Available: ${entry.versions.join(", ")}`,
        };
      }
      return {
        name,
        version: resolved,
        activeVersion: entry.activeVersion,
        versions: entry.versions,
        description: entry.description,
        publisher: entry.publisher,
        yaml,
      };
    },

    async publishWorkflow(name, yaml, description, category) {
      const entry = await findWorkflow(name);
      if (entry && entry.publisher !== AI_PUBLISHER) {
        return {
          error:
            `Workflow "${name}" exists and is not agent-authored (publisher: ` +
            `${entry.publisher ?? "none"}). The meta surface never edits workflows it ` +
            `didn't publish — choose a new name to author a candidate.`,
        };
      }
      try {
        const result = await workspace.publishWorkflowByContent(
          name,
          yaml,
          description,
          category,
          AI_PUBLISHER,
        );
        return {
          ok: true,
          name,
          version: result.version,
          changed: result.changed,
          created: !entry,
        };
      } catch (err) {
        return { error: err instanceof Error ? err.message : String(err) };
      }
    },

    async runWorkflow(name, input, params, version, opts) {
      const gate = await notOwned(name, "runs");
      if (gate) return { error: gate };
      let flow;
      try {
        flow = version
          ? await workspace.getWorkflowVersion(name, version)
          : await workspace.getWorkflow(name);
      } catch (err) {
        return { error: err instanceof Error ? err.message : String(err) };
      }
      // FRESH registry, same reason as runStep: steps published mid-run are
      // invisible to the enclosing run's registry snapshot.
      const registry = await deps.getRegistry();
      const runId = generateRunId();
      // Tree linkage: attach this nested run's controller to the launching
      // run's (opts.parentRunId = the calling step's ctx.runId), so controls
      // on the parent reach it (RUN_CONTROL_SPEC §2.2).
      const tracked = deps.trackRun?.(flow.name, runId, opts?.parentRunId);
      try {
        return await runWorkflow(flow, coerceJsonArg(input) ?? {}, registry, {
          runId,
          store,
          workspace,
          services: deps.services,
          params: coerceJsonArg(params) as Record<string, unknown> | undefined,
          controller: tracked?.controller,
          workflowHash:
            (await workspace.getWorkflowHash(flow.name, version)) ?? undefined,
        });
      } finally {
        tracked?.untrack();
      }
    },

    async listRuns(name, limit = 20) {
      const gate = await notOwned(name, "reads run history of");
      if (gate) return { error: gate };
      return { workflow: name, runs: await listRunSummaries(store, name, limit) };
    },

    async getRun(name, runId, fullEvents = false) {
      const gate = await notOwned(name, "reads run history of");
      if (gate) return { error: gate };
      return readRun(store, name, runId, fullEvents);
    },

    async searchRuns(name, pattern, opts) {
      const gate = await notOwned(name, "reads run history of");
      if (gate) return { error: gate };
      return searchRunEvents(store, name, pattern, opts);
    },

    async listSecrets() {
      if (!deps.secrets) {
        return { error: "Secret store is not available in this deployment." };
      }
      const secrets = await deps.secrets.list();
      return { secrets: secrets.map((s) => ({ name: s.name, updatedAt: s.updatedAt })) };
    },
  };
}
