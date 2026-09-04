import { Hono } from "hono";
import { logger } from "hono/logger";
import { streamSSE } from "hono/streaming";
import { serveStatic } from "@hono/node-server/serve-static";
import { serve } from "@hono/node-server";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";

import type { Flow, StepRegistry, RunEvent, RunResult } from "./core.js";
import type { RunStore } from "./store.js";
import { FileRunStore, MemoryRunStore, generateRunId, summarizeFromEvents } from "./store.js";
import type { ChatStore, ChatEvent, ChatMeta } from "./chat-store.js";
import {
  FileChatStore,
  MemoryChatStore,
  generateChatId,
  truncateToolMessages,
} from "./chat-store.js";
import { FileWorkspaceStore, type WorkspaceStore } from "./workspace.js";
import { buildRegistry } from "./steps/registry.js";
import { maxOutputTokensFor } from "./pricing.js";
import type { StepSources } from "./steps/registry.js";
import { runWorkflow } from "./runner.js";
import { RunController } from "./run-control.js";
import { buildJournal, invalidateFrom, readRunStart } from "./journal.js";
import { requireApiKey, warnIfUnconfigured } from "./auth.js";
import { standardServices, fileArtifactsCapability } from "./capabilities.js";
import type { ArtifactsCapability } from "./capabilities.js";
import type { SecretStore } from "./secret-store.js";
import {
  FileSecretStore,
  MemorySecretStore,
  isValidSecretName,
} from "./secret-store.js";
import { runSingleStep, cassettePath } from "./run-step.js";
import { buildAuthoringCapability } from "./authoring.js";
import type { CassetteMode } from "./cassette.js";
// Type-only: the graph backend stays a lazy, opt-in dependency.
import type { GraphBackend } from "./graph/backend.js";
// Static import is safe: notifier depends only on chat-store, never the AI
// SDK (which stays lazy-loaded inside launchChatTurn).
import { createChatNotifier, formatRunNotification } from "./ai/notifier.js";

// ── Public types ───────────────────────────────────────────────────────────

/**
 * Options for constructing a Vein instance. Everything is optional — pass
 * nothing for the default "filesystem-backed server" behavior, or supply
 * any subset to embed vein in your own app.
 */
export interface VeinOptions<TServices = unknown> {
  /** Persistent store for workflows and steps. Defaults to a new
   *  `FileWorkspaceStore()` (reads `VEIN_WORKSPACE` env, falls back to
   *  `./workspace`). Any `WorkspaceStore` implementation works. */
  workspace?: WorkspaceStore;

  /** Local directory for the inherently-local things: run artifacts,
   *  step cassettes, the chat builder's shell cwd + scratch/. Defaults to
   *  the file workspace's root (so a file-backed deployment keeps one
   *  directory), else `VEIN_WORKSPACE` / `./workspace`. A non-file
   *  workspace can point this at any scratch volume — losing it loses
   *  blobs and cassettes, never workspace records. */
  dataDir?: string;

  /** Step registry. If supplied, used as-is and `rebuildRegistry` becomes
   *  a no-op (the consumer owns step composition). If omitted, vein
   *  discovers steps via `buildRegistry(await workspace.materializeCustomSteps())`. */
  registry?: StepRegistry;

  /** Where to persist run events + summaries. Defaults to a `FileRunStore`
   *  rooted at the workspace path. Pass `new MemoryRunStore()` for
   *  ephemeral / test environments. */
  store?: RunStore;

  /** Consumer-defined capabilities bag exposed to every step via
   *  `ctx.services`. Use this to inject environment-specific
   *  implementations (Neo4j vs in-memory store, real vs fake LLM, …)
   *  without changing the workflow or registry. */
  services?: TServices;

  /** When true, mount the static web UI under `/` (SPA fallback) and
   *  `/assets/*`. Defaults to true. Disable when embedding vein under a
   *  larger app that owns its own UI routes. */
  serveUi?: boolean;

  /** Mount the `POST /chat` AI workflow-builder endpoint. Defaults to
   *  true; disable to avoid pulling in the `ai`/`@ai-sdk/anthropic` deps. */
  enableChat?: boolean;

  /** Where to persist chat sessions (the detached AI-builder background jobs:
   *  `messages.jsonl` + `events.jsonl` + `meta.json`). Defaults to a
   *  `FileChatStore` rooted at the workspace path (or `MemoryChatStore` when
   *  `store` is a `MemoryRunStore`). */
  chatStore?: ChatStore;

  /** Deployment-scoped secret store backing `ctx.services.secrets` and the
   *  `/secrets` admin endpoints. Defaults to an encrypted `FileSecretStore`
   *  rooted at the workspace path (or `MemorySecretStore` when `store` is a
   *  `MemoryRunStore`). The default `secrets` capability reads this store with
   *  `process.env` as fallback. Pass your own `services.secrets` to bypass
   *  entirely (then the `/secrets` endpoints return 501). */
  secretStore?: SecretStore;

  /** Max agent steps (tool-call iterations) per chat turn. Raise for longer
   *  autonomous "let it rip" loops. Defaults to `VEIN_CHAT_MAX_STEPS` or 100. */
  chatMaxSteps?: number;

  /** Anthropic model id for the chat agent. Defaults to `VEIN_CHAT_MODEL` or
   *  `claude-sonnet-5`. */
  chatModel?: string;


  /** How long the chat agent's `run_workflow` tool waits before a still-
   *  running workflow converts to a DETACHED run (the tool returns a
   *  `{ status: "running", runId }` stub and the chat is woken with a
   *  `[run-notification]` message when the run settles). Defaults to
   *  `VEIN_CHAT_RUN_WAIT_MS` or 60000. */
  chatRunWaitMs?: number;

  /** Max consecutive notification-triggered chat turns since the last human
   *  message before the chat PARKS (notifications still append to the
   *  transcript, but no turn launches until a human replies) — the runaway
   *  guard for autonomous launch→wake→relaunch loops. Defaults to
   *  `VEIN_CHAT_MAX_AUTO_TURNS` or 10. */
  chatMaxAutoTurns?: number;

  /** Boot-time auto-resume of runs cut off by a crash/restart
   *  (RUN_CONTROL_SPEC §5.3). Defaults to ON for a file-backed store unless
   *  `VEIN_AUTO_RESUME=0`; pass `false` to disable, or an object to tune
   *  the guards. Only the NEWEST root run per workflow is considered. */
  autoResume?: boolean | AutoResumeOptions;

  /** The vein graph backend, when the deployment has one (server.ts passes
   *  the one behind its graph-backed workspace). Enables the chat builder's
   *  read-only `graph_query` tool. Omit and the tool isn't offered. */
  graph?: GraphBackend;

  /** Directory containing the built web UI (the `dist` folder). Defaults
   *  to vein's own bundled UI resolved relative to this module, so it
   *  works regardless of the host process's CWD. The built UI uses
   *  relative asset paths, so it can be mounted at any sub-path (e.g.
   *  `/lab`) as long as the host serves it with a trailing slash. */
  webDist?: string;
}

export interface AutoResumeOptions {
  /** Ignore cut-off runs whose last event is older than this. Default 7 days. */
  maxAgeMs?: number;
  /** Give up on a run that has already been resumed this many times — a step
   *  that deterministically kills the server must not loop forever. Default 5. */
  maxResumes?: number;
  /** Delay after construction before the scan runs (lets the host finish
   *  booting). Default 3000ms; 0 runs it on the next tick. */
  delayMs?: number;
}

/** One line of the boot-time auto-resume report. */
export interface AutoResumeOutcome {
  workflow: string;
  runId: string;
  action: "resumed" | "finalized" | "skipped";
  reason: string;
}

/**
 * A configured vein instance. Carries the Hono `app` (mount it under your
 * own router, or call `listen()`), the underlying workspace / store /
 * services bag, and a typed `run()` helper that automatically threads
 * `services` into every workflow execution.
 */
export interface Vein<TServices = unknown> {
  /** Hono app with all vein routes mounted. Mount under your own router
   *  with `parent.route("/vein", vein.app)`, or call `vein.listen(port)`. */
  app: Hono;
  workspace: WorkspaceStore;
  /** Resolved local data directory (see `VeinOptions.dataDir`). */
  dataDir: string;
  store: RunStore;
  /** Deployment-scoped secret store backing `ctx.services.secrets` + the
   *  `/secrets` endpoints. */
  secretStore: SecretStore;
  services: TServices;

  /** Current registry. Reads through the closure so callers always see
   *  the latest after `rebuildRegistry()`. */
  getRegistry: () => StepRegistry;

  /** Re-scan the workspace for newly-published custom steps. No-op when
   *  the instance was constructed with an explicit `registry`. */
  rebuildRegistry: () => Promise<void>;

  /** Resume every run cut off by the previous process (RUN_CONTROL_SPEC
   *  §5.3): the newest root run of each workflow that has a log but no
   *  summary, was not paused or cancelling, is younger than `maxAgeMs`,
   *  and has fewer than `maxResumes` prior resumes. Runs automatically
   *  after boot when `autoResume` is enabled; callable directly. */
  autoResumeStaleRuns: (opts?: AutoResumeOptions) => Promise<AutoResumeOutcome[]>;

  /** Run a workflow by name (resolves through the workspace) or by Flow
   *  object. `services` is auto-injected from the instance; pass a
   *  `services` override in `opts` to use a different bag for one run. */
  run: (
    workflow: string | Flow,
    input?: unknown,
    opts?: VeinRunOptions<TServices>,
  ) => Promise<RunResult>;

  /** Boot the Hono server with `@hono/node-server`. Resolves to the
   *  bound port. Convenience wrapper — feel free to mount `app` yourself. */
  listen: (port?: number) => Promise<number>;
}

export interface VeinRunOptions<TServices = unknown> {
  runId?: string;
  /** Workflow version (only meaningful when `workflow` is a string). */
  version?: string;
  /** The launching run's id (the calling step's `ctx.runId`) — attaches this
   *  run's controller under the parent's, so cancel/pause on the parent
   *  reach it (RUN_CONTROL_SPEC §2.2 tree linkage). */
  parentRunId?: string;
  /** Per-event hook — useful for SSE streaming. */
  onEvent?: (event: RunEvent) => void | Promise<void>;
  /** Override the instance-level services for a single run. */
  services?: TServices;
  /** Per-run overrides for the workflow's `params` knobs (shallow-merged
   *  over the flow's `params` defaults). */
  params?: Record<string, unknown>;
  /** Per-run overrides keyed by workflow name, applied at every level of the
   *  execution tree (entry + nested subflows). See `RunOptions.paramOverrides`. */
  paramOverrides?: Record<string, Record<string, unknown>>;
}

// ── Zod → field descriptors (UI helper, used by /steps/:type/schema) ──────

interface FieldDesc {
  name: string;
  kind: "string" | "number" | "boolean" | "enum" | "json";
  required: boolean;
  default?: unknown;
  enumValues?: string[];
  description?: string;
}

function zodToFields(schema: z.ZodTypeAny): FieldDesc[] {
  const shape = getObjectShape(schema);
  if (!shape) return [];
  return Object.entries(shape).map(([name, s]) => describeField(name, s as z.ZodTypeAny));
}

/** Resolve a dotted/bracketed path (`a.b`, `a[0].b`) into a nested value.
 *  Used to pull a promote spec's `from` value out of a run's output. */
function getByPath(obj: unknown, path: string): unknown {
  if (!path) return obj;
  const parts = path.replace(/\[(\d+)\]/g, ".$1").split(".").filter(Boolean);
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

/** Split a promote spec's `to` ("<workflow>.<param>") on the FIRST dot. */
function parsePromoteTarget(to: string): { workflow: string; param: string } | null {
  const dot = to.indexOf(".");
  if (dot <= 0 || dot >= to.length - 1) return null;
  return { workflow: to.slice(0, dot), param: to.slice(dot + 1) };
}

// zod v4 def layout: `_def.type` is a lowercase kind string ("object",
// "optional", "default", ...), an object's `_def.shape` is a plain record,
// a default's `_def.defaultValue` is the VALUE (not a thunk), and `.refine`
// no longer wraps the schema (transforms become a "pipe" whose input is
// `_def.in`).
function getObjectShape(s: z.ZodTypeAny): Record<string, z.ZodTypeAny> | null {
  const def = s._def as any;
  if (def.type === "object") return def.shape;
  if (def.type === "pipe") return getObjectShape(def.in);
  return null;
}

function describeField(name: string, s: z.ZodTypeAny): FieldDesc {
  let required = true;
  let defaultVal: unknown = undefined;
  let inner = s;

  for (;;) {
    const def = inner._def as any;
    if (def.type === "optional") {
      required = false;
      inner = def.innerType;
    } else if (def.type === "default" || def.type === "prefault") {
      required = false;
      defaultVal = def.defaultValue;
      inner = def.innerType;
    } else if (def.type === "nullable") {
      required = false;
      inner = def.innerType;
    } else {
      break;
    }
  }

  const kind = (inner._def as any).type as string;

  if (kind === "enum") {
    return { name, kind: "enum", required, default: defaultVal, enumValues: (inner as any).options };
  }
  if (kind === "string") return { name, kind: "string", required, default: defaultVal };
  if (kind === "number") return { name, kind: "number", required, default: defaultVal };
  if (kind === "boolean") return { name, kind: "boolean", required, default: defaultVal };
  return { name, kind: "json", required, default: defaultVal };
}

// ── Factory ────────────────────────────────────────────────────────────────

/**
 * Build a configured Vein instance. This is the primary entry point for
 * using vein as a library: pass your registry (or let it be discovered
 * from disk), your services bag, and mount the returned Hono `app`
 * wherever you like.
 *
 * ```ts
 * import { createVein, createRegistry, defineStep } from "vein";
 *
 * interface MyServices { graph: GraphStore; llm: LLMClient }
 *
 * const vein = await createVein<MyServices>({
 *   registry: await createRegistry([myStep, anotherStep]),
 *   services: { graph: new Neo4jGraph(), llm: new Anthropic() },
 * });
 *
 * await vein.listen(3000);
 * ```
 *
 * The returned `app` can also be mounted under a parent Hono / Express
 * app — vein owns its routes (`/workflows`, `/steps`, `/chat`, `/health`)
 * but nothing else.
 */
export async function createVein<TServices = unknown>(
  opts: VeinOptions<TServices> = {},
): Promise<Vein<TServices>> {
  const workspace: WorkspaceStore = opts.workspace ?? new FileWorkspaceStore();
  // Backend mode, used ONLY to pick unspecified defaults: the run/chat/secret
  // stores follow the workspace's kind (file-backed → file stores under
  // dataDir; anything else → in-memory). No capability is gated on it —
  // every store is the full interface. An explicitly passed store always wins.
  const fileBacked = workspace instanceof FileWorkspaceStore;
  const dataDir =
    opts.dataDir ??
    (fileBacked ? workspace.path : (process.env["VEIN_WORKSPACE"] ?? "./workspace"));
  const store: RunStore = opts.store ?? (fileBacked ? new FileRunStore(dataDir) : new MemoryRunStore());
  // Controllers for runs currently executing **in this process** (keyed
  // `${workflow}/${runId}`) — RUN_CONTROL_SPEC §2.2. A controller's presence
  // IS "in-flight" (superseding the old `activeRuns` set): detached execution
  // is in-memory, so a run with no `run.json` summary is only genuinely live
  // if it's in this map; otherwise it never finalized (server restart /
  // crash) and is reported as "stale" — i.e. resumable (§5).
  const controllers = new Map<string, RunController>();
  // Secondary index for tree linkage: a nested launch names only its
  // `parentRunId` (the calling step's ctx.runId), not the parent workflow.
  const controllersByRunId = new Map<string, RunController>();
  /** Register a run as in-flight, creating its controller (attached to the
   *  launching run's controller when `parentRunId` resolves — controls apply
   *  to whole subtrees). Every launch path must register: HTTP
   *  (launchDetached), programmatic (vein.run), in-process nested runs
   *  (authoring's meta/run-workflow), and chat-detached runs. The returned
   *  untrack fn belongs in the launcher's finally. */
  const trackRun = (workflow: string, runId: string, parentRunId?: string) => {
    const key = `${workflow}/${runId}`;
    const parent = parentRunId ? controllersByRunId.get(parentRunId) : undefined;
    const controller = new RunController(runId, workflow, parent);
    controllers.set(key, controller);
    controllersByRunId.set(runId, controller);
    const untrack = () => {
      controllers.delete(key);
      if (controllersByRunId.get(runId) === controller) controllersByRunId.delete(runId);
      controller.detach();
    };
    return { controller, untrack };
  };
  /** Listing/stream status for a summary-less run: the controller's live
   *  state, or "stale" (orphaned by a crash/restart — resumable). */
  const liveStatus = (workflow: string, runId: string): string => {
    const controller = controllers.get(`${workflow}/${runId}`);
    return controller ? controller.state : "stale";
  };
  /** Append a control marker (run.paused / run.resumed) to a run's log so
   *  the parked gap is visible in the record (§4 observability). */
  const appendControlEvent = async (
    workflow: string,
    runId: string,
    type: "run.paused" | "run.resumed" | "run.cancelling",
  ) => {
    await store.append(workflow, runId, {
      ts: new Date().toISOString(),
      runId,
      path: workflow,
      type,
    });
  };
  // Deployment-scoped secret store backing the `secrets` capability + the
  // `/secrets` admin endpoints. Mirrors the run/chat store defaults: encrypted
  // file store for the standard server, in-memory when runs are in-memory.
  const secretStore: SecretStore =
    opts.secretStore ??
    (fileBacked ? new FileSecretStore(dataDir) : new MemorySecretStore());
  // Did the consumer inject their own `secrets` capability? If so we leave it
  // alone and the `/secrets` endpoints (which manage *our* store) report 501.
  const secretsInjected =
    opts.services != null &&
    typeof (opts.services as Record<string, unknown>)["secrets"] === "object";
  // Auto-provide the standard capabilities (http + secrets) every adapter step
  // builds on, with the consumer's bag spread on top so they can override or
  // extend any capability. The default `secrets` capability reads the secret
  // store (UI-managed) with `process.env` as fallback. This is what lets an
  // LLM-authored adapter rely on `ctx.services.http` / `ctx.services.secrets`
  // existing out of the box.
  const services = {
    ...(standardServices({ secretStore }) as unknown as Record<string, unknown>),
    // Per-run artifact files, rooted in the local data dir. A consumer bag
    // can override with its own ArtifactsCapability (spread below wins).
    artifacts: fileArtifactsCapability(join(dataDir, "artifacts")),
    ...((opts.services ?? {}) as Record<string, unknown>),
  } as TServices;
  const artifacts = (services as Record<string, unknown>)["artifacts"] as
    | ArtifactsCapability
    | undefined;
  const serveUi = opts.serveUi ?? true;
  const enableChat = opts.enableChat ?? true;
  const chatStore: ChatStore =
    opts.chatStore ?? (fileBacked ? new FileChatStore(dataDir) : new MemoryChatStore());
  const chatMaxSteps =
    opts.chatMaxSteps ?? Number(process.env["VEIN_CHAT_MAX_STEPS"] ?? 100);
  const chatModel =
    opts.chatModel ?? process.env["VEIN_CHAT_MODEL"] ?? "claude-sonnet-5";
  const chatRunWaitMs =
    opts.chatRunWaitMs ?? Number(process.env["VEIN_CHAT_RUN_WAIT_MS"] ?? 60_000);
  // Provider-derived infra constant (see pricing.ts — NOT an option: a wrong
  // value is only ever a bug). Without it the AI SDK defaults max_tokens to
  // 4096, which truncates a create_step tool call MID-JSON (it carries a
  // whole TS file in its `code` arg) — the turn dies with finish=length.
  const chatMaxOutputTokens = maxOutputTokensFor("anthropic");
  const chatMaxAutoTurns =
    opts.chatMaxAutoTurns ?? Number(process.env["VEIN_CHAT_MAX_AUTO_TURNS"] ?? 10);
  const webDist =
    opts.webDist ??
    resolve(dirname(fileURLToPath(import.meta.url)), "../web/dist");
  const registryWasInjected = opts.registry !== undefined;

  // Mutable closure state — `app` handlers read through these.
  let registry: StepRegistry = opts.registry ?? {};
  let stepSources: StepSources = {};

  async function rebuildRegistry(): Promise<void> {
    if (registryWasInjected) return; // consumer owns the registry
    const bundle = await buildRegistry(await workspace.materializeCustomSteps());
    registry = bundle.registry;
    stepSources = bundle.sources;
  }

  if (!registryWasInjected) {
    await rebuildRegistry();
  }

  // Auto-provide the AUTHORING capability (the workspace's author/test/inspect
  // operations as one service) unless the consumer injected their own — same
  // spirit as http/secrets/artifacts above, added here because it closes over
  // the registry state initialized just before. This is what the meta/* lib
  // steps reach via ctx.services.authoring, letting an in-workflow agent
  // (agentTools: ["meta/*"]) author and evaluate candidate workflows
  // (EVOLVE_SPEC §5). Everything it publishes is stamped publisher "ai", and
  // its publish/run/run-history operations are closed over that stamped set.
  if (!(services as Record<string, unknown>)["authoring"]) {
    (services as Record<string, unknown>)["authoring"] = buildAuthoringCapability({
      workspace,
      dataDir,
      store,
      services,
      trackRun,
      publishingEnabled: !registryWasInjected,
      getRegistry: async () => {
        await rebuildRegistry();
        return registry;
      },
      ...(secretsInjected ? {} : { secrets: secretStore }),
    });
  }

  const app = new Hono();
  app.use(logger());

  // ── Workflows ────────────────────────────────────────────────────────────

  app.get("/workflows", async (c) => {
    // The workspace lists what it stores; the run store decorates with each
    // workflow's newest run time. Composed here so neither layer reads the
    // other's records.
    const workflows = await workspace.listWorkflows();
    const decorated = await Promise.all(
      workflows.map(async (w) => {
        const lastRunAt = await store.lastRunAt(w.name);
        return lastRunAt != null ? { ...w, lastRunAt } : w;
      }),
    );
    return c.json(decorated);
  });

  app.post("/workflows", async (c) => {
    const body = await c.req.json<{
      name: string;
      steps?: any[];
      params?: Record<string, unknown>;
      yaml?: string;
      description?: string;
      category?: string;
    }>();

    if (!body.name) return c.json({ error: "name is required" }, 400);

    let result;
    if (body.yaml) {
      result = await workspace.createWorkflow(body.name, body.yaml, body.description, body.category);
    } else if (body.steps) {
      result = await workspace.createWorkflow(
        body.name,
        { steps: body.steps, ...(body.params != null ? { params: body.params } : {}) },
        body.description,
        body.category,
      );
    } else {
      return c.json({ error: "either steps or yaml is required" }, 400);
    }

    await rebuildRegistry();

    return c.json(
      {
        ok: true,
        workflow: result.name,
        version: result.version,
        active: result.version,
        renamed: result.name !== body.name,
        requested: body.name,
      },
      201,
    );
  });

  app.get("/workflows/:name", async (c) => {
    const name = c.req.param("name");
    const meta = await workspace.getWorkflowMetadata(name);
    if (!meta) return c.json({ error: `Workflow "${name}" not found` }, 404);
    return c.json(meta);
  });

  app.get("/workflows/:name/runs", async (c) => {
    const name = c.req.param("name");
    const runIds = await store.listRuns(name);
    const runs = [];
    for (const runId of runIds) {
      const summary = await store.getRunSummary(name, runId);
      if (summary) {
        runs.push(summary);
      } else {
        runs.push({ runId, workflow: name, status: liveStatus(name, runId) });
      }
    }
    return c.json(runs);
  });

  app.get("/workflows/:name/runs/:runId", async (c) => {
    const { name, runId } = c.req.param();
    const summary = await store.getRunSummary(name, runId);
    if (summary) return c.json(summary);
    // No run.json — in-flight, or orphaned before finalize (crash/restart).
    // The event log is durable per-step, so serve a summary reconstructed
    // from it (`partial: true` is the discriminator) instead of a 404: a
    // 17-hour run that dies mid-generation must not cost its whole report.
    const events = await store.getRunEvents(name, runId);
    const partial = summarizeFromEvents(name, runId, events, liveStatus(name, runId));
    if (!partial) {
      return c.json({ error: `Run "${runId}" not found for workflow "${name}"` }, 404);
    }
    return c.json(partial);
  });

  app.get("/workflows/:name/runs/:runId/events", async (c) => {
    const { name, runId } = c.req.param();
    const events = await store.getRunEvents(name, runId);
    return c.json(events);
  });

  // Reattach to a run (live or completed) — SSE tail of its event log.
  // Replays history from the start of the file, then follows appends until
  // the terminal event, then sends a final `done` carrying the RunResult.
  // One path serves in-flight and finished runs (see `RunStore.tailEvents`).
  app.get("/workflows/:name/runs/:runId/stream", async (c) => {
    const { name, runId } = c.req.param();
    return streamSSE(c, async (stream) => {
      const ac = new AbortController();
      stream.onAbort(() => ac.abort());
      for await (const event of store.tailEvents(name, runId, {
        signal: ac.signal,
        // A resumed run appends past its old terminal event — keep following
        // while a live controller exists (§5.2 tail terminality).
        stillLive: () => controllers.has(`${name}/${runId}`),
      })) {
        await stream.writeSSE({ data: JSON.stringify(event) });
      }
      if (ac.signal.aborted) return;
      const summary = await store.getRunSummary(name, runId);
      const result = summary
        ? { runId, status: summary.status, output: summary.output, error: summary.error }
        : { runId, status: liveStatus(name, runId) };
      await stream.writeSSE({ event: "done", data: JSON.stringify(result) });
    });
  });

  // ── Run control (RUN_CONTROL_SPEC §3–§5) ────────────────────────────────
  //
  // Cancel / pause / resume act on the LIVE controller (whole subtree via the
  // effective-state walk); resume additionally covers DURABLE resume (§5):
  // with no live controller, it replays the journal of a "stale" (crashed),
  // `error`, or `cancelled` run — or, with `from`, re-runs a completed run
  // from a chosen step.

  /** The stored workflow version whose content hash is `hash`, or null.
   *  Lets a durable resume run against the exact DAG a run executed when
   *  the active version has since moved on (§5 validity guard). */
  const findWorkflowVersionByHash = async (name: string, hash: string): Promise<string | null> => {
    const meta = await workspace.getWorkflowMetadata(name);
    if (!meta) return null;
    // Prefer the active version when it matches; then newest first so a
    // republished identical version wins over an older twin.
    const labels = Object.keys(meta.versions).sort((a, b) => b.localeCompare(a));
    for (const v of [meta.active, ...labels]) {
      if ((await workspace.getWorkflowHash(name, v)) === hash) return v;
    }
    return null;
  };

  type PreparedResume = {
    ok: true;
    name: string;
    runId: string;
    flow: Flow;
    version?: string;
    journal: Record<string, unknown>;
    runStart: NonNullable<ReturnType<typeof readRunStart>>;
    events: RunEvent[];
  };
  type ResumeRefusal = { ok: false; status: 400 | 404 | 409; error: string };

  /** Everything a durable resume (§5) needs, or a refusal with the HTTP
   *  status it maps to. Shared by the resume endpoint and boot-time
   *  auto-resume (§5.3). Heals a torn log, recovers the run's input/params
   *  from `run.start`, resolves the DAG the run actually executed, and
   *  builds (optionally `from`-invalidated) the journal. */
  const prepareDurableResume = async (
    name: string,
    runId: string,
    body: { from?: string; force?: boolean },
  ): Promise<PreparedResume | ResumeRefusal> => {
    // Crash hardening (§5.1): a kill mid-append leaves a torn final line.
    // Heal it BEFORE reading, or the resumed run's first append would be
    // glued onto the fragment and the whole log would turn unreadable.
    if (await store.repairLog?.(name, runId)) {
      console.warn(`[run ${runId}] repaired a torn event log before resume`);
    }
    const events = await store.getRunEvents(name, runId);
    const runStart = readRunStart(events);
    if (!runStart) {
      return { ok: false, status: 409, error: "Run log has no run.start event — cannot resume" };
    }

    let flow: Flow;
    let version: string | undefined;
    try {
      flow = await workspace.getWorkflow(name);
    } catch (err) {
      return { ok: false, status: 404, error: err instanceof Error ? err.message : String(err) };
    }

    // Validity guard (§5): replaying outputs into a DIFFERENT DAG is
    // undefined. When the active version's hash differs from the one
    // recorded at run.start, resume against the version the run actually
    // executed if it is still on record (a run of a pinned or since-
    // superseded version must not dead-end on "content changed"); refuse
    // only when no stored version matches, unless forced.
    const currentHash = await workspace.getWorkflowHash(name);
    if (runStart.workflowHash && currentHash && runStart.workflowHash !== currentHash) {
      const pinned = await findWorkflowVersionByHash(name, runStart.workflowHash);
      if (pinned) {
        flow = await workspace.getWorkflowVersion(name, pinned);
        version = pinned;
      } else if (!body.force) {
        return {
          ok: false,
          status: 409,
          error:
            `Workflow content changed since this run started (recorded ${runStart.workflowHash}, ` +
            `active ${currentHash}) and no stored version matches — replaying its journal into a ` +
            `different DAG is refused. Pass { force: true } to override.`,
        };
      }
    }
    if (!runStart.workflowHash) {
      console.warn(
        `[run ${runId}] no workflow hash recorded at run.start (pre-run-control log) — resuming without the DAG guard`,
      );
    }

    let journal = buildJournal(events);
    if (body.from) {
      try {
        const inv = await invalidateFrom(journal, body.from, flow, workspace);
        journal = inv.journal;
        for (const w of inv.warnings) console.warn(`[run ${runId}] resume from=${body.from}: ${w}`);
      } catch (err) {
        return { ok: false, status: 400, error: err instanceof Error ? err.message : String(err) };
      }
    }
    return { ok: true, name, runId, flow, version, journal, runStart, events };
  };

  /** Relaunch a prepared durable resume under its ORIGINAL runId (§5). */
  const launchDurableResume = (p: PreparedResume) => {
    launchDetached(
      p.flow,
      {
        input: p.runStart.input,
        runId: p.runId,
        params: p.runStart.params,
        paramOverrides: p.runStart.paramOverrides,
      },
      { journal: p.journal, resume: true, ...(p.version ? { version: p.version } : {}) },
    );
  };

  const isTerminalEvent = (e: RunEvent) =>
    e.type === "run.end" || e.type === "run.error" || e.type === "run.cancelled";

  /** Write the summary a dead run never got to write: its log already ends
   *  in a terminal event (crash between the event append and `finalize`),
   *  or it was cancelling when the process died (finalize as cancelled,
   *  appending the `run.cancelled` the boundary would have emitted). */
  const finalizeDeadRun = async (
    name: string,
    runId: string,
    events: RunEvent[],
    status: "success" | "error" | "cancelled",
  ) => {
    const last = events[events.length - 1]!;
    const start = events.find((e) => e.type === "run.start");
    if (status === "cancelled" && last.type !== "run.cancelled") {
      await store.append(name, runId, { ts: new Date().toISOString(), runId, path: name, type: "run.cancelled" });
    }
    const startedAt = start?.ts ?? events[0]!.ts;
    const finishedAt = last.type === "run.end" || last.type === "run.error" || last.type === "run.cancelled"
      ? last.ts
      : new Date().toISOString();
    await store.finalize(name, runId, {
      runId,
      workflow: name,
      startedAt,
      finishedAt,
      durationMs: Math.max(0, Date.parse(finishedAt) - Date.parse(startedAt)),
      status,
      input: start?.input,
      ...(status === "success" ? { output: last.output } : {}),
      ...(status === "error" && last.error ? { error: last.error } : {}),
    });
  };

  /** Boot-time auto-resume (RUN_CONTROL_SPEC §5.3). Every run that died
   *  with the previous process has a log and no summary. For each
   *  workflow, look only at the NEWEST root run (children are relaunched by
   *  their parent's re-executed step) and:
   *    - log already terminal → just write the missing summary;
   *    - last control marker `run.cancelling` → finalize as cancelled;
   *    - last control marker `run.paused` → leave parked (a human chose that);
   *    - older than `maxAgeMs`, or resumed ≥ `maxResumes` times → skip, loudly;
   *    - otherwise → durable resume, same path as the endpoint. */
  const autoResumeStaleRuns = async (o: AutoResumeOptions = {}): Promise<AutoResumeOutcome[]> => {
    const maxAgeMs = o.maxAgeMs ?? 7 * 24 * 60 * 60 * 1000;
    const maxResumes = o.maxResumes ?? 5;
    const out: AutoResumeOutcome[] = [];
    const report = (workflow: string, runId: string, action: AutoResumeOutcome["action"], reason: string) => {
      out.push({ workflow, runId, action, reason });
      console.warn(`[auto-resume] ${workflow}/${runId}: ${action} — ${reason}`);
    };

    let workflows: Array<{ name: string }>;
    try {
      workflows = await workspace.listWorkflows();
    } catch (err) {
      console.error(`[auto-resume] could not list workflows:`, err);
      return out;
    }

    for (const { name } of workflows) {
      let runIds: string[];
      try {
        runIds = await store.listRuns(name); // newest first
      } catch {
        continue;
      }
      for (const runId of runIds) {
        if (controllers.has(`${name}/${runId}`)) break; // live in this process
        if (await store.getRunSummary(name, runId)) break; // newest finished → nothing was cut off
        await store.repairLog?.(name, runId);
        const events = await store.getRunEvents(name, runId);
        const runStart = readRunStart(events);
        if (!runStart) break; // unreadable / empty — a human's call
        if (runStart.parentRunId) continue; // nested: its parent relaunches it

        const last = events[events.length - 1]!;
        if (isTerminalEvent(last)) {
          const status = last.type === "run.end" ? "success" : last.type === "run.error" ? "error" : "cancelled";
          await finalizeDeadRun(name, runId, events, status);
          report(name, runId, "finalized", `log already ended in ${last.type}; wrote the missing summary`);
          break;
        }
        const marker = [...events].reverse().find(
          (e) => e.type === "run.paused" || e.type === "run.resumed" || e.type === "run.cancelling",
        );
        if (marker?.type === "run.cancelling") {
          await finalizeDeadRun(name, runId, events, "cancelled");
          report(name, runId, "finalized", "was cancelling when the process died; finalized as cancelled");
          break;
        }
        if (marker?.type === "run.paused") {
          report(name, runId, "skipped", "was paused when the process died — resume it by hand");
          break;
        }
        const ageMs = Date.now() - Date.parse(last.ts);
        if (!(ageMs <= maxAgeMs)) {
          report(name, runId, "skipped", `last event ${Math.round(ageMs / 3_600_000)}h ago exceeds the ${Math.round(maxAgeMs / 3_600_000)}h age cap`);
          break;
        }
        const resumes = events.filter((e) => e.type === "run.resumed").length;
        if (resumes >= maxResumes) {
          report(name, runId, "skipped", `already resumed ${resumes} times (cap ${maxResumes}) — a step may be crashing the server; resume by hand`);
          break;
        }
        const prepared = await prepareDurableResume(name, runId, {});
        if (!prepared.ok) {
          report(name, runId, "skipped", prepared.error);
          break;
        }
        if (controllers.has(`${name}/${runId}`)) break;
        launchDurableResume(prepared);
        report(name, runId, "resumed", `replaying ${Object.keys(prepared.journal).length} journaled step(s)${prepared.version ? ` against version ${prepared.version}` : ""}`);
        break;
      }
    }
    return out;
  };

  /** Resolve a control request target: its live controller (if any) and
   *  whether the run exists at all. */
  const findRun = async (name: string, runId: string) => {
    const controller = controllers.get(`${name}/${runId}`) ?? null;
    const summary = await store.getRunSummary(name, runId);
    const events = await store.getRunEvents(name, runId);
    return { controller, summary, exists: controller != null || events.length > 0 };
  };

  app.post("/workflows/:name/runs/:runId/cancel", async (c) => {
    const { name, runId } = c.req.param();
    const { controller, summary, exists } = await findRun(name, runId);
    if (!exists) return c.json({ error: `Run "${runId}" not found` }, 404);
    if (!controller) {
      return c.json(
        { error: summary ? `Run already terminal (${summary.status})` : "Run is not live (stale) — nothing to cancel" },
        409,
      );
    }
    controller.cancel();
    // Marker: if the process dies before the run reaches its boundary and
    // finalizes as `run.cancelled`, boot-time auto-resume must know this run
    // was being cancelled, not cut off (§5.3).
    await appendControlEvent(name, runId, "run.cancelling");
    return c.json({ ok: true, runId, state: controller.state }, 202);
  });

  app.post("/workflows/:name/runs/:runId/pause", async (c) => {
    const { name, runId } = c.req.param();
    const { controller, summary, exists } = await findRun(name, runId);
    if (!exists) return c.json({ error: `Run "${runId}" not found` }, 404);
    if (!controller) {
      return c.json(
        { error: summary ? `Run already terminal (${summary.status})` : "Run is not live (stale) — nothing to pause" },
        409,
      );
    }
    controller.pause();
    await appendControlEvent(name, runId, "run.paused");
    return c.json(
      { ok: true, runId, state: controller.state, quiesced: controller.quiesced() },
      202,
    );
  });

  app.post("/workflows/:name/runs/:runId/resume", async (c) => {
    const { name, runId } = c.req.param();
    const body = await c.req
      .json<{ from?: string; force?: boolean }>()
      .catch(() => ({}) as { from?: string; force?: boolean });

    const { controller, summary, exists } = await findRun(name, runId);

    // Live controller → in-memory resume of a paused run (§4).
    if (controller) {
      if (body.from) {
        return c.json(
          { error: "Run is live — `from` applies to durable resume of a dead run. Pause/cancel it first." },
          409,
        );
      }
      controller.resume();
      await appendControlEvent(name, runId, "run.resumed");
      return c.json({ ok: true, runId, state: controller.state, resumed: "in-memory" }, 202);
    }

    // No controller → durable resume: journal replay (§5).
    if (!exists) return c.json({ error: `Run "${runId}" not found` }, 404);
    if (summary?.status === "success" && !body.from) {
      return c.json(
        { error: "Run completed successfully — nothing to resume (pass `from` to force re-execution from a step)" },
        400,
      );
    }

    const prepared = await prepareDurableResume(name, runId, body);
    if (!prepared.ok) return c.json({ error: prepared.error }, prepared.status);

    // Re-check liveness after the awaits above: two concurrent resume
    // requests must not both launch onto the same log.
    if (controllers.has(`${name}/${runId}`)) {
      return c.json({ error: "Resume already in flight for this run" }, 409);
    }
    const { version, journal } = prepared;
    launchDurableResume(prepared);
    return c.json({
      ok: true,
      runId,
      resumed: "journal",
      replaying: Object.keys(journal).length,
      ...(version ? { version } : {}),
    }, 202);
  });

  // Resolve this workflow's declared `promotes` against a run's OUTPUT — the
  // review surface for "promote a winner". For each spec: the resolved value
  // from the run output (`value`) and the target param's CURRENT default
  // (`current`), so the UI can diff current → value. Pure read; nothing is
  // written until the human POSTs to `/promote`.
  app.get("/workflows/:name/runs/:runId/promotions", async (c) => {
    const { name, runId } = c.req.param();
    let flow: Flow;
    try {
      flow = await workspace.getWorkflow(name);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
    const specs = flow.promotes ?? [];
    if (specs.length === 0) return c.json([]);

    const summary = await store.getRunSummary(name, runId);
    if (!summary) {
      return c.json({ error: `Run "${runId}" not found for workflow "${name}"` }, 404);
    }
    const output = summary.output;

    const promotions = [];
    for (const spec of specs) {
      const target = parsePromoteTarget(spec.to);
      if (!target) continue;
      const value = getByPath(output, spec.from);
      let current: unknown = undefined;
      try {
        current = (await workspace.getWorkflow(target.workflow)).params?.[target.param];
      } catch {
        current = undefined; // target workflow may not exist yet
      }
      promotions.push({
        from: spec.from,
        to: spec.to,
        target,
        label: spec.label ?? spec.to,
        value,
        current,
        resolved: value !== undefined,
      });
    }
    return c.json(promotions);
  });

  // Apply a single declared promotion (human-approved): resolve the run output
  // value for the spec whose `to` === body.to, write it to the target's param
  // default, publish a new version, and rebuild the registry. Returns the new
  // version + before/after.
  app.post("/workflows/:name/runs/:runId/promote", async (c) => {
    const { name, runId } = c.req.param();
    const body = await c.req.json<{ to?: string }>();
    if (!body.to) return c.json({ error: "to is required" }, 400);

    let flow: Flow;
    try {
      flow = await workspace.getWorkflow(name);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
    const spec = (flow.promotes ?? []).find((s) => s.to === body.to);
    if (!spec) {
      return c.json({ error: `No promote spec with to="${body.to}" on workflow "${name}"` }, 404);
    }
    const target = parsePromoteTarget(spec.to);
    if (!target) return c.json({ error: `Invalid promote target "${spec.to}"` }, 400);

    const summary = await store.getRunSummary(name, runId);
    if (!summary) {
      return c.json({ error: `Run "${runId}" not found for workflow "${name}"` }, 404);
    }
    const value = getByPath(summary.output, spec.from);
    if (value === undefined) {
      return c.json({ error: `Promote value at "${spec.from}" is undefined in this run's output` }, 422);
    }

    let result;
    try {
      result = await workspace.setParam(target.workflow, target.param, value);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 400);
    }
    await rebuildRegistry();
    return c.json({
      ok: true,
      workflow: target.workflow,
      param: target.param,
      version: result.version,
      before: result.before,
      after: result.after,
    });
  });

  app.get("/workflows/:name/flow", async (c) => {
    const name = c.req.param("name");
    // ?version= pins a historical version (the UI's version picker);
    // omitted = the active version, as before.
    const version = c.req.query("version");
    try {
      const flow = version
        ? await workspace.getWorkflowVersion(name, version)
        : await workspace.getWorkflow(name);
      return c.json({
        name: flow.name,
        steps: flow.steps,
        ...(flow.params != null ? { params: flow.params } : {}),
        ...(flow.promotes != null ? { promotes: flow.promotes } : {}),
      });
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
  });

  app.get("/workflows/:name/:version", async (c) => {
    const { name, version } = c.req.param();
    try {
      const src = await workspace.getWorkflowSource(name, version);
      return c.text(src, 200, { "Content-Type": "text/yaml" });
    } catch {
      return c.json(
        { error: `Version "${version}" of workflow "${name}" not found` },
        404,
      );
    }
  });

  app.post("/workflows/:name", async (c) => {
    const name = c.req.param("name");
    const body = await c.req.json<{
      version: string;
      steps?: any[];
      params?: Record<string, unknown>;
      yaml?: string;
      description?: string;
    }>();

    if (!body.version) return c.json({ error: "version is required" }, 400);

    if (body.yaml) {
      await workspace.publishWorkflow(name, body.version, body.yaml, body.description);
    } else if (body.steps) {
      await workspace.publishWorkflow(
        name,
        body.version,
        { steps: body.steps, ...(body.params != null ? { params: body.params } : {}) },
        body.description,
      );
    } else {
      return c.json({ error: "either steps or yaml is required" }, 400);
    }

    await rebuildRegistry();

    return c.json({ ok: true, workflow: name, version: body.version, active: body.version }, 201);
  });

  // Set or clear a workflow's grouping category. Metadata-only — no new
  // version is published (unlike POST /workflows/:name).
  app.put("/workflows/:name/category", async (c) => {
    const name = c.req.param("name");
    const body = await c.req.json<{ category?: string | null }>();
    try {
      await workspace.setWorkflowCategory(name, body.category ?? null);
      return c.json({ ok: true, workflow: name, category: body.category ?? null });
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
  });

  app.put("/workflows/:name/active", async (c) => {
    const name = c.req.param("name");
    const body = await c.req.json<{ version: string }>();
    if (!body.version) return c.json({ error: "version is required" }, 400);
    try {
      await workspace.setActiveVersion(name, body.version);
      return c.json({ ok: true, workflow: name, active: body.version });
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
  });

  // ── Artifacts ────────────────────────────────────────────────────────────
  // Files a run wrote via `ctx.services.artifacts` (keyed by runId alone —
  // artifacts are run-scoped, not workflow-scoped). Read-only: steps are the
  // only writers.

  app.get("/artifacts/:runId", async (c) => {
    if (!artifacts) return c.json({ error: "artifacts capability not available" }, 501);
    const runId = c.req.param("runId");
    try {
      return c.json({ runId, files: await artifacts.list(runId) });
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 400);
    }
  });

  app.get("/artifacts/:runId/:path{.+}", async (c) => {
    if (!artifacts) return c.json({ error: "artifacts capability not available" }, 501);
    const runId = c.req.param("runId");
    const relPath = c.req.param("path");
    try {
      const bytes = await artifacts.read(runId, relPath);
      return c.body(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, 200, {
        "content-type": contentTypeFor(relPath),
      });
    } catch (err: any) {
      if (err?.code === "ENOENT") {
        return c.json({ error: `artifact not found: ${relPath}` }, 404);
      }
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 400);
    }
  });

  // ── Secrets ──────────────────────────────────────────────────────────────
  // Deployment-scoped credential store behind `ctx.services.secrets`. Values
  // are write-only over the API: GET returns NAMES + metadata only, never the
  // value. All routes are gated by `VEIN_API_KEY` (permissive in dev mode).
  // 501 when the consumer injected their own `secrets` capability (we don't
  // own that store).

  app.get("/secrets", requireApiKey, async (c) => {
    if (secretsInjected) {
      return c.json({ error: "secrets are managed by an injected capability" }, 501);
    }
    return c.json({ secrets: await secretStore.list() });
  });

  app.put("/secrets/:name", requireApiKey, async (c) => {
    if (secretsInjected) {
      return c.json({ error: "secrets are managed by an injected capability" }, 501);
    }
    const name = c.req.param("name");
    if (!name || !isValidSecretName(name)) {
      return c.json(
        { error: `invalid secret name "${name}" — use letters, digits, underscore (not starting with a digit)` },
        400,
      );
    }
    const body = await c.req.json<{ value?: unknown }>().catch(() => ({ value: undefined }));
    if (typeof body.value !== "string" || body.value.length === 0) {
      return c.json({ error: "value (non-empty string) is required" }, 400);
    }
    await secretStore.set(name, body.value);
    return c.json({ ok: true, name });
  });

  app.delete("/secrets/:name", requireApiKey, async (c) => {
    if (secretsInjected) {
      return c.json({ error: "secrets are managed by an injected capability" }, 501);
    }
    const name = c.req.param("name");
    if (!name) return c.json({ error: "secret name is required" }, 400);
    const existed = await secretStore.delete(name);
    if (!existed) return c.json({ error: `secret "${name}" not found` }, 404);
    return c.json({ ok: true, name });
  });

  // ── Steps ────────────────────────────────────────────────────────────────

  app.get("/steps", async (c) => {
    const allSteps = Object.keys(registry).map((type) => ({
      type,
      source: stepSources[type] ?? "core",
      description: registry[type]?.description,
    }));
    const workspaceSteps = await workspace.listSteps();
    return c.json({ core: allSteps, workspace: workspaceSteps });
  });

  app.get("/steps/:type{.+}/schema", async (c) => {
    const type = c.req.param("type");
    const def = registry[type];
    if (!def) return c.json({ error: `Step type "${type}" not found` }, 404);
    return c.json({ type, fields: zodToFields(def.input) });
  });

  // Source code for a step. In-code steps (injected via createRegistry) carry
  // their source on the def; everything else comes through the workspace
  // boundary (core / lib / custom). `source` is null when none is available.
  app.get("/steps/:type{.+}/source", async (c) => {
    const type = c.req.param("type");
    const def = registry[type];
    if (!def) return c.json({ error: `Step type "${type}" not found` }, 404);
    if (def.source) {
      return c.json({ type, source: def.source, origin: "registry" });
    }
    const found = await workspace.getStepSource(type);
    return c.json({
      type,
      source: found?.code ?? null,
      origin: found?.origin ?? null,
    });
  });

  // List a step's versions + its active version id (parallels workflows).
  app.get("/steps/:type{.+}/versions", async (c) => {
    const type = c.req.param("type");
    try {
      return c.json({ type, ...(await workspace.listStepVersions(type)) });
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
  });

  // Source for a specific archived step version.
  app.get("/steps/:type{.+}/version/:version", async (c) => {
    const { type, version } = c.req.param();
    try {
      const src = await workspace.getStepVersionSource(type, version);
      return c.json({ type, version, source: src });
    } catch {
      return c.json(
        { error: `Version "${version}" of step "${type}" not found` },
        404,
      );
    }
  });

  // Switch a step's active version.
  app.put("/steps/:type{.+}/active", requireApiKey, async (c) => {
    if (registryWasInjected) {
      return c.json(
        { error: "Step versioning is disabled when the registry is provided at construction time" },
        409,
      );
    }
    const type = c.req.param("type");
    const body = await c.req.json<{ version: string }>();
    if (!type) return c.json({ error: "step type is required" }, 400);
    if (!body.version) return c.json({ error: "version is required" }, 400);
    try {
      await workspace.setActiveStepVersion(type, body.version);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
    await rebuildRegistry();
    return c.json({ ok: true, type, active: body.version });
  });

  app.post("/steps", requireApiKey, async (c) => {
    if (registryWasInjected) {
      return c.json(
        { error: "Step publishing is disabled when the registry is provided at construction time" },
        409,
      );
    }
    const body = await c.req.json<{
      name: string;
      code: string;
      description?: string;
      publisher?: string;
    }>();
    if (!body.name || !body.code) return c.json({ error: "name and code are required" }, 400);
    let result: { version: string; changed: boolean };
    try {
      result = await workspace.publishStep(body.name, body.code, body.description, body.publisher);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 400);
    }
    if (result.changed) await rebuildRegistry();
    return c.json({ ok: true, type: body.name, version: result.version, changed: result.changed }, 201);
  });

  // Run a SINGLE step in isolation (synchronous) — the adapter author's inner
  // loop. Body: { config?, input?, params?, cassette?: "record"|"replay",
  // cassetteName? }. With `cassette`, external `ctx.services` calls are recorded
  // to / replayed from `steps/_cassettes/<name>.json` (secrets scrubbed), so the
  // step can be iterated offline. Returns { status, output?, error?, events,
  // recorded? }. Unlike workflow runs, this awaits and returns the result.
  app.post("/steps/:type{.+}/run", async (c) => {
    const type = c.req.param("type");
    if (!registry[type]) return c.json({ error: `Step type "${type}" not found` }, 404);
    const body = await c.req
      .json<{
        config?: Record<string, unknown>;
        input?: unknown;
        params?: Record<string, unknown>;
        cassette?: CassetteMode;
        cassetteName?: string;
      }>()
      .catch(() => ({}) as Record<string, never>);

    const mode = body.cassette;
    if (mode && mode !== "record" && mode !== "replay") {
      return c.json({ error: `cassette must be "record" or "replay"` }, 400);
    }

    const result = await runSingleStep(type, registry, services, {
      config: body.config,
      input: body.input,
      params: body.params,
      workspace,
      ...(mode
        ? { cassette: { mode, path: cassettePath(dataDir, body.cassetteName ?? type) } }
        : {}),
    });
    return c.json(result);
  });

  app.delete("/steps", requireApiKey, async (c) => {
    const publisher = c.req.query("publisher");
    if (!publisher) return c.json({ error: "publisher query parameter is required" }, 400);
    const deleted = await workspace.deleteStepsByPublisher(publisher);
    if (deleted.length > 0) await rebuildRegistry();
    return c.json({ ok: true, deleted });
  });

  app.delete("/steps/:name{.+}", requireApiKey, async (c) => {
    const name = c.req.param("name");
    if (!name) return c.json({ error: "step name is required" }, 400);
    let removed: boolean;
    try {
      removed = await workspace.deleteStep(name);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 400);
    }
    if (!removed) return c.json({ error: `Step "${name}" not found` }, 404);
    await rebuildRegistry();
    return c.json({ ok: true, type: name });
  });

  // ── Run workflows ────────────────────────────────────────────────────────

  interface RunBody {
    input?: unknown;
    params?: Record<string, unknown>;
    paramOverrides?: Record<string, Record<string, unknown>>;
    runId?: string;
  }

  /**
   * Launch a run **detached** (§8): kick off `runWorkflow` without awaiting it
   * in the request and return the `runId` immediately. The run's liveness is
   * decoupled from the connection — it executes server-side and persists every
   * event to the store's append-only log, which `GET …/runs/:runId/stream`
   * tails for live or after-the-fact viewing. `runWorkflow` finalizes its own
   * errors into the log; the `.catch` is only a safety net for unexpected
   * throws (e.g. a store write failure) so they don't become unhandled
   * rejections.
   */
  function launchDetached(
    flow: Flow,
    body: RunBody,
    extra?: { journal?: Record<string, unknown>; resume?: boolean; version?: string },
  ): string {
    const runId = body.runId ?? generateRunId();
    const { controller, untrack } = trackRun(flow.name, runId);
    void (async () => {
      const workflowHash =
        (await workspace.getWorkflowHash(flow.name, extra?.version)) ?? undefined;
      return runWorkflow(flow, body.input ?? {}, registry, {
        runId,
        store,
        workspace,
        services,
        params: body.params,
        paramOverrides: body.paramOverrides,
        controller,
        ...(workflowHash ? { workflowHash } : {}),
        ...(extra?.journal ? { journal: extra.journal } : {}),
        ...(extra?.resume ? { resume: true } : {}),
      });
    })()
      .catch((err) => {
        console.error(`[run ${runId}] launch failed:`, err);
      })
      .finally(untrack);
    return runId;
  }

  app.post("/workflows/:name/run", async (c) => {
    const name = c.req.param("name");
    const body = await c.req.json<RunBody>();
    let flow;
    try {
      flow = await workspace.getWorkflow(name);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
    const runId = launchDetached(flow, body);
    return c.json({ runId }, 202);
  });

  app.post("/workflows/:name/:version/run", async (c) => {
    const { name, version } = c.req.param();
    const body = await c.req.json<RunBody>();
    let flow;
    try {
      flow = await workspace.getWorkflowVersion(name, version);
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : String(err) }, 404);
    }
    const runId = launchDetached(flow, body, { version });
    return c.json({ runId }, 202);
  });

  // ── Chat (AI workflow builder) ───────────────────────────────────────────
  //
  // A chat is a DETACHED background job (EVAL_SPEC §8), mirroring workflow
  // runs: `POST /chat` launches a turn server-side and returns immediately;
  // the turn keeps running (and persists to `messages.jsonl` + `events.jsonl`)
  // regardless of the connection. Watch it — live or after the fact — by
  // tailing `GET /chat/:id/stream`. Close the browser and the agent keeps
  // working; reopen and reattach. See `chat-store.ts`.

  if (enableChat) {
    // Wakes the chat when a detached `run_workflow` run settles: queues
    // while a turn is live (drained into one wake-up turn), appends the
    // `[run-notification]` message, and launches the next turn through the
    // same launchChatTurn path a human message uses. `meta.autoTurns`
    // (reset by POST /chat) caps consecutive machine-triggered turns.
    const notifier = createChatNotifier({
      chatStore,
      maxAutoTurns: chatMaxAutoTurns,
      startTurn: (chatId, turn, modelMessages) =>
        launchChatTurn(chatId, turn, modelMessages),
    });

    /**
     * Run one chat turn detached: build the agent, stream it server-side,
     * persist each fine-grained part to `events.jsonl`, then append the new
     * conversation messages to `messages.jsonl` and mark the turn terminal.
     * Not awaited by the request (`launchChatTurn` returns void) — its
     * liveness is decoupled from any connection, exactly like `launchDetached`.
     */
    function launchChatTurn(chatId: string, turn: number, modelMessages: any[]): void {
      void (async () => {
        // Synchronous (before any await): run-notifications arriving during
        // this turn must queue rather than launching a concurrent turn.
        notifier.turnStarted(chatId);
        const emit = (e: Partial<ChatEvent> & { type: ChatEvent["type"] }) =>
          chatStore.appendEvent(chatId, {
            ts: new Date().toISOString(),
            chatId,
            turn,
            ...e,
          });

        try {
          const { ToolLoopAgent, stepCountIs } = await import("ai");
          const { anthropic } = await import("@ai-sdk/anthropic");
          const { buildTools, buildSystem } = await import("./ai/index.js");

          const deps = {
            workspace,
            dataDir,
            registry,
            store,
            services,
            secrets: secretsInjected ? undefined : secretStore,
            // Build-time bash for the chat builder, cwd'd at the local data
            // dir (scrubbed env — see shell.ts).
            shell: { cwd: dataDir },
            webSearch: true,
            // Read-only graph_query, when the host wired a graph backend.
            graph: opts.graph,
            publishingEnabled: !registryWasInjected,
            getRegistry: async () => {
              if (registryWasInjected) return registry;
              const bundle = await buildRegistry(await workspace.materializeCustomSteps());
              registry = bundle.registry;
              stepSources = bundle.sources;
              return bundle.registry;
            },
            // Controller registration for chat-launched runs — cancellable/
            // pausable like any other launch path (tracked from launch inside
            // the run_workflow tool, not only on detach).
            trackRun,
            // Dispatch-mode run_workflow: a run that outlives the wait window
            // converts to detached — wake this chat with a [run-notification]
            // when it settles.
            detach: {
              waitMs: chatRunWaitMs,
              onDetach: ({
                workflow,
                runId,
                startedAt,
                promise,
              }: {
                workflow: string;
                runId: string;
                startedAt: number;
                promise: Promise<RunResult>;
              }) => {
                promise
                  .then(
                    (res) =>
                      notifier.deliver(
                        chatId,
                        formatRunNotification({
                          workflow,
                          runId,
                          status: res.status,
                          durationMs: Date.now() - startedAt,
                          output: res.output,
                          ...(res.error ? { error: res.error } : {}),
                        }),
                      ),
                    // runWorkflow finalizes its own errors into a resolved
                    // result; a rejection here is an unexpected throw (e.g.
                    // store write failure) — still wake the chat with it.
                    (err) =>
                      notifier.deliver(
                        chatId,
                        formatRunNotification({
                          workflow,
                          runId,
                          status: "error",
                          durationMs: Date.now() - startedAt,
                          error: {
                            message: err instanceof Error ? err.message : String(err),
                          },
                        }),
                      ),
                  )
                  .catch((err) =>
                    console.error(`[chat ${chatId}] run-notification delivery failed:`, err),
                  );
              },
            },
          };

          const agent = new ToolLoopAgent({
            model: anthropic(chatModel),
            instructions: await buildSystem(deps),
            tools: buildTools(deps),
            maxOutputTokens: chatMaxOutputTokens,
            stopWhen: stepCountIs(chatMaxSteps),
            onFinish: () => {
              registry = deps.registry;
            },
          });

          console.log(`[chat ${chatId}] turn ${turn} start (${modelMessages.length} msgs)`);

          const result = await agent.stream({
            messages: modelMessages,
            onStepFinish: (step) => {
              const u = step.usage;
              console.log(
                `[chat ${chatId}] turn ${turn} step ${step.stepNumber} finish=${step.finishReason} tokens=in:${u?.inputTokens ?? "?"}/out:${u?.outputTokens ?? "?"}`,
              );
              // finish=length is a TRUNCATED generation: a cut-off tool call
              // never executes and the turn dies silently. Say why, loudly.
              if (step.finishReason === "length") {
                console.warn(
                  `[chat ${chatId}] turn ${turn} step ${step.stepNumber} TRUNCATED at maxOutputTokens=${chatMaxOutputTokens} — a cut-off tool call never executed; the turn likely ended incomplete. Raise VEIN_MAX_OUTPUT_TOKENS if this recurs.`,
                );
              }
            },
          });

          for await (const part of result.fullStream) {
            switch (part.type) {
              case "text-delta":
                if (part.text) await emit({ type: "text-delta", delta: part.text });
                break;
              case "tool-call":
                await emit({
                  type: "tool-input",
                  toolName: part.toolName,
                  toolCallId: part.toolCallId,
                  input: part.input,
                });
                break;
              case "tool-result":
                await emit({
                  type: "tool-output",
                  toolName: part.toolName,
                  toolCallId: part.toolCallId,
                  output: part.output,
                });
                break;
              case "tool-error":
                await emit({
                  type: "tool-output",
                  toolName: part.toolName,
                  toolCallId: part.toolCallId,
                  output: part.error instanceof Error ? part.error.message : String(part.error),
                  isError: true,
                });
                break;
              case "finish-step":
                await emit({ type: "step.finish" });
                break;
              case "error":
                throw part.error;
            }
          }

          const resp = await result.response;
          await chatStore.appendMessages(chatId, resp.messages as any);
          await emit({ type: "chat.end" });
          await chatStore.setMeta(chatId, { status: "done" });
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`[chat ${chatId}] turn ${turn} failed:`, err);
          await emit({ type: "chat.error", error: { message } });
          await chatStore.setMeta(chatId, { status: "error" });
        } finally {
          // Drain run-notifications that queued during this turn — delivers
          // them all in ONE follow-up turn (launched via startTurn above).
          try {
            await notifier.turnEnded(chatId);
          } catch (err) {
            console.error(`[chat ${chatId}] notification drain failed:`, err);
          }
        }
      })();
    }

    /**
     * A turn that was live when this process died never wrote its
     * `chat.end`/`chat.error` — `meta.status` stays `"live"` on disk and any
     * tail of that turn waits forever (which is what pins the UI: the client
     * treats a live chat as loading until its stream ends). Liveness is only
     * knowable in-process (`notifier.isLive`), so reconcile lazily on every
     * read: no running turn → emit the missing terminal and mark the chat
     * `"error"`. Returns the (possibly patched) meta.
     */
    async function reconcileStaleChat(meta: ChatMeta): Promise<ChatMeta> {
      if (meta.status !== "live" || meta.currentTurn < 0 || notifier.isLive(meta.id)) {
        return meta;
      }
      console.warn(
        `[chat ${meta.id}] turn ${meta.currentTurn} marked live but no turn is running — interrupted by a restart; marking error.`,
      );
      await chatStore.appendEvent(meta.id, {
        ts: new Date().toISOString(),
        chatId: meta.id,
        turn: meta.currentTurn,
        type: "chat.error",
        error: { message: "Turn interrupted: the server restarted mid-turn." },
      });
      return (await chatStore.setMeta(meta.id, { status: "error" })) ?? meta;
    }

    // Send a message: append it, launch the turn detached, return ids (202).
    app.post("/chat", async (c) => {
      const body = await c.req.json<{ chatId?: string; message?: string; title?: string }>();
      if (!body.message || typeof body.message !== "string") {
        return c.json({ error: "message (string) is required" }, 400);
      }

      let chatId = body.chatId;
      let meta = chatId ? await chatStore.getMeta(chatId) : null;
      if (chatId && !meta) {
        return c.json({ error: `Chat "${chatId}" not found` }, 404);
      }
      if (meta) {
        // One turn per chat at a time — two agents appending to the same
        // transcript would interleave. (Different chats run concurrently.)
        if (notifier.isLive(meta.id)) {
          return c.json({ error: `Chat "${meta.id}" has a turn in progress` }, 409);
        }
        meta = await reconcileStaleChat(meta);
      }
      if (!chatId) {
        chatId = generateChatId();
        meta = await chatStore.createChat({
          id: chatId,
          title: body.title ?? body.message.slice(0, 80),
          model: chatModel,
        });
      }

      const prior = await chatStore.loadMessages(chatId);
      const userMsg = { role: "user", content: body.message };
      await chatStore.appendMessages(chatId, [userMsg]);

      const turn = (meta!.currentTurn ?? -1) + 1;
      // A human message resets the consecutive-auto-turn counter (the
      // notification runaway guard) — see `ai/notifier.ts`.
      await chatStore.setMeta(chatId, { status: "live", currentTurn: turn, autoTurns: 0 });

      // Lossless on disk (transcript); truncated copy re-fed to the model.
      const modelMessages = truncateToolMessages([...prior, userMsg]);
      launchChatTurn(chatId, turn, modelMessages);

      return c.json({ chatId, turn }, 202);
    });

    // List chat sessions (newest first).
    app.get("/chats", async (c) => {
      const list = await chatStore.listChats();
      return c.json(await Promise.all(list.map(reconcileStaleChat)));
    });

    // Reattach to a chat turn's event stream (live or completed) — SSE tail.
    // Defaults to the latest turn; pass ?turn=N to follow a specific one.
    app.get("/chat/:chatId/stream", async (c) => {
      const chatId = c.req.param("chatId");
      const stored = await chatStore.getMeta(chatId);
      if (!stored) return c.json({ error: `Chat "${chatId}" not found` }, 404);
      const meta = await reconcileStaleChat(stored);

      const turnParam = c.req.query("turn");
      const turn = turnParam != null ? Number(turnParam) : meta.currentTurn;

      return streamSSE(c, async (stream) => {
        // No such turn yet — nothing to tail; report current status.
        if (!(turn >= 0) || turn > meta.currentTurn) {
          await stream.writeSSE({
            event: "done",
            data: JSON.stringify({ chatId, turn, status: meta.status }),
          });
          return;
        }
        const ac = new AbortController();
        stream.onAbort(() => ac.abort());
        for await (const event of chatStore.tailEvents(chatId, turn, { signal: ac.signal })) {
          await stream.writeSSE({ data: JSON.stringify(event) });
        }
        if (ac.signal.aborted) return;
        const fresh = await chatStore.getMeta(chatId);
        await stream.writeSSE({
          event: "done",
          data: JSON.stringify({ chatId, turn, status: fresh?.status ?? "done" }),
        });
      });
    });

    // Full chat transcript + meta (for reload / reattach).
    app.get("/chat/:chatId", async (c) => {
      const chatId = c.req.param("chatId");
      let meta = await chatStore.getMeta(chatId);
      if (!meta) return c.json({ error: `Chat "${chatId}" not found` }, 404);
      meta = await reconcileStaleChat(meta);
      const messages = await chatStore.loadMessages(chatId);
      return c.json({ meta, messages });
    });
  }

  // ── Health ───────────────────────────────────────────────────────────────

  app.get("/health", (c) => {
    return c.json({
      ok: true,
      dataDir,
      stepCount: Object.keys(registry).length,
    });
  });

  // ── Static files (web UI) ────────────────────────────────────────────────

  if (serveUi) {
    app.use("/assets/*", serveStatic({ root: webDist }));

    app.get("*", async (c) => {
      const path = c.req.path;
      if (
        path.startsWith("/workflows") ||
        path.startsWith("/steps") ||
        path.startsWith("/chat") ||
        path.startsWith("/health")
      ) {
        return c.notFound();
      }
      try {
        const html = await readFile(join(webDist, "index.html"), "utf-8");
        return c.html(html);
      } catch {
        return c.text("UI not built. Run: cd web && npm run build", 404);
      }
    });
  }

  // ── Programmatic run helper ──────────────────────────────────────────────

  async function run(
    workflow: string | Flow,
    input: unknown = {},
    runOpts?: VeinRunOptions<TServices>,
  ): Promise<RunResult> {
    const flow =
      typeof workflow === "string"
        ? runOpts?.version
          ? await workspace.getWorkflowVersion(workflow, runOpts.version)
          : await workspace.getWorkflow(workflow)
        : workflow;

    // Generate the runId here (rather than letting runWorkflow default it) so
    // the run can be registered as in-flight — otherwise nested runs (e.g. the
    // optimizer capability's generation runs) list as "stale" while running.
    const runId = runOpts?.runId ?? generateRunId();
    // Tree linkage (RUN_CONTROL_SPEC §2.2): `parentRunId` (the calling
    // step's ctx.runId — set by e.g. the lab's optimizer capability) attaches
    // this run's controller under the launching run's.
    const { controller, untrack } = trackRun(flow.name, runId, runOpts?.parentRunId);
    try {
      const workflowHash =
        typeof workflow === "string"
          ? ((await workspace.getWorkflowHash(workflow, runOpts?.version)) ?? undefined)
          : undefined;
      return await runWorkflow(flow, input, registry, {
        runId,
        store,
        workspace,
        services: runOpts?.services ?? services,
        params: runOpts?.params,
        paramOverrides: runOpts?.paramOverrides,
        onEvent: runOpts?.onEvent,
        controller,
        ...(workflowHash ? { workflowHash } : {}),
      });
    } finally {
      untrack();
    }
  }

  // ── Listener ─────────────────────────────────────────────────────────────

  async function listen(port?: number): Promise<number> {
    warnIfUnconfigured();
    const p = port ?? parseInt(process.env["VEIN_PORT"] ?? "3000", 10);
    console.log(
      fileBacked
        ? `vein workspace: ${dataDir}`
        : `vein workspace: ${workspace.constructor.name} (data dir: ${dataDir})`,
    );
    console.log(`vein steps: ${Object.keys(registry).length} registered`);
    console.log(`vein server: http://localhost:${p}`);
    serve({ fetch: app.fetch, port: p });
    return p;
  }

  // Boot-time auto-resume (§5.3): on by default for a file-backed store
  // (an in-memory store cannot hold a cut-off run), off with
  // `VEIN_AUTO_RESUME=0` or `autoResume: false`. Deferred and unref'd so a
  // host that constructs vein and exits (tests, CLIs) is never held open.
  const autoResumeEnabled =
    opts.autoResume === undefined
      ? fileBacked && process.env["VEIN_AUTO_RESUME"] !== "0"
      : opts.autoResume !== false;
  if (autoResumeEnabled) {
    const o = typeof opts.autoResume === "object" ? opts.autoResume : {};
    const timer = setTimeout(() => {
      void autoResumeStaleRuns(o).catch((err) => console.error(`[auto-resume] scan failed:`, err));
    }, o.delayMs ?? 3000);
    timer.unref?.();
  }

  return {
    app,
    workspace,
    dataDir,
    store,
    secretStore,
    services,
    getRegistry: () => registry,
    rebuildRegistry,
    autoResumeStaleRuns,
    run,
    listen,
  };
}

/** Minimal content-type map for serving artifacts; octet-stream otherwise. */
function contentTypeFor(path: string): string {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  const map: Record<string, string> = {
    json: "application/json",
    txt: "text/plain; charset=utf-8",
    md: "text/markdown; charset=utf-8",
    csv: "text/csv; charset=utf-8",
    html: "text/html; charset=utf-8",
    yaml: "text/yaml; charset=utf-8",
    yml: "text/yaml; charset=utf-8",
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    svg: "image/svg+xml",
    pdf: "application/pdf",
  };
  return map[ext] ?? "application/octet-stream";
}
