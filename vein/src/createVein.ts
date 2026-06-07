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
import { FileRunStore, generateRunId } from "./store.js";
import type { ChatStore, ChatEvent } from "./chat-store.js";
import {
  FileChatStore,
  MemoryChatStore,
  generateChatId,
  truncateToolMessages,
} from "./chat-store.js";
import { WorkspaceManager } from "./workspace.js";
import { buildRegistry, readStepSourceFromDisk } from "./steps/registry.js";
import type { StepSources } from "./steps/registry.js";
import { runWorkflow } from "./runner.js";
import { requireApiKey, warnIfUnconfigured } from "./auth.js";
import { standardServices } from "./capabilities.js";
import { runSingleStep, cassettePath } from "./run-step.js";
import type { CassetteMode } from "./cassette.js";

// ── Public types ───────────────────────────────────────────────────────────

/**
 * Options for constructing a Vein instance. Everything is optional — pass
 * nothing for the default "filesystem-backed server" behavior, or supply
 * any subset to embed vein in your own app.
 */
export interface VeinOptions<TServices = unknown> {
  /** Persistent workspace for workflows/runs. Defaults to a new
   *  `WorkspaceManager()` (reads `VEIN_WORKSPACE` env, falls back to
   *  `./workspace`). */
  workspace?: WorkspaceManager;

  /** Step registry. If supplied, used as-is and `rebuildRegistry` becomes
   *  a no-op (the consumer owns step composition). If omitted, vein
   *  discovers steps from disk via `buildRegistry(workspace.path)`. */
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

  /** Max agent steps (tool-call iterations) per chat turn. Raise for longer
   *  autonomous "let it rip" loops. Defaults to `VEIN_CHAT_MAX_STEPS` or 30. */
  chatMaxSteps?: number;

  /** Anthropic model id for the chat agent. Defaults to `VEIN_CHAT_MODEL` or
   *  `claude-sonnet-4-20250514`. */
  chatModel?: string;

  /** Directory containing the built web UI (the `dist` folder). Defaults
   *  to vein's own bundled UI resolved relative to this module, so it
   *  works regardless of the host process's CWD. The built UI uses
   *  relative asset paths, so it can be mounted at any sub-path (e.g.
   *  `/lab`) as long as the host serves it with a trailing slash. */
  webDist?: string;
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
  workspace: WorkspaceManager;
  store: RunStore;
  services: TServices;

  /** Current registry. Reads through the closure so callers always see
   *  the latest after `rebuildRegistry()`. */
  getRegistry: () => StepRegistry;

  /** Re-scan the workspace for newly-published custom steps. No-op when
   *  the instance was constructed with an explicit `registry`. */
  rebuildRegistry: () => Promise<void>;

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

function getObjectShape(s: z.ZodTypeAny): Record<string, z.ZodTypeAny> | null {
  const def = s._def;
  if (def.typeName === "ZodObject") return (def as any).shape();
  if (def.typeName === "ZodEffects") return getObjectShape(def.schema);
  return null;
}

function describeField(name: string, s: z.ZodTypeAny): FieldDesc {
  let required = true;
  let defaultVal: unknown = undefined;
  let inner = s;

  for (;;) {
    const def = inner._def;
    if (def.typeName === "ZodOptional") {
      required = false;
      inner = def.innerType;
    } else if (def.typeName === "ZodDefault") {
      required = false;
      defaultVal = def.defaultValue();
      inner = def.innerType;
    } else if (def.typeName === "ZodNullable") {
      required = false;
      inner = def.innerType;
    } else {
      break;
    }
  }

  const typeName = inner._def.typeName as string;

  if (typeName === "ZodEnum") {
    return { name, kind: "enum", required, default: defaultVal, enumValues: inner._def.values };
  }
  if (typeName === "ZodString") return { name, kind: "string", required, default: defaultVal };
  if (typeName === "ZodNumber") return { name, kind: "number", required, default: defaultVal };
  if (typeName === "ZodBoolean") return { name, kind: "boolean", required, default: defaultVal };
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
  const workspace = opts.workspace ?? new WorkspaceManager();
  const store = opts.store ?? new FileRunStore(workspace.path);
  // Run IDs currently executing **in this process** (keyed `${workflow}/${runId}`).
  // Detached execution is in-memory, so a run with no `run.json` summary is only
  // genuinely "running" if it's in this set; otherwise it never finalized (server
  // restart / crash / cancellation) and is reported as "stale" rather than a
  // perpetual "running".
  const activeRuns = new Set<string>();
  // Auto-provide the standard capabilities (http + secrets) every adapter step
  // builds on, with the consumer's bag spread on top so they can override or
  // extend any capability. This is what lets an LLM-authored adapter rely on
  // `ctx.services.http` / `ctx.services.secrets` existing out of the box.
  const services = {
    ...(standardServices() as unknown as Record<string, unknown>),
    ...((opts.services ?? {}) as Record<string, unknown>),
  } as TServices;
  const serveUi = opts.serveUi ?? true;
  const enableChat = opts.enableChat ?? true;
  const chatStore: ChatStore =
    opts.chatStore ??
    (store instanceof FileRunStore
      ? new FileChatStore(workspace.path)
      : new MemoryChatStore());
  const chatMaxSteps =
    opts.chatMaxSteps ?? Number(process.env["VEIN_CHAT_MAX_STEPS"] ?? 30);
  const chatModel =
    opts.chatModel ?? process.env["VEIN_CHAT_MODEL"] ?? "claude-sonnet-4-20250514";
  const webDist =
    opts.webDist ??
    resolve(dirname(fileURLToPath(import.meta.url)), "../web/dist");
  const registryWasInjected = opts.registry !== undefined;

  // Mutable closure state — `app` handlers read through these.
  let registry: StepRegistry = opts.registry ?? {};
  let stepSources: StepSources = {};

  async function rebuildRegistry(): Promise<void> {
    if (registryWasInjected) return; // consumer owns the registry
    const bundle = await buildRegistry(workspace.path);
    registry = bundle.registry;
    stepSources = bundle.sources;
  }

  if (!registryWasInjected) {
    await rebuildRegistry();
  }

  const app = new Hono();
  app.use(logger());

  // ── Workflows ────────────────────────────────────────────────────────────

  app.get("/workflows", async (c) => {
    const workflows = await workspace.listWorkflows();
    return c.json(workflows);
  });

  app.post("/workflows", async (c) => {
    const body = await c.req.json<{
      name: string;
      steps?: any[];
      params?: Record<string, unknown>;
      yaml?: string;
      description?: string;
    }>();

    if (!body.name) return c.json({ error: "name is required" }, 400);

    let result;
    if (body.yaml) {
      result = await workspace.createWorkflow(body.name, body.yaml, body.description);
    } else if (body.steps) {
      result = await workspace.createWorkflow(
        body.name,
        { steps: body.steps, ...(body.params != null ? { params: body.params } : {}) },
        body.description,
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
    try {
      const meta = await readFile(
        join(workspace.path, "workflows", name, "_metadata.json"),
        "utf-8",
      );
      return c.json(JSON.parse(meta));
    } catch {
      return c.json({ error: `Workflow "${name}" not found` }, 404);
    }
  });

  app.get("/workflows/:name/runs", async (c) => {
    const name = c.req.param("name");
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Run listing requires a FileRunStore" }, 501);
    }
    const runIds = await store.listRuns(name);
    const runs = [];
    for (const runId of runIds) {
      const summary = await store.getRunSummary(name, runId);
      if (summary) {
        runs.push(summary);
      } else {
        const status = activeRuns.has(`${name}/${runId}`) ? "running" : "stale";
        runs.push({ runId, workflow: name, status });
      }
    }
    return c.json(runs);
  });

  app.get("/workflows/:name/runs/:runId", async (c) => {
    const { name, runId } = c.req.param();
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Run lookup requires a FileRunStore" }, 501);
    }
    const summary = await store.getRunSummary(name, runId);
    if (!summary) {
      return c.json({ error: `Run "${runId}" not found for workflow "${name}"` }, 404);
    }
    return c.json(summary);
  });

  app.get("/workflows/:name/runs/:runId/events", async (c) => {
    const { name, runId } = c.req.param();
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Event lookup requires a FileRunStore" }, 501);
    }
    const events = await store.getRunEvents(name, runId);
    return c.json(events);
  });

  // Reattach to a run (live or completed) — SSE tail of its event log.
  // Replays history from the start of the file, then follows appends until
  // the terminal event, then sends a final `done` carrying the RunResult.
  // One path serves in-flight and finished runs (see `FileRunStore.tailEvents`).
  app.get("/workflows/:name/runs/:runId/stream", async (c) => {
    const { name, runId } = c.req.param();
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Run streaming requires a FileRunStore" }, 501);
    }
    return streamSSE(c, async (stream) => {
      const ac = new AbortController();
      stream.onAbort(() => ac.abort());
      for await (const event of store.tailEvents(name, runId, { signal: ac.signal })) {
        await stream.writeSSE({ data: JSON.stringify(event) });
      }
      if (ac.signal.aborted) return;
      const summary = await store.getRunSummary(name, runId);
      const result = summary
        ? { runId, status: summary.status, output: summary.output, error: summary.error }
        : { runId, status: activeRuns.has(`${name}/${runId}`) ? "running" : "stale" };
      await stream.writeSSE({ event: "done", data: JSON.stringify(result) });
    });
  });

  // Resolve this workflow's declared `promotes` against a run's OUTPUT — the
  // review surface for "promote a winner". For each spec: the resolved value
  // from the run output (`value`) and the target param's CURRENT default
  // (`current`), so the UI can diff current → value. Pure read; nothing is
  // written until the human POSTs to `/promote`.
  app.get("/workflows/:name/runs/:runId/promotions", async (c) => {
    const { name, runId } = c.req.param();
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Promotions require a FileRunStore" }, 501);
    }
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
    if (!(store instanceof FileRunStore)) {
      return c.json({ error: "Promotions require a FileRunStore" }, 501);
    }
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
    try {
      const flow = await workspace.getWorkflow(name);
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

  // ── Steps ────────────────────────────────────────────────────────────────

  app.get("/steps", async (c) => {
    const allSteps = Object.keys(registry).map((type) => ({
      type,
      source: stepSources[type] ?? (registryWasInjected ? "core" : "core"),
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
  // their source on the def; everything else is read from disk
  // (core / lib / workspace custom). `source` is null when none is available.
  app.get("/steps/:type{.+}/source", async (c) => {
    const type = c.req.param("type");
    const def = registry[type];
    if (!def) return c.json({ error: `Step type "${type}" not found` }, 404);
    if (def.source) {
      return c.json({ type, source: def.source, origin: "registry" });
    }
    const onDisk = await readStepSourceFromDisk(type, workspace.path);
    return c.json({
      type,
      source: onDisk?.code ?? null,
      origin: onDisk?.origin ?? null,
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
        ? { cassette: { mode, path: cassettePath(workspace.path, body.cassetteName ?? type) } }
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
  function launchDetached(flow: Flow, body: RunBody): string {
    const runId = body.runId ?? generateRunId();
    const key = `${flow.name}/${runId}`;
    activeRuns.add(key);
    void runWorkflow(flow, body.input ?? {}, registry, {
      runId,
      store,
      workspace,
      services,
      params: body.params,
      paramOverrides: body.paramOverrides,
    })
      .catch((err) => {
        console.error(`[run ${runId}] launch failed:`, err);
      })
      .finally(() => {
        activeRuns.delete(key);
      });
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
    const runId = launchDetached(flow, body);
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
    /**
     * Run one chat turn detached: build the agent, stream it server-side,
     * persist each fine-grained part to `events.jsonl`, then append the new
     * conversation messages to `messages.jsonl` and mark the turn terminal.
     * Not awaited by the request (`launchChatTurn` returns void) — its
     * liveness is decoupled from any connection, exactly like `launchDetached`.
     */
    function launchChatTurn(chatId: string, turn: number, modelMessages: any[]): void {
      void (async () => {
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
            registry,
            store,
            services,
            publishingEnabled: !registryWasInjected,
            getRegistry: async () => {
              if (registryWasInjected) return registry;
              const bundle = await buildRegistry(workspace.path);
              registry = bundle.registry;
              stepSources = bundle.sources;
              return bundle.registry;
            },
          };

          const agent = new ToolLoopAgent({
            model: anthropic(chatModel),
            instructions: await buildSystem(deps),
            tools: buildTools(deps),
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
        }
      })();
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
      await chatStore.setMeta(chatId, { status: "live", currentTurn: turn });

      // Lossless on disk (transcript); truncated copy re-fed to the model.
      const modelMessages = truncateToolMessages([...prior, userMsg]);
      launchChatTurn(chatId, turn, modelMessages);

      return c.json({ chatId, turn }, 202);
    });

    // List chat sessions (newest first).
    app.get("/chats", async (c) => {
      return c.json(await chatStore.listChats());
    });

    // Reattach to a chat turn's event stream (live or completed) — SSE tail.
    // Defaults to the latest turn; pass ?turn=N to follow a specific one.
    app.get("/chat/:chatId/stream", async (c) => {
      const chatId = c.req.param("chatId");
      const meta = await chatStore.getMeta(chatId);
      if (!meta) return c.json({ error: `Chat "${chatId}" not found` }, 404);

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
      const meta = await chatStore.getMeta(chatId);
      if (!meta) return c.json({ error: `Chat "${chatId}" not found` }, 404);
      const messages = await chatStore.loadMessages(chatId);
      return c.json({ meta, messages });
    });
  }

  // ── Health ───────────────────────────────────────────────────────────────

  app.get("/health", (c) => {
    return c.json({
      ok: true,
      workspace: workspace.path,
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

    return runWorkflow(flow, input, registry, {
      runId: runOpts?.runId,
      store,
      workspace,
      services: runOpts?.services ?? services,
      params: runOpts?.params,
      paramOverrides: runOpts?.paramOverrides,
      onEvent: runOpts?.onEvent,
    });
  }

  // ── Listener ─────────────────────────────────────────────────────────────

  async function listen(port?: number): Promise<number> {
    warnIfUnconfigured();
    const p = port ?? parseInt(process.env["VEIN_PORT"] ?? "3000", 10);
    console.log(`vein workspace: ${workspace.path}`);
    console.log(`vein steps: ${Object.keys(registry).length} registered`);
    console.log(`vein server: http://localhost:${p}`);
    serve({ fetch: app.fetch, port: p });
    return p;
  }

  return {
    app,
    workspace,
    store,
    services,
    getRegistry: () => registry,
    rebuildRegistry,
    run,
    listen,
  };
}
