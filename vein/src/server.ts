import { Hono } from "hono";
import { logger } from "hono/logger";
import { streamSSE } from "hono/streaming";
import { serveStatic } from "@hono/node-server/serve-static";
import { readFile, readdir, access } from "node:fs/promises";
import { join, resolve } from "node:path";
import { serve } from "@hono/node-server";
import { z } from "zod";
import { WorkspaceManager } from "./workspace.js";
import { FileRunStore } from "./store.js";
import { buildRegistry, CORE_STEP_TYPES } from "./steps/registry.js";
import { runWorkflow } from "./runner.js";
import type { StepRegistry } from "./core.js";
import { buildTools, SYSTEM } from "./ai.js";

// ── Zod → field descriptors ────────────────────────────────────────────────

interface FieldDesc {
  name: string;
  kind: "string" | "number" | "boolean" | "enum" | "json";
  required: boolean;
  default?: unknown;
  enumValues?: string[];
  description?: string;
}

/** Walk a ZodObject and produce a flat field list for the UI. */
function zodToFields(schema: z.ZodTypeAny): FieldDesc[] {
  const shape = getObjectShape(schema);
  if (!shape) return [];

  const fields: FieldDesc[] = [];
  for (const [name, fieldSchema] of Object.entries(shape)) {
    fields.push(describeField(name, fieldSchema as z.ZodTypeAny));
  }
  return fields;
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

  // Unwrap optional / default wrappers
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
  if (typeName === "ZodString") {
    return { name, kind: "string", required, default: defaultVal };
  }
  if (typeName === "ZodNumber") {
    return { name, kind: "number", required, default: defaultVal };
  }
  if (typeName === "ZodBoolean") {
    return { name, kind: "boolean", required, default: defaultVal };
  }

  // Anything complex (ZodAny, ZodRecord, ZodArray, etc.) → json textarea
  return { name, kind: "json", required, default: defaultVal };
}

// ── Bootstrap ──────────────────────────────────────────────────────────────

const workspace = new WorkspaceManager();
const store = new FileRunStore(workspace.path);
let registry: StepRegistry;

const app = new Hono();
app.use(logger());

// ── Workflows ──────────────────────────────────────────────────────────────

/** List all workflows */
app.get("/workflows", async (c) => {
  const workflows = await workspace.listWorkflows();
  return c.json(workflows);
});

/** Get workflow metadata (versions, active) */
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

/** List runs for a workflow */
app.get("/workflows/:name/runs", async (c) => {
  const name = c.req.param("name");
  const runIds = await store.listRuns(name);

  const runs = [];
  for (const runId of runIds) {
    const summary = await store.getRunSummary(name, runId);
    if (summary) {
      runs.push(summary);
    } else {
      runs.push({ runId, workflow: name, status: "running" });
    }
  }

  return c.json(runs);
});

/** Get run summary */
app.get("/workflows/:name/runs/:runId", async (c) => {
  const { name, runId } = c.req.param();
  const summary = await store.getRunSummary(name, runId);
  if (!summary) {
    return c.json({ error: `Run "${runId}" not found for workflow "${name}"` }, 404);
  }
  return c.json(summary);
});

/** Get run events (JSON array) */
app.get("/workflows/:name/runs/:runId/events", async (c) => {
  const { name, runId } = c.req.param();
  const events = await store.getRunEvents(name, runId);
  if (events.length === 0) {
    return c.json({ error: `Events for run "${runId}" not found` }, 404);
  }
  return c.json(events);
});

/** Get the flow structure (parsed JSON) for the active version */
app.get("/workflows/:name/flow", async (c) => {
  const name = c.req.param("name");
  try {
    const flow = await workspace.getWorkflow(name);
    // Serialize to plain JSON (strip Zod schemas)
    return c.json({
      name: flow.name,
      steps: flow.steps,
    });
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : String(err) },
      404,
    );
  }
});

/** Get workflow YAML source for a specific version */
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

/** Publish a new workflow version. Body: { version, steps, description? } or { version, yaml, description? } */
app.post("/workflows/:name", async (c) => {
  const name = c.req.param("name");
  const body = await c.req.json<{
    version: string;
    steps?: any[];
    yaml?: string;
    description?: string;
  }>();

  if (!body.version) {
    return c.json({ error: "version is required" }, 400);
  }

  if (body.yaml) {
    // Raw YAML string
    await workspace.publishWorkflow(name, body.version, body.yaml, body.description);
  } else if (body.steps) {
    // Steps array — workspace will serialize to YAML
    await workspace.publishWorkflow(name, body.version, { steps: body.steps }, body.description);
  } else {
    return c.json({ error: "either steps or yaml is required" }, 400);
  }

  // Rebuild registry to pick up any new step types
  registry = await buildRegistry(workspace.path);

  return c.json({
    ok: true,
    workflow: name,
    version: body.version,
    active: body.version,
  }, 201);
});

/** Set the active version of a workflow */
app.put("/workflows/:name/active", async (c) => {
  const name = c.req.param("name");
  const body = await c.req.json<{ version: string }>();

  if (!body.version) {
    return c.json({ error: "version is required" }, 400);
  }

  try {
    await workspace.setActiveVersion(name, body.version);
    return c.json({ ok: true, workflow: name, active: body.version });
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : String(err) },
      404,
    );
  }
});

// ── Steps ──────────────────────────────────────────────────────────────────

/** List all steps (core + lib + custom) */
app.get("/steps", async (c) => {
  const coreSet = new Set<string>(CORE_STEP_TYPES);
  const allSteps = Object.keys(registry).map((type) => ({
    type,
    source: coreSet.has(type) ? "core" : type.includes("/") ? "lib" : "custom",
  }));

  const workspaceSteps = await workspace.listSteps();

  return c.json({ core: allSteps, workspace: workspaceSteps });
});

/** Get input schema for a step type as a simple field descriptor */
app.get("/steps/:type/schema", async (c) => {
  const type = c.req.param("type");
  const def = registry[type];
  if (!def) {
    return c.json({ error: `Step type "${type}" not found` }, 404);
  }
  const fields = zodToFields(def.input);
  return c.json({ type, fields });
});

/** Publish a new custom step. Lib steps ship with the engine and cannot be created here. */
app.post("/steps", async (c) => {
  const body = await c.req.json<{
    name: string;
    code: string;
    description?: string;
  }>();

  if (!body.name || !body.code) {
    return c.json({ error: "name and code are required" }, 400);
  }

  await workspace.publishStep(body.name, body.code, body.description);

  // Rebuild registry to pick up the new step
  registry = await buildRegistry(workspace.path);

  return c.json({ ok: true, type: body.name }, 201);
});

// ── Run workflows ──────────────────────────────────────────────────────────

/** Run a workflow (active version) — SSE stream of events */
app.post("/workflows/:name/run", async (c) => {
  const name = c.req.param("name");
  const body = await c.req.json<{ input?: unknown; runId?: string }>();

  let flow;
  try {
    flow = await workspace.getWorkflow(name);
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : String(err) },
      404,
    );
  }

  return streamSSE(c, async (stream) => {
    const result = await runWorkflow(flow, body.input ?? {}, registry, {
      runId: body.runId,
      store,
      workspace,
      onEvent: async (event) => {
        await stream.writeSSE({ data: JSON.stringify(event) });
      },
    });
    await stream.writeSSE({ event: "done", data: JSON.stringify(result) });
  });
});

/** Run a specific workflow version — SSE stream of events */
app.post("/workflows/:name/:version/run", async (c) => {
  const { name, version } = c.req.param();
  const body = await c.req.json<{ input?: unknown; runId?: string }>();

  let flow;
  try {
    flow = await workspace.getWorkflowVersion(name, version);
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : String(err) },
      404,
    );
  }

  return streamSSE(c, async (stream) => {
    const result = await runWorkflow(flow, body.input ?? {}, registry, {
      runId: body.runId,
      store,
      workspace,
      onEvent: async (event) => {
        await stream.writeSSE({ data: JSON.stringify(event) });
      },
    });
    await stream.writeSSE({ event: "done", data: JSON.stringify(result) });
  });
});

// ── Chat (AI workflow builder) ─────────────────────────────────────────────

app.post("/chat", async (c) => {
  const body = await c.req.json<{ messages: any[] }>();
  const messages = (body.messages ?? []).map((m: any) => ({
    role: m.role,
    content: m.content,
  }));

  const { ToolLoopAgent, stepCountIs } = await import("ai");
  const { anthropic } = await import("@ai-sdk/anthropic");

  const deps = {
    workspace,
    registry,
    store,
    getRegistry: () => buildRegistry(workspace.path),
  };
  const tools = buildTools(deps);

  const agent = new ToolLoopAgent({
    model: anthropic("claude-sonnet-4-20250514"),
    instructions: SYSTEM,
    tools,
    stopWhen: stepCountIs(10),
    onFinish: () => { registry = deps.registry; },
  });

  const result = await agent.stream({ messages });
  return result.toUIMessageStreamResponse();
});

// ── Health ─────────────────────────────────────────────────────────────────

app.get("/health", (c) => {
  return c.json({
    ok: true,
    workspace: workspace.path,
    stepCount: Object.keys(registry).length,
  });
});

// ── Static files (web UI) ──────────────────────────────────────────────────

// Serve built web UI from web/dist/
app.use("/assets/*", serveStatic({ root: "./web/dist" }));

// SPA fallback: serve index.html for all non-API routes
app.get("*", async (c) => {
  // Don't catch API routes
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
    const html = await readFile(resolve("./web/dist/index.html"), "utf-8");
    return c.html(html);
  } catch {
    return c.text("UI not built. Run: cd web && npm run build", 404);
  }
});

// ── Start ──────────────────────────────────────────────────────────────────

export { app };

export async function startServer(port?: number) {
  registry = await buildRegistry(workspace.path);
  const p = port ?? parseInt(process.env["VEIN_PORT"] ?? "3000", 10);

  console.log(`vein workspace: ${workspace.path}`);
  console.log(`vein steps: ${Object.keys(registry).length} registered`);
  console.log(`vein server: http://localhost:${p}`);

  serve({ fetch: app.fetch, port: p });
}

// Run directly
const isMain =
  process.argv[1]?.endsWith("server.ts") ||
  process.argv[1]?.endsWith("server.js");
if (isMain) {
  startServer();
}
