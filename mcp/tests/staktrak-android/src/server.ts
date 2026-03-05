import express from "express";
import { z, ZodError } from "zod";
import {
  ensureSession,
  getConnectionState,
  getPageSource,
  getSessionMeta,
  pressBack,
  pressHome,
  startSession,
  stopSession,
  swipe,
  takeScreenshot,
  tapByCoordinates,
  tapBySelector,
  typeIntoElement,
} from "./appium";
import { generateAppiumScript, parseActionsFromScript } from "./generator";
import { recorder } from "./recorder";
import { replayActions } from "./replayer";
import { parseAccessibilityTree } from "./tree";
import { AndroidSelector, RecordedAction, ReplayEvent } from "./types";
import { logError, logInfo } from "./utils";

const app = express();
app.use(express.json({ limit: "10mb" }));

const PORT = Number(process.env.PORT || 4724);
const SSE_HEARTBEAT_MS = 15000;
const AUTOSTART_BACKOFF_MS = [1000, 2000, 5000, 10000];

const sseClients = new Set<express.Response>();
const sseHeartbeats = new Map<express.Response, NodeJS.Timeout>();

const autostartState: {
  retrying: boolean;
  attempts: number;
  lastError?: string;
  nextRetryInMs: number;
} = {
  retrying: false,
  attempts: 0,
  nextRetryInMs: 0,
};

let autostartTimer: NodeJS.Timeout | null = null;
let autostartMonitor: NodeJS.Timeout | null = null;

class RequestValidationError extends Error {
  public readonly details: Array<{ path: string; message: string }>;

  constructor(details: Array<{ path: string; message: string }>) {
    super("Invalid request payload");
    this.details = details;
  }
}

const selectorSchema = z
  .object({
    resourceId: z.string().min(1).optional(),
    accessibilityId: z.string().min(1).optional(),
    text: z.string().optional(),
    xpath: z.string().min(1).optional(),
  })
  .strict()
  .refine(
    (value) =>
      value.resourceId !== undefined ||
      value.accessibilityId !== undefined ||
      value.text !== undefined ||
      value.xpath !== undefined,
    { message: "Selector must include at least one field." }
  );

const tapBodySchema = z
  .object({
    selector: selectorSchema.optional(),
    x: z.number().optional(),
    y: z.number().optional(),
  })
  .strict()
  .refine((value) => value.selector !== undefined || (value.x !== undefined && value.y !== undefined), {
    message: "Provide selector or x/y coordinates.",
  });

const typeBodySchema = z
  .object({
    selector: selectorSchema,
    text: z.string(),
    replace: z.boolean().optional(),
  })
  .strict();

const swipeBodySchema = z
  .object({
    startX: z.number(),
    startY: z.number(),
    endX: z.number(),
    endY: z.number(),
    durationMs: z.number().int().positive().optional(),
  })
  .strict();

const sessionStartBodySchema = z
  .object({
    package: z.string().min(1).optional(),
    activity: z.string().min(1).optional(),
    deviceName: z.string().min(1).optional(),
  })
  .strict();

const sessionStopBodySchema = z
  .object({
    teardown: z.boolean().optional(),
  })
  .strict();

const tapActionSchema = z
  .object({
    type: z.literal("tap"),
    timestamp: z.number().optional(),
    selector: selectorSchema.optional(),
    x: z.number().optional(),
    y: z.number().optional(),
  })
  .strict()
  .refine((value) => value.selector !== undefined || (value.x !== undefined && value.y !== undefined), {
    message: "Tap action needs selector or x/y.",
  });

const typeActionSchema = z
  .object({
    type: z.literal("type"),
    timestamp: z.number().optional(),
    selector: selectorSchema,
    text: z.string(),
    replace: z.boolean(),
  })
  .strict();

const swipeActionSchema = z
  .object({
    type: z.literal("swipe"),
    timestamp: z.number().optional(),
    startX: z.number(),
    startY: z.number(),
    endX: z.number(),
    endY: z.number(),
    durationMs: z.number(),
  })
  .strict();

const backActionSchema = z
  .object({
    type: z.literal("back"),
    timestamp: z.number().optional(),
  })
  .strict();

const homeActionSchema = z
  .object({
    type: z.literal("home"),
    timestamp: z.number().optional(),
  })
  .strict();

const recordedActionSchema = z.union([
  tapActionSchema,
  typeActionSchema,
  swipeActionSchema,
  backActionSchema,
  homeActionSchema,
]);

const sessionReplayBodySchema = z
  .object({
    actions: z.array(recordedActionSchema).min(1).optional(),
    script: z.string().min(1).optional(),
  })
  .strict()
  .refine((value) => value.actions !== undefined || value.script !== undefined, {
    message: "Provide non-empty actions array or script.",
  });

function parseBody<T>(schema: z.ZodType<T>, input: unknown): T {
  try {
    return schema.parse(input);
  } catch (error) {
    if (error instanceof ZodError) {
      throw new RequestValidationError(
        error.issues.map((issue) => ({
          path: issue.path.join("."),
          message: issue.message,
        }))
      );
    }
    throw error;
  }
}

function sendError(
  res: express.Response,
  error: unknown,
  fallbackMessage: string,
  context: string
): void {
  logError(context, error);

  if (error instanceof RequestValidationError) {
    res.status(400).json({ error: error.message, details: error.details });
    return;
  }

  res.status(500).json({ error: error instanceof Error ? error.message : fallbackMessage });
}

function sendSse(replayId: string, event: ReplayEvent): void {
  if (event.type === "started") {
    logInfo("replay.started", { replayId, total: event.total });
  } else if (event.type === "error") {
    logError("replay.step_failed", event.error, {
      replayId,
      current: event.current,
      total: event.total,
      actionType: event.action.type,
    });
  } else if (event.type === "completed") {
    logInfo("replay.completed", { replayId, total: event.total, errors: event.errors });
  }

  const payload = JSON.stringify({ replayId, ...event });
  for (const response of sseClients) {
    response.write(`event: replay\n`);
    response.write(`data: ${payload}\n\n`);
  }
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function clearAutostartTimer(): void {
  if (!autostartTimer) {
    return;
  }
  clearTimeout(autostartTimer);
  autostartTimer = null;
}

function markAutostartConnected(): void {
  clearAutostartTimer();
  autostartState.retrying = false;
  autostartState.lastError = undefined;
  autostartState.nextRetryInMs = 0;
}

function scheduleAutostart(delayMs: number): void {
  clearAutostartTimer();
  autostartState.retrying = true;
  autostartState.nextRetryInMs = delayMs;

  autostartTimer = setTimeout(() => {
    void runAutostartAttempt();
  }, delayMs);
}

function getAutostartBackoffMs(attempt: number): number {
  const index = Math.min(Math.max(attempt - 1, 0), AUTOSTART_BACKOFF_MS.length - 1);
  return AUTOSTART_BACKOFF_MS[index];
}

async function runAutostartAttempt(): Promise<void> {
  autostartTimer = null;
  autostartState.attempts += 1;

  try {
    const session = await ensureSession();
    markAutostartConnected();
    logInfo("session.autostarted", {
      sessionId: session.sessionId,
      reused: session.reused,
      attempt: autostartState.attempts,
    });
  } catch (error) {
    const retryInMs = getAutostartBackoffMs(autostartState.attempts);
    autostartState.lastError = toErrorMessage(error);
    logError("session.autostart_failed", error, {
      attempt: autostartState.attempts,
      retryInMs,
    });
    scheduleAutostart(retryInMs);
  }
}

async function ensureReadySession(context: string): Promise<void> {
  try {
    await ensureSession();
  } catch (error) {
    logError(`${context}.ensure_session_failed`, error);
    throw error;
  }
}

function resolveStartConfig(body: Record<string, unknown>) {
  const packageName =
    (typeof body.package === "string" ? body.package : undefined) ||
    process.env.APPIUM_APP_PACKAGE;

  const activity =
    (typeof body.activity === "string" ? body.activity : undefined) ||
    process.env.APPIUM_APP_ACTIVITY;

  const deviceName =
    (typeof body.deviceName === "string" ? body.deviceName : undefined) ||
    process.env.APPIUM_DEVICE_NAME ||
    "Android";

  if (!packageName) {
    throw new Error("Missing app package. Provide { package: \"com.example.app\" } or APPIUM_APP_PACKAGE.");
  }

  return {
    packageName,
    activity,
    deviceName,
  };
}

app.get("/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  sseClients.add(res);
  const heartbeat = setInterval(() => {
    res.write(`event: heartbeat\n`);
    res.write(`data: ${JSON.stringify({ ts: Date.now() })}\n\n`);
  }, SSE_HEARTBEAT_MS);
  sseHeartbeats.set(res, heartbeat);
  res.write(`event: connected\ndata: ${JSON.stringify({ ok: true })}\n\n`);

  req.on("close", () => {
    sseClients.delete(res);
    const timer = sseHeartbeats.get(res);
    if (timer) {
      clearInterval(timer);
      sseHeartbeats.delete(res);
    }
  });
});

app.get("/health", (_req, res) => {
  const connection = getConnectionState();
  res.json({
    ok: true,
    connection,
    session: getSessionMeta(),
    recording: recorder.isActive(),
    autostart: {
      retrying: autostartState.retrying,
      attempts: autostartState.attempts,
      nextRetryInMs: autostartState.nextRetryInMs,
      lastError: autostartState.lastError,
    },
  });
});

app.get("/tree", async (_req, res) => {
  try {
    await ensureReadySession("tree");
    const xml = await getPageSource();
    const parsed = parseAccessibilityTree(xml);
    logInfo("tree.fetched", { elements: parsed.elements.length });
    res.json({ xml, tree: parsed });
  } catch (error) {
    logError("tree.failed", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Failed to get tree" });
  }
});

app.post("/tap", async (req, res) => {
  try {
    await ensureReadySession("tap");
    const { selector, x, y } = parseBody(tapBodySchema, req.body);

    if (selector) {
      await tapBySelector(selector as AndroidSelector);
      recorder.record({ type: "tap", selector });
      res.json({ ok: true, mode: "selector" });
      return;
    }

    if (x !== undefined && y !== undefined) {
      await tapByCoordinates(x, y);
      recorder.record({ type: "tap", x, y });
      res.json({ ok: true, mode: "coordinates" });
      return;
    }

    res.status(400).json({ error: "Provide selector or x/y coordinates." });
  } catch (error) {
    sendError(res, error, "Tap failed", "tap.failed");
  }
});

app.post("/type", async (req, res) => {
  try {
    await ensureReadySession("type");
    const { selector, text, replace } = parseBody(typeBodySchema, req.body);

    const shouldReplace = replace !== false;
    await typeIntoElement(selector as AndroidSelector, text, shouldReplace);
    recorder.record({ type: "type", selector, text, replace: shouldReplace });
    res.json({ ok: true });
  } catch (error) {
    sendError(res, error, "Type failed", "type.failed");
  }
});

app.post("/swipe", async (req, res) => {
  try {
    await ensureReadySession("swipe");
    const { startX, startY, endX, endY, durationMs } = parseBody(swipeBodySchema, req.body);

    const duration = typeof durationMs === "number" ? durationMs : 400;
    await swipe(startX, startY, endX, endY, duration);
    recorder.record({ type: "swipe", startX, startY, endX, endY, durationMs: duration });
    res.json({ ok: true });
  } catch (error) {
    sendError(res, error, "Swipe failed", "swipe.failed");
  }
});

app.post("/screenshot", async (_req, res) => {
  try {
    await ensureReadySession("screenshot");
    const screenshot = await takeScreenshot();
    res.json({ screenshot });
  } catch (error) {
    logError("screenshot.failed", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Screenshot failed" });
  }
});

app.post("/back", async (_req, res) => {
  try {
    await ensureReadySession("back");
    await pressBack();
    recorder.record({ type: "back" });
    res.json({ ok: true });
  } catch (error) {
    logError("back.failed", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Back failed" });
  }
});

app.post("/home", async (_req, res) => {
  try {
    await ensureReadySession("home");
    await pressHome();
    recorder.record({ type: "home" });
    res.json({ ok: true });
  } catch (error) {
    logError("home.failed", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Home failed" });
  }
});

app.post("/session/start", async (req, res) => {
  try {
    const body = parseBody(sessionStartBodySchema, req.body || {});
    const hasExplicitTarget =
      typeof body.package === "string" ||
      typeof body.activity === "string" ||
      typeof body.deviceName === "string";
    let ensured: { sessionId: string; reused: boolean };

    if (hasExplicitTarget) {
      const config = resolveStartConfig(body);
      logInfo("session.starting", {
        packageName: config.packageName,
        activity: config.activity,
        deviceName: config.deviceName,
      });
      const started = await startSession(config);
      ensured = { sessionId: started.sessionId, reused: false };
    } else {
      ensured = await ensureSession();
    }

    const session = getSessionMeta();
    if (!session) {
      throw new Error("Failed to initialize session.");
    }

    recorder.start({
      packageName: session.packageName,
      activity: session.activity,
      deviceName: session.deviceName,
    });

    logInfo("session.started", {
      sessionId: ensured.sessionId,
      reused: ensured.reused,
      packageName: session.packageName,
      activity: session.activity,
      deviceName: session.deviceName,
      recording: true,
    });
    markAutostartConnected();

    res.json({ ok: true, session, recording: true });
  } catch (error) {
    if (error instanceof RequestValidationError) {
      logError("session.start.validation_failed", error);
      res.status(400).json({ error: error.message, details: error.details });
      return;
    }
    logError("session.start.failed", error);
    res.status(400).json({ error: error instanceof Error ? error.message : "Failed to start session" });
  }
});

app.post("/session/stop", async (req, res) => {
  try {
    const { teardown } = parseBody(sessionStopBodySchema, req.body || {});
    const stopped = recorder.stop();
    const shouldTeardown = teardown === true;
    logInfo("session.stopping", {
      actions: stopped.actions.length,
      teardown: shouldTeardown,
    });
    const script = generateAppiumScript(stopped.actions, stopped.context);

    if (shouldTeardown) {
      await stopSession();
      logInfo("session.driver_torn_down");
    }

    logInfo("session.stopped", {
      actions: stopped.actions.length,
      startedAt: stopped.startedAt,
      stoppedAt: stopped.stoppedAt,
      teardown: shouldTeardown,
    });

    res.json({
      ok: true,
      actions: stopped.actions,
      startedAt: stopped.startedAt,
      stoppedAt: stopped.stoppedAt,
      script,
      teardown: shouldTeardown,
      session: getSessionMeta(),
    });
  } catch (error) {
    logError("session.stop.failed", error);
    res.status(500).json({ error: error instanceof Error ? error.message : "Failed to stop session" });
  }
});

app.post("/session/replay", async (req, res) => {
  try {
    await ensureReadySession("replay");
    const body = parseBody(sessionReplayBodySchema, req.body);

    const actionsFromBody = Array.isArray(body.actions)
      ? (body.actions as RecordedAction[])
      : undefined;
    const actionsFromScript =
      typeof body.script === "string" ? parseActionsFromScript(body.script) : null;

    const actions = actionsFromBody || actionsFromScript;

    if (!actions || actions.length === 0) {
      logError("replay.invalid_input", "No actions/script provided");
      res
        .status(400)
        .json({ error: "Provide non-empty actions array or script generated by /session/stop." });
      return;
    }

    const replayId = `replay-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    logInfo("replay.starting", { replayId, total: actions.length });
    const summary = await replayActions(actions, (event) => sendSse(replayId, event));
    logInfo("replay.finished", {
      replayId,
      total: summary.total,
      completed: summary.completed,
      errors: summary.errors.length,
    });
    res.json({ ok: true, replayId, ...summary });
  } catch (error) {
    sendError(res, error, "Replay failed", "replay.failed");
  }
});

app.get("/session", (_req, res) => {
  res.json({
    session: getSessionMeta(),
    recording: recorder.isActive(),
    actions: recorder.getActions().length,
  });
});

app.listen(PORT, () => {
  logInfo("server.started", { url: `http://localhost:${PORT}` });
  scheduleAutostart(0);

  autostartMonitor = setInterval(() => {
    const connection = getConnectionState();
    if (connection.status !== "disconnected") {
      return;
    }
    if (autostartTimer) {
      return;
    }
    scheduleAutostart(getAutostartBackoffMs(autostartState.attempts + 1));
  }, 10000);
});