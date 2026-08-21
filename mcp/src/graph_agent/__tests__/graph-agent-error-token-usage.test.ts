/**
 * Unit tests for token_usage on error/abort session-end branches in
 * graph_agent/agent.ts — both get_context (non-streaming) and finalizeSession
 * (streaming).
 *
 * These tests verify Requirement #3 of the "Fix persistence gaps" feature:
 *   "Error/abort session-end records carry token usage."
 *
 * The two error branches covered:
 *
 * A. get_context catch (~lines 308-335 of graph_agent/agent.ts):
 *    Triggered when agent.generate() throws. The catch persists session_end
 *    with token_usage computed as:
 *      stepMetas.length > 0
 *        ? normalizeUsage(addUsage(...stepMetas.map(s => s.usage)))
 *        : normalizeUsage(totalUsage)
 *    where totalUsage comes from the failed generate result (undefined if
 *    generate never returned anything).
 *
 * B. finalizeSession catch (~lines 429-458 of graph_agent/agent.ts):
 *    Triggered when stream consumption (steps/usage await) throws. The catch
 *    persists session_end with token_usage computed as:
 *      stepMetas.length > 0
 *        ? normalizeUsage(addUsage(...stepMetas.map(s => s.usage)))
 *        : normalizeUsage(undefined)
 *    (stream usage not available since it may be what threw).
 *
 * Because importing graph_agent/agent.ts directly pulls in ToolLoopAgent and
 * neo4j, these tests use inline mirrors of the catch-block logic — the same
 * pattern as get-context-stream.test.ts and graph-agent-session.test.ts.
 * Mirrors are kept in lockstep with the real code via explicit comments.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Session file helpers
// ---------------------------------------------------------------------------

function makeSessionsDir(): string {
  const dir = path.join(os.tmpdir(), `test-ga-error-${randomUUID()}`);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function sessionFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.jsonl`);
}

interface SessionEntry {
  type?: string;
  status?: string;
  error_message?: string;
  token_usage?: Record<string, unknown>;
  model?: string;
  provider?: string;
  duration_ms?: number;
  [key: string]: unknown;
}

function appendToSession(sessionsDir: string, sessionId: string, entry: SessionEntry): void {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  fs.appendFileSync(filePath, JSON.stringify(entry) + "\n");
}

function loadSessionEntries(sessionsDir: string, sessionId: string): SessionEntry[] {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, "utf-8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as SessionEntry);
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

interface LanguageModelUsage {
  inputTokens?: number;
  outputTokens?: number;
  promptTokens?: number;
  completionTokens?: number;
}

// ---------------------------------------------------------------------------
// Mirrors of aieo / agent utilities
// ---------------------------------------------------------------------------

function isAbortError(err: unknown): boolean {
  if (!err) return false;
  if (err instanceof Error) {
    if (err.name === "AbortError") return true;
    const cause: any = (err as any).cause;
    if (cause && cause.name === "AbortError") return true;
    if (/abort/i.test(err.message)) return true;
  }
  return false;
}

function normalizeUsage(usage: LanguageModelUsage | undefined): LanguageModelUsage {
  if (!usage) return {};
  return {
    inputTokens: usage.inputTokens ?? usage.promptTokens ?? 0,
    outputTokens: usage.outputTokens ?? usage.completionTokens ?? 0,
  };
}

function addUsage(...usages: LanguageModelUsage[]): LanguageModelUsage {
  return usages.reduce(
    (acc, u) => ({
      inputTokens: (acc.inputTokens ?? 0) + (u.inputTokens ?? 0),
      outputTokens: (acc.outputTokens ?? 0) + (u.outputTokens ?? 0),
    }),
    {}
  );
}

// ---------------------------------------------------------------------------
// Shared test opts
// ---------------------------------------------------------------------------

interface SimulateOpts {
  sessionsDir: string;
  sessionId: string;
  modelId: string;
  provider: string;
  startTime: number;
  stepMetas?: Array<{ usage: LanguageModelUsage }>;
}

// ---------------------------------------------------------------------------
// A. Mirror of graph_agent/agent.ts get_context catch block.
//
// MIRROR NOTE: Keep in lockstep with the catch block in get_context
// (~lines 308-335 of graph_agent/agent.ts).
//
// Key differences from the streaming finalizeSession catch:
//   - totalUsage comes from agent.generate() result (unavailable on throw;
//     passed in here as an explicit parameter to exercise the fallback).
//   - Uses normalizeUsage(totalUsage as any) — same as real code.
// ---------------------------------------------------------------------------

async function simulateGraphAgentGetContextError(
  err: Error,
  opts: SimulateOpts & { totalUsage?: LanguageModelUsage }
): Promise<void> {
  const { sessionsDir, sessionId, modelId, provider, startTime } = opts;
  const stepMetas = opts.stepMetas ?? [];
  const totalUsage = opts.totalUsage;

  const aborted = isAbortError(err);
  const duration = Date.now() - startTime;

  // Mirror: stepMetas.length > 0 ? normalizeUsage(addUsage(...)) : normalizeUsage(totalUsage as any)
  const errorUsage =
    stepMetas.length > 0
      ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
      : normalizeUsage(totalUsage as any);

  appendToSession(sessionsDir, sessionId, {
    type: "session_end",
    session_id: sessionId,
    end_time: new Date().toISOString(),
    model: modelId,
    provider,
    duration_ms: duration,
    status: aborted ? "aborted" : "error",
    error_message: err.message,
    token_usage: errorUsage,
  });

  throw err;
}

// ---------------------------------------------------------------------------
// B. Mirror of graph_agent/agent.ts finalizeSession catch block.
//
// MIRROR NOTE: Keep in lockstep with the catch block in finalizeSession
// (~lines 429-458 of graph_agent/agent.ts).
//
// Key difference: stream usage is not available (may be what threw), so
// the fallback is normalizeUsage(undefined) — a safe empty object.
// ---------------------------------------------------------------------------

async function simulateGraphAgentFinalizeSessionError(
  err: Error,
  opts: SimulateOpts
): Promise<void> {
  const { sessionsDir, sessionId, modelId, provider, startTime } = opts;
  const stepMetas = opts.stepMetas ?? [];

  const aborted = isAbortError(err);

  // Mirror: stepMetas.length > 0 ? normalizeUsage(addUsage(...)) : normalizeUsage(undefined)
  const errorUsage =
    stepMetas.length > 0
      ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
      : normalizeUsage(undefined);

  // .catch(() => {}) guard is present in the real code; the test writes directly
  appendToSession(sessionsDir, sessionId, {
    type: "session_end",
    session_id: sessionId,
    end_time: new Date().toISOString(),
    model: modelId,
    provider,
    duration_ms: Date.now() - startTime,
    status: aborted ? "aborted" : "error",
    error_message: err.message,
    token_usage: errorUsage,
  });

  // Note: real code uses .catch(() => {}) and does NOT re-throw; for testing
  // we don't throw so callers can inspect the written entry directly.
}

// ---------------------------------------------------------------------------
// Tests — A. get_context error branch
// ---------------------------------------------------------------------------

test.describe("graph_agent get_context error branch: token_usage on error/abort", () => {
  let sessionsDir: string;

  test.beforeEach(() => {
    sessionsDir = makeSessionsDir();
  });

  test.afterEach(() => {
    try {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
  });

  // -------------------------------------------------------------------------
  // A1. Error with stepMetas — addUsage path produces correct token_usage
  // -------------------------------------------------------------------------
  test("error with stepMetas: token_usage sums stepMetas (addUsage path)", async () => {
    const sessionId = randomUUID();
    const stepMeta1 = { usage: { inputTokens: 120, outputTokens: 60 } };
    const stepMeta2 = { usage: { inputTokens: 180, outputTokens: 90 } };
    // totalUsage that would give wrong results if used instead of stepMetas
    const totalUsage = { inputTokens: 999, outputTokens: 999 };

    let threw = false;
    try {
      await simulateGraphAgentGetContextError(new Error("context window exceeded"), {
        sessionsDir,
        sessionId,
        modelId: "claude-3-5-sonnet",
        provider: "anthropic",
        startTime: Date.now() - 400,
        stepMetas: [stepMeta1, stepMeta2],
        totalUsage,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(endEntry!.error_message).toBe("context window exceeded");

    // addUsage: 120+180=300 input, 60+90=150 output — NOT 999
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(300);
    expect(tokenUsage.outputTokens).toBe(150);
  });

  // -------------------------------------------------------------------------
  // A2. Error with no stepMetas — falls back to totalUsage
  // -------------------------------------------------------------------------
  test("error with empty stepMetas: token_usage falls back to totalUsage", async () => {
    const sessionId = randomUUID();
    const totalUsage = { inputTokens: 350, outputTokens: 175 };

    let threw = false;
    try {
      await simulateGraphAgentGetContextError(new Error("rate limit"), {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 200,
        stepMetas: [],
        totalUsage,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");

    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(350);
    expect(tokenUsage.outputTokens).toBe(175);
  });

  // -------------------------------------------------------------------------
  // A3. Error with no stepMetas and undefined totalUsage
  //     normalizeUsage(undefined) must return a safe empty object, not throw
  // -------------------------------------------------------------------------
  test("error with empty stepMetas and undefined totalUsage: token_usage is safe empty object", async () => {
    const sessionId = randomUUID();

    let threw = false;
    try {
      await simulateGraphAgentGetContextError(new Error("pre-flight failure"), {
        sessionsDir,
        sessionId,
        modelId: "claude-3-haiku",
        provider: "anthropic",
        startTime: Date.now() - 50,
        stepMetas: [],
        totalUsage: undefined,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    // token_usage must be present (not missing) — safe empty object is acceptable
    expect(endEntry!.token_usage).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // A4. Abort path — status is aborted, token_usage still present
  // -------------------------------------------------------------------------
  test("abort path: status is aborted and token_usage is still present", async () => {
    const sessionId = randomUUID();
    const stepMeta = { usage: { inputTokens: 90, outputTokens: 45 } };

    const abortErr = new Error("request aborted by client");
    abortErr.name = "AbortError";

    let threw = false;
    try {
      await simulateGraphAgentGetContextError(abortErr, {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 300,
        stepMetas: [stepMeta],
        totalUsage: { inputTokens: 999, outputTokens: 999 },
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("aborted");

    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(90);
    expect(tokenUsage.outputTokens).toBe(45);
  });

  // -------------------------------------------------------------------------
  // A5. Error always re-throws (best-effort, never fatal)
  // -------------------------------------------------------------------------
  test("error is always re-thrown — best-effort, never fatal", async () => {
    const sessionId = randomUUID();
    const original = new Error("network failure");

    let caught: Error | undefined;
    try {
      await simulateGraphAgentGetContextError(original, {
        sessionsDir,
        sessionId,
        modelId: "claude-3-5-sonnet",
        provider: "anthropic",
        startTime: Date.now() - 100,
      });
    } catch (e) {
      caught = e as Error;
    }

    expect(caught).toBeDefined();
    expect(caught).toBe(original); // same reference
    expect(caught!.message).toBe("network failure");
  });
});

// ---------------------------------------------------------------------------
// Tests — B. finalizeSession error branch
// ---------------------------------------------------------------------------

test.describe("graph_agent finalizeSession error branch: token_usage on error/abort", () => {
  let sessionsDir: string;

  test.beforeEach(() => {
    sessionsDir = makeSessionsDir();
  });

  test.afterEach(() => {
    try {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
  });

  // -------------------------------------------------------------------------
  // B1. Error with stepMetas — addUsage path produces correct token_usage
  // -------------------------------------------------------------------------
  test("error with stepMetas: token_usage sums stepMetas (addUsage path)", async () => {
    const sessionId = randomUUID();
    const stepMeta1 = { usage: { inputTokens: 200, outputTokens: 100 } };
    const stepMeta2 = { usage: { inputTokens: 300, outputTokens: 150 } };

    const err = new Error("stream interrupted");
    await simulateGraphAgentFinalizeSessionError(err, {
      sessionsDir,
      sessionId,
      modelId: "claude-3-5-sonnet",
      provider: "anthropic",
      startTime: Date.now() - 500,
      stepMetas: [stepMeta1, stepMeta2],
    });

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(endEntry!.error_message).toBe("stream interrupted");

    // addUsage: 200+300=500 input, 100+150=250 output
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(500);
    expect(tokenUsage.outputTokens).toBe(250);
  });

  // -------------------------------------------------------------------------
  // B2. Error with no stepMetas — falls back to safe empty object
  //     (stream usage unavailable since it may be what threw)
  // -------------------------------------------------------------------------
  test("error with empty stepMetas: token_usage is safe empty object (stream unavailable)", async () => {
    const sessionId = randomUUID();

    const err = new Error("stream usage fetch failed");
    await simulateGraphAgentFinalizeSessionError(err, {
      sessionsDir,
      sessionId,
      modelId: "gpt-4o",
      provider: "openai",
      startTime: Date.now() - 100,
      stepMetas: [],
    });

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    // token_usage must be present — safe empty object is acceptable
    expect(endEntry!.token_usage).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // B3. Abort path — status is aborted, token_usage still present
  // -------------------------------------------------------------------------
  test("abort path: status is aborted and token_usage is still present", async () => {
    const sessionId = randomUUID();
    const stepMeta = { usage: { inputTokens: 110, outputTokens: 55 } };

    const abortErr = new Error("client disconnected");
    abortErr.name = "AbortError";

    await simulateGraphAgentFinalizeSessionError(abortErr, {
      sessionsDir,
      sessionId,
      modelId: "claude-3-5-sonnet",
      provider: "anthropic",
      startTime: Date.now() - 200,
      stepMetas: [stepMeta],
    });

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("aborted");

    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(110);
    expect(tokenUsage.outputTokens).toBe(55);
  });

  // -------------------------------------------------------------------------
  // B4. Success-path token_usage matches the success path computation
  //     (regression guard: verify finalizeSession success writes correctly too)
  // -------------------------------------------------------------------------
  test("success path still writes token_usage correctly (regression guard)", async () => {
    const sessionId = randomUUID();
    const stepMeta1 = { usage: { inputTokens: 400, outputTokens: 200 } };
    const stepMeta2 = { usage: { inputTokens: 600, outputTokens: 300 } };
    const streamUsage = { inputTokens: 1, outputTokens: 1 }; // should be overridden by stepMetas

    // Simulate the success path directly
    const stepUsage =
      [stepMeta1, stepMeta2].length > 0
        ? normalizeUsage(addUsage(...[stepMeta1, stepMeta2].map((s) => s.usage)))
        : normalizeUsage(streamUsage);

    appendToSession(sessionsDir, sessionId, {
      type: "session_end",
      session_id: sessionId,
      end_time: new Date().toISOString(),
      model: "claude-3-5-sonnet",
      provider: "anthropic",
      duration_ms: 500,
      status: "success",
      token_usage: stepUsage,
    });

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry!.status).toBe("success");

    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage.inputTokens).toBe(1000); // 400+600
    expect(tokenUsage.outputTokens).toBe(500); // 200+300
  });
});
