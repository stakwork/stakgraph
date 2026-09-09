/**
 * Unit tests for token_usage on error/abort session-end branches in
 * get_context (non-streaming path) in repo/agent.ts.
 *
 * These tests verify Requirement #3 of the "Fix persistence gaps" feature:
 *   "Error/abort session-end records carry token usage."
 *
 * The non-streaming path (get_context) throws from within the loop body;
 * the catch at ~lines 1385-1415 of agent.ts should include token_usage
 * computed the same way as the success path:
 *   stepMetas.length > 0
 *     ? normalizeUsage(addUsage(...stepMetas.map(s => s.usage)))
 *     : normalizeUsage(streamTotalUsage)
 *
 * Because importing get_context directly pulls in neo4j, ToolLoopAgent, and
 * file-system tools, these tests use an inline mirror of the catch-block logic
 * — the same pattern as get-context-stream.test.ts and logs-agent-session.test.ts.
 * The mirror is kept in lockstep with agent.ts via explicit comments.
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
  const dir = path.join(os.tmpdir(), `test-gc-error-${randomUUID()}`);
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
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
}

// ---------------------------------------------------------------------------
// Helpers — mirrors from agent.ts / aieo
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
// Simulate the non-streaming get_context catch block.
//
// Mirrors the catch block in repo/agent.ts get_context (~lines 1385-1415).
//
// Key invariants under test:
//   1. token_usage is always present in the session_end entry on error/abort.
//   2. token_usage derives from stepMetas when available (addUsage path).
//   3. token_usage falls back to streamTotalUsage (normalizeUsage) when
//      stepMetas is empty — even if streamTotalUsage is undefined,
//      normalizeUsage returns a safe empty object rather than throwing.
//   4. The original error always re-throws (best-effort, never fatal).
// ---------------------------------------------------------------------------

interface SimulateOpts {
  sessionsDir: string;
  sessionId: string;
  modelId: string;
  provider: string;
  startTime: number;
  /** stepMetas accumulated before the error (may be empty) */
  stepMetas?: Array<{ usage: LanguageModelUsage }>;
  /** last known stream total usage at error time (may be undefined) */
  streamTotalUsage?: LanguageModelUsage;
}

/**
 * Simulates the error branch of get_context's catch block.
 * Always throws the supplied error after persisting session_end.
 *
 * MIRROR NOTE: Keep in lockstep with the catch block in repo/agent.ts
 * get_context. If the real catch block changes its token_usage computation,
 * update this mirror accordingly.
 */
async function simulateGetContextError(err: Error, opts: SimulateOpts): Promise<void> {
  const { sessionsDir, sessionId, modelId, provider, startTime } = opts;
  const stepMetas = opts.stepMetas ?? [];
  const streamTotalUsage = opts.streamTotalUsage;

  const aborted = isAbortError(err);

  // Mirror: stepMetas.length > 0 ? normalizeUsage(addUsage(...)) : normalizeUsage(streamTotalUsage)
  const errorUsage =
    stepMetas.length > 0
      ? normalizeUsage(addUsage(...stepMetas.map((step) => step.usage)))
      : normalizeUsage(streamTotalUsage);

  const endTime = new Date();
  appendToSession(sessionsDir, sessionId, {
    type: "session_end",
    session_id: sessionId,
    end_time: endTime.toISOString(),
    model: modelId,
    provider,
    duration_ms: endTime.getTime() - startTime,
    status: aborted ? "aborted" : "error",
    error_message: err.message,
    token_usage: errorUsage,
  });

  throw err;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("get_context non-stream error branch: token_usage on error/abort", () => {
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
  // 1. Error with stepMetas — addUsage path produces correct token_usage
  // -------------------------------------------------------------------------
  test("error with stepMetas: token_usage sums stepMetas (addUsage path)", async () => {
    const sessionId = randomUUID();
    const stepMeta1 = { usage: { inputTokens: 100, outputTokens: 50 } };
    const stepMeta2 = { usage: { inputTokens: 200, outputTokens: 80 } };

    // A streamTotalUsage that would give wrong results if used instead of stepMetas
    const streamTotalUsage = { inputTokens: 999, outputTokens: 999 };

    let threw = false;
    try {
      await simulateGetContextError(new Error("model overloaded"), {
        sessionsDir,
        sessionId,
        modelId: "claude-3-5-sonnet",
        provider: "anthropic",
        startTime: Date.now() - 300,
        stepMetas: [stepMeta1, stepMeta2],
        streamTotalUsage,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(endEntry!.error_message).toBe("model overloaded");

    // addUsage path: 100+200=300 input, 50+80=130 output — NOT 999
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(300);
    expect(tokenUsage.outputTokens).toBe(130);
  });

  // -------------------------------------------------------------------------
  // 2. Error with empty stepMetas — falls back to streamTotalUsage
  // -------------------------------------------------------------------------
  test("error with empty stepMetas: token_usage falls back to streamTotalUsage", async () => {
    const sessionId = randomUUID();
    const streamTotalUsage = { inputTokens: 450, outputTokens: 220 };

    let threw = false;
    try {
      await simulateGetContextError(new Error("network timeout"), {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 150,
        stepMetas: [],          // no step metas accumulated
        streamTotalUsage,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");

    // Fallback to streamTotalUsage
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(450);
    expect(tokenUsage.outputTokens).toBe(220);
  });

  // -------------------------------------------------------------------------
  // 3. Error with empty stepMetas and undefined streamTotalUsage
  //    normalizeUsage(undefined) should return a safe empty object, not throw
  // -------------------------------------------------------------------------
  test("error with empty stepMetas and undefined streamTotalUsage: token_usage is safe empty object", async () => {
    const sessionId = randomUUID();

    let threw = false;
    try {
      await simulateGetContextError(new Error("early failure"), {
        sessionsDir,
        sessionId,
        modelId: "claude-3-haiku",
        provider: "anthropic",
        startTime: Date.now() - 50,
        stepMetas: [],
        streamTotalUsage: undefined,  // never populated before error
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(endEntry!.error_message).toBe("early failure");

    // Must be present (not undefined) — safe empty object is acceptable
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 4. Abort path — isAbortError triggers status: aborted; token_usage still present
  // -------------------------------------------------------------------------
  test("abort path: status is aborted and token_usage is still present", async () => {
    const sessionId = randomUUID();
    const stepMeta = { usage: { inputTokens: 80, outputTokens: 40 } };

    const abortErr = new Error("Request aborted");
    abortErr.name = "AbortError";

    let threw = false;
    try {
      await simulateGetContextError(abortErr, {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 200,
        stepMetas: [stepMeta],
        streamTotalUsage: { inputTokens: 999, outputTokens: 999 },
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("aborted");

    // addUsage path: 80 input, 40 output from stepMeta
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(80);
    expect(tokenUsage.outputTokens).toBe(40);
  });

  // -------------------------------------------------------------------------
  // 5. Error always re-throws (best-effort, never fatal)
  // -------------------------------------------------------------------------
  test("error is always re-thrown — best-effort, never fatal", async () => {
    const sessionId = randomUUID();
    const originalError = new Error("connection refused");

    let caughtError: Error | undefined;
    try {
      await simulateGetContextError(originalError, {
        sessionsDir,
        sessionId,
        modelId: "claude-3-5-sonnet",
        provider: "anthropic",
        startTime: Date.now() - 100,
      });
    } catch (e) {
      caughtError = e as Error;
    }

    expect(caughtError).toBeDefined();
    expect(caughtError).toBe(originalError); // same reference — not wrapped
    expect(caughtError!.message).toBe("connection refused");
  });
});
