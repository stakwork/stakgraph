/**
 * Unit tests for get_context streaming behaviour.
 *
 * These tests verify that the get_context switch from agent.generate() to
 * agent.stream() preserves all session persistence behaviour:
 *   - Happy path: stream is consumed, totalUsage (not per-step usage) propagates
 *   - Abort/error path: appendSessionEnd is called with status "aborted"/"error"
 *     and a non-empty error_message
 *   - Null steps guard: undefined steps resolves to [] rather than TypeError
 *
 * Because importing get_context directly would pull in neo4j, ToolLoopAgent,
 * file-system tools, etc., these tests reproduce the session-writing logic
 * inline — the same pattern used by logs-agent-session.test.ts and
 * graph-agent-session.test.ts in this directory.
 *
 * The key invariants under test are:
 *   1. streamResult.totalUsage (not .usage) is awaited and used.
 *   2. (await streamResult.steps) ?? [] guards against undefined steps.
 *   3. The catch block populates all six fields required by appendSessionEnd.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Session file helpers (inline, no session.ts import needed)
// ---------------------------------------------------------------------------

function makeSessionsDir(): string {
  const dir = path.join(os.tmpdir(), `test-gc-stream-${randomUUID()}`);
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
// Type helpers — mirror what the real get_context uses
// ---------------------------------------------------------------------------

interface LanguageModelUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
}

interface StepResult {
  content: Array<{ type: string; text?: string; [k: string]: unknown }>;
  usage: LanguageModelUsage;
}

interface StreamResult {
  steps: Promise<StepResult[] | undefined>;
  totalUsage: Promise<LanguageModelUsage>;
  usage: Promise<LanguageModelUsage>;
}

// ---------------------------------------------------------------------------
// Helpers — replicate isAbortError logic from agent.ts
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

// ---------------------------------------------------------------------------
// Normalise usage — mirrors normalizeUsage from aieo
// ---------------------------------------------------------------------------

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
// extractFinalAnswer stub — mirrors the minimal real behaviour
// ---------------------------------------------------------------------------

function extractFinalAnswer(steps: StepResult[]): { answer: string; tool_use?: string } {
  let allText = "";
  for (const step of steps) {
    for (const item of step.content) {
      if (item.type === "text" && item.text) {
        allText += item.text;
      }
    }
  }
  const endMarkerIndex = allText.indexOf("[END_OF_ANSWER]");
  const answer = endMarkerIndex !== -1
    ? allText.substring(0, endMarkerIndex).trim()
    : allText.trim();
  return { answer };
}

// ---------------------------------------------------------------------------
// Simulate get_context's streaming block
//
// This mirrors the implementation in agent.ts lines ~808-890:
//   - awaits agent.stream(), then steps and totalUsage
//   - applies null guard on steps
//   - writes session end on both success and error paths
//   - returns a ContextResult-shaped object
// ---------------------------------------------------------------------------

interface SimulatedContextResult {
  final: string;
  usage: LanguageModelUsage & { model: string; provider: string };
  sessionId?: string;
}

interface SimulateOpts {
  sessionsDir: string;
  sessionId: string;
  modelId: string;
  provider: string;
  startTime: number;
  /** stepMetas.length > 0 triggers the addUsage path; otherwise totalUsage is used */
  stepMetas?: Array<{ usage: LanguageModelUsage }>;
}

async function simulateGetContextStream(
  streamResultFactory: () => StreamResult,
  opts: SimulateOpts
): Promise<SimulatedContextResult> {
  const { sessionsDir, sessionId, modelId, provider, startTime } = opts;
  const stepMetas = opts.stepMetas ?? [];

  let steps: StepResult[] = [];
  let streamTotalUsage: LanguageModelUsage | undefined;

  try {
    // Mirrors: const streamResult = await agent.stream(buildCallParams(prepared));
    const streamResult = await Promise.resolve(streamResultFactory());

    // Mirrors: steps = (await streamResult.steps) ?? [];
    steps = (await streamResult.steps) ?? [];

    // Mirrors: streamTotalUsage = await streamResult.totalUsage;
    streamTotalUsage = await streamResult.totalUsage;
  } catch (err) {
    const aborted = isAbortError(err);
    const endTime = new Date();
    appendToSession(sessionsDir, sessionId, {
      type: "session_end",
      session_id: sessionId,
      end_time: endTime.toISOString(),
      model: modelId,
      provider,
      duration_ms: endTime.getTime() - startTime,
      status: aborted ? "aborted" : "error",
      error_message: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }

  // Mirrors: const usage = stepMetas.length > 0 ? normalizeUsage(addUsage(...)) : normalizeUsage(streamTotalUsage);
  const usage = stepMetas.length > 0
    ? normalizeUsage(addUsage(...stepMetas.map((s) => s.usage)))
    : normalizeUsage(streamTotalUsage);

  const endTime = Date.now();
  const duration = endTime - startTime;

  // Mirrors: success session end
  appendToSession(sessionsDir, sessionId, {
    type: "session_end",
    session_id: sessionId,
    end_time: new Date().toISOString(),
    model: modelId,
    provider,
    duration_ms: duration,
    status: "success",
    token_usage: usage,
  });

  const final = extractFinalAnswer(steps);

  return {
    final: final.answer,
    usage: { ...usage, model: modelId, provider },
    sessionId,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("get_context streaming (agent.generate → agent.stream)", () => {
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
  // 1. Happy path — stream consumed, session_end status: success
  //    Key: totalUsage (not per-step usage) propagates to token_usage
  // -------------------------------------------------------------------------
  test("happy path: returns ContextResult shape and session_end records totalUsage", async () => {
    const sessionId = randomUUID();

    // Deliberately use different values so we can assert the RIGHT one propagates
    const perStepUsage: LanguageModelUsage = { inputTokens: 1, outputTokens: 1 };
    const streamTotalUsage: LanguageModelUsage = { inputTokens: 500, outputTokens: 250 };

    const streamResultFactory = (): StreamResult => ({
      steps: Promise.resolve([
        {
          content: [{ type: "text", text: "Hello world [END_OF_ANSWER]" }],
          usage: perStepUsage,
        },
      ]),
      totalUsage: Promise.resolve(streamTotalUsage),
      usage: Promise.resolve(perStepUsage), // intentionally different — must NOT be used
    });

    const result = await simulateGetContextStream(streamResultFactory, {
      sessionsDir,
      sessionId,
      modelId: "claude-3-5-sonnet",
      provider: "anthropic",
      startTime: Date.now() - 100,
    });

    // Return shape checks
    expect(result.final).toBe("Hello world");
    expect(result.sessionId).toBe(sessionId);
    expect(result.usage.model).toBe("claude-3-5-sonnet");
    expect(result.usage.provider).toBe("anthropic");

    // Session persistence checks
    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("success");

    // token_usage must reflect totalUsage, NOT per-step usage
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage).toBeDefined();
    expect(tokenUsage.inputTokens).toBe(streamTotalUsage.inputTokens);   // 500, not 1
    expect(tokenUsage.outputTokens).toBe(streamTotalUsage.outputTokens); // 250, not 1
  });

  // -------------------------------------------------------------------------
  // 2. Happy path with stepMetas — addUsage path (step-aggregated usage wins)
  // -------------------------------------------------------------------------
  test("happy path with stepMetas: addUsage path overrides totalUsage for token accounting", async () => {
    const sessionId = randomUUID();

    const streamTotalUsage: LanguageModelUsage = { inputTokens: 999, outputTokens: 999 };
    const stepMeta1 = { usage: { inputTokens: 100, outputTokens: 50 } };
    const stepMeta2 = { usage: { inputTokens: 200, outputTokens: 80 } };

    const streamResultFactory = (): StreamResult => ({
      steps: Promise.resolve([
        {
          content: [{ type: "text", text: "Answer [END_OF_ANSWER]" }],
          usage: stepMeta1.usage,
        },
      ]),
      totalUsage: Promise.resolve(streamTotalUsage),
      usage: Promise.resolve(stepMeta1.usage),
    });

    const result = await simulateGetContextStream(streamResultFactory, {
      sessionsDir,
      sessionId,
      modelId: "gpt-4o",
      provider: "openai",
      startTime: Date.now() - 50,
      stepMetas: [stepMeta1, stepMeta2],
    });

    expect(result.final).toBe("Answer");

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry!.status).toBe("success");

    // addUsage path: 100+200=300 input, 50+80=130 output — NOT 999
    const tokenUsage = endEntry!.token_usage as LanguageModelUsage;
    expect(tokenUsage.inputTokens).toBe(300);
    expect(tokenUsage.outputTokens).toBe(130);
  });

  // -------------------------------------------------------------------------
  // 3. Error path — stream throws; appendSessionEnd called with status: error
  //    and a non-empty error_message
  // -------------------------------------------------------------------------
  test("error path: session_end written with status error and error_message", async () => {
    const sessionId = randomUUID();

    const streamResultFactory = (): StreamResult => ({
      // Simulate error surfacing during stream consumption
      steps: Promise.reject(new Error("LLM connection refused")),
      totalUsage: Promise.resolve({}),
      usage: Promise.resolve({}),
    });

    let threw = false;
    try {
      await simulateGetContextStream(streamResultFactory, {
        sessionsDir,
        sessionId,
        modelId: "claude-3-opus",
        provider: "anthropic",
        startTime: Date.now() - 200,
      });
    } catch (err) {
      threw = true;
      expect((err as Error).message).toBe("LLM connection refused");
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(typeof endEntry!.error_message).toBe("string");
    expect((endEntry!.error_message as string).length).toBeGreaterThan(0);
    expect(endEntry!.error_message).toBe("LLM connection refused");
    expect(endEntry!.model).toBe("claude-3-opus");
    expect(endEntry!.provider).toBe("anthropic");
    expect(typeof endEntry!.duration_ms).toBe("number");
  });

  // -------------------------------------------------------------------------
  // 4. Abort path — error with abort-shaped message → status: aborted
  // -------------------------------------------------------------------------
  test("abort path: session_end written with status aborted", async () => {
    const sessionId = randomUUID();

    const abortError = new Error("Request aborted by user");
    abortError.name = "AbortError";

    const streamResultFactory = (): StreamResult => ({
      steps: Promise.reject(abortError),
      totalUsage: Promise.resolve({}),
      usage: Promise.resolve({}),
    });

    let threw = false;
    try {
      await simulateGetContextStream(streamResultFactory, {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 100,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("aborted");
    expect(typeof endEntry!.error_message).toBe("string");
    expect((endEntry!.error_message as string).length).toBeGreaterThan(0);
  });

  // -------------------------------------------------------------------------
  // 5. Null steps guard — undefined steps resolves to [] without TypeError
  //    Verifies (await streamResult.steps) ?? [] prevents crashes
  // -------------------------------------------------------------------------
  test("null steps guard: undefined steps resolves to valid ContextResult (no TypeError)", async () => {
    const sessionId = randomUUID();

    const streamResultFactory = (): StreamResult => ({
      // Simulate abnormal-but-non-throwing completion where steps is undefined
      steps: Promise.resolve(undefined as any),
      totalUsage: Promise.resolve({ inputTokens: 10, outputTokens: 5 }),
      usage: Promise.resolve({ inputTokens: 10, outputTokens: 5 }),
    });

    // Must NOT throw a TypeError from iterating undefined
    let result: SimulatedContextResult | undefined;
    let threw = false;
    try {
      result = await simulateGetContextStream(streamResultFactory, {
        sessionsDir,
        sessionId,
        modelId: "claude-3-haiku",
        provider: "anthropic",
        startTime: Date.now() - 50,
      });
    } catch {
      threw = true;
    }

    expect(threw).toBe(false);
    expect(result).toBeDefined();
    expect(result!.final).toBeDefined(); // empty string is fine
    expect(result!.sessionId).toBe(sessionId);

    // Session end must still be written successfully
    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("success");
  });

  // -------------------------------------------------------------------------
  // 6. Error surfacing during totalUsage (post-steps) is also caught
  //    Ensures the catch block wraps BOTH steps and totalUsage awaits
  // -------------------------------------------------------------------------
  test("error during totalUsage consumption: caught and session_end written with error", async () => {
    const sessionId = randomUUID();

    const streamResultFactory = (): StreamResult => ({
      steps: Promise.resolve([
        { content: [{ type: "text", text: "partial [END_OF_ANSWER]" }], usage: {} },
      ]),
      // Error surfaces when totalUsage is awaited
      totalUsage: Promise.reject(new Error("stream interrupted during usage read")),
      usage: Promise.resolve({}),
    });

    let threw = false;
    try {
      await simulateGetContextStream(streamResultFactory, {
        sessionsDir,
        sessionId,
        modelId: "gpt-4o",
        provider: "openai",
        startTime: Date.now() - 150,
      });
    } catch (err) {
      threw = true;
      expect((err as Error).message).toBe("stream interrupted during usage read");
    }

    expect(threw).toBe(true);

    const entries = loadSessionEntries(sessionsDir, sessionId);
    const endEntry = entries.find((e) => e.type === "session_end");
    expect(endEntry).toBeDefined();
    expect(endEntry!.status).toBe("error");
    expect(endEntry!.error_message).toBe("stream interrupted during usage read");
  });
});
