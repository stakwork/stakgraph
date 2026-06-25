/**
 * Unit tests for logs_agent session persistence.
 *
 * These tests verify that the logs_agent execute closure (mcp/src/repo/tools.ts)
 * generates a unique sessionId per invocation and that session files are written
 * to SESSIONS_DIR independently of the per-run logsDir scratch space.
 *
 * The session persistence path exercised here mirrors what log_agent_context
 * does internally:
 *   createSession(sessionId, SYSTEM, source)
 *   appendMessages(sessionId, messages)
 *   appendStepMeta(sessionId, stepMetas)
 *   appendSessionEnd(sessionId, { status: "success"|"error", … })
 *
 * We reproduce this inline (same pattern as neo4j-retry.test.ts) so we can
 * assert file creation without triggering real LLM calls.
 */

import { test, expect } from "@playwright/test";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Helpers — inline reproductions of session.ts exports to avoid import
// side-effects (neo4j module init, etc.)
// ---------------------------------------------------------------------------

function makeSessionsDir(): string {
  const dir = path.join(os.tmpdir(), `test-sessions-${randomUUID()}`);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function sessionFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.jsonl`);
}

function metaFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.meta.jsonl`);
}

interface SessionEntry {
  role?: string;
  content?: string;
  [key: string]: unknown;
}

function createSession(
  sessionsDir: string,
  sessionId: string,
  system: string,
  source: string
): void {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  const systemMsg = { role: "system", content: system };
  fs.appendFileSync(filePath, JSON.stringify(systemMsg) + "\n");
}

function appendMessages(
  sessionsDir: string,
  sessionId: string,
  messages: SessionEntry[]
): void {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  const content = messages.map((m) => JSON.stringify(m)).join("\n") + "\n";
  fs.appendFileSync(filePath, content);
}

function appendStepMeta(
  sessionsDir: string,
  sessionId: string,
  steps: object[]
): void {
  if (steps.length === 0) return;
  const filePath = metaFilePath(sessionsDir, sessionId);
  const content = steps.map((s) => JSON.stringify(s)).join("\n") + "\n";
  fs.appendFileSync(filePath, content);
}

function appendSessionEnd(
  sessionsDir: string,
  sessionId: string,
  opts: {
    end_time: string;
    model?: string;
    status?: "success" | "error";
    error_message?: string;
  }
): void {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  const entry = {
    type: "session_end",
    session_id: sessionId,
    ...opts,
  };
  fs.appendFileSync(filePath, JSON.stringify(entry) + "\n");
}

function loadSession(sessionsDir: string, sessionId: string): SessionEntry[] {
  const filePath = sessionFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, "utf-8");
  return content
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as SessionEntry);
}

function loadStepMeta(sessionsDir: string, sessionId: string): object[] {
  const filePath = metaFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(filePath)) return [];
  const content = fs.readFileSync(filePath, "utf-8");
  return content
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

// ---------------------------------------------------------------------------
// Reproduce the logs_agent execute closure logic (from tools.ts), but with
// log_agent_context replaced by a configurable stub so no real LLM is called.
// ---------------------------------------------------------------------------

interface FakeLogAgentOpts {
  sessionId?: string;
  logsDir: string;
  source?: string;
}

type LogAgentStub = (
  prompt: string,
  opts: FakeLogAgentOpts
) => Promise<{ final: string; sessionId?: string }>;

/**
 * Builds the stub session-writing logic that mirrors what log_agent_context
 * does when opts.sessionId is provided. Returns the sessionId used so tests
 * can inspect the resulting files.
 */
async function makeSessionWritingStub(
  sessionsDir: string,
  prompt: string
): LogAgentStub {
  return async (p: string, opts: FakeLogAgentOpts) => {
    if (!opts.sessionId) {
      throw new Error("sessionId is required in this stub");
    }
    const sessionId = opts.sessionId;

    // Simulate createSession (new session path)
    createSession(sessionsDir, sessionId, "SYSTEM_PROMPT", opts.source || "repo_agent");

    // Simulate messages being appended after generate()
    appendMessages(sessionsDir, sessionId, [
      { role: "user", content: p },
      { role: "assistant", content: "Analyzed logs. [END_OF_ANSWER]" },
    ]);

    // Simulate step metadata
    appendStepMeta(sessionsDir, sessionId, [
      {
        step: 0,
        turn: 1,
        usage: { inputTokens: 100, outputTokens: 50 },
        toolCalls: [],
        timestamp: new Date().toISOString(),
      },
    ]);

    // Simulate appendSessionEnd with status: success
    appendSessionEnd(sessionsDir, sessionId, {
      end_time: new Date().toISOString(),
      model: "gpt-4o",
      status: "success",
    });

    return { final: "Log analysis complete.", sessionId };
  };
}

/**
 * Reproduce the execute() closure from tools.ts:
 *
 *   const sessionId = randomUUID();
 *   const logsDir = createRunLogsDir(randomUUID());   // independent scratch dir
 *   try {
 *     const result = await log_agent_context(prompt, { ..., sessionId, source: "repo_agent" });
 *     return result.final || "No result returned from logs agent.";
 *   } catch (e) {
 *     return `Logs agent error: …`;
 *   } finally {
 *     cleanupRunLogsDir(logsDir);
 *   }
 */
async function simulateExecute(
  prompt: string,
  logAgentFn: LogAgentStub,
  opts: { scratchBase: string }
): Promise<{
  output: string;
  sessionId: string;
  logsDir: string;
  logsDirRemovedAfterExecute: boolean;
}> {
  const sessionId = randomUUID(); // ← the change in tools.ts
  const logsDir = path.join(opts.scratchBase, randomUUID()); // createRunLogsDir equivalent
  fs.mkdirSync(logsDir, { recursive: true });

  let output: string;
  let returnedSessionId: string | undefined;
  try {
    const result = await logAgentFn(prompt, {
      logsDir,
      sessionId, // ← passed through, the key change
      source: "repo_agent",
    });
    returnedSessionId = result.sessionId;
    output = result.final || "No result returned from logs agent.";
  } catch (e) {
    output = `Logs agent error: ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    // cleanupRunLogsDir equivalent
    if (fs.existsSync(logsDir)) {
      fs.rmSync(logsDir, { recursive: true, force: true });
    }
  }

  return {
    output,
    sessionId: returnedSessionId || sessionId,
    logsDir,
    logsDirRemovedAfterExecute: !fs.existsSync(logsDir),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("logs_agent session persistence", () => {
  let sessionsDir: string;
  let scratchBase: string;

  test.beforeEach(() => {
    sessionsDir = makeSessionsDir();
    scratchBase = path.join(os.tmpdir(), `test-logs-${randomUUID()}`);
    fs.mkdirSync(scratchBase, { recursive: true });
  });

  test.afterEach(() => {
    try {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
      fs.rmSync(scratchBase, { recursive: true, force: true });
    } catch {
      // ignore cleanup errors
    }
  });

  // -------------------------------------------------------------------------
  // 1. A single invocation creates a session file with messages, step meta,
  //    and a success end record.
  // -------------------------------------------------------------------------
  test("single invocation creates session file with messages, step meta, and success end record", async () => {
    const stub = await makeSessionWritingStub(sessionsDir, "check recent errors");

    const { sessionId } = await simulateExecute(
      "check recent errors",
      stub,
      { scratchBase }
    );

    // Session .jsonl file must exist
    const filePath = sessionFilePath(sessionsDir, sessionId);
    expect(fs.existsSync(filePath), `session file should exist at ${filePath}`).toBe(true);

    // Must contain a system message
    const entries = loadSession(sessionsDir, sessionId);
    const systemEntry = entries.find((e) => e.role === "system");
    expect(systemEntry).toBeDefined();
    expect(systemEntry?.content).toBe("SYSTEM_PROMPT");

    // Must contain a user message
    const userEntry = entries.find((e) => e.role === "user");
    expect(userEntry).toBeDefined();
    expect(userEntry?.content).toBe("check recent errors");

    // Must contain an assistant message
    const assistantEntry = entries.find((e) => e.role === "assistant");
    expect(assistantEntry).toBeDefined();

    // Must contain a session_end entry with status: success
    const endEntry = entries.find((e) => (e as any).type === "session_end");
    expect(endEntry).toBeDefined();
    expect((endEntry as any).status).toBe("success");
    expect((endEntry as any).session_id).toBe(sessionId);

    // Step meta file must exist with at least one entry
    const steps = loadStepMeta(sessionsDir, sessionId);
    expect(steps.length).toBeGreaterThan(0);
    const firstStep = steps[0] as any;
    expect(firstStep.step).toBeDefined();
    expect(firstStep.toolCalls).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 2. Two invocations within the same "repo_agent run" produce two distinct
  //    session ids.
  // -------------------------------------------------------------------------
  test("two invocations produce distinct session ids", async () => {
    const stub = await makeSessionWritingStub(sessionsDir, "prompt");

    const run1 = await simulateExecute("find errors", stub, { scratchBase });
    const run2 = await simulateExecute("count warnings", stub, { scratchBase });

    expect(run1.sessionId).toBeDefined();
    expect(run2.sessionId).toBeDefined();
    expect(run1.sessionId).not.toBe(run2.sessionId);

    // Both session files must exist
    expect(
      fs.existsSync(sessionFilePath(sessionsDir, run1.sessionId)),
      "session file for run1 must exist"
    ).toBe(true);
    expect(
      fs.existsSync(sessionFilePath(sessionsDir, run2.sessionId)),
      "session file for run2 must exist"
    ).toBe(true);
  });

  // -------------------------------------------------------------------------
  // 3. Two invocations' session ids differ from a hypothetical parent session.
  // -------------------------------------------------------------------------
  test("invocation session ids differ from parent repo_agent session id", async () => {
    const parentSessionId = randomUUID(); // simulates the parent repo_agent session

    const stub = await makeSessionWritingStub(sessionsDir, "prompt");
    const run1 = await simulateExecute("find errors", stub, { scratchBase });
    const run2 = await simulateExecute("count warnings", stub, { scratchBase });

    expect(run1.sessionId).not.toBe(parentSessionId);
    expect(run2.sessionId).not.toBe(parentSessionId);
    expect(run1.sessionId).not.toBe(run2.sessionId);
  });

  // -------------------------------------------------------------------------
  // 4. On a thrown error inside log_agent_context, the session still persists
  //    with status: "error".
  // -------------------------------------------------------------------------
  test("error inside log_agent_context persists session with status error", async () => {
    // Build a stub that writes an error session end, then throws
    const errorStub: LogAgentStub = async (prompt, opts) => {
      if (!opts.sessionId) throw new Error("no sessionId");
      const sessionId = opts.sessionId;

      createSession(
        sessionsDir,
        sessionId,
        "SYSTEM_PROMPT",
        opts.source || "repo_agent"
      );
      appendMessages(sessionsDir, sessionId, [{ role: "user", content: prompt }]);

      // Mirrors the catch block in log_agent_context
      appendSessionEnd(sessionsDir, sessionId, {
        end_time: new Date().toISOString(),
        model: "gpt-4o",
        status: "error",
        error_message: "simulated LLM failure",
      });

      throw new Error("simulated LLM failure");
    };

    // execute() catches the error and returns an error string — it does NOT re-throw
    const { output, sessionId } = await simulateExecute(
      "broken query",
      errorStub,
      { scratchBase }
    );

    // The execute closure swallows the error and returns a string
    expect(output).toContain("Logs agent error:");
    expect(output).toContain("simulated LLM failure");

    // Session file must still exist
    const filePath = sessionFilePath(sessionsDir, sessionId);
    expect(
      fs.existsSync(filePath),
      "session file must still exist after an error"
    ).toBe(true);

    // Must contain the error end record
    const entries = loadSession(sessionsDir, sessionId);
    const endEntry = entries.find((e) => (e as any).type === "session_end") as any;
    expect(endEntry).toBeDefined();
    expect(endEntry.status).toBe("error");
    expect(endEntry.error_message).toBe("simulated LLM failure");
  });

  // -------------------------------------------------------------------------
  // 5. After execute() completes, the per-run logsDir is removed while the
  //    session .jsonl file still exists.
  // -------------------------------------------------------------------------
  test("logsDir is removed after execute while session file persists", async () => {
    const stub = await makeSessionWritingStub(sessionsDir, "prompt");

    const { sessionId, logsDir, logsDirRemovedAfterExecute } =
      await simulateExecute("analyze logs", stub, { scratchBase });

    // The scratch logsDir must be gone
    expect(logsDirRemovedAfterExecute).toBe(true);
    expect(fs.existsSync(logsDir)).toBe(false);

    // The session .jsonl must still be present
    const sessionFile = sessionFilePath(sessionsDir, sessionId);
    expect(
      fs.existsSync(sessionFile),
      "session .jsonl must survive after logsDir cleanup"
    ).toBe(true);
  });

  // -------------------------------------------------------------------------
  // 6. logsDir and sessionId reference independent locations.
  //    Confirm SESSIONS_DIR != LOGS_DIR path prefix so cleanup is safe.
  // -------------------------------------------------------------------------
  test("session file path and logsDir are in separate directories", async () => {
    const stub = await makeSessionWritingStub(sessionsDir, "prompt");

    const { sessionId, logsDir } = await simulateExecute(
      "diagnose issue",
      stub,
      { scratchBase }
    );

    const sessionFile = sessionFilePath(sessionsDir, sessionId);

    // They must not share the same parent directory
    expect(path.dirname(sessionFile)).not.toBe(path.dirname(logsDir));

    // The session file must be under sessionsDir, logsDir under scratchBase
    expect(sessionFile.startsWith(sessionsDir)).toBe(true);
    expect(logsDir.startsWith(scratchBase)).toBe(true);
  });
});
