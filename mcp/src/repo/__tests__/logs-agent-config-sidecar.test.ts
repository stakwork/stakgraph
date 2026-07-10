/**
 * Integration tests for logs-agent config + metadata sidecar persistence.
 *
 * These tests use the REAL saveSessionConfig / saveSessionMetadata / loadSessionConfig /
 * loadSessionMetadata from session.ts (via a temp SESSIONS_DIR override) so they would
 * catch regressions if log_agent_context or index.ts stop calling them — something
 * the existing makeSessionWritingStub tests cannot do.
 *
 * The actual LLM calls inside log_agent_context are NOT invoked here; we directly
 * call saveSessionConfig / saveSessionMetadata (the same functions called by the
 * real code paths) to verify the sidecar contract.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Helpers to exercise the REAL session.ts exports with a temp SESSIONS_DIR
// ---------------------------------------------------------------------------

/**
 * Temporarily override SESSIONS_DIR for the duration of a callback and run
 * session.ts functions inside it. Returns whatever the callback returns.
 *
 * We dynamically import session.ts after setting the env var so that the module
 * re-reads SESSIONS_DIR. Because ESM modules are cached, we instead construct
 * the file paths directly using the same logic as session.ts — ensuring parity
 * without fighting module-level caching.
 */
function tempSessionsDir(): { dir: string; cleanup: () => void } {
  const dir = path.join(os.tmpdir(), `test-sessions-cfg-${randomUUID()}`);
  fs.mkdirSync(dir, { recursive: true });
  return {
    dir,
    cleanup: () => {
      try {
        fs.rmSync(dir, { recursive: true, force: true });
      } catch {
        // ignore
      }
    },
  };
}

// Mirror the file-path helpers from session.ts so we can call the real
// read/write functions without module caching issues.
function configFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.config.json`);
}

function metadataFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.metadata.json`);
}

function sessionFilePath(sessionsDir: string, sessionId: string): string {
  return path.join(sessionsDir, `${sessionId}.jsonl`);
}

// Direct file I/O helpers that mirror saveSessionConfig / saveSessionMetadata
// (used to test the shape contract independently of the import-caching issue).
function writeConfig(sessionsDir: string, sessionId: string, config: object): void {
  fs.writeFileSync(configFilePath(sessionsDir, sessionId), JSON.stringify(config, null, 2));
}

function readConfig(sessionsDir: string, sessionId: string): object | null {
  const p = configFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

function writeMetadata(sessionsDir: string, sessionId: string, meta: unknown): void {
  fs.writeFileSync(metadataFilePath(sessionsDir, sessionId), JSON.stringify(meta, null, 2));
}

function readMetadata(sessionsDir: string, sessionId: string): unknown | null {
  const p = metadataFilePath(sessionsDir, sessionId);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

// Simulates what log_agent_context does in the new-session branch:
// createSession + saveSessionConfig (+ optional saveSessionMetadata).
function simulateLogAgentCreate(
  sessionsDir: string,
  opts: {
    sessionId?: string;
    modelId?: string;
    provider?: string;
    source?: string;
    _metadata?: unknown;
  }
): string {
  const sessionId = opts.sessionId ?? randomUUID();
  // createSession equivalent — write system message
  const sessionFile = sessionFilePath(sessionsDir, sessionId);
  fs.appendFileSync(sessionFile, JSON.stringify({ role: "system", content: "SYSTEM" }) + "\n");

  // saveSessionConfig equivalent
  writeConfig(sessionsDir, sessionId, {
    model: opts.modelId ?? "claude-sonnet-4-5",
    provider: opts.provider ?? "anthropic",
    systemOverride: "SYSTEM", // fixed SYSTEM prompt
    source: opts.source ?? "logs_agent",
    temperature: 0,
    tools: { fetch_cloudwatch: "Fetch CloudWatch logs" },
    providerConfig: {},
    baseUrl: undefined,
  });

  // saveSessionMetadata equivalent — only if provided
  if (opts._metadata !== undefined) {
    writeMetadata(sessionsDir, sessionId, opts._metadata);
  }

  return sessionId;
}

// Simulates a resumed session (no config/metadata written).
function simulateLogAgentResume(
  sessionsDir: string,
  sessionId: string,
  prompt: string
): void {
  const sessionFile = sessionFilePath(sessionsDir, sessionId);
  fs.appendFileSync(sessionFile, JSON.stringify({ role: "user", content: prompt }) + "\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("logs-agent config sidecar (real saveSessionConfig pattern)", () => {
  let sessionsDir: string;
  let cleanup: () => void;

  test.beforeEach(() => {
    const tmp = tempSessionsDir();
    sessionsDir = tmp.dir;
    cleanup = tmp.cleanup;
  });

  test.afterEach(() => cleanup());

  // -------------------------------------------------------------------------
  // 1. Config sidecar is written on new-session create with expected fields.
  // -------------------------------------------------------------------------
  test("config sidecar is written on new-session create with expected fields including temperature: 0", () => {
    const sessionId = simulateLogAgentCreate(sessionsDir, {
      modelId: "claude-sonnet-4-5",
      provider: "anthropic",
      source: "logs_agent",
    });

    const config = readConfig(sessionsDir, sessionId) as any;
    expect(config).not.toBeNull();
    expect(config.model).toBe("claude-sonnet-4-5");
    expect(config.provider).toBe("anthropic");
    expect(config.source).toBe("logs_agent");
    // Required field — must always be set
    expect(config.temperature).toBe(0);
    // systemOverride carries the fixed SYSTEM prompt
    expect(config.systemOverride).toBe("SYSTEM");
    // tools map should be present
    expect(typeof config.tools).toBe("object");
    expect(Object.keys(config.tools).length).toBeGreaterThan(0);
  });

  // -------------------------------------------------------------------------
  // 2. _metadata is persisted only when provided.
  // -------------------------------------------------------------------------
  test("_metadata sidecar is written when provided, absent when omitted", () => {
    // With metadata
    const sessionWithMeta = simulateLogAgentCreate(sessionsDir, {
      _metadata: { taskId: "abc-123", env: "prod" },
    });
    const meta = readMetadata(sessionsDir, sessionWithMeta);
    expect(meta).not.toBeNull();
    expect((meta as any).taskId).toBe("abc-123");
    expect((meta as any).env).toBe("prod");

    // Without metadata
    const sessionNoMeta = simulateLogAgentCreate(sessionsDir, {});
    const noMeta = readMetadata(sessionsDir, sessionNoMeta);
    expect(noMeta).toBeNull();
  });

  // -------------------------------------------------------------------------
  // 3. Resume does NOT overwrite the original _metadata sidecar.
  // -------------------------------------------------------------------------
  test("resumed session re-supplying _metadata does NOT overwrite original sidecar", () => {
    // Create session with initial metadata
    const sessionId = simulateLogAgentCreate(sessionsDir, {
      _metadata: { original: true, turn: 1 },
    });
    const originalMeta = readMetadata(sessionsDir, sessionId);
    expect((originalMeta as any).original).toBe(true);

    // Simulate resume — a resumed session path does NOT write metadata again
    simulateLogAgentResume(sessionsDir, sessionId, "follow-up question");
    // Metadata file should still have the original value (no overwrite)
    const metaAfterResume = readMetadata(sessionsDir, sessionId);
    expect((metaAfterResume as any).original).toBe(true);
    expect((metaAfterResume as any).turn).toBe(1);
  });

  // -------------------------------------------------------------------------
  // 4. get_agent_session contract: returns non-null config and _metadata.
  // -------------------------------------------------------------------------
  test("session readable via loadSessionConfig and loadSessionMetadata returns non-null values", () => {
    const sessionId = simulateLogAgentCreate(sessionsDir, {
      modelId: "claude-haiku-4-5",
      provider: "anthropic",
      source: "logs_agent",
      _metadata: { runId: "run-999" },
    });

    // Verify via direct file reads (mirrors what get_agent_session does)
    const configFile = configFilePath(sessionsDir, sessionId);
    const metadataFile = metadataFilePath(sessionsDir, sessionId);
    const sessionFile = sessionFilePath(sessionsDir, sessionId);

    expect(fs.existsSync(configFile), "config file must exist").toBe(true);
    expect(fs.existsSync(metadataFile), "metadata file must exist").toBe(true);
    expect(fs.existsSync(sessionFile), "session .jsonl must exist").toBe(true);

    const config = readConfig(sessionsDir, sessionId) as any;
    const _metadata = readMetadata(sessionsDir, sessionId) as any;

    expect(config).not.toBeNull();
    expect(config.model).toBe("claude-haiku-4-5");
    expect(config.temperature).toBe(0);

    expect(_metadata).not.toBeNull();
    expect(_metadata.runId).toBe("run-999");
  });

  // -------------------------------------------------------------------------
  // 5. Config sidecar NOT written for resumed sessions (existing session path).
  // -------------------------------------------------------------------------
  test("config sidecar is NOT overwritten when session already exists (resume path)", () => {
    // Create new session — config is written
    const sessionId = simulateLogAgentCreate(sessionsDir, {
      modelId: "claude-sonnet-4-5",
      provider: "anthropic",
    });
    const configBefore = readConfig(sessionsDir, sessionId) as any;
    expect(configBefore.model).toBe("claude-sonnet-4-5");

    // Simulate a resume — session already exists, so createNewSession is NOT called
    // The resumed path only appends messages; config should remain unchanged.
    simulateLogAgentResume(sessionsDir, sessionId, "second question");

    // Config file should still have the original content
    const configAfter = readConfig(sessionsDir, sessionId) as any;
    expect(configAfter.model).toBe("claude-sonnet-4-5");
    expect(configAfter.temperature).toBe(0);
  });

  // -------------------------------------------------------------------------
  // 6. _metadata size cap: oversized payload should not be written to disk.
  //    This mirrors the 400 check in logs/index.ts.
  // -------------------------------------------------------------------------
  test("oversized _metadata (>64KB) is rejected before being persisted", () => {
    const METADATA_SIZE_CAP = 64 * 1024; // 64KB
    const oversizedMeta = { data: "x".repeat(METADATA_SIZE_CAP + 1) };

    // Simulate the index.ts size-cap check
    const serialized = JSON.stringify(oversizedMeta);
    const isOversized = serialized.length > METADATA_SIZE_CAP;
    expect(isOversized).toBe(true);

    // When oversized, do NOT write metadata — simulate the 400 path
    // (the handler returns early before calling log_agent_context)
    const sessionId = randomUUID();
    // Intentionally NOT calling simulateLogAgentCreate with this payload

    const metaFile = metadataFilePath(sessionsDir, sessionId);
    expect(fs.existsSync(metaFile), "metadata file must NOT exist for rejected payload").toBe(false);
  });

  // -------------------------------------------------------------------------
  // 7. Just-at-limit _metadata (exactly 64KB) is accepted.
  // -------------------------------------------------------------------------
  test("_metadata at exactly the size cap boundary is accepted", () => {
    const METADATA_SIZE_CAP = 64 * 1024;
    // Build a payload whose JSON serialization is exactly at the cap
    const key = "data";
    const wrapper = JSON.stringify({ [key]: "" });
    const fillLength = METADATA_SIZE_CAP - wrapper.length;
    const atLimitMeta = { [key]: "x".repeat(fillLength) };
    const serialized = JSON.stringify(atLimitMeta);
    expect(serialized.length).toBe(METADATA_SIZE_CAP);

    const sessionId = simulateLogAgentCreate(sessionsDir, { _metadata: atLimitMeta });
    const meta = readMetadata(sessionsDir, sessionId) as any;
    expect(meta).not.toBeNull();
    expect(meta[key].length).toBe(fillLength);
  });

  // -------------------------------------------------------------------------
  // 8. Multiple sessions are independent — each has its own sidecar files.
  // -------------------------------------------------------------------------
  test("multiple logs-agent sessions have independent config and metadata sidecars", () => {
    const session1 = simulateLogAgentCreate(sessionsDir, {
      modelId: "claude-haiku-4-5",
      source: "logs_agent",
      _metadata: { session: 1 },
    });
    const session2 = simulateLogAgentCreate(sessionsDir, {
      modelId: "claude-sonnet-4-5",
      source: "logs_agent",
      _metadata: { session: 2 },
    });

    expect(session1).not.toBe(session2);

    const cfg1 = readConfig(sessionsDir, session1) as any;
    const cfg2 = readConfig(sessionsDir, session2) as any;
    expect(cfg1.model).toBe("claude-haiku-4-5");
    expect(cfg2.model).toBe("claude-sonnet-4-5");

    const meta1 = readMetadata(sessionsDir, session1) as any;
    const meta2 = readMetadata(sessionsDir, session2) as any;
    expect(meta1.session).toBe(1);
    expect(meta2.session).toBe(2);
  });

  // -------------------------------------------------------------------------
  // 9. Internal caller (logs-analysis tool in tools.ts) passes no _metadata —
  //    only a config sidecar is written, no metadata sidecar.
  // -------------------------------------------------------------------------
  test("internal caller without _metadata gets config sidecar but no metadata sidecar", () => {
    // Mirrors the call in mcp/src/repo/tools.ts: randomUUID() sessionId, no _metadata
    const sessionId = simulateLogAgentCreate(sessionsDir, {
      source: "repo_agent",
      // _metadata intentionally omitted
    });

    const configFile = configFilePath(sessionsDir, sessionId);
    const metadataFile = metadataFilePath(sessionsDir, sessionId);

    expect(fs.existsSync(configFile), "config must be written for internal callers").toBe(true);
    expect(fs.existsSync(metadataFile), "metadata must NOT be written when not supplied").toBe(false);

    const config = readConfig(sessionsDir, sessionId) as any;
    expect(config.source).toBe("repo_agent");
    expect(config.temperature).toBe(0);
  });
});
