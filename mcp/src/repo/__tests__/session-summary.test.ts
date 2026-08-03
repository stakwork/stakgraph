/**
 * Unit tests for the session-summary sidecar added alongside the opt-in
 * `summarize` flag on /repo/agent.
 *
 * The summary is written to `<sessionId>.summary.json` next to the session
 * JSONL and surfaced via GET /repo/agent/session?summarize=true. These tests
 * cover the persistence layer directly: round-trip, the two null paths, the
 * graph write being a no-op without a database, and sidecar cleanup on delete
 * (a leaked summary file would outlive the session it describes).
 *
 * session.ts reads SESSIONS_DIR at module load, so it is imported dynamically
 * after the env var is set — the same pattern as graph-agent-session.test.ts.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

const tmpSessionsDir = path.join(os.tmpdir(), `test-session-summary-${randomUUID()}`);

type SessionModule = typeof import("../session.js");
let sessionModule: SessionModule;

async function getSessionModule(): Promise<SessionModule> {
  process.env.SESSIONS_DIR = tmpSessionsDir;
  if (!sessionModule) {
    sessionModule = await import("../session.js");
  }
  return sessionModule;
}

function summaryPath(sessionId: string): string {
  return path.join(tmpSessionsDir, `${sessionId}.summary.json`);
}

test.describe("session summary sidecar", () => {
  test.beforeEach(() => {
    fs.mkdirSync(tmpSessionsDir, { recursive: true });
  });

  test.afterEach(() => {
    try {
      fs.rmSync(tmpSessionsDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
  });

  test("saveSessionSummary then loadSessionSummary round-trips the summary text", async () => {
    const { saveSessionSummary, loadSessionSummary } = await getSessionModule();
    const sessionId = randomUUID();
    const summary = "## What happened\nTraced a regression.\n\n## Open threads\nNone.";

    saveSessionSummary(sessionId, { summary });

    expect(fs.existsSync(summaryPath(sessionId))).toBe(true);
    expect(loadSessionSummary(sessionId)?.summary).toBe(summary);
  });

  test("loadSessionSummary returns null when no summary was ever written", async () => {
    const { loadSessionSummary } = await getSessionModule();

    expect(loadSessionSummary(randomUUID())).toBe(null);
  });

  test("loadSessionSummary returns null rather than throwing on a corrupt sidecar", async () => {
    const { loadSessionSummary } = await getSessionModule();
    const sessionId = randomUUID();

    fs.writeFileSync(summaryPath(sessionId), "{ not json");

    // A truncated write must degrade to "no summary", not break the session read.
    expect(loadSessionSummary(sessionId)).toBe(null);
  });

  test("saveSummaryToGraph writes only to the graph, and is a no-op without a database", async () => {
    const { saveSummaryToGraph } = await getSessionModule();
    const sessionId = randomUUID();

    // Runs under NO_DB=true, so `db` is undefined and the optional chain short
    // -circuits before `.catch`. A throw here fails the test directly; testkit's
    // `.not.toThrow()` is double-negated and cannot express this.
    saveSummaryToGraph(sessionId, "a summary");

    // The graph write is independent of the file sidecar — it must not create one.
    expect(fs.existsSync(summaryPath(sessionId))).toBe(false);
  });

  test("deleteSession removes the summary sidecar along with the session", async () => {
    const { saveSessionSummary, loadSessionSummary, deleteSession } = await getSessionModule();
    const sessionId = randomUUID();

    saveSessionSummary(sessionId, { summary: "to be deleted" });
    expect(fs.existsSync(summaryPath(sessionId))).toBe(true);

    deleteSession(sessionId);

    expect(fs.existsSync(summaryPath(sessionId))).toBe(false);
    expect(loadSessionSummary(sessionId)).toBe(null);
  });
});
