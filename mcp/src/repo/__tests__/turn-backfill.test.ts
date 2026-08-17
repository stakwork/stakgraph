/**
 * Turn backfill (turnBackfill.ts): the sweep that gives pre-live-emission
 * sessions the same Turn chains the live path now writes.
 *
 * Runs with NO_DB=true, so the db-touching sweep itself must no-op cleanly;
 * what's exercised here is the label resolution (must match what the live
 * path would have stamped) and the no-db guard.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

const tmpSessionsDir = path.join(os.tmpdir(), `test-turn-backfill-${randomUUID()}`);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

test.describe("turn backfill", () => {
  test("no-ops without a db and writes no marker", async () => {
    const { backfillTurns } = await import("../turnBackfill.js");
    const result = await backfillTurns();
    expect(result).toEqual({ scanned: 0, sessions: 0, turns: 0 });
    expect(
      fs.existsSync(path.join(tmpSessionsDir, ".turn-backfill.json")),
    ).toBe(false);
  });

  test("agent label: config source, then -sub- pattern, then fallback", async () => {
    const { backfillAgentLabel } = await import("../turnBackfill.js");
    const { createSession, saveSessionConfig } = await import("../session.js");

    // Top-level session with a config sidecar recording its source — the
    // same value the live emitter stamps into turn_ids.
    const top = createSession(`top-${randomUUID().slice(0, 8)}`);
    saveSessionConfig(top, { source: "repo_agent", temperature: 0 } as any);
    expect(backfillAgentLabel(top)).toBe("repo_agent");

    // Child sessions never get a config sidecar; the id pattern names them.
    expect(backfillAgentLabel(`${top}-sub-1a2b3c4d`)).toBe("graph_sub_agent");

    // No sidecar, no pattern: the live emitter's own fallback.
    expect(backfillAgentLabel(`bare-${randomUUID().slice(0, 8)}`)).toBe("agent");
  });
});
