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

/** Minimal express Response double capturing the JSON body and status. */
function mockRes() {
  const captured: { status: number; body: any } = { status: 200, body: null };
  const res: any = {
    status(code: number) {
      captured.status = code;
      return res;
    },
    json(body: any) {
      captured.body = body;
      return res;
    },
  };
  return { res, captured };
}

test.describe("GET /api/sessions/:id/turns", () => {
  test("cursor semantics via the transcript fallback (NO_DB)", async () => {
    const { createSession, appendMessages, saveSessionConfig } = await import(
      "../session.js"
    );
    const { get_session_turns } = await import("../../benchmark/sessions.js");

    const id = createSession(`turns-ep-${randomUUID().slice(0, 8)}`);
    saveSessionConfig(id, { source: "repo_agent", temperature: 0 } as any);
    appendMessages(id, [
      { role: "user", content: "list the files" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Listing now." },
          { type: "tool-call", toolCallId: "c1", toolName: "bash", input: { command: "ls" } },
        ],
      },
      {
        role: "tool",
        content: [
          { type: "tool-result", toolCallId: "c1", toolName: "bash", output: { type: "text", value: "a.ts b.ts" } },
        ],
      },
      { role: "assistant", content: [{ type: "text", text: "Two files." }] },
    ] as any);

    // History load: everything from the start.
    const full = mockRes();
    await get_session_turns({ params: { id }, query: {} } as any, full.res);
    expect(full.captured.status).toBe(200);
    expect(full.captured.body.turn_count).toBe(5);
    expect(full.captured.body.status).toBe("unknown"); // no graph under NO_DB
    expect(full.captured.body.turns.map((t: any) => t.turn_type)).toEqual([
      "user_input",
      "reasoning",
      "tool_call",
      "tool_result",
      "response",
    ]);
    // Agent label came from the config sidecar, like live emission would.
    expect(full.captured.body.turns[0].turn_id).toBe(`repo_agent-${id}-turn-0`);

    // Poll: only what's after the cursor.
    const delta = mockRes();
    await get_session_turns(
      { params: { id }, query: { after: "2" } } as any,
      delta.res,
    );
    expect(delta.captured.body.turns.map((t: any) => t.order)).toEqual([3, 4]);
  });

  test("404 for a session that exists nowhere", async () => {
    const { get_session_turns } = await import("../../benchmark/sessions.js");
    const { res, captured } = mockRes();
    await get_session_turns(
      { params: { id: "no-such-session" }, query: {} } as any,
      res,
    );
    expect(captured.status).toBe(404);
  });
});

test.describe("turn backfill", () => {
  test("no-ops without a db and writes no marker", async () => {
    const { backfillTurns } = await import("../turnBackfill.js");
    const result = await backfillTurns();
    expect(result).toEqual({ scanned: 0, sessions: 0, turns: 0 });
    expect(
      fs.existsSync(path.join(tmpSessionsDir, ".turn-backfill.json")),
    ).toBe(false);
  });

  test("agent label: agentName, then source, then -sub- pattern, then fallback", async () => {
    const { backfillAgentLabel } = await import("../turnBackfill.js");
    const { createSession, saveSessionConfig } = await import("../session.js");

    // Caller-assigned identity wins — it's what the live emitter labels
    // turn_ids with when the workflow passes agentName.
    const named = createSession(`named-${randomUUID().slice(0, 8)}`);
    saveSessionConfig(named, {
      agentName: "repair-agent-147813394",
      source: "repo_agent",
      temperature: 0,
    } as any);
    expect(backfillAgentLabel(named)).toBe("repair-agent-147813394");

    // Without agentName, the recorded source — the live fallback.
    const top = createSession(`top-${randomUUID().slice(0, 8)}`);
    saveSessionConfig(top, { source: "repo_agent", temperature: 0 } as any);
    expect(backfillAgentLabel(top)).toBe("repo_agent");

    // Child sessions never get a config sidecar; the id pattern names them.
    expect(backfillAgentLabel(`${top}-sub-1a2b3c4d`)).toBe("graph_sub_agent");

    // No sidecar, no pattern: the live emitter's own fallback.
    expect(backfillAgentLabel(`bare-${randomUUID().slice(0, 8)}`)).toBe("agent");
  });
});
