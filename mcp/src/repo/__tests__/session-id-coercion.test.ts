/**
 * Regression tests for numeric session ids reporting zero token usage.
 *
 * External clients supply their own session ids. When a client sent one as a
 * JSON number ({"sessionId": 151395375}), the `as string` cast in
 * repo/index.ts was compile-time only, so a JS number flowed all the way into
 * `MERGE (n:AgentSession {node_key: $session_id})` and Neo4j stored node_key
 * as a Float. Cypher equality is type-strict, so the detail endpoint's lookup
 * with the string id from the URL path never matched. `buildFullSession` then
 * fell through to its file-only branch, which hardcoded token_usage to zeros —
 * while the list endpoint (which matches on label, not key) kept reporting the
 * real numbers. Every session with a numeric id showed 0 tokens on
 * GET /api/sessions/:id.
 *
 * Two invariants are covered here:
 *   1. Session ids are coerced to strings before they reach the Neo4j key, so
 *      no new node can be written with a non-string node_key.
 *   2. When the node lookup misses for any reason, the detail endpoint recovers
 *      usage from the `.meta.jsonl` sidecar instead of reporting zeros.
 *
 * These run with NO_DB=true, so `db` is undefined and the detail endpoint takes
 * exactly the file-only branch that used to zero everything out.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// session.ts reads SESSIONS_DIR once at module load, so it must be set before
// the dynamic import below and stay fixed for the lifetime of the process.
const tmpSessionsDir = path.join(
  os.tmpdir(),
  `test-session-id-coercion-${randomUUID()}`,
);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

const NUMERIC_ID = 151395375;

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

test.describe("numeric session ids", () => {
  test("createSession coerces a numeric id to a string", async () => {
    const { createSession, sessionExists } = await import("../session.js");

    // Cast mirrors reality: the type says string, the JSON body carried a number.
    const returned = createSession(NUMERIC_ID as unknown as string);

    expect(typeof returned).toBe("string");
    expect(returned).toBe("151395375");
    // The string id — the one the URL path will carry — must resolve.
    expect(sessionExists("151395375")).toBe(true);
  });

  test("appendSessionEnd accepts a numeric id without losing the session", async () => {
    const { createSession, appendSessionEnd } = await import("../session.js");

    const id = 151395376;
    createSession(id as unknown as string);

    // sessionMeta is keyed by the coerced string. If appendSessionEnd skipped
    // the same coercion, this lookup would miss and it would bail out with
    // "was never registered via createSession()" — writing no node at all.
    const errors: string[] = [];
    const originalError = console.error;
    console.error = (...args: unknown[]) => errors.push(args.join(" "));
    try {
      await appendSessionEnd(id as unknown as string, {
        end_time: new Date().toISOString(),
        model: "claude-sonnet-5",
        provider: "anthropic",
        duration_ms: 1234,
        status: "success",
        token_usage: {
          input: 10,
          cache_read: 20,
          cache_write: 30,
          output: 40,
          total: 100,
        } as any,
      });
    } finally {
      console.error = originalError;
    }

    expect(errors.join("\n")).not.toContain("never registered");
  });

  test("detail endpoint recovers usage from step meta when the node is missing", async () => {
    const { createSession, appendMessages, appendStepMeta } = await import(
      "../session.js"
    );
    const { get_session } = await import("../../benchmark/sessions.js");

    const id = createSession(151395377 as unknown as string);
    appendMessages(id, [{ role: "user", content: "hi" }] as any);

    // Two steps of provider-reported usage in the sidecar. Under NO_DB there is
    // no AgentSession node, so this is the only surviving record of the spend.
    appendStepMeta(id, [
      {
        step: 0,
        turn: 2,
        usage: {
          input: 2,
          cache_read: 0,
          cache_write: 33886,
          output: 161,
          total: 34049,
        },
        cumulativeInput: 33888,
        cumulativeOutput: 161,
        toolCalls: ["bash"],
        timestamp: "2026-08-07T15:24:49.432Z",
      },
      {
        step: 1,
        turn: 2,
        usage: {
          input: 3,
          cache_read: 1000,
          cache_write: 500,
          output: 200,
          total: 1703,
        },
        cumulativeInput: 35391,
        cumulativeOutput: 361,
        toolCalls: [],
        timestamp: "2026-08-07T15:25:49.432Z",
      },
    ] as any);

    const { res, captured } = mockRes();
    await get_session({ params: { id }, query: {} } as any, res);

    expect(captured.status).toBe(200);
    const usage = captured.body.token_usage;

    // The regression: this whole object used to be hardcoded zeros.
    expect(usage.total).toBe(34049 + 1703);
    expect(usage.input).toBe(5);
    expect(usage.cache_read).toBe(1000);
    expect(usage.cache_write).toBe(34386);
    expect(usage.output).toBe(361);

    // Duration is derived from the step timestamps (60s apart), not left at 0.
    expect(captured.body.duration_ms).toBe(60000);
  });

  test("detail and list endpoints agree on usage for the same session", async () => {
    const { createSession, appendMessages, appendStepMeta } = await import(
      "../session.js"
    );
    const { get_session, list_sessions } = await import(
      "../../benchmark/sessions.js"
    );

    const id = createSession(151395378 as unknown as string);
    appendMessages(id, [{ role: "user", content: "hi" }] as any);
    appendStepMeta(id, [
      {
        step: 0,
        turn: 2,
        usage: {
          input: 7,
          cache_read: 900,
          cache_write: 80,
          output: 13,
          total: 1000,
        },
        cumulativeInput: 987,
        cumulativeOutput: 13,
        toolCalls: [],
        timestamp: "2026-08-07T15:24:49.432Z",
      },
    ] as any);

    const detail = mockRes();
    await get_session({ params: { id }, query: {} } as any, detail.res);

    const list = mockRes();
    await list_sessions({ query: {} } as any, list.res);
    const row = (list.captured.body as any[]).find((r) => r.id === id);

    // The original bug was precisely this disagreement: list reported the real
    // spend while detail reported zeros for the very same session.
    expect(row).toBeTruthy();
    expect(detail.captured.body.token_usage).toEqual(row.token_usage);
    expect(detail.captured.body.token_usage.total).toBe(1000);
  });
});
