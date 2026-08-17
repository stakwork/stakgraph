/**
 * Session ingest (benchmark/ingest.ts + the buildExternalTurns half of
 * turns.ts): the chain an out-of-process agent posts over HTTP must be
 * indistinguishable from the one the in-process emitter writes.
 *
 * The parity test is the load-bearing one — same session id, same agent
 * label, same content produces the same turn_ids and node_keys either way,
 * which is what makes a hive chain and a local chain the same data.
 *
 * Runs with NO_DB=true: `db` is undefined, so the handlers are exercised for
 * validation and the no-graph guard, and the builder (pure) for everything
 * about turn shape.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

const tmpSessionsDir = path.join(os.tmpdir(), `test-ingest-${randomUUID()}`);
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

test.describe("buildExternalTurns", () => {
  test("numbers from startOrder and chains each turn to its predecessor", async () => {
    const { buildExternalTurns, turnNodeKey } = await import("../turns.js");
    const sid = "sess-abc";

    const turns = buildExternalTurns(sid, "hive", 4, [
      { turn_type: "reasoning", content: "thinking" },
      { turn_type: "tool_call", content: '{"q":"x"}', tool: "search" },
    ]);

    expect(turns.map((t) => t.order)).toEqual([4, 5]);
    expect(turns[0].turn_id).toBe("hive-sess-abc-turn-4");
    expect(turns[0].prev_node_key).toBe(turnNodeKey("hive-sess-abc-turn-3"));
    expect(turns[1].prev_node_key).toBe(turns[0].node_key);
    expect(turns[1].tool).toBe("search");
  });

  test("order 0 opens the chain with no predecessor", async () => {
    const { buildExternalTurns } = await import("../turns.js");
    const turns = buildExternalTurns("s1", "hive", 0, [
      { turn_type: "user_input", content: "go" },
    ]);
    expect(turns[0].prev_node_key).toBe(null);
  });

  test("ingested turns match what the live emitter would have written", async () => {
    const { buildExternalTurns, emitUserTurn, emitStepTurns } = await import(
      "../turns.js"
    );
    const local = `sess-${randomUUID().slice(0, 8)}`;
    const remote = local; // same id, other process

    const localTurns = [
      ...emitUserTurn(local, "hive", { role: "user", content: "fix it" }),
      ...emitStepTurns(local, "hive", [
        { type: "text", text: "looking" },
        { type: "tool-call", toolCallId: "c1", toolName: "search", input: { q: "x" } },
      ]),
    ];
    const ingested = buildExternalTurns(remote, "hive", 0, [
      { turn_type: "user_input", content: "fix it" },
      { turn_type: "reasoning", content: "looking" },
      { turn_type: "tool_call", content: '{"q":"x"}', tool: "search", tool_call_id: "c1" },
    ]);

    expect(ingested.map((t) => t.node_key)).toEqual(
      localTurns.map((t) => t.node_key),
    );
    expect(ingested.map((t) => t.turn_id)).toEqual(
      localTurns.map((t) => t.turn_id),
    );
    expect(ingested.map((t) => t.turn_type)).toEqual(
      localTurns.map((t) => t.turn_type),
    );
  });

  test("tool_result content is wrapped and truncated like the live path", async () => {
    const { buildExternalTurns } = await import("../turns.js");
    const [wrapped] = buildExternalTurns("s1", "hive", 0, [
      { turn_type: "tool_result", content: "ok", tool: "search" },
    ]);
    expect(wrapped.content).toBe('{"type":"text","value":"ok"}');

    const [long] = buildExternalTurns("s1", "hive", 0, [
      { turn_type: "tool_result", content: "x".repeat(500), tool: "search" },
    ]);
    expect(long.content.length).toBe(103); // 100 chars + "..."
    expect(long.content.endsWith("...")).toBe(true);
  });

  test("concepts without any identifier are dropped", async () => {
    const { buildExternalTurns } = await import("../turns.js");
    const [turn] = buildExternalTurns("s1", "hive", 0, [
      {
        turn_type: "tool_result",
        content: "ok",
        concepts: [{ ref_id: "r1" }, { id: "c2" }, {}, { ref_id: null, id: null }],
      },
    ]);
    expect(turn.concepts).toEqual([
      { ref_id: "r1", id: null, repo: null },
      { ref_id: null, id: "c2", repo: null },
    ]);
  });

  test("a concept's repo rides along, so a bare gitree id can still resolve", async () => {
    const { buildExternalTurns } = await import("../turns.js");
    const [turn] = buildExternalTurns("s1", "hive", 0, [
      {
        turn_type: "tool_result",
        content: "ok",
        concepts: [{ id: "auth-flow", repo: "stakwork/hive" }],
      },
    ]);
    expect(turn.concepts).toEqual([
      { ref_id: null, id: "auth-flow", repo: "stakwork/hive" },
    ]);
  });
});

test.describe("agentFromTurnId", () => {
  test("recovers the label so a batch without `agent` continues the chain", async () => {
    const { agentFromTurnId } = await import("../turns.js");
    expect(agentFromTurnId("hive-sess-abc-turn-7", "sess-abc")).toBe("hive");
    // Session ids containing the marker text still resolve off the last one.
    expect(agentFromTurnId("hive-a-turn-1-turn-2", "a-turn-1")).toBe("hive");
    expect(agentFromTurnId("nonsense", "sess-abc")).toBe(null);
  });
});

test.describe("ingest endpoints", () => {
  test("POST /api/sessions rejects a missing or path-like session_id", async () => {
    const { create_session } = await import("../../benchmark/ingest.js");

    for (const session_id of [undefined, "", "../etc", "a/b"]) {
      const { res, captured } = mockRes();
      await create_session({ body: { session_id } } as any, res);
      expect(captured.status).toBe(400);
    }
  });

  test("POST /sessions/:id/turns validates the batch before touching the graph", async () => {
    const { append_turns } = await import("../../benchmark/ingest.js");
    const req = (body: any) => ({ params: { id: "s1" }, body }) as any;

    const empty = mockRes();
    await append_turns(req({ turns: [] }), empty.res);
    expect(empty.captured.status).toBe(400);

    const badType = mockRes();
    await append_turns(
      req({ turns: [{ turn_type: "nope", content: "x" }] }),
      badType.res,
    );
    expect(badType.captured.status).toBe(400);
    expect(String(badType.captured.body.error)).toContain("turns[0].turn_type");

    const tooMany = mockRes();
    await append_turns(
      req({
        turns: Array.from({ length: 501 }, () => ({
          turn_type: "reasoning",
          content: "x",
        })),
      }),
      tooMany.res,
    );
    expect(tooMany.captured.status).toBe(400);

    const negative = mockRes();
    await append_turns(
      req({ start_order: -1, turns: [{ turn_type: "reasoning", content: "x" }] }),
      negative.res,
    );
    expect(negative.captured.status).toBe(400);
  });

  test("POST /sessions/:id/end rejects an unknown status", async () => {
    const { end_session } = await import("../../benchmark/ingest.js");
    const { res, captured } = mockRes();
    await end_session(
      { params: { id: "s1" }, body: { status: "finished" } } as any,
      res,
    );
    expect(captured.status).toBe(400);
  });

  test("a valid request reports the graph as unavailable rather than silently dropping", async () => {
    const { append_turns, end_session, link_session_concepts, create_session } =
      await import("../../benchmark/ingest.js");

    const create = mockRes();
    await create_session({ body: { session_id: "s1" } } as any, create.res);
    expect(create.captured.status).toBe(503);

    const turns = mockRes();
    await append_turns(
      {
        params: { id: "s1" },
        body: { turns: [{ turn_type: "reasoning", content: "x" }] },
      } as any,
      turns.res,
    );
    expect(turns.captured.status).toBe(503);

    const end = mockRes();
    await end_session({ params: { id: "s1" }, body: {} } as any, end.res);
    expect(end.captured.status).toBe(503);

    const concepts = mockRes();
    await link_session_concepts(
      { params: { id: "s1" }, body: { concepts: [] } } as any,
      concepts.res,
    );
    expect(concepts.captured.status).toBe(503);
  });
});
