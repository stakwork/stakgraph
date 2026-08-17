/**
 * Live Turn emission (turns.ts): the node/edge shape must match what the
 * post-hoc build_trace_edges workflow produced in prod, byte-for-byte where
 * it matters:
 *
 *   - turn_id  = "{agent}-{session}-turn-{order}"
 *   - node_key = "turn-" + turn_id stripped of non-alphanumerics
 *     (prod: "repair-agent-147813394-turn-0" -> "turn-repairagent147813394turn0")
 *   - one Turn per content part; tool results truncated to 100 chars + "..."
 *   - orders monotonic, each turn chained to its predecessor's node_key
 *
 * Runs with NO_DB=true: `db` is undefined, so emitters build and return the
 * turns without writing — exactly the surface these tests need.
 */

import { test, expect } from "../../testkit.js";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// turns.ts reads SESSIONS_DIR once at module load, so set it before import.
const tmpSessionsDir = path.join(os.tmpdir(), `test-turns-${randomUUID()}`);
fs.mkdirSync(tmpSessionsDir, { recursive: true });
process.env.SESSIONS_DIR = tmpSessionsDir;
process.env.NO_DB = "true";

function sidecar(sessionId: string): any {
  return JSON.parse(
    fs.readFileSync(path.join(tmpSessionsDir, `${sessionId}.turns.json`), "utf-8"),
  );
}

test.describe("turn emission", () => {
  test("node_key sanitization matches the prod backfill scheme", async () => {
    const { turnNodeKey } = await import("../turns.js");
    expect(turnNodeKey("repair-agent-147813394-turn-0")).toBe(
      "turn-repairagent147813394turn0",
    );
    expect(turnNodeKey("plan-agent-cmr2lsldx0001jp041nlg1bp8-turn-3")).toBe(
      "turn-planagentcmr2lsldx0001jp041nlg1bp8turn3",
    );
  });

  test("user turn starts the chain at order 0 with no predecessor", async () => {
    const { emitUserTurn } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;

    const turns = emitUserTurn(sid, "hive", { role: "user", content: "fix the bug" });

    expect(turns.length).toBe(1);
    expect(turns[0].turn_id).toBe(`hive-${sid}-turn-0`);
    expect(turns[0].order).toBe(0);
    expect(turns[0].prev_node_key).toBe(null);
    expect(turns[0].turn_type).toBe("user_input");
    expect(turns[0].content).toBe("fix the bug");
    expect(turns[0].tool).toBe(null);
    expect(sidecar(sid)).toEqual({
      agent: "hive",
      next_order: 1,
      last_reasoning_order: null,
    });
  });

  test("step content becomes reasoning/tool_call/tool_result turns, results last", async () => {
    const { emitUserTurn, emitStepTurns, turnNodeKey } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;
    emitUserTurn(sid, "hive", { role: "user", content: "go" });

    // Interleaved as the SDK yields them: call, result, text. Emission must
    // reorder to assistant-parts-first to match the persisted message layout.
    const turns = emitStepTurns(sid, "hive", [
      {
        type: "tool-call",
        toolCallId: "c1",
        toolName: "bash",
        input: { command: "ls" },
      },
      {
        type: "tool-result",
        toolCallId: "c1",
        toolName: "bash",
        output: { type: "text", value: "x".repeat(300) },
      },
      { type: "text", text: "I'll list files first." },
    ]);

    expect(turns.map((t) => t.turn_type)).toEqual([
      "tool_call",
      "reasoning",
      "tool_result",
    ]);
    expect(turns.map((t) => t.order)).toEqual([1, 2, 3]);
    // Chain: every turn points at the node_key of order-1.
    expect(turns[0].prev_node_key).toBe(turnNodeKey(`hive-${sid}-turn-0`));
    expect(turns[2].prev_node_key).toBe(turnNodeKey(`hive-${sid}-turn-2`));

    expect(turns[0].tool).toBe("bash");
    expect(turns[0].content).toBe('{"command":"ls"}');
    // Call/result pairing survives reordering via tool_call_id.
    expect(turns[0].tool_call_id).toBe("c1");
    expect(turns[2].tool_call_id).toBe("c1");
    expect(turns[0].timestamp).toBeGreaterThan(0);
    expect(turns[1].tool).toBe(null);
    expect(turns[1].tool_call_id).toBe(null);
    // Tool result: output wrapper stringified, truncated at 100 + "...".
    expect(turns[2].tool).toBe("bash");
    expect(turns[2].content.length).toBe(103);
    expect(turns[2].content.endsWith("...")).toBe(true);
    expect(turns[2].content.startsWith('{"type":"text","value":"xxx')).toBe(true);
  });

  test("empty text and thinking parts emit nothing", async () => {
    const { emitStepTurns } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;
    const turns = emitStepTurns(sid, "hive", [
      { type: "text", text: "" },
      { type: "reasoning", text: "internal thinking" },
    ]);
    expect(turns).toEqual([]);
  });

  test("a Concept-bearing tool result carries the concept link", async () => {
    const { emitStepTurns } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;

    const conceptNode = {
      node_type: "Concept",
      ref_id: "abc-123",
      name: "auth-flow",
      properties: { id: "gitree-9" },
    };
    const turns = emitStepTurns(sid, "hive", [
      { type: "tool-call", toolCallId: "c2", toolName: "graph_get", input: { ref_id: "abc-123" } },
      {
        type: "tool-result",
        toolCallId: "c2",
        toolName: "graph_get",
        output: { type: "json", value: conceptNode },
      },
    ]);

    const result = turns.find((t) => t.turn_type === "tool_result")!;
    expect(result.concepts).toEqual([{ ref_id: "abc-123", id: "gitree-9" }]);
    // Non-concept turns carry none.
    expect(turns.find((t) => t.turn_type === "tool_call")!.concepts).toEqual([]);
  });

  test("finalize retypes the LAST reasoning turn and is one-shot per run", async () => {
    const { emitUserTurn, emitStepTurns, finalizeTurns, turnNodeKey } =
      await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;
    emitUserTurn(sid, "hive", { role: "user", content: "go" });
    emitStepTurns(sid, "hive", [{ type: "text", text: "thinking..." }]); // order 1
    emitStepTurns(sid, "hive", [{ type: "text", text: "the answer" }]); // order 2

    expect(finalizeTurns(sid)).toBe(turnNodeKey(`hive-${sid}-turn-2`));
    // Cleared: a second finalize (or the next run's, before new reasoning)
    // has nothing to retype.
    expect(finalizeTurns(sid)).toBe(null);
    expect(sidecar(sid).last_reasoning_order).toBe(null);
  });

  test("a resumed session continues its chain from the sidecar", async () => {
    const { emitUserTurn, turnNodeKey } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;
    // Simulate a prior run in another process: sidecar exists, no memory state.
    fs.writeFileSync(
      path.join(tmpSessionsDir, `${sid}.turns.json`),
      JSON.stringify({ agent: "hive", next_order: 7, last_reasoning_order: null }),
    );

    // The agent hint is ignored on resume — turn_ids must stay stable.
    const turns = emitUserTurn(sid, "other-label", {
      role: "user",
      content: "follow-up",
    });

    expect(turns[0].turn_id).toBe(`hive-${sid}-turn-7`);
    expect(turns[0].order).toBe(7);
    expect(turns[0].prev_node_key).toBe(turnNodeKey(`hive-${sid}-turn-6`));
    expect(turns[0].concepts).toEqual([]);
  });

  test("deleteTurnState removes the sidecar", async () => {
    const { emitUserTurn, deleteTurnState } = await import("../turns.js");
    const sid = `sess-${randomUUID().slice(0, 8)}`;
    emitUserTurn(sid, "hive", { role: "user", content: "hi" });
    const file = path.join(tmpSessionsDir, `${sid}.turns.json`);
    expect(fs.existsSync(file)).toBe(true);
    deleteTurnState(sid);
    expect(fs.existsSync(file)).toBe(false);
  });
});
