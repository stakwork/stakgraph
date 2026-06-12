import { test, expect } from "@playwright/test";
import { ModelMessage } from "ai";
import { stubOldToolResults, loadCompactState } from "../context.js";
import { buildRestorableStub, buildToolInputMap, CLEARED_PREFIX } from "../utils.js";
import { getSessionSidecarFile } from "../session.js";
import { writeFileSync, unlinkSync, existsSync } from "fs";

function toolTurn(
  turn: number,
  toolName: string,
  input: unknown,
  output: string
): ModelMessage[] {
  const id = `call_${turn}_${toolName}`;
  return [
    {
      role: "assistant",
      content: [{ type: "tool-call", toolCallId: id, toolName, input }],
    } as ModelMessage,
    {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: id,
          toolName,
          output: { type: "text", value: output },
        },
      ],
    } as ModelMessage,
  ];
}

function userMsg(text: string): ModelMessage {
  return { role: "user", content: text };
}

function assistantMsg(text: string): ModelMessage {
  return { role: "assistant", content: text };
}

const BIG = "x".repeat(5000);

function threeTurnSession(): ModelMessage[] {
  return [
    userMsg("turn 1 question"),
    ...toolTurn(1, "stakgraph_code", { ref_id: "abc123" }, BIG),
    assistantMsg("turn 1 answer"),
    userMsg("turn 2 question"),
    ...toolTurn(2, "bash", { command: "cat src/auth.ts" }, BIG),
    assistantMsg("turn 2 answer"),
    userMsg("turn 3 question"),
    ...toolTurn(3, "stakgraph_search", { query: "auth flow" }, BIG),
    assistantMsg("turn 3 answer"),
  ];
}

test.describe("buildRestorableStub", () => {
  test("formats stakgraph_code with ref_id", () => {
    const stub = buildRestorableStub("stakgraph_code", { ref_id: "abc123" });
    expect(stub).toContain(CLEARED_PREFIX);
    expect(stub).toContain("ref_id=abc123");
    expect(stub).toContain("stakgraph_code");
  });

  test("formats bash with command", () => {
    const stub = buildRestorableStub("bash", { command: "cat src/auth.ts" });
    expect(stub).toContain('cmd="cat src/auth.ts"');
  });

  test("formats search tools with query", () => {
    const stub = buildRestorableStub("stakgraph_search", { query: "auth flow" });
    expect(stub).toContain('query="auth flow"');
  });

  test("falls back to compact input JSON", () => {
    const stub = buildRestorableStub("some_tool", { foo: "bar" });
    expect(stub).toContain('input={"foo":"bar"}');
  });

  test("is deterministic", () => {
    const a = buildRestorableStub("bash", { command: "ls" });
    const b = buildRestorableStub("bash", { command: "ls" });
    expect(a).toBe(b);
  });
});

test.describe("buildToolInputMap", () => {
  test("maps toolCallId to input across messages", () => {
    const messages = threeTurnSession();
    const map = buildToolInputMap(messages);
    expect(map.get("call_1_stakgraph_code")).toEqual({ ref_id: "abc123" });
    expect(map.get("call_2_bash")).toEqual({ command: "cat src/auth.ts" });
  });
});

test.describe("stubOldToolResults", () => {
  test("no-op when under threshold", () => {
    const messages = threeTurnSession();
    // Huge limit -> way under 35%
    const result = stubOldToolResults(messages, 10_000_000);
    expect(result).toBe(messages);
  });

  test("stubs old turns, keeps recent turns intact", () => {
    const messages = threeTurnSession();
    // Small limit -> over threshold. 3 turns, keep last 2 -> only turn 1 stubbed.
    const result = stubOldToolResults(messages, 1000);
    expect(result).not.toBe(messages);

    const turn1Result = (result[2] as any).content[0];
    expect(turn1Result.output.value).toContain(CLEARED_PREFIX);
    expect(turn1Result.output.value).toContain("ref_id=abc123");

    const turn2Result = (result[6] as any).content[0];
    expect(turn2Result.output.value).toBe(BIG);
    const turn3Result = (result[10] as any).content[0];
    expect(turn3Result.output.value).toBe(BIG);
  });

  test("preserves tool-call/result pairing and message count", () => {
    const messages = threeTurnSession();
    const result = stubOldToolResults(messages, 1000);
    expect(result.length).toBe(messages.length);
    for (let i = 0; i < result.length; i++) {
      expect(result[i].role).toBe(messages[i].role);
    }
    const stubbed = (result[2] as any).content[0];
    expect(stubbed.toolCallId).toBe("call_1_stakgraph_code");
    expect(stubbed.type).toBe("tool-result");
  });

  test("does not mutate the original messages", () => {
    const messages = threeTurnSession();
    stubOldToolResults(messages, 1000);
    expect((messages[2] as any).content[0].output.value).toBe(BIG);
  });

  test("protects error outputs", () => {
    const messages: ModelMessage[] = [
      userMsg("turn 1"),
      ...toolTurn(1, "bash", { command: "run tests" }, "Error: tests failed\n" + BIG),
      assistantMsg("a1"),
      userMsg("turn 2"),
      ...toolTurn(2, "bash", { command: "ls" }, BIG),
      assistantMsg("a2"),
      userMsg("turn 3"),
      ...toolTurn(3, "bash", { command: "pwd" }, BIG),
      assistantMsg("a3"),
    ];
    const result = stubOldToolResults(messages, 1000);
    const errResult = (result[2] as any).content[0];
    expect(errResult.output.value).toContain("Error: tests failed");
  });

  test("is idempotent (already-stubbed results untouched)", () => {
    const messages = threeTurnSession();
    const once = stubOldToolResults(messages, 1000);
    const twice = stubOldToolResults(once, 1000);
    expect(twice).toBe(once);
  });

  test("no-op with fewer turns than keepRecentTurns", () => {
    const messages: ModelMessage[] = [
      userMsg("only turn"),
      ...toolTurn(1, "bash", { command: "ls" }, BIG.repeat(3)),
      assistantMsg("answer"),
    ];
    const result = stubOldToolResults(messages, 1000);
    expect(result).toBe(messages);
  });
});

test.describe("loadCompactState", () => {
  const sessionId = "context-test-session";

  test.afterEach(() => {
    const p = getSessionSidecarFile(sessionId, ".compact.json");
    if (existsSync(p)) unlinkSync(p);
  });

  test("returns undefined when no sidecar exists", () => {
    expect(loadCompactState("does-not-exist")).toBeUndefined();
  });

  test("round-trips a valid sidecar and normalizes state", () => {
    const p = getSessionSidecarFile(sessionId, ".compact.json");
    writeFileSync(
      p,
      JSON.stringify({
        compactedThroughIndex: 7,
        state: {
          summary: "did stuff",
          goals: ["finish feature"],
          decisions: [],
          importantRefs: [{ kind: "file", value: "src/auth.ts", reason: "core" }],
          checked: ["dead end A"],
          openQuestions: [],
          nextSteps: [],
          warnings: [],
        },
        tokensBefore: 50000,
        tokensAfter: 4000,
        updated_at: "2026-06-12T00:00:00.000Z",
      })
    );
    const loaded = loadCompactState(sessionId);
    expect(loaded?.compactedThroughIndex).toBe(7);
    expect(loaded?.state.summary).toBe("did stuff");
    expect(loaded?.state.importantRefs[0].value).toBe("src/auth.ts");
    expect(loaded?.tokensBefore).toBe(50000);
  });

  test("returns undefined for corrupt sidecar", () => {
    const p = getSessionSidecarFile(sessionId, ".compact.json");
    writeFileSync(p, "{ not valid json");
    expect(loadCompactState(sessionId)).toBeUndefined();
  });

  test("returns undefined when compactedThroughIndex missing", () => {
    const p = getSessionSidecarFile(sessionId, ".compact.json");
    writeFileSync(p, JSON.stringify({ state: {} }));
    expect(loadCompactState(sessionId)).toBeUndefined();
  });
});
