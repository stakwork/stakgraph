/**
 * Tests for the tool-loop exit guard (utils.ts):
 *   unresolvedToolCalls — which tool calls on a step the loop could not resolve
 *   stripToolCallParts  — removing those calls from a replayable transcript
 *
 * Background: the AI SDK ends its tool loop on a step whose client-side call
 * and output counts disagree, without an error or a stop condition. Seen in
 * production when an LLM gateway rewrote a server tool's version and the API
 * answered with a server_tool_use (code_execution) the run never registered.
 *
 * Harness: node:test (tsx --test). Pure functions, no fixtures.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { ModelMessage, StepResult, ToolSet } from "ai";

import { unresolvedToolCalls, stripToolCallParts } from "../utils.js";

const TOOLS = ["bash", "web_search"];

function step(content: unknown[]): StepResult<ToolSet> {
  return { content, finishReason: "tool-calls" } as unknown as StepResult<ToolSet>;
}

describe("unresolvedToolCalls", () => {
  it("returns nothing for an undefined step", () => {
    assert.deepEqual(unresolvedToolCalls(undefined, TOOLS), []);
  });

  it("a registered tool with a result is resolved", () => {
    const s = step([
      { type: "tool-call", toolCallId: "a", toolName: "bash", input: {} },
      { type: "tool-result", toolCallId: "a", toolName: "bash", output: "ok" },
    ]);
    assert.deepEqual(unresolvedToolCalls(s, TOOLS), []);
  });

  it("a registered tool whose execution errored is still resolved", () => {
    const s = step([
      { type: "tool-call", toolCallId: "a", toolName: "bash", input: {} },
      { type: "tool-error", toolCallId: "a", toolName: "bash", error: "boom" },
    ]);
    assert.deepEqual(unresolvedToolCalls(s, TOOLS), []);
  });

  it("the production shape: an invalid provider-executed call next to a resolved bash call", () => {
    // What the SDK recorded on 2026-09-15: code_execution (never registered)
    // marked invalid with a tool-error, plus the bash edit that did run.
    const s = step([
      { type: "text", text: "Now let's apply the edit." },
      {
        type: "tool-call",
        toolCallId: "srv_1",
        toolName: "code_execution",
        input: { type: "text_editor_code_execution", command: "str_replace" },
        providerExecuted: true,
        dynamic: true,
        invalid: true,
      },
      { type: "tool-error", toolCallId: "srv_1", toolName: "code_execution", error: "No such tool", dynamic: true },
      { type: "tool-call", toolCallId: "b", toolName: "bash", input: { command: "python3 - <<'EOF' ..." } },
      { type: "tool-result", toolCallId: "b", toolName: "bash", output: "done" },
    ]);
    assert.deepEqual(unresolvedToolCalls(s, TOOLS), [
      { toolCallId: "srv_1", toolName: "code_execution" },
    ]);
  });

  it("an unknown tool name is unresolved even when it has a result", () => {
    const s = step([
      { type: "tool-call", toolCallId: "x", toolName: "apply_patch", input: {} },
      { type: "tool-result", toolCallId: "x", toolName: "apply_patch", output: "ok" },
    ]);
    assert.deepEqual(unresolvedToolCalls(s, TOOLS), [{ toolCallId: "x", toolName: "apply_patch" }]);
  });

  it("a registered tool with no output is unresolved", () => {
    const s = step([{ type: "tool-call", toolCallId: "a", toolName: "bash", input: {} }]);
    assert.deepEqual(unresolvedToolCalls(s, TOOLS), [{ toolCallId: "a", toolName: "bash" }]);
  });

  it("accepts a ToolSet as well as a name list", () => {
    const tools = { bash: {}, web_search: {} } as unknown as ToolSet;
    const s = step([
      { type: "tool-call", toolCallId: "a", toolName: "bash", input: {} },
      { type: "tool-result", toolCallId: "a", toolName: "bash", output: "ok" },
      { type: "tool-call", toolCallId: "z", toolName: "nope", input: {} },
      { type: "tool-result", toolCallId: "z", toolName: "nope", output: "ok" },
    ]);
    assert.deepEqual(unresolvedToolCalls(s, tools), [{ toolCallId: "z", toolName: "nope" }]);
  });
});

describe("stripToolCallParts", () => {
  const messages: ModelMessage[] = [
    {
      role: "assistant",
      content: [
        { type: "text", text: "Now let's apply the edit." },
        { type: "tool-call", toolCallId: "srv_1", toolName: "code_execution", input: {} },
        { type: "tool-call", toolCallId: "b", toolName: "bash", input: { command: "true" } },
      ],
    },
    {
      role: "tool",
      content: [
        { type: "tool-result", toolCallId: "srv_1", toolName: "code_execution", output: { type: "error-text", value: "no such tool" } },
        { type: "tool-result", toolCallId: "b", toolName: "bash", output: { type: "text", value: "done" } },
      ],
    },
  ] as ModelMessage[];

  it("returns the input untouched when there is nothing to strip", () => {
    assert.equal(stripToolCallParts(messages, new Set()), messages);
  });

  it("removes the call and its result, keeping everything else in order", () => {
    const out = stripToolCallParts(messages, new Set(["srv_1"]));
    assert.equal(out.length, 2);
    const asst = out[0].content as Array<{ type: string; toolCallId?: string }>;
    assert.deepEqual(asst.map((p) => p.type), ["text", "tool-call"]);
    assert.equal(asst[1].toolCallId, "b");
    const tool = out[1].content as Array<{ toolCallId?: string }>;
    assert.deepEqual(tool.map((p) => p.toolCallId), ["b"]);
  });

  it("drops a message left with no content", () => {
    const only: ModelMessage[] = [
      { role: "assistant", content: [{ type: "tool-call", toolCallId: "srv_1", toolName: "code_execution", input: {} }] },
      { role: "tool", content: [{ type: "tool-result", toolCallId: "srv_1", toolName: "code_execution", output: { type: "text", value: "" } }] },
    ] as ModelMessage[];
    assert.deepEqual(stripToolCallParts(only, new Set(["srv_1"])), []);
  });

  it("leaves string-content messages alone", () => {
    const msgs: ModelMessage[] = [{ role: "user", content: "hi" }];
    assert.deepEqual(stripToolCallParts(msgs, new Set(["srv_1"])), msgs);
  });
});
