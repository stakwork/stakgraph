import { test } from "node:test";
import assert from "node:assert/strict";
import { generateText, isStepCount, tool, type ModelMessage } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { z } from "zod";
import { extractMessagesFromSteps } from "../utils.js";

// Runs a REAL two-step tool loop (mock model) so the steps have the SDK's own
// shape. Pins the AI SDK v7 change: each step's `response.messages` is
// per-step, so a transcript built from the last step alone drops the first
// step's tool call + result — the orphaned tool_use that breaks session replay.

const usage = {
  inputTokens: { total: 10, noCache: 10, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 5, text: 5, reasoning: undefined },
};

test("extractMessagesFromSteps keeps every step's messages, in order", async () => {
  const model = new MockLanguageModelV4({
    doGenerate: [
      {
        content: [{ type: "tool-call", toolCallId: "call_1", toolName: "lookup", input: '{"q":"x"}' }],
        finishReason: { unified: "tool-calls", raw: "tool_use" },
        usage,
        warnings: [],
      },
      {
        content: [{ type: "text", text: "done" }],
        finishReason: { unified: "stop", raw: "end_turn" },
        usage,
        warnings: [],
      },
    ],
  });

  const result = await generateText({
    model,
    prompt: "go",
    tools: {
      lookup: tool({
        inputSchema: z.object({ q: z.string() }),
        execute: async ({ q }) => `found ${q}`,
      }),
    },
    stopWhen: isStepCount(5),
  });
  assert.equal(result.steps.length, 2);

  const user: ModelMessage = { role: "user", content: "go" };
  const messages = extractMessagesFromSteps(user, result.steps);

  assert.deepEqual(
    messages.map((m) => m.role),
    ["user", "assistant", "tool", "assistant"],
  );
  const toolCall = (messages[1].content as any[]).find((p) => p.type === "tool-call");
  const toolResult = (messages[2].content as any[]).find((p) => p.type === "tool-result");
  assert.equal(toolCall?.toolCallId, "call_1");
  assert.equal(toolResult?.toolCallId, "call_1");
  // Same transcript the SDK itself reports for the whole call.
  assert.deepEqual(messages.slice(1), result.responseMessages);
});
