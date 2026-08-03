/**
 * Regression tests for the system-prompt wiring of /repo/agent.
 *
 * PR #1494 silently dropped `instructions` from the `new ToolLoopAgent({...})`
 * call in src/repo/agent.ts. Nothing caught it: `instructions` is optional so
 * tsc stayed green, and the existing agent tests reproduce logic inline rather
 * than exercising the real construction. The agent ran with no system prompt at
 * all — no repo info, no org-agent block, no sub-agent block, no skills index,
 * no systemOverride — until the change was reverted in #1497.
 *
 * These tests pin the two facts that made that one-line deletion fatal:
 *   1. the constructor's `instructions` is what becomes the model's `system`;
 *   2. there is no per-call channel for it, so the constructor is the only one.
 *
 * The last test guards the construction site itself, which is the only way to
 * catch a re-deletion: the type system cannot, because the field is optional.
 */

import { test, expect } from "../../testkit.js";
import { ToolLoopAgent } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const INSTRUCTIONS = "SYSTEM_PROMPT_UNDER_TEST";

function mockModel(): MockLanguageModelV3 {
  return new MockLanguageModelV3({
    doGenerate: async () =>
      ({
        content: [{ type: "text", text: "ok" }],
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        warnings: [],
      }) as any,
  });
}

function systemOf(model: MockLanguageModelV3, callIndex = 0): string | undefined {
  const prompt = model.doGenerateCalls[callIndex]?.prompt as any[];
  return prompt?.find((m) => m.role === "system")?.content;
}

test.describe("/repo/agent system-prompt wiring (ToolLoopAgent contract)", () => {
  test("constructor `instructions` reaches the model as the system message", async () => {
    const model = mockModel();
    const agent = new ToolLoopAgent({ model, instructions: INSTRUCTIONS });

    await agent.generate({ prompt: "hello" });

    expect(model.doGenerateCalls.length).toBe(1);
    expect(systemOf(model)).toBe(INSTRUCTIONS);
  });

  test("omitting `instructions` sends no system message at all — the #1494 regression", async () => {
    const model = mockModel();
    const agent = new ToolLoopAgent({ model });

    await agent.generate({ prompt: "hello" });

    // This is exactly what shipped in #1494: a model call with no system turn.
    expect(systemOf(model)).toBe(undefined);
  });

  test("a per-call `messages` array cannot supply the system prompt — the constructor is the only channel", async () => {
    const model = mockModel();
    const agent = new ToolLoopAgent({ model, instructions: INSTRUCTIONS });

    await agent.generate({
      messages: [{ role: "user", content: "hello" }],
    });

    // Callers pass `messages`; the system turn still comes from the constructor.
    expect(systemOf(model)).toBe(INSTRUCTIONS);
  });

  test("`instructions` applies to every call on the same agent, including the summarization pass", async () => {
    const model = mockModel();
    const agent = new ToolLoopAgent({ model, instructions: INSTRUCTIONS });

    await agent.generate({ prompt: "the turn" });
    // summarizeAfterTurn re-uses `prepared.agent` and passes only `messages`.
    await agent.generate({
      messages: [{ role: "user", content: "summarize the session" }],
    });

    expect(model.doGenerateCalls.length).toBe(2);
    expect(systemOf(model, 0)).toBe(INSTRUCTIONS);
    expect(systemOf(model, 1)).toBe(INSTRUCTIONS);
  });

  test("agent.ts still passes `instructions` when constructing ToolLoopAgent", () => {
    const agentSrc = fs.readFileSync(
      path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "agent.ts"),
      "utf-8",
    );

    const start = agentSrc.indexOf("new ToolLoopAgent({");
    expect(start).not.toBe(-1);

    // The constructor object literal ends at the first line that closes it at
    // the same indentation as the `const agent = ...` statement.
    const end = agentSrc.indexOf("\n  });", start);
    expect(end).not.toBe(-1);

    const ctorBlock = agentSrc.slice(start, end);
    expect(ctorBlock.includes("instructions:")).toBe(true);
  });
});
