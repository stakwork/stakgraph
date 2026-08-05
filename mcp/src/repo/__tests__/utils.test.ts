import { test, expect } from "../../testkit.js";
import {
  deepParseJsonStrings,
  extractLeadingJsonObject,
  matchesSchemaShape,
  collectEnumConstraints,
  extractFinalAnswer,
  needsContinuation,
} from "../utils.js";
import type { StepResult, ToolSet } from "ai";

// Minimal StepResult factory — only the fields the functions under test read.
function step(opts: {
  content?: any[];
  toolCalls?: any[];
  finishReason?: string;
  rawFinishReason?: string;
}): StepResult<ToolSet> {
  return {
    content: opts.content ?? [],
    toolCalls: opts.toolCalls ?? [],
    finishReason: opts.finishReason ?? "stop",
    rawFinishReason: opts.rawFinishReason,
  } as unknown as StepResult<ToolSet>;
}

const toolCallStep = (narration?: string) =>
  step({
    content: [
      ...(narration ? [{ type: "text", text: narration }] : []),
      { type: "tool-call", toolName: "bash" },
      { type: "tool-result", toolName: "bash", output: "ok" },
    ],
    toolCalls: [{ toolName: "bash" }],
    finishReason: "tool-calls",
  });

test.describe("deepParseJsonStrings", () => {
  test("should parse stringified array into parsed array", () => {
    const input = { phases: '[{"name": "Phase 1"}]' };
    const result = deepParseJsonStrings(input);
    expect(result.phases).toEqual([{ name: "Phase 1" }]);
    expect(Array.isArray(result.phases)).toBe(true);
  });

  test("should leave plain string values unchanged", () => {
    const input = { priority: "HIGH" };
    const result = deepParseJsonStrings(input);
    expect(result.priority).toBe("HIGH");
    expect(typeof result.priority).toBe("string");
  });

  test("should parse stringified object into parsed object", () => {
    const input = { task: '{"title": "Fix bug"}' };
    const result = deepParseJsonStrings(input);
    expect(result.task).toEqual({ title: "Fix bug" });
    expect(typeof result.task).toBe("object");
  });

  test("should return null without throwing", () => {
    const result = deepParseJsonStrings(null);
    expect(result).toBeNull();
  });

  test("should leave already parsed structures unchanged", () => {
    const input = { phases: [{ name: "Phase 1" }] };
    const result = deepParseJsonStrings(input);
    expect(result.phases).toEqual([{ name: "Phase 1" }]);
    expect(Array.isArray(result.phases)).toBe(true);
  });

  test("should leave invalid JSON strings as-is", () => {
    const input = { description: "{ not valid json" };
    const result = deepParseJsonStrings(input);
    expect(result.description).toBe("{ not valid json");
    expect(typeof result.description).toBe("string");
  });

  test("should handle nested stringified JSON", () => {
    const input = {
      outer: '{"inner": "[{\\"id\\": 1}]"}',
    };
    const result = deepParseJsonStrings(input);
    expect(result.outer).toEqual({ inner: [{ id: 1 }] });
    expect(Array.isArray(result.outer.inner)).toBe(true);
  });

  test("should preserve markdown descriptions", () => {
    const input = {
      description: "# Heading\n\nThis is markdown text with **bold** and *italic*.",
    };
    const result = deepParseJsonStrings(input);
    expect(result.description).toBe(
      "# Heading\n\nThis is markdown text with **bold** and *italic*."
    );
  });

  test("should handle arrays at top level", () => {
    const input = ['{"id": 1}', '{"id": 2}'];
    const result = deepParseJsonStrings(input);
    expect(result).toEqual([{ id: 1 }, { id: 2 }]);
  });

  test("should not parse JSON primitives", () => {
    const tests = [
      { input: { value: '"hello"' }, expected: { value: '"hello"' } },
      { input: { value: "123" }, expected: { value: "123" } },
      { input: { value: "true" }, expected: { value: "true" } },
      { input: { value: "false" }, expected: { value: "false" } },
    ];

    tests.forEach(({ input, expected }) => {
      const result = deepParseJsonStrings(input);
      expect(result).toEqual(expected);
    });
  });

  test("should handle complex nested structure", () => {
    const input = {
      title: "Feature Plan",
      priority: "HIGH",
      phases: '[{"name": "Phase 1", "tasks": "[{\\"id\\": 1, \\"title\\": \\"Task 1\\"}]"}]',
      metadata: {
        created: "2024-01-01",
        tags: '["feature", "urgent"]',
      },
    };

    const result = deepParseJsonStrings(input);

    expect(result.title).toBe("Feature Plan");
    expect(result.priority).toBe("HIGH");
    expect(Array.isArray(result.phases)).toBe(true);
    expect(result.phases[0].name).toBe("Phase 1");
    expect(Array.isArray(result.phases[0].tasks)).toBe(true);
    expect(result.phases[0].tasks[0].id).toBe(1);
    expect(result.metadata.created).toBe("2024-01-01");
    expect(Array.isArray(result.metadata.tags)).toBe(true);
    expect(result.metadata.tags).toEqual(["feature", "urgent"]);
  });

  test("should handle empty strings", () => {
    const input = { value: "" };
    const result = deepParseJsonStrings(input);
    expect(result.value).toBe("");
  });

  test("should handle whitespace-only strings", () => {
    const input = { value: "   " };
    const result = deepParseJsonStrings(input);
    expect(result.value).toBe("   ");
  });

  test("should handle strings with JSON-like characters but not valid JSON", () => {
    const tests = [
      { input: { value: "{incomplete" }, expected: { value: "{incomplete" } },
      { input: { value: "[incomplete" }, expected: { value: "[incomplete" } },
      { input: { value: "{}" }, expected: { value: {} } },
      { input: { value: "[]" }, expected: { value: [] } },
    ];

    tests.forEach(({ input, expected }) => {
      const result = deepParseJsonStrings(input);
      expect(result).toEqual(expected);
    });
  });
});

test.describe("extractLeadingJsonObject", () => {
  test("extracts a bare JSON object", () => {
    const input = '{"type":"user_question","content":"/tmp/answer.json"}';
    expect(extractLeadingJsonObject(input)).toEqual({
      type: "user_question",
      content: "/tmp/answer.json",
    });
  });

  test("extracts the leading object followed by markdown", () => {
    const input =
      '{"type":"user_question","content":"/tmp/workspace_55741/answer_debug_146354633.json"}\n\n---\n\n## Debug Report — Run 146354633\n\nSome **markdown** here.';
    expect(extractLeadingJsonObject(input)).toEqual({
      type: "user_question",
      content: "/tmp/workspace_55741/answer_debug_146354633.json",
    });
  });

  test("handles leading whitespace before the object", () => {
    const input = '   \n  {"a":1}\n\nrest';
    expect(extractLeadingJsonObject(input)).toEqual({ a: 1 });
  });

  test("handles nested objects", () => {
    const input =
      '{"node_type":"Show","node_data":{"show_title":null,"meta":{"x":1}}} trailing';
    expect(extractLeadingJsonObject(input)).toEqual({
      node_type: "Show",
      node_data: { show_title: null, meta: { x: 1 } },
    });
  });

  test("does not get confused by braces inside strings", () => {
    const input = '{"text":"a } b { c","ok":true} after';
    expect(extractLeadingJsonObject(input)).toEqual({
      text: "a } b { c",
      ok: true,
    });
  });

  test("handles escaped quotes inside strings", () => {
    const input = '{"text":"she said \\"hi\\" } now"} tail';
    expect(extractLeadingJsonObject(input)).toEqual({
      text: 'she said "hi" } now',
    });
  });

  test("returns null when there is no object", () => {
    expect(extractLeadingJsonObject("just some text")).toBeNull();
  });

  test("returns null for an unbalanced/incomplete object", () => {
    expect(extractLeadingJsonObject('{"a":1')).toBeNull();
  });

  test("returns null for invalid JSON inside braces", () => {
    expect(extractLeadingJsonObject("{not valid json}")).toBeNull();
  });

  test("ignores text before the first object", () => {
    const input = 'Here is the answer: {"type":"run_debug"} done';
    expect(extractLeadingJsonObject(input)).toEqual({ type: "run_debug" });
  });
});

test.describe("matchesSchemaShape", () => {
  const schema = {
    type: "object",
    properties: {
      type: { type: "string", enum: ["user_question", "run_debug"] },
      content: { type: "string" },
    },
    required: ["type", "content"],
  };

  test("matches an object with exactly the schema keys", () => {
    expect(
      matchesSchemaShape(
        { type: "user_question", content: "/tmp/x.json" },
        schema
      )
    ).toBe(true);
  });

  test("matches when optional props are omitted", () => {
    const optionalSchema = {
      type: "object",
      properties: { type: { type: "string" }, content: { type: "string" } },
      required: ["type"],
    };
    expect(matchesSchemaShape({ type: "run_debug" }, optionalSchema)).toBe(true);
  });

  test("rejects when a required key is missing", () => {
    expect(matchesSchemaShape({ type: "user_question" }, schema)).toBe(false);
  });

  test("rejects when there is an unknown key", () => {
    expect(
      matchesSchemaShape(
        { type: "user_question", content: "x", extra: 1 },
        schema
      )
    ).toBe(false);
  });

  test("rejects arrays, null, and primitives", () => {
    expect(matchesSchemaShape([], schema)).toBe(false);
    expect(matchesSchemaShape(null, schema)).toBe(false);
    expect(matchesSchemaShape("str", schema)).toBe(false);
    expect(matchesSchemaShape(42, schema)).toBe(false);
  });

  test("rejects an empty object", () => {
    expect(matchesSchemaShape({}, schema)).toBe(false);
  });

  test("rejects when schema is not an object schema", () => {
    expect(matchesSchemaShape({ a: 1 }, { type: "string" })).toBe(false);
    expect(matchesSchemaShape({ a: 1 }, {} as any)).toBe(false);
  });

  test("treats missing required as no required keys", () => {
    const noRequired = {
      type: "object",
      properties: { a: { type: "string" } },
    };
    expect(matchesSchemaShape({ a: "x" }, noRequired)).toBe(true);
  });
});

test.describe("collectEnumConstraints", () => {
  test("collects a top-level enum field", () => {
    const schema = {
      type: "object",
      properties: {
        type: { type: "string", enum: ["user_question", "run_debug"] },
        content: { type: "string" },
      },
    };
    expect(collectEnumConstraints(schema)).toEqual([
      { path: "type", values: ["user_question", "run_debug"] },
    ]);
  });

  test("returns empty when there are no enums", () => {
    const schema = {
      type: "object",
      properties: { content: { type: "string" } },
    };
    expect(collectEnumConstraints(schema)).toEqual([]);
  });

  test("collects nested and array-item enums with dotted paths", () => {
    const schema = {
      type: "object",
      properties: {
        status: { type: "string", enum: ["ok", "error"] },
        meta: {
          type: "object",
          properties: {
            level: { type: "string", enum: ["low", "high"] },
          },
        },
        items: {
          type: "array",
          items: {
            type: "object",
            properties: {
              kind: { type: "string", enum: ["a", "b"] },
            },
          },
        },
      },
    };
    expect(collectEnumConstraints(schema)).toEqual([
      { path: "status", values: ["ok", "error"] },
      { path: "meta.level", values: ["low", "high"] },
      { path: "items[].kind", values: ["a", "b"] },
    ]);
  });

  test("collects enums inside anyOf/oneOf combiners", () => {
    const schema = {
      type: "object",
      properties: {
        choice: {
          anyOf: [
            { type: "string", enum: ["x", "y"] },
            { type: "number" },
          ],
        },
      },
    };
    expect(collectEnumConstraints(schema)).toEqual([
      { path: "choice", values: ["x", "y"] },
    ]);
  });
});

test.describe("extractFinalAnswer", () => {
  test("marker in final text: narration before the last tool call is excluded", () => {
    const steps = [
      toolCallStep("Let me look at the code."),
      toolCallStep("Now reading session.ts."),
      step({
        content: [
          { type: "text", text: "# Report\nThe answer.\n[END_OF_ANSWER]" },
        ],
      }),
    ];
    const result = extractFinalAnswer(steps);
    expect(result.answer).toBe("# Report\nThe answer.");
    expect(result.tool_use).toBe("text_with_end_marker");
  });

  test("answer that QUOTES the marker mid-text is not truncated at the quote", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [
          {
            type: "text",
            text:
              'The agent uses stopSequences: ["[END_OF_ANSWER]"] to stop generation.\n' +
              "That is the whole mechanism.\n[END_OF_ANSWER]",
          },
        ],
      }),
    ];
    const result = extractFinalAnswer(steps);
    expect(result.answer).toContain("That is the whole mechanism.");
    expect(result.answer).toContain('stopSequences: ["[END_OF_ANSWER]"]');
  });

  test("no marker (stripped by stop sequence): falls back to text after last tool call", () => {
    const steps = [
      toolCallStep("Investigating."),
      step({ content: [{ type: "text", text: "Final answer without marker." }] }),
    ];
    const result = extractFinalAnswer(steps);
    expect(result.answer).toBe("Final answer without marker.");
  });

  test("no tools at all: falls back to all text", () => {
    const steps = [
      step({ content: [{ type: "text", text: "Just a direct reply." }] }),
    ];
    expect(extractFinalAnswer(steps).answer).toBe("Just a direct reply.");
  });
});

test.describe("needsContinuation", () => {
  test("anthropic stop_sequence finish is a proper termination", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "The answer." }],
        rawFinishReason: "stop_sequence",
      }),
    ];
    expect(needsContinuation(steps, false)).toBe(false);
  });

  test("anthropic end_turn with only a plan is a stall", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Let me now run six graph searches." }],
        rawFinishReason: "end_turn",
      }),
    ];
    expect(needsContinuation(steps, false)).toBe(true);
  });

  test("truncation (length) is a stall", () => {
    const steps = [
      toolCallStep(),
      step({ content: [{ type: "reasoning", text: "hmm" }], finishReason: "length" }),
    ];
    expect(needsContinuation(steps, false)).toBe(true);
  });

  test("marker mode: plain stop without marker is a stall", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Batch 1 (parallel): 6 searches." }],
        rawFinishReason: "stop",
      }),
    ];
    expect(needsContinuation(steps, true)).toBe(true);
  });

  test("marker mode: marker in text is a proper termination", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Answer.\n[END_OF_ANSWER]" }],
        rawFinishReason: "stop",
      }),
    ];
    expect(needsContinuation(steps, true)).toBe(false);
  });

  test("non-marker mode: ambiguous raw stop is left alone", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Some answer." }],
        rawFinishReason: "stop",
      }),
    ];
    expect(needsContinuation(steps, false)).toBe(false);
  });

  test("last step with tool calls (e.g. maxTurns stop) is not a stall", () => {
    const steps = [toolCallStep(), toolCallStep()];
    expect(needsContinuation(steps, true)).toBe(false);
  });

  test("ask_clarifying_questions is a proper termination", () => {
    const steps = [
      step({
        content: [
          { type: "tool-result", toolName: "ask_clarifying_questions", output: "{}" },
        ],
      }),
    ];
    expect(needsContinuation(steps, true)).toBe(false);
  });
});
