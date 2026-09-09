import { test, expect } from "../../testkit.js";
import {
  deepParseJsonStrings,
  extractLeadingJsonObject,
  matchesSchemaShape,
  collectEnumConstraints,
  extractFinalAnswer,
  needsContinuation,
  isContinuationNudge,
  timeBudgetNudge,
  redactCredentials,
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

test.describe("redactCredentials", () => {
  test("strips a user:token pair from a git auth failure", () => {
    const msg =
      "fatal: Authentication failed for 'https://x-access-token:ghp_SECRET123@github.com/org/repo.git/'";
    const out = redactCredentials(msg);
    expect(out).not.toContain("ghp_SECRET123");
    expect(out).not.toContain("x-access-token");
    expect(out).toBe(
      "fatal: Authentication failed for 'https://***@github.com/org/repo.git/'"
    );
  });

  test("strips a bare token used as the username", () => {
    const out = redactCredentials("remote: https://ghp_SECRET123@github.com/o/r.git");
    expect(out).not.toContain("ghp_SECRET123");
    expect(out).toBe("remote: https://***@github.com/o/r.git");
  });

  test("redacts every occurrence in a multi-repo failure summary", () => {
    const out = redactCredentials(
      "a: https://u:t1@github.com/o/a.git failed; b: https://u:t2@github.com/o/b.git failed"
    );
    expect(out).not.toContain("t1");
    expect(out).not.toContain("t2");
  });

  test("leaves credential-free urls and plain text untouched", () => {
    expect(redactCredentials("cloning https://github.com/org/repo.git")).toBe(
      "cloning https://github.com/org/repo.git"
    );
    expect(redactCredentials("no url here at all")).toBe("no url here at all");
  });

  test("does not treat a bare email address as a url userinfo", () => {
    expect(redactCredentials("contact dev@example.com")).toBe(
      "contact dev@example.com"
    );
  });

  // New extended patterns -------------------------------------------------------

  test("strips bare ghp_ token appearing mid-string", () => {
    const out = redactCredentials("stderr: ghp_AbCdEfGhIjKlMnOpQ123 was rejected");
    expect(out).not.toContain("ghp_AbCdEfGhIjKlMnOpQ123");
    expect(out).toContain("[REDACTED]");
    expect(out).toContain("stderr:");
    expect(out).toContain("was rejected");
  });

  test("strips bare gho_ token", () => {
    const out = redactCredentials("token: gho_TestToken9999");
    expect(out).not.toContain("gho_TestToken9999");
    expect(out).toContain("[REDACTED]");
  });

  test("strips bare ghu_ token", () => {
    const out = redactCredentials("error using ghu_UserToken0001 for auth");
    expect(out).not.toContain("ghu_UserToken0001");
  });

  test("strips bare ghs_ token", () => {
    const out = redactCredentials("ghs_ServerToken1234abcd");
    expect(out).not.toContain("ghs_ServerToken1234abcd");
  });

  test("strips bare ghr_ token", () => {
    const out = redactCredentials("refresh ghr_RefreshToken5678");
    expect(out).not.toContain("ghr_RefreshToken5678");
  });

  test("strips bare github_pat_ token", () => {
    const out = redactCredentials("github_pat_LongPatToken_ABCDEF123456");
    expect(out).not.toContain("github_pat_LongPatToken_ABCDEF123456");
    expect(out).toContain("[REDACTED]");
  });

  test("strips bare token appearing multiple times", () => {
    const out = redactCredentials(
      "first ghp_MULTI123 then again ghp_MULTI123 in output"
    );
    expect(out).not.toContain("ghp_MULTI123");
    // Both occurrences replaced
    expect(out.split("[REDACTED]").length - 1).toBe(2);
  });

  test("strips Authorization: Basic header value", () => {
    const b64 = Buffer.from("x-access-token:ghp_secret123").toString("base64");
    const out = redactCredentials(`Authorization: Basic ${b64}`);
    expect(out).not.toContain(b64);
    expect(out).not.toContain("ghp_secret123");
    expect(out).toContain("[REDACTED]");
  });

  test("strips Authorization: token header value (case-insensitive)", () => {
    const out = redactCredentials("authorization: token ghp_TokenValue1234");
    expect(out).not.toContain("ghp_TokenValue1234");
    expect(out).toContain("[REDACTED]");
  });

  test("strips Authorization: Bearer header value", () => {
    const out = redactCredentials("Authorization: Bearer myBearerToken99");
    expect(out).not.toContain("myBearerToken99");
    expect(out).toContain("[REDACTED]");
  });

  test("strips literal pat when supplied as second argument", () => {
    const pat = "my-unusual-token-that-matches-no-pattern";
    const out = redactCredentials(`error: ${pat} was rejected by server`, pat);
    expect(out).not.toContain(pat);
    expect(out).toContain("[REDACTED]");
    expect(out).toContain("error:");
  });

  test("strips literal pat appearing multiple times", () => {
    const pat = "tok_special_ABCDEF";
    const out = redactCredentials(`used ${pat} here and ${pat} again`, pat);
    expect(out).not.toContain(pat);
    expect(out.split("[REDACTED]").length - 1).toBe(2);
  });

  test("surrounding text is preserved when redacting bare token", () => {
    const out = redactCredentials("git: error: remote rejected (ghp_SECRET1234ABCD) — permission denied");
    expect(out).not.toContain("ghp_SECRET1234ABCD");
    expect(out).toContain("git: error: remote rejected");
    expect(out).toContain("permission denied");
  });

  test("redacts embedded-credential URL alongside bare token", () => {
    const out = redactCredentials(
      "clone https://user:ghp_Token123@github.com/o/r.git failed, token=ghp_Token123"
    );
    expect(out).not.toContain("ghp_Token123");
    expect(out).not.toContain("user:");
    // URL part redacted with ***@
    expect(out).toContain("***@");
  });
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
  test("marker in text: answer is everything before [END_OF_ANSWER]", () => {
    const steps = [
      toolCallStep("Investigating."),
      step({
        content: [
          { type: "text", text: "# Report\nThe answer.\n[END_OF_ANSWER]" },
        ],
      }),
    ];
    const result = extractFinalAnswer(steps);
    expect(result.answer.endsWith("The answer.")).toBe(true);
    expect(result.answer).not.toContain("[END_OF_ANSWER]");
    expect(result.tool_use).toBe("text_with_end_marker");
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
    expect(needsContinuation(steps)).toBe(false);
  });

  test("anthropic end_turn with only a plan is a stall", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Let me now run six graph searches." }],
        rawFinishReason: "end_turn",
      }),
    ];
    expect(needsContinuation(steps)).toBe(true);
  });

  test("truncation (length) is a stall", () => {
    const steps = [
      toolCallStep(),
      step({ content: [{ type: "reasoning", text: "hmm" }], finishReason: "length" }),
    ];
    expect(needsContinuation(steps)).toBe(true);
  });

  test("mid-stream error step (e.g. connection drop after reasoning) is a stall", () => {
    const steps = [
      toolCallStep(),
      step({ content: [{ type: "reasoning", text: "I now have all six documents..." }], finishReason: "error" }),
    ];
    expect(needsContinuation(steps)).toBe(true);
  });

  test("ambiguous raw stop (OpenAI-compatible providers) is left alone", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Some answer." }],
        rawFinishReason: "stop",
      }),
    ];
    expect(needsContinuation(steps)).toBe(false);
  });

  test("last step with tool calls (e.g. maxTurns stop) is not a stall", () => {
    const steps = [toolCallStep(), toolCallStep()];
    expect(needsContinuation(steps)).toBe(false);
  });

  test("ask_clarifying_questions is a proper termination", () => {
    const steps = [
      step({
        content: [
          { type: "tool-result", toolName: "ask_clarifying_questions", output: "{}" },
        ],
      }),
    ];
    expect(needsContinuation(steps)).toBe(false);
  });

  test("text containing [END_OF_ANSWER] is a proper termination", () => {
    const steps = [
      toolCallStep(),
      step({
        content: [{ type: "text", text: "Answer.\n[END_OF_ANSWER]" }],
        rawFinishReason: "end_turn",
      }),
    ];
    expect(needsContinuation(steps)).toBe(false);
  });
});

test.describe("timeBudgetNudge", () => {
  const MIN = 60_000;

  test("nothing before 50% of budget", () => {
    expect(timeBudgetNudge(59 * MIN, 120, new Set(), false)).toBe(null);
  });

  test("each threshold fires once, in order, with escalating urgency", () => {
    const fired = new Set<number>();
    const first = timeBudgetNudge(61 * MIN, 120, fired, false);
    expect(first).toContain("61 of 120 minutes");
    expect(timeBudgetNudge(62 * MIN, 120, fired, false)).toBe(null);
    const second = timeBudgetNudge(91 * MIN, 120, fired, false);
    expect(second).toContain("Start converging");
    const third = timeBudgetNudge(111 * MIN, 120, fired, false);
    expect(third).toContain("~9 minutes remain");
    expect(third).toContain("[END_OF_ANSWER]");
    expect(timeBudgetNudge(115 * MIN, 120, fired, false)).toBe(null);
  });

  test("a slow step that jumps past several thresholds fires only the most urgent", () => {
    const fired = new Set<number>();
    const nudge = timeBudgetNudge(112 * MIN, 120, fired, false);
    expect(nudge).toContain("minutes remain");
    expect(timeBudgetNudge(113 * MIN, 120, fired, false)).toBe(null);
  });

  test("final warning offers ask_clarifying_questions when enabled", () => {
    const withAsk = timeBudgetNudge(111 * MIN, 120, new Set(), true);
    expect(withAsk).toContain("ask_clarifying_questions");
    const withoutAsk = timeBudgetNudge(111 * MIN, 120, new Set(), false);
    expect(withoutAsk).not.toContain("ask_clarifying_questions");
  });

  test("thresholds scale with a non-default budget", () => {
    const fired = new Set<number>();
    expect(timeBudgetNudge(14 * MIN, 30, fired, false)).toBe(null);
    expect(timeBudgetNudge(15 * MIN, 30, fired, false)).toContain("15 of 30 minutes");
  });
});

test.describe("isContinuationNudge", () => {
  test("detects a tagged nudge message", () => {
    expect(
      isContinuationNudge({
        role: "user",
        content: "You ended your turn without completing the task...",
        providerOptions: { stakgraph: { continuationNudge: true } },
      })
    ).toBe(true);
  });

  test("real user messages are not nudges", () => {
    expect(isContinuationNudge({ role: "user", content: "hi" })).toBe(false);
    expect(
      isContinuationNudge({
        role: "user",
        content: "hi",
        providerOptions: { anthropic: { cacheControl: { type: "ephemeral" } } },
      })
    ).toBe(false);
  });
});
