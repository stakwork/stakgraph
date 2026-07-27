import { test, expect } from "../testkit.js";
import { logStep } from "./utils.js";

/** Capture console.log for the duration of one logStep call. */
function capture(contents: any): string[] {
  const lines: string[] = [];
  const orig = console.log;
  console.log = (...args: any[]) => { lines.push(args.join(" ")); };
  try {
    logStep(contents);
  } finally {
    console.log = orig;
  }
  return lines;
}

test.describe("logStep tool-call arguments", () => {
  test("logs which skill was loaded, not just the tool name", () => {
    const [line] = capture([
      { type: "tool-call", toolName: "load_skill", input: { name: "commercial-legal/nda-review" } },
    ]);
    expect(line).toBe('[repo_agent] tool_call: load_skill {"name":"commercial-legal/nda-review"}');
  });

  test("omits empty and absent inputs rather than printing {}", () => {
    const lines = capture([
      { type: "tool-call", toolName: "list_skills", input: {} },
      { type: "tool-call", toolName: "repo_overview", input: undefined },
      { type: "tool-call", toolName: "repo_overview" },
    ]);
    expect(lines).toEqual([
      "[repo_agent] tool_call: list_skills",
      "[repo_agent] tool_call: repo_overview",
      "[repo_agent] tool_call: repo_overview",
    ]);
  });

  test("truncates long inputs so a file body can't flood the log", () => {
    const [line] = capture([
      { type: "tool-call", toolName: "bash", input: { command: "x".repeat(500) } },
    ]);
    expect(line.endsWith("…")).toBe(true);
    expect(line.length).toBeLessThan(220);
  });

  test("a multi-line object input stays on one log line", () => {
    // JSON.stringify escapes the newlines to a literal backslash-n, so the
    // whitespace collapse never sees them — the single-line property holds
    // either way, which is what actually matters for log readability.
    const [line] = capture([
      { type: "tool-call", toolName: "bash", input: { command: "line1\nline2\n\tline3" } },
    ]);
    expect(line.includes("\n")).toBe(false);
    expect(line).toContain("line1");
    expect(line).toContain("line3");
  });

  test("a raw string input has its real newlines collapsed", () => {
    const [line] = capture([
      { type: "tool-call", toolName: "bash", input: "line1\nline2\n\tline3" },
    ]);
    expect(line.includes("\n")).toBe(false);
    expect(line).toBe("[repo_agent] tool_call: bash line1 line2 line3");
  });

  test("an unserializable input still logs the tool name", () => {
    const circular: any = {};
    circular.self = circular;
    const [line] = capture([{ type: "tool-call", toolName: "weird", input: circular }]);
    expect(line).toBe("[repo_agent] tool_call: weird");
  });

  test("non-tool-call content is unaffected", () => {
    const lines = capture([
      { type: "text", text: "hello there" },
      { type: "reasoning", text: "ignored" },
    ]);
    expect(lines).toEqual(["[repo_agent] text: hello there..."]);
  });

  test("non-array content is a no-op", () => {
    expect(capture(undefined)).toEqual([]);
    expect(capture({ type: "tool-call", toolName: "x" })).toEqual([]);
  });
});
