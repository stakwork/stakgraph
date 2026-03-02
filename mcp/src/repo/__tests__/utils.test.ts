import { test, expect } from "@playwright/test";
import { deepParseJsonStrings } from "../utils.js";

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
