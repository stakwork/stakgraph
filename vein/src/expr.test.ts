import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  evaluateExpr,
  resolveTemplate,
  resolveConfig,
  hasTemplates,
  TemplateError,
} from "./expr.js";

// ── evaluateExpr ───────────────────────────────────────────────────────────

describe("evaluateExpr", () => {
  describe("literals", () => {
    it("evaluates number literals", () => {
      assert.equal(evaluateExpr("42", {}), 42);
      assert.equal(evaluateExpr("3.14", {}), 3.14);
      assert.equal(evaluateExpr("0", {}), 0);
    });

    it("evaluates string literals (single quotes)", () => {
      assert.equal(evaluateExpr("'hello'", {}), "hello");
    });

    it("evaluates string literals (double quotes)", () => {
      assert.equal(evaluateExpr('"world"', {}), "world");
    });

    it("evaluates escaped characters in strings", () => {
      assert.equal(evaluateExpr("'it\\'s'", {}), "it's");
    });

    it("evaluates boolean literals", () => {
      assert.equal(evaluateExpr("true", {}), true);
      assert.equal(evaluateExpr("false", {}), false);
    });

    it("evaluates null literal", () => {
      assert.equal(evaluateExpr("null", {}), null);
    });

    it("evaluates undefined literal", () => {
      assert.equal(evaluateExpr("undefined", {}), undefined);
    });
  });

  describe("identifier resolution", () => {
    it("resolves a simple identifier from scope", () => {
      assert.equal(evaluateExpr("x", { x: 10 }), 10);
    });

    it("resolves an object from scope", () => {
      const obj = { a: 1, b: 2 };
      assert.deepEqual(evaluateExpr("data", { data: obj }), obj);
    });

    it("throws TemplateError for undefined references", () => {
      assert.throws(() => evaluateExpr("missing", {}), TemplateError);
    });

    it("resolves $current (special loop variable)", () => {
      assert.equal(evaluateExpr("$current", { $current: "prev" }), "prev");
    });

    it("resolves $error (special onError variable)", () => {
      const err = { message: "boom", stack: "..." };
      assert.deepEqual(evaluateExpr("$error", { $error: err }), err);
    });
  });

  describe("property access", () => {
    it("resolves dot access", () => {
      assert.equal(
        evaluateExpr("input.url", { input: { url: "/api" } }),
        "/api",
      );
    });

    it("resolves nested dot access", () => {
      assert.equal(
        evaluateExpr("a.b.c", { a: { b: { c: 42 } } }),
        42,
      );
    });

    it("resolves bracket access with string", () => {
      assert.equal(
        evaluateExpr('obj["key"]', { obj: { key: "value" } }),
        "value",
      );
    });

    it("resolves bracket access with number (array index)", () => {
      assert.equal(
        evaluateExpr("arr[0]", { arr: ["first", "second"] }),
        "first",
      );
    });

    it("resolves bracket access with computed index", () => {
      assert.equal(
        evaluateExpr("arr[1]", { arr: [10, 20, 30] }),
        20,
      );
    });

    it("resolves mixed dot and bracket access", () => {
      assert.equal(
        evaluateExpr('data.items[0].name', {
          data: { items: [{ name: "Alice" }] },
        }),
        "Alice",
      );
    });

    it("throws on property access of null", () => {
      assert.throws(
        () => evaluateExpr("x.y", { x: null }),
        TemplateError,
      );
    });

    it("throws on property access of undefined", () => {
      assert.throws(
        () => evaluateExpr("x.y", { x: undefined }),
        TemplateError,
      );
    });

    it("returns undefined for missing property on object", () => {
      assert.equal(
        evaluateExpr("obj.missing", { obj: { a: 1 } }),
        undefined,
      );
    });
  });

  describe("arithmetic operators", () => {
    it("adds numbers", () => {
      assert.equal(evaluateExpr("a + b", { a: 3, b: 4 }), 7);
    });

    it("subtracts numbers", () => {
      assert.equal(evaluateExpr("a - b", { a: 10, b: 3 }), 7);
    });

    it("multiplies numbers", () => {
      assert.equal(evaluateExpr("a * b", { a: 6, b: 7 }), 42);
    });

    it("divides numbers", () => {
      assert.equal(evaluateExpr("a / b", { a: 10, b: 2 }), 5);
    });

    it("modulo", () => {
      assert.equal(evaluateExpr("a % b", { a: 10, b: 3 }), 1);
    });

    it("respects operator precedence (* before +)", () => {
      assert.equal(evaluateExpr("2 + 3 * 4", {}), 14);
    });

    it("respects operator precedence (/ before -)", () => {
      assert.equal(evaluateExpr("10 - 6 / 2", {}), 7);
    });

    it("handles parenthesized expressions", () => {
      assert.equal(evaluateExpr("(2 + 3) * 4", {}), 20);
    });

    it("concatenates strings with +", () => {
      assert.equal(
        evaluateExpr("a + b", { a: "hello ", b: "world" }),
        "hello world",
      );
    });
  });

  describe("comparison operators", () => {
    it("less than", () => {
      assert.equal(evaluateExpr("a < b", { a: 1, b: 2 }), true);
      assert.equal(evaluateExpr("a < b", { a: 2, b: 1 }), false);
    });

    it("less than or equal", () => {
      assert.equal(evaluateExpr("a <= b", { a: 2, b: 2 }), true);
      assert.equal(evaluateExpr("a <= b", { a: 3, b: 2 }), false);
    });

    it("greater than", () => {
      assert.equal(evaluateExpr("a > b", { a: 5, b: 3 }), true);
      assert.equal(evaluateExpr("a > b", { a: 1, b: 3 }), false);
    });

    it("greater than or equal", () => {
      assert.equal(evaluateExpr("a >= b", { a: 5, b: 5 }), true);
      assert.equal(evaluateExpr("a >= b", { a: 4, b: 5 }), false);
    });
  });

  describe("equality operators", () => {
    it("strict equality (===)", () => {
      assert.equal(evaluateExpr("a === b", { a: 1, b: 1 }), true);
      assert.equal(evaluateExpr("a === b", { a: 1, b: "1" }), false);
    });

    it("strict inequality (!==)", () => {
      assert.equal(evaluateExpr("a !== b", { a: 1, b: 2 }), true);
      assert.equal(evaluateExpr("a !== b", { a: 1, b: 1 }), false);
    });

    it("loose equality (==)", () => {
      assert.equal(evaluateExpr("a == b", { a: 1, b: "1" }), true);
    });

    it("loose inequality (!=)", () => {
      assert.equal(evaluateExpr("a != b", { a: 1, b: "2" }), true);
    });

    it("string comparison with ===", () => {
      assert.equal(
        evaluateExpr("status === 'complete'", { status: "complete" }),
        true,
      );
      assert.equal(
        evaluateExpr("status === 'complete'", { status: "pending" }),
        false,
      );
    });
  });

  describe("logical operators", () => {
    it("logical AND (&&)", () => {
      assert.equal(evaluateExpr("a && b", { a: true, b: true }), true);
      assert.equal(evaluateExpr("a && b", { a: true, b: false }), false);
      assert.equal(evaluateExpr("a && b", { a: false, b: true }), false);
    });

    it("logical OR (||)", () => {
      assert.equal(evaluateExpr("a || b", { a: false, b: true }), true);
      assert.equal(evaluateExpr("a || b", { a: false, b: false }), false);
    });

    it("logical NOT (!)", () => {
      assert.equal(evaluateExpr("!a", { a: true }), false);
      assert.equal(evaluateExpr("!a", { a: false }), true);
    });

    it("double NOT (!!)", () => {
      assert.equal(evaluateExpr("!!a", { a: 0 }), false);
      assert.equal(evaluateExpr("!!a", { a: 1 }), true);
    });

    it("&& does not short-circuit (one-pass parser evaluates both sides)", () => {
      // The one-pass parser evaluates eagerly — both sides are resolved.
      // This means referencing undefined vars on the right side throws.
      assert.throws(
        () => evaluateExpr("false && missing", {}),
        TemplateError,
      );
    });

    it("|| does not short-circuit (one-pass parser evaluates both sides)", () => {
      assert.throws(
        () => evaluateExpr("true || missing", {}),
        TemplateError,
      );
    });
  });

  describe("ternary operator", () => {
    it("returns then branch when condition is true", () => {
      assert.equal(evaluateExpr("true ? 'yes' : 'no'", {}), "yes");
    });

    it("returns else branch when condition is false", () => {
      assert.equal(evaluateExpr("false ? 'yes' : 'no'", {}), "no");
    });

    it("works with expression conditions", () => {
      assert.equal(
        evaluateExpr("x > 5 ? 'big' : 'small'", { x: 10 }),
        "big",
      );
      assert.equal(
        evaluateExpr("x > 5 ? 'big' : 'small'", { x: 3 }),
        "small",
      );
    });

    it("works with nested ternaries", () => {
      assert.equal(
        evaluateExpr("a ? 'one' : b ? 'two' : 'three'", {
          a: false,
          b: true,
        }),
        "two",
      );
    });
  });

  describe("unary negation", () => {
    it("negates a number", () => {
      assert.equal(evaluateExpr("-x", { x: 5 }), -5);
    });

    it("negates a literal", () => {
      assert.equal(evaluateExpr("-42", {}), -42);
    });
  });

  describe("complex expressions (from spec examples)", () => {
    it("{{ input.url }}", () => {
      assert.equal(
        evaluateExpr("input.url", { input: { url: "https://example.com" } }),
        "https://example.com",
      );
    });

    it("{{ poll.body.status === 'complete' }}", () => {
      assert.equal(
        evaluateExpr("poll.body.status === 'complete'", {
          poll: { body: { status: "complete" } },
        }),
        true,
      );
    });

    it("{{ fan.left.count + fan.right.count }}", () => {
      assert.equal(
        evaluateExpr("fan.left.count + fan.right.count", {
          fan: { left: { count: 3 }, right: { count: 7 } },
        }),
        10,
      );
    });

    it("{{ items[0].name }}", () => {
      assert.equal(
        evaluateExpr("items[0].name", {
          items: [{ name: "Alice" }, { name: "Bob" }],
        }),
        "Alice",
      );
    });

    it("{{ $current.body.status === 'complete' }} (loop condition)", () => {
      assert.equal(
        evaluateExpr("$current.body.status === 'complete'", {
          $current: { body: { status: "pending" } },
        }),
        false,
      );
    });
  });

  describe("error handling", () => {
    it("throws TemplateError on empty expression", () => {
      assert.throws(() => evaluateExpr("", {}), TemplateError);
      assert.throws(() => evaluateExpr("   ", {}), TemplateError);
    });

    it("throws TemplateError on unexpected characters", () => {
      assert.throws(() => evaluateExpr("@foo", {}), TemplateError);
    });

    it("throws TemplateError on unexpected tokens after expression", () => {
      assert.throws(() => evaluateExpr("1 2", {}), TemplateError);
    });
  });
});

// ── hasTemplates ───────────────────────────────────────────────────────────

describe("hasTemplates", () => {
  it("returns true for strings with {{ }}", () => {
    assert.equal(hasTemplates("{{ input.url }}"), true);
    assert.equal(hasTemplates("hello {{ name }}!"), true);
    assert.equal(hasTemplates("{{ a }} and {{ b }}"), true);
  });

  it("returns false for strings without templates", () => {
    assert.equal(hasTemplates("hello world"), false);
    assert.equal(hasTemplates("no templates here"), false);
    assert.equal(hasTemplates(""), false);
  });

  it("returns false for partial template syntax", () => {
    assert.equal(hasTemplates("{ notTemplate }"), false);
    assert.equal(hasTemplates("{{ unclosed"), false);
  });
});

// ── resolveTemplate ────────────────────────────────────────────────────────

describe("resolveTemplate", () => {
  it("resolves a single expression preserving type (number)", () => {
    const result = resolveTemplate("{{ x }}", { x: 42 });
    assert.equal(result, 42);
    assert.equal(typeof result, "number");
  });

  it("resolves a single expression preserving type (object)", () => {
    const obj = { a: 1, b: 2 };
    const result = resolveTemplate("{{ data }}", { data: obj });
    assert.deepEqual(result, obj);
  });

  it("resolves a single expression preserving type (boolean)", () => {
    const result = resolveTemplate("{{ x > 5 }}", { x: 10 });
    assert.equal(result, true);
    assert.equal(typeof result, "boolean");
  });

  it("resolves a single expression preserving type (null)", () => {
    const result = resolveTemplate("{{ x }}", { x: null });
    assert.equal(result, null);
  });

  it("concatenates multi-segment templates as strings", () => {
    const result = resolveTemplate("Hello {{ name }}!", { name: "World" });
    assert.equal(result, "Hello World!");
    assert.equal(typeof result, "string");
  });

  it("handles multiple template segments", () => {
    const result = resolveTemplate("{{ a }} + {{ b }} = {{ a + b }}", {
      a: 3,
      b: 4,
    });
    assert.equal(result, "3 + 4 = 7");
  });

  it("stringifies objects in multi-segment templates", () => {
    const result = resolveTemplate("data: {{ obj }}", {
      obj: { key: "val" },
    });
    assert.equal(result, 'data: {"key":"val"}');
  });

  it("renders null/undefined as empty in multi-segment", () => {
    const result = resolveTemplate("value: {{ x }}", { x: null });
    assert.equal(result, "value: ");
  });

  it("handles whitespace around expression", () => {
    const result = resolveTemplate("{{  x  }}", { x: "trimmed" });
    assert.equal(result, "trimmed");
  });

  it("preserves prefix/suffix text", () => {
    const result = resolveTemplate(
      "/deploy/{{ input.service }}",
      { input: { service: "api" } },
    );
    assert.equal(result, "/deploy/api");
  });
});

// ── resolveConfig ──────────────────────────────────────────────────────────

describe("resolveConfig", () => {
  it("resolves template strings in flat object", () => {
    const config = { url: "{{ input.url }}", method: "GET" };
    const scope = { input: { url: "/api" } };
    assert.deepEqual(resolveConfig(config, scope), {
      url: "/api",
      method: "GET",
    });
  });

  it("resolves nested template strings", () => {
    const config = {
      body: { name: "{{ input.name }}", count: "{{ input.count }}" },
    };
    const scope = { input: { name: "test", count: 5 } };
    const result = resolveConfig(config, scope) as any;
    assert.equal(result.body.name, "test");
    assert.equal(result.body.count, 5);
  });

  it("resolves arrays", () => {
    const config = ["{{ a }}", "{{ b }}", "literal"];
    const scope = { a: 1, b: 2 };
    assert.deepEqual(resolveConfig(config, scope), [1, 2, "literal"]);
  });

  it("passes through non-template strings", () => {
    const config = { url: "/static/path", method: "POST" };
    assert.deepEqual(resolveConfig(config, {}), config);
  });

  it("passes through numbers", () => {
    assert.equal(resolveConfig(42, {}), 42);
  });

  it("passes through booleans", () => {
    assert.equal(resolveConfig(true, {}), true);
    assert.equal(resolveConfig(false, {}), false);
  });

  it("passes through null", () => {
    assert.equal(resolveConfig(null, {}), null);
  });

  it("does not resolve Flow objects (objects with name + steps)", () => {
    const flowObj = { name: "child", steps: [], input: {} };
    const result = resolveConfig(flowObj, { name: "WRONG" });
    assert.equal((result as any).name, "child"); // not resolved
  });

  it("resolves references to previous step outputs", () => {
    const config = {
      url: "{{ input.webhookUrl }}",
      body: "{{ poll.body.result }}",
    };
    const scope = {
      input: { webhookUrl: "/webhook" },
      poll: { body: { result: { data: "done" } } },
    };
    const result = resolveConfig(config, scope) as any;
    assert.equal(result.url, "/webhook");
    assert.deepEqual(result.body, { data: "done" });
  });
});
