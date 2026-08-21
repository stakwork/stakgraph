/**
 * Expression evaluator for {{ ... }} templates.
 *
 * Supports:
 *  - Property access: foo.bar.baz, foo["bar"], arr[0]
 *  - Literals: numbers, strings ('...' or "..."), true, false, null
 *  - Operators: === !== == != < <= > >= && || ! + - * / %
 *  - Ternary: a ? b : c
 *  - Array methods (whitelist): map, filter, find, join, includes, slice —
 *    with single-param arrow lambdas, e.g. {{ search.map(n => n.ref_id) }}.
 *    Expression bodies only (no statements, no user-defined functions), so
 *    every expression still terminates by construction. The whitelist exists
 *    because the language's primary AUTHORS are LLM agents, whose first
 *    attempt is always the JS idiom — grow it only when a real agent misses.
 */

export class TemplateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TemplateError";
  }
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

type TokenType =
  | "number"
  | "string"
  | "ident"
  | "op"
  | "dot"
  | "lbracket"
  | "rbracket"
  | "lparen"
  | "rparen"
  | "question"
  | "colon"
  | "not"
  | "comma"
  | "arrow"
  | "eof";

interface Token {
  type: TokenType;
  value: string;
}

const OPERATORS = [
  "===",
  "!==",
  "==",
  "!=",
  "<=",
  ">=",
  "&&",
  "||",
  "<",
  ">",
  "+",
  "-",
  "*",
  "/",
  "%",
];

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < input.length) {
    if (/\s/.test(input[i]!)) {
      i++;
      continue;
    }

    // String literals
    if (input[i] === "'" || input[i] === '"') {
      const quote = input[i]!;
      let str = "";
      i++;
      while (i < input.length && input[i] !== quote) {
        if (input[i] === "\\") {
          i++;
          str += input[i] ?? "";
        } else {
          str += input[i];
        }
        i++;
      }
      i++;
      tokens.push({ type: "string", value: str });
      continue;
    }

    // Numbers
    if (/[0-9]/.test(input[i]!)) {
      let num = "";
      while (i < input.length && /[0-9.]/.test(input[i]!)) {
        num += input[i];
        i++;
      }
      tokens.push({ type: "number", value: num });
      continue;
    }

    // Arrow (lambda) — before the operator loop so '=' never reaches the
    // unexpected-character error.
    if (input.slice(i, i + 2) === "=>") {
      tokens.push({ type: "arrow", value: "=>" });
      i += 2;
      continue;
    }

    // Multi-char operators
    let matched = false;
    for (const op of OPERATORS) {
      if (input.slice(i, i + op.length) === op) {
        tokens.push({ type: "op", value: op });
        i += op.length;
        matched = true;
        break;
      }
    }
    if (matched) continue;

    const ch = input[i]!;
    if (ch === ".") {
      tokens.push({ type: "dot", value: "." });
      i++;
    } else if (ch === "[") {
      tokens.push({ type: "lbracket", value: "[" });
      i++;
    } else if (ch === "]") {
      tokens.push({ type: "rbracket", value: "]" });
      i++;
    } else if (ch === "(") {
      tokens.push({ type: "lparen", value: "(" });
      i++;
    } else if (ch === ")") {
      tokens.push({ type: "rparen", value: ")" });
      i++;
    } else if (ch === "?") {
      tokens.push({ type: "question", value: "?" });
      i++;
    } else if (ch === ":") {
      tokens.push({ type: "colon", value: ":" });
      i++;
    } else if (ch === "!") {
      tokens.push({ type: "not", value: "!" });
      i++;
    } else if (ch === ",") {
      tokens.push({ type: "comma", value: "," });
      i++;
    } else if (/[a-zA-Z_$]/.test(ch)) {
      let ident = "";
      while (i < input.length && /[a-zA-Z0-9_$]/.test(input[i]!)) {
        ident += input[i];
        i++;
      }
      tokens.push({ type: "ident", value: ident });
    } else {
      throw new TemplateError(`Unexpected character '${ch}' in expression`);
    }
  }

  tokens.push({ type: "eof", value: "" });
  return tokens;
}

// ── Array methods (whitelist) ──────────────────────────────────────────────

const ARRAY_METHODS = ["map", "filter", "find", "join", "includes", "slice"] as const;

/** The teaching suffix: every method error states the supported surface, so
 *  an agent that steps outside it can self-correct from the error alone. */
const METHODS_HINT =
  `supported on arrays: ${ARRAY_METHODS.join(", ")} ` +
  `(single-param arrow lambdas only, e.g. items.map(x => x.ref_id))`;

function callMethod(obj: unknown, method: string, args: unknown[]): unknown {
  if (obj === null || obj === undefined) {
    throw new TemplateError(`Cannot call method '${method}' of ${obj}`);
  }
  if (!Array.isArray(obj)) {
    throw new TemplateError(
      `Cannot call '${method}' on ${typeof obj} — methods are ${METHODS_HINT}`,
    );
  }
  const lambdaArg = (): ((el: unknown) => unknown) => {
    const f = args[0];
    if (typeof f !== "function") {
      throw new TemplateError(
        `${method}() requires a lambda argument, e.g. items.${method}(x => x.name)`,
      );
    }
    return f as (el: unknown) => unknown;
  };
  const plainArgs = (): unknown[] => {
    if (args.some((a) => typeof a === "function")) {
      throw new TemplateError(`${method}() does not take a lambda argument`);
    }
    return args;
  };
  switch (method) {
    case "map": {
      const f = lambdaArg();
      return obj.map((el) => f(el));
    }
    case "filter": {
      const f = lambdaArg();
      return obj.filter((el) => Boolean(f(el)));
    }
    case "find": {
      const f = lambdaArg();
      return obj.find((el) => Boolean(f(el)));
    }
    case "join": {
      const [sep] = plainArgs();
      return obj.join(sep === undefined ? "," : String(sep));
    }
    case "includes":
      return obj.includes(plainArgs()[0]);
    case "slice": {
      const [a, b] = plainArgs();
      return obj.slice(a as number | undefined, b as number | undefined);
    }
    default:
      throw new TemplateError(`Unknown method '${method}' — ${METHODS_HINT}`);
  }
}

// ── Parser (recursive descent, evaluates in one pass) ─────────────────────

function parse(
  tokens: Token[],
  scope: Record<string, unknown>,
): unknown {
  let pos = 0;

  function peek(): Token {
    return tokens[pos]!;
  }

  function advance(): Token {
    const t = tokens[pos]!;
    pos++;
    return t;
  }

  function expect(type: TokenType): Token {
    const t = peek();
    if (t.type !== type) {
      throw new TemplateError(
        `Expected ${type} but got ${t.type} ('${t.value}')`,
      );
    }
    return advance();
  }

  function parseTernary(): unknown {
    const cond = parseOr();
    if (peek().type === "question") {
      advance();
      const then = parseTernary();
      expect("colon");
      const els = parseTernary();
      return cond ? then : els;
    }
    return cond;
  }

  function parseOr(): unknown {
    let left = parseAnd();
    while (peek().type === "op" && peek().value === "||") {
      advance();
      const right = parseAnd();
      left = left || right;
    }
    return left;
  }

  function parseAnd(): unknown {
    let left = parseEquality();
    while (peek().type === "op" && peek().value === "&&") {
      advance();
      const right = parseEquality();
      left = left && right;
    }
    return left;
  }

  function parseEquality(): unknown {
    let left = parseComparison();
    while (
      peek().type === "op" &&
      ["===", "!==", "==", "!="].includes(peek().value)
    ) {
      const op = advance().value;
      const right = parseComparison();
      switch (op) {
        case "===":
          left = left === right;
          break;
        case "!==":
          left = left !== right;
          break;
        case "==":
          left = left == right;
          break;
        case "!=":
          left = left != right;
          break;
      }
    }
    return left;
  }

  function parseComparison(): unknown {
    let left = parseAdditive();
    while (
      peek().type === "op" &&
      ["<", "<=", ">", ">="].includes(peek().value)
    ) {
      const op = advance().value;
      const right = parseAdditive();
      switch (op) {
        case "<":
          left = (left as number) < (right as number);
          break;
        case "<=":
          left = (left as number) <= (right as number);
          break;
        case ">":
          left = (left as number) > (right as number);
          break;
        case ">=":
          left = (left as number) >= (right as number);
          break;
      }
    }
    return left;
  }

  function parseAdditive(): unknown {
    let left = parseMultiplicative();
    while (peek().type === "op" && ["+", "-"].includes(peek().value)) {
      const op = advance().value;
      const right = parseMultiplicative();
      if (op === "+") left = (left as number) + (right as number);
      else left = (left as number) - (right as number);
    }
    return left;
  }

  function parseMultiplicative(): unknown {
    let left = parseUnary();
    while (peek().type === "op" && ["*", "/", "%"].includes(peek().value)) {
      const op = advance().value;
      const right = parseUnary();
      if (op === "*") left = (left as number) * (right as number);
      else if (op === "/") left = (left as number) / (right as number);
      else left = (left as number) % (right as number);
    }
    return left;
  }

  function parseUnary(): unknown {
    if (peek().type === "not") {
      advance();
      return !parseUnary();
    }
    if (peek().type === "op" && peek().value === "-") {
      advance();
      return -(parseUnary() as number);
    }
    return parseAccess();
  }

  function peekAhead(): Token {
    return tokens[pos + 1] ?? { type: "eof", value: "" };
  }

  /** Capture a lambda body's tokens (to the depth-0 `,` or `)` that ends the
   *  argument) WITHOUT evaluating them — the parameter isn't bound yet.
   *  Returns a closure that re-parses the slice per element with the
   *  parameter bound in a child scope (shadowing any outer binding). */
  function parseLambda(): (el: unknown) => unknown {
    const param = expect("ident").value;
    expect("arrow");
    const body: Token[] = [];
    let depth = 0;
    while (true) {
      const t = peek();
      if (t.type === "eof") {
        throw new TemplateError(`Unterminated lambda body for parameter '${param}'`);
      }
      if (depth === 0 && (t.type === "comma" || t.type === "rparen")) break;
      if (t.type === "lparen" || t.type === "lbracket") depth++;
      if (t.type === "rparen" || t.type === "rbracket") depth--;
      body.push(advance());
    }
    if (body.length === 0) {
      throw new TemplateError(`Empty lambda body for parameter '${param}'`);
    }
    const bodyTokens: Token[] = [...body, { type: "eof", value: "" }];
    return (el: unknown) => parse(bodyTokens, { ...scope, [param]: el });
  }

  /** Parse a call's arguments: each is a single-param arrow lambda
   *  (`x => expr`) or an ordinary expression. */
  function parseCallArgs(): unknown[] {
    expect("lparen");
    const args: unknown[] = [];
    while (peek().type !== "rparen") {
      if (peek().type === "ident" && peekAhead().type === "arrow") {
        args.push(parseLambda());
      } else {
        args.push(parseTernary());
      }
      if (peek().type === "comma") advance();
      else break;
    }
    expect("rparen");
    return args;
  }

  function parseAccess(): unknown {
    let obj = parsePrimary();

    while (true) {
      if (peek().type === "dot") {
        advance();
        const prop = expect("ident").value;
        if (peek().type === "lparen") {
          obj = callMethod(obj, prop, parseCallArgs());
          continue;
        }
        if (obj === null || obj === undefined) {
          throw new TemplateError(
            `Cannot access property '${prop}' of ${obj}`,
          );
        }
        obj = (obj as Record<string, unknown>)[prop];
      } else if (peek().type === "lbracket") {
        advance();
        const idx = parseTernary();
        expect("rbracket");
        if (obj === null || obj === undefined) {
          throw new TemplateError(`Cannot index into ${obj}`);
        }
        obj = (obj as Record<string, unknown>)[idx as string];
      } else {
        break;
      }
    }

    return obj;
  }

  function parsePrimary(): unknown {
    const t = peek();

    if (t.type === "number") {
      advance();
      return parseFloat(t.value);
    }

    if (t.type === "string") {
      advance();
      return t.value;
    }

    if (t.type === "ident") {
      advance();
      if (t.value === "true") return true;
      if (t.value === "false") return false;
      if (t.value === "null") return null;
      if (t.value === "undefined") return undefined;

      if (!(t.value in scope)) {
        throw new TemplateError(`Undefined reference '${t.value}'`);
      }
      return scope[t.value];
    }

    if (t.type === "lparen") {
      advance();
      const val = parseTernary();
      expect("rparen");
      return val;
    }

    throw new TemplateError(`Unexpected token '${t.value}' (${t.type})`);
  }

  const result = parseTernary();
  if (peek().type !== "eof") {
    throw new TemplateError(
      `Unexpected token '${peek().value}' after expression`,
    );
  }
  return result;
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Evaluate a single expression string against a scope.
 */
export function evaluateExpr(
  expr: string,
  scope: Record<string, unknown>,
): unknown {
  const trimmed = expr.trim();
  if (!trimmed) {
    throw new TemplateError("Empty expression");
  }
  const tokens = tokenize(trimmed);
  return parse(tokens, scope);
}

const TEMPLATE_RE = /\{\{(.*?)\}\}/g;

/**
 * Check if a string contains template expressions.
 */
export function hasTemplates(value: string): boolean {
  TEMPLATE_RE.lastIndex = 0;
  return TEMPLATE_RE.test(value);
}

/**
 * Resolve a template string. If the entire string is a single `{{ expr }}`,
 * the result preserves the expression's type. Otherwise, segments are
 * stringified and concatenated.
 */
export function resolveTemplate(
  template: string,
  scope: Record<string, unknown>,
): unknown {
  // Check if the entire string is a single {{ expr }}.
  // We trim and check that the string starts with {{ and ends with }}
  // and contains no other {{ }} pairs (which would make it multi-segment).
  const trimmed = template.trim();
  if (
    trimmed.startsWith("{{") &&
    trimmed.endsWith("}}") &&
    trimmed.indexOf("{{", 2) === -1
  ) {
    const expr = trimmed.slice(2, -2);
    return evaluateExpr(expr, scope);
  }

  // Multiple segments — stringify and concat
  TEMPLATE_RE.lastIndex = 0;
  return template.replace(TEMPLATE_RE, (_match, expr: string) => {
    const val = evaluateExpr(expr, scope);
    if (val === null || val === undefined) return "";
    if (typeof val === "object") return JSON.stringify(val);
    return String(val);
  });
}

/**
 * Recursively resolve all template strings in a config object.
 * Returns a deep copy with all templates resolved.
 */
export function resolveConfig(
  config: unknown,
  scope: Record<string, unknown>,
): unknown {
  if (typeof config === "string") {
    if (hasTemplates(config)) {
      return resolveTemplate(config, scope);
    }
    return config;
  }

  if (Array.isArray(config)) {
    return config.map((item) => resolveConfig(item, scope));
  }

  if (config !== null && typeof config === "object") {
    // Don't resolve Zod schemas or Flow objects
    if ("_def" in config || ("name" in config && "steps" in config)) {
      return config;
    }
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(config)) {
      result[key] = resolveConfig(value, scope);
    }
    return result;
  }

  return config;
}
