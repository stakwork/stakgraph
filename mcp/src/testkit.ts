/**
 * Minimal Playwright-compatible `test` / `expect` shim backed by `node:test`.
 *
 * These unit tests use Playwright only as a generic test runner (no browser).
 * Importing this shim instead of `@playwright/test` lets them run under
 * `node --test` (via `npm run test:node`) without rewriting every assertion.
 */
import {
  describe,
  it,
  before,
  after,
  beforeEach as nodeBeforeEach,
  afterEach as nodeAfterEach,
} from "node:test";
import assert from "node:assert/strict";

type Fn = (...args: any[]) => any;

interface TestApi {
  (name: string, fn: Fn): void;
  describe: (name: string, fn: Fn) => void;
  beforeEach: (fn: Fn) => void;
  afterEach: (fn: Fn) => void;
  beforeAll: (fn: Fn) => void;
  afterAll: (fn: Fn) => void;
  skip: (name: string, fn?: Fn) => void;
  only: (name: string, fn: Fn) => void;
}

const test = ((name: string, fn: Fn) => {
  it(name, fn as any);
}) as TestApi;

test.describe = (name: string, fn: Fn) => {
  describe(name, fn as any);
};
test.beforeEach = (fn: Fn) => nodeBeforeEach(fn as any);
test.afterEach = (fn: Fn) => nodeAfterEach(fn as any);
test.beforeAll = (fn: Fn) => before(fn as any);
test.afterAll = (fn: Fn) => after(fn as any);
test.skip = (name: string, fn?: Fn) => it.skip(name, (fn ?? (() => {})) as any);
test.only = (name: string, fn: Fn) => it.only(name, fn as any);

function isPrimitiveNumber(v: unknown): v is number {
  return typeof v === "number" || typeof v === "bigint";
}

function makeMatchers(actual: any, negate: boolean) {
  const ok = (pass: boolean, message: string) => {
    assert.ok(negate ? !pass : pass, (negate ? "NOT: " : "") + message);
  };
  return {
    toBe(expected: any) {
      if (negate) assert.notStrictEqual(actual, expected);
      else assert.strictEqual(actual, expected);
    },
    toEqual(expected: any) {
      if (negate) assert.notDeepStrictEqual(actual, expected);
      else assert.deepStrictEqual(actual, expected);
    },
    toStrictEqual(expected: any) {
      if (negate) assert.notDeepStrictEqual(actual, expected);
      else assert.deepStrictEqual(actual, expected);
    },
    toContain(sub: any) {
      const has =
        typeof actual === "string"
          ? actual.includes(sub)
          : Array.isArray(actual)
          ? actual.includes(sub)
          : false;
      ok(has, `expected ${JSON.stringify(actual)} to contain ${JSON.stringify(sub)}`);
    },
    toMatch(re: RegExp | string) {
      const pass = re instanceof RegExp ? re.test(actual) : String(actual).includes(re);
      ok(pass, `expected ${JSON.stringify(actual)} to match ${re}`);
    },
    toBeTruthy() {
      ok(Boolean(actual), `expected ${JSON.stringify(actual)} to be truthy`);
    },
    toBeFalsy() {
      ok(!actual, `expected ${JSON.stringify(actual)} to be falsy`);
    },
    toBeDefined() {
      ok(actual !== undefined, `expected value to be defined`);
    },
    toBeUndefined() {
      ok(actual === undefined, `expected value to be undefined`);
    },
    toBeNull() {
      ok(actual === null, `expected value to be null`);
    },
    toBeNaN() {
      ok(Number.isNaN(actual), `expected ${actual} to be NaN`);
    },
    toBeGreaterThan(n: number) {
      ok(isPrimitiveNumber(actual) && actual > n, `expected ${actual} > ${n}`);
    },
    toBeGreaterThanOrEqual(n: number) {
      ok(isPrimitiveNumber(actual) && actual >= n, `expected ${actual} >= ${n}`);
    },
    toBeLessThan(n: number) {
      ok(isPrimitiveNumber(actual) && actual < n, `expected ${actual} < ${n}`);
    },
    toBeLessThanOrEqual(n: number) {
      ok(isPrimitiveNumber(actual) && actual <= n, `expected ${actual} <= ${n}`);
    },
    toBeCloseTo(n: number, digits = 2) {
      const pass = Math.abs(actual - n) < Math.pow(10, -digits) / 2;
      ok(pass, `expected ${actual} to be close to ${n}`);
    },
    toHaveLength(n: number) {
      ok(actual?.length === n, `expected length ${actual?.length} to be ${n}`);
    },
    toBeInstanceOf(cls: any) {
      ok(actual instanceof cls, `expected value to be instance of ${cls?.name}`);
    },
    toThrow(expected?: RegExp | string) {
      let thrown: unknown;
      let threw = false;
      try {
        (actual as Fn)();
      } catch (e) {
        threw = true;
        thrown = e;
      }
      if (negate) {
        ok(!threw, `expected function not to throw, but it threw ${thrown}`);
        return;
      }
      ok(threw, `expected function to throw`);
      if (expected !== undefined) {
        const msg = thrown instanceof Error ? thrown.message : String(thrown);
        const pass = expected instanceof RegExp ? expected.test(msg) : msg.includes(expected);
        assert.ok(pass, `expected error message ${JSON.stringify(msg)} to match ${expected}`);
      }
    },
  };
}

/** Awaits `promise`, applying matchers to the resolved value (resolves) or the rejection error (rejects). */
function makeAsyncMatchers(promise: any, wantReject: boolean) {
  return new Proxy(
    {},
    {
      get(_t, prop: string) {
        return async (...args: any[]) => {
          let value: unknown;
          let error: unknown;
          let rejected = false;
          try {
            value = await promise;
          } catch (e) {
            rejected = true;
            error = e;
          }
          if (wantReject) {
            assert.ok(rejected, `expected promise to reject`);
            if (prop === "toThrow") {
              if (args[0] !== undefined) {
                const msg = error instanceof Error ? error.message : String(error);
                const pass =
                  args[0] instanceof RegExp ? args[0].test(msg) : msg.includes(args[0]);
                assert.ok(pass, `expected rejection message ${JSON.stringify(msg)} to match ${args[0]}`);
              }
              return;
            }
            return (makeMatchers(error, false) as any)[prop](...args);
          }
          assert.ok(!rejected, `expected promise to resolve, but it rejected with ${error}`);
          return (makeMatchers(value, false) as any)[prop](...args);
        };
      },
    }
  );
}

export function expect(actual: any) {
  const matchers = makeMatchers(actual, false) as ReturnType<typeof makeMatchers> & {
    not: ReturnType<typeof makeMatchers>;
    rejects: any;
    resolves: any;
  };
  matchers.not = makeMatchers(actual, true);
  matchers.rejects = makeAsyncMatchers(actual, true);
  matchers.resolves = makeAsyncMatchers(actual, false);
  return matchers;
}

export { test };
