import { test } from "node:test";
import assert from "node:assert/strict";
import {
  usageFromResult,
  coerceUsage,
  addUsage,
  computeCost,
  emptyUsage,
  TOKEN_PRICING,
} from "./pricing.js";

test("usageFromResult prefers the v6 inputTokenDetails breakdown", () => {
  const u = usageFromResult({
    inputTokens: 1000,
    outputTokens: 200,
    totalTokens: 1200,
    inputTokenDetails: { noCacheTokens: 600, cacheReadTokens: 300, cacheWriteTokens: 100 },
  });
  assert.deepEqual(u, {
    inputTokens: 600,
    cacheReadTokens: 300,
    cacheWriteTokens: 100,
    outputTokens: 200,
    totalTokens: 1200,
  });
});

test("usageFromResult falls back to flat inputTokens minus cache when details absent", () => {
  const u = usageFromResult({ inputTokens: 1000, outputTokens: 200, cachedInputTokens: 300 });
  // non-cached = 1000 - 300 cacheRead - 0 cacheWrite
  assert.equal(u.inputTokens, 700);
  assert.equal(u.cacheReadTokens, 300);
  assert.equal(u.cacheWriteTokens, 0);
  assert.equal(u.outputTokens, 200);
  assert.equal(u.totalTokens, 700 + 300 + 200);
});

test("usageFromResult is safe on null / garbage", () => {
  assert.deepEqual(usageFromResult(null), emptyUsage());
  assert.deepEqual(usageFromResult(undefined), emptyUsage());
  assert.deepEqual(usageFromResult(42 as unknown), emptyUsage());
});

test("computeCost applies anthropic per-1M rates across all token classes", () => {
  const cost = computeCost("anthropic", {
    inputTokens: 1_000_000,
    cacheReadTokens: 1_000_000,
    cacheWriteTokens: 1_000_000,
    outputTokens: 1_000_000,
    totalTokens: 4_000_000,
  });
  const p = TOKEN_PRICING.anthropic;
  assert.equal(cost, p.inputTokenPrice + p.cacheReadPrice! + p.cacheWritePrice! + p.outputTokenPrice);
});

test("computeCost defaults unknown providers to anthropic pricing", () => {
  const usage = { inputTokens: 1_000_000, cacheReadTokens: 0, cacheWriteTokens: 0, outputTokens: 0, totalTokens: 1_000_000 };
  assert.equal(computeCost("nope", usage), TOKEN_PRICING.anthropic.inputTokenPrice);
});

test("addUsage / coerceUsage sum token classes for cross-call totals", () => {
  const a = coerceUsage({ inputTokens: 10, cacheReadTokens: 5, outputTokens: 2 });
  const b = usageFromResult({ inputTokens: 20, outputTokens: 3, inputTokenDetails: { noCacheTokens: 20 } });
  const sum = addUsage(a, b);
  assert.equal(sum.inputTokens, 30);
  assert.equal(sum.cacheReadTokens, 5);
  assert.equal(sum.outputTokens, 5);
});
