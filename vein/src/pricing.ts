// ── LLM token usage + cost ───────────────────────────────────────────────────
//
// A tiny, provider-aware pricing helper shared by the LLM steps (the core
// `agent` step, plus the lab's eval/reflect + gitsee/score-setup). It normalizes
// the Vercel AI SDK's usage object into a flat { input, cacheRead, cacheWrite,
// output } token count and turns that into a dollar cost.
//
// Pricing table copied from `mcp/src/aieo/src/provider.ts` ($ per 1M tokens).
// Keep it in sync when that table changes.

export type LLMProvider = "anthropic" | "openai" | "google" | "openrouter";

export interface TokenPricing {
  inputTokenPrice: number; // $ per 1M non-cached input tokens
  outputTokenPrice: number; // $ per 1M output tokens
  cacheReadPrice?: number; // $ per 1M cache-read tokens (defaults to input)
  cacheWritePrice?: number; // $ per 1M cache-write tokens (defaults to input)
}

// $ per 1,000,000 tokens. Source of truth: mcp/src/aieo/src/provider.ts.
export const TOKEN_PRICING: Record<LLMProvider, TokenPricing> = {
  anthropic: {
    inputTokenPrice: 3.0,
    outputTokenPrice: 15.0,
    cacheReadPrice: 0.3,
    cacheWritePrice: 3.75,
  },
  google: {
    inputTokenPrice: 1.25,
    outputTokenPrice: 5.0,
  },
  openai: {
    inputTokenPrice: 2.5,
    outputTokenPrice: 10.0,
  },
  openrouter: {
    inputTokenPrice: 0.6,
    outputTokenPrice: 3.0,
  },
};

/** Normalized, provider-agnostic token counts (flat, addable across calls). */
export interface TokenUsage {
  inputTokens: number; // non-cached input (billed at full input price)
  cacheReadTokens: number;
  cacheWriteTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export function emptyUsage(): TokenUsage {
  return { inputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, outputTokens: 0, totalTokens: 0 };
}

const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);

/**
 * Normalize a Vercel AI SDK `LanguageModelUsage` (the `.usage` / `.totalUsage`
 * on a generate result) into a flat {@link TokenUsage}. Prefers the v6
 * `inputTokenDetails` breakdown (noCache / cacheRead / cacheWrite); falls back
 * to the flat `inputTokens` + deprecated `cachedInputTokens` when details are
 * absent (other providers), treating the remainder as non-cached input.
 */
export function usageFromResult(usage: unknown): TokenUsage {
  if (!usage || typeof usage !== "object") return emptyUsage();
  const u = usage as Record<string, any>;
  const details = (u.inputTokenDetails ?? {}) as Record<string, any>;

  const cacheReadTokens = num(details.cacheReadTokens ?? u.cachedInputTokens);
  const cacheWriteTokens = num(details.cacheWriteTokens);
  // Non-cached input: the detailed noCacheTokens when present, else the flat
  // total input minus what we already accounted as cache read/write.
  const inputTokens =
    details.noCacheTokens != null
      ? num(details.noCacheTokens)
      : Math.max(0, num(u.inputTokens) - cacheReadTokens - cacheWriteTokens);
  const outputTokens = num(u.outputTokens);
  const totalTokens =
    num(u.totalTokens) || inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens;

  return { inputTokens, cacheReadTokens, cacheWriteTokens, outputTokens, totalTokens };
}

/** Sum two normalized usages (e.g. across multiple LLM calls in one run). */
export function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    totalTokens: a.totalTokens + b.totalTokens,
  };
}

/** Coerce an unknown (e.g. a usage object threaded through a workflow template)
 *  back into a {@link TokenUsage} so it can be safely summed. */
export function coerceUsage(usage: unknown): TokenUsage {
  if (!usage || typeof usage !== "object") return emptyUsage();
  const u = usage as Record<string, any>;
  return {
    inputTokens: num(u.inputTokens),
    cacheReadTokens: num(u.cacheReadTokens),
    cacheWriteTokens: num(u.cacheWriteTokens),
    outputTokens: num(u.outputTokens),
    totalTokens: num(u.totalTokens) || num(u.inputTokens) + num(u.cacheReadTokens) + num(u.cacheWriteTokens) + num(u.outputTokens),
  };
}

/** Dollar cost of a usage at a provider's rates. Unknown providers default to
 *  anthropic pricing; cache prices default to the input price when unset. */
export function computeCost(provider: string, usage: TokenUsage): number {
  const p = TOKEN_PRICING[provider as LLMProvider] ?? TOKEN_PRICING.anthropic;
  const M = 1_000_000;
  return (
    (usage.inputTokens / M) * p.inputTokenPrice +
    (usage.cacheReadTokens / M) * (p.cacheReadPrice ?? p.inputTokenPrice) +
    (usage.cacheWriteTokens / M) * (p.cacheWritePrice ?? p.inputTokenPrice) +
    (usage.outputTokens / M) * p.outputTokenPrice
  );
}
