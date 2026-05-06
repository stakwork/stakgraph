import type { LanguageModelUsage } from "ai";

export interface TokenPricing {
  inputTokenPrice: number;
  outputTokenPrice: number;
  cacheReadPrice?: number;
  cacheWritePrice?: number;
}

export interface TokenUsageForCost {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
}

export interface SessionTokenUsage extends TokenUsageForCost {
  total: number;
}

export interface AiUsage extends SessionTokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  token_usage: SessionTokenUsage;
}

export type ModelUsage = AiUsage;

export function usageForSession(
  usage?: LanguageModelUsage | Partial<AiUsage> | SessionTokenUsage | null,
): SessionTokenUsage {
  if (!usage) {
    return { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 };
  }

  const tokenUsage = (usage as Partial<AiUsage>).token_usage;
  if (tokenUsage) {
    return {
      input: tokenUsage.input ?? 0,
      cache_read: tokenUsage.cache_read ?? 0,
      cache_write: tokenUsage.cache_write ?? 0,
      output: tokenUsage.output ?? 0,
      total: tokenUsage.total ?? 0,
    };
  }

  const languageUsage = usage as LanguageModelUsage;
  const input =
    languageUsage.inputTokenDetails?.noCacheTokens ??
    (usage as Partial<SessionTokenUsage>).input ??
    languageUsage.inputTokens ??
    0;
  const cache_read =
    languageUsage.inputTokenDetails?.cacheReadTokens ??
    (usage as Partial<SessionTokenUsage>).cache_read ??
    0;
  const cache_write =
    languageUsage.inputTokenDetails?.cacheWriteTokens ??
    (usage as Partial<SessionTokenUsage>).cache_write ??
    0;
  const output =
    (usage as Partial<SessionTokenUsage>).output ??
    languageUsage.outputTokens ??
    0;
  const total =
    (usage as Partial<SessionTokenUsage>).total ??
    languageUsage.totalTokens ??
    input + cache_read + cache_write + output;

  return { input, cache_read, cache_write, output, total };
}

export function normalizeUsage(
  usage?: LanguageModelUsage | Partial<AiUsage> | SessionTokenUsage | null,
): AiUsage {
  const token_usage = usageForSession(usage);
  const languageUsage = usage as LanguageModelUsage | undefined;
  const existing = usage as Partial<AiUsage> | undefined;

  return {
    ...token_usage,
    inputTokens:
      existing?.inputTokens ??
      languageUsage?.inputTokens ??
      token_usage.input + token_usage.cache_read + token_usage.cache_write,
    outputTokens: existing?.outputTokens ?? languageUsage?.outputTokens ?? token_usage.output,
    totalTokens: existing?.totalTokens ?? languageUsage?.totalTokens ?? token_usage.total,
    token_usage,
  };
}

export function emptyUsage(): AiUsage {
  return normalizeUsage(null);
}

export function addUsage(first: Partial<AiUsage>, second: Partial<AiUsage>): AiUsage {
  const a = normalizeUsage(first);
  const b = normalizeUsage(second);
  return normalizeUsage({
    input: a.input + b.input,
    cache_read: a.cache_read + b.cache_read,
    cache_write: a.cache_write + b.cache_write,
    output: a.output + b.output,
    total: a.total + b.total,
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    totalTokens: a.totalTokens + b.totalTokens,
  });
}

export function usageForLegacyResponse(usage: Partial<AiUsage>): AiUsage {
  return normalizeUsage(usage);
}

export function computeUsageCost(
  pricing: TokenPricing,
  usage: TokenUsageForCost,
): number {
  const inputCost = (usage.input / 1_000_000) * pricing.inputTokenPrice;
  const cacheReadCost =
    (usage.cache_read / 1_000_000) *
    (pricing.cacheReadPrice ?? pricing.inputTokenPrice);
  const cacheWriteCost =
    (usage.cache_write / 1_000_000) *
    (pricing.cacheWritePrice ?? pricing.inputTokenPrice);
  const outputCost = (usage.output / 1_000_000) * pricing.outputTokenPrice;
  return inputCost + cacheReadCost + cacheWriteCost + outputCost;
}