export interface AiUsage {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
}

export interface AiUsageWithLegacy extends AiUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

type RawUsage = {
  inputTokens?: number;
  inputTokenDetails?: {
    noCacheTokens?: number;
    cacheReadTokens?: number;
    cacheWriteTokens?: number;
  };
  outputTokens?: number;
  totalTokens?: number;
};

function tokenCount(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function emptyUsage(): AiUsage {
  return {
    input: 0,
    cache_read: 0,
    cache_write: 0,
    output: 0,
    total: 0,
  };
}

export function withLegacyUsage(usage: AiUsage): AiUsageWithLegacy {
  const inputTokens = usage.input + usage.cache_read + usage.cache_write;
  return {
    ...usage,
    inputTokens,
    outputTokens: usage.output,
    totalTokens: usage.total,
  };
}

export function normalizeUsage(raw?: RawUsage | null): AiUsageWithLegacy {
  const inputTokens = tokenCount(raw?.inputTokens);
  const cache_read = tokenCount(raw?.inputTokenDetails?.cacheReadTokens);
  const cache_write = tokenCount(raw?.inputTokenDetails?.cacheWriteTokens);
  const input =
    raw?.inputTokenDetails?.noCacheTokens !== undefined
      ? tokenCount(raw.inputTokenDetails.noCacheTokens)
      : Math.max(inputTokens - cache_read - cache_write, 0);
  const output = tokenCount(raw?.outputTokens);
  const total = tokenCount(raw?.totalTokens) || input + cache_read + cache_write + output;

  return withLegacyUsage({
    input,
    cache_read,
    cache_write,
    output,
    total,
  });
}

export function addUsage(...usages: Array<AiUsage | undefined | null>): AiUsage {
  return usages.reduce<AiUsage>((total, usage) => {
    if (!usage) return total;
    return {
      input: total.input + usage.input,
      cache_read: total.cache_read + usage.cache_read,
      cache_write: total.cache_write + usage.cache_write,
      output: total.output + usage.output,
      total: total.total + usage.total,
    };
  }, emptyUsage());
}