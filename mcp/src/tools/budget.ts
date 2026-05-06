import { computeSessionCost, Provider } from "../aieo/src/provider.js";

export interface BudgetTracker {
  budgetLimit: number;
  inputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  outputTokens: number;
  totalTokens: number;
  provider: Provider;
}

export interface BudgetInfo {
  totalCost: number;
  budgetExceeded: boolean;
  remainingBudget: number;
  inputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export function createBudgetTracker(
  budgetDollars: number,
  provider: Provider = "anthropic"
): BudgetTracker {
  return {
    budgetLimit: budgetDollars,
    inputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    provider,
  };
}

export function addUsage(
  tracker: BudgetTracker,
  usage: { input: number; output: number; totalTokens?: number; cache_read?: number; cache_write?: number },
  provider?: Provider
): BudgetTracker {
  return {
    ...tracker,
    inputTokens: tracker.inputTokens + usage.input,
    cacheReadTokens: tracker.cacheReadTokens + (usage.cache_read ?? 0),
    cacheWriteTokens: tracker.cacheWriteTokens + (usage.cache_write ?? 0),
    outputTokens: tracker.outputTokens + usage.output,
    totalTokens: tracker.totalTokens + (usage.totalTokens ?? usage.input + (usage.cache_read ?? 0) + (usage.cache_write ?? 0) + usage.output),
    provider: provider || tracker.provider,
  };
}

export function getTotalCost(tracker: BudgetTracker): number {
  return computeSessionCost(tracker.provider, {
    input: tracker.inputTokens,
    cache_read: tracker.cacheReadTokens,
    cache_write: tracker.cacheWriteTokens,
    output: tracker.outputTokens,
  });
}

export function isBudgetExceeded(tracker: BudgetTracker): boolean {
  return getTotalCost(tracker) >= tracker.budgetLimit;
}

export function getRemainingBudget(tracker: BudgetTracker): number {
  return Math.max(0, tracker.budgetLimit - getTotalCost(tracker));
}

export function getBudgetInfo(tracker: BudgetTracker): BudgetInfo {
  const totalCost = getTotalCost(tracker);
  return {
    totalCost,
    budgetExceeded: totalCost >= tracker.budgetLimit,
    remainingBudget: Math.max(0, tracker.budgetLimit - totalCost),
    inputTokens: tracker.inputTokens,
    cacheReadTokens: tracker.cacheReadTokens,
    cacheWriteTokens: tracker.cacheWriteTokens,
    outputTokens: tracker.outputTokens,
    totalTokens: tracker.totalTokens,
  };
}
