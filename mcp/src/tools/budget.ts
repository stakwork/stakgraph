import { computeSessionCost, Provider } from "../aieo/src/provider.js";
import { AiUsage, emptyUsage } from "../aieo/src/usage.js";

export interface BudgetTracker {
  budgetLimit: number;
  usage: AiUsage;
  provider: Provider;
}

export interface BudgetInfo {
  totalCost: number;
  budgetExceeded: boolean;
  remainingBudget: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export function createBudgetTracker(
  budgetDollars: number,
  provider: Provider = "anthropic"
): BudgetTracker {
  return {
    budgetLimit: budgetDollars,
    usage: emptyUsage(),
    provider,
  };
}

export function addUsage(
  tracker: BudgetTracker,
  usage: AiUsage,
  provider?: Provider
): BudgetTracker {
  return {
    ...tracker,
    usage: {
      input: tracker.usage.input + usage.input,
      cache_read: tracker.usage.cache_read + usage.cache_read,
      cache_write: tracker.usage.cache_write + usage.cache_write,
      output: tracker.usage.output + usage.output,
      total: tracker.usage.total + usage.total,
    },
    provider: provider || tracker.provider,
  };
}

export function getTotalCost(tracker: BudgetTracker): number {
  return computeSessionCost(tracker.provider, tracker.usage);
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
    inputTokens: tracker.usage.input + tracker.usage.cache_read + tracker.usage.cache_write,
    outputTokens: tracker.usage.output,
    totalTokens: tracker.usage.total,
  };
}
