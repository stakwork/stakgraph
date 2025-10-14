import { getTokenPricing, Provider } from "../aieo/src/provider.js";

export interface BudgetTracker {
  budgetLimit: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
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
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    provider,
  };
}

export function addUsage(
  tracker: BudgetTracker,
  inputTokens: number,
  outputTokens: number,
  provider?: Provider
): BudgetTracker {
  return {
    ...tracker,
    inputTokens: tracker.inputTokens + inputTokens,
    outputTokens: tracker.outputTokens + outputTokens,
    totalTokens: tracker.totalTokens + inputTokens + outputTokens,
    provider: provider || tracker.provider,
  };
}

export function getTotalCost(tracker: BudgetTracker): number {
  const pricing = getTokenPricing(tracker.provider);
  const inputCost = (tracker.inputTokens / 1_000_000) * pricing.inputTokenPrice;
  const outputCost =
    (tracker.outputTokens / 1_000_000) * pricing.outputTokenPrice;
  return inputCost + outputCost;
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
    outputTokens: tracker.outputTokens,
    totalTokens: tracker.totalTokens,
  };
}
