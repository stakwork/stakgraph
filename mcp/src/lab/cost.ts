/**
 * Dollar cost of an LLM call at aieo's rates. aieo owns the pricing table
 * (`computeSessionCost` / `getTokenPricing` in `aieo/src/provider.ts`); strut
 * only normalizes the AI SDK's usage object (`usageFromResult` →
 * `{ inputTokens, cacheReadTokens, cacheWriteTokens, outputTokens }`). This is
 * the shape adapter between the two, so the lab steps carry no price table
 * and strut needs none either.
 *
 * An unknown provider string is priced as anthropic (what strut's old
 * `computeCost` did). Pass `modelId` for per-model OpenRouter rates once
 * `loadModelPricing()` has run; otherwise the provider default applies.
 */
// NOT importable from a SEEDED step (src/lab/*/steps/*.ts): those files are
// published verbatim into the workspace's step dir, where `../../cost.js` does
// not exist and the step silently fails to load. Seeded steps inline this
// helper against the `aieo` package instead (see seed-imports.test.ts).
import { computeSessionCost, PROVIDERS, type Provider } from "../aieo/src/provider.js";
import type { TokenUsage } from "strut";

export function costOf(provider: string, usage: TokenUsage, modelId?: string): number {
  const p = PROVIDERS.includes(provider as Provider) ? (provider as Provider) : "anthropic";
  return computeSessionCost(
    p,
    {
      input: usage.inputTokens,
      cache_read: usage.cacheReadTokens,
      cache_write: usage.cacheWriteTokens,
      output: usage.outputTokens,
    },
    modelId,
  );
}
