import type { LanguageModelUsage, ModelMessage, SystemModelMessage, ToolSet } from "ai";
import {
  applyProviderDefaultsToMessages,
  applyProviderDefaultsToSystem,
  applyProviderDefaultsToTools,
  getProviderOptions,
  Provider,
  ThinkingSpeed,
} from "./provider.js";
import { AiUsage, normalizeUsage } from "./usage.js";

export interface ProviderPolicy {
  providerOptions: ReturnType<typeof getProviderOptions>;
  applySystem(
    system: string | SystemModelMessage | SystemModelMessage[] | undefined,
  ): ReturnType<typeof applyProviderDefaultsToSystem>;
  applyMessages(messages: ModelMessage[]): ModelMessage[];
  applyTools<T extends ToolSet | undefined>(tools: T): T;
  normalizeUsage(usage?: LanguageModelUsage | Partial<AiUsage> | null): AiUsage;
}

export function getProviderPolicy(
  provider: Provider,
  thinkingSpeed?: ThinkingSpeed,
): ProviderPolicy {
  return {
    providerOptions: getProviderOptions(provider, thinkingSpeed),
    applySystem: (system) => applyProviderDefaultsToSystem(provider, system),
    applyMessages: (messages) => applyProviderDefaultsToMessages(provider, messages),
    applyTools: (tools) => applyProviderDefaultsToTools(provider, tools),
    normalizeUsage,
  };
}