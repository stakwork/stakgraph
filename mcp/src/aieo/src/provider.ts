import { createAnthropic, AnthropicProviderOptions } from "@ai-sdk/anthropic";
import {
  createGoogleGenerativeAI,
  GoogleGenerativeAIProviderOptions,
} from "@ai-sdk/google";
import { createOpenAI, OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { LanguageModel } from "ai";
import { Logger } from "./logger.js";
import { createOpenRouter, OpenRouterProviderOptions } from "@openrouter/ai-sdk-provider";

export type Provider = "anthropic" | "google" | "openai" | "openrouter";

export const PROVIDERS: Provider[] = [
  "anthropic",
  "google",
  "openai",
  "openrouter",
];

// shortcuts to latest models
export type ModelName = "sonnet" | "opus" | "haiku" | "gemini" | "gpt" | "kimi";

type ModelId = string;

const MODELS: Record<Provider, Partial<Record<ModelName, ModelId>>> = {
  anthropic: {
    sonnet: "claude-sonnet-4-5",
    opus: "claude-opus-4-5",
    haiku: "claude-haiku-4-5",
  },
  google: {
    gemini: "gemini-3-pro-preview",
  },
  openai: {
    gpt: "gpt-5",
  },
  openrouter: {
    kimi: "moonshotai/kimi-k2.5"
  }
};

const DEFAULT_MODELS: Record<Provider, string> = {
  anthropic: MODELS.anthropic.sonnet!,
  google: MODELS.google.gemini!,
  openai: MODELS.openai.gpt!,
  openrouter: MODELS.openrouter.kimi!,
};

export interface TokenPricing {
  inputTokenPrice: number;
  outputTokenPrice: number;
}

export function getProviderForModel(modelName?: ModelName | string): Provider {
  // Handle format like "anthropic/claude-sonnet-4-5" or "openrouter/moonshotai/kimi-k2.5"
  if (modelName && modelName.includes("/")) {
    const provider = modelName.split("/")[0] as Provider;
    if (PROVIDERS.includes(provider)) {
      return provider;
    }
  }
  switch (modelName) {
    case "kimi":
      return "openrouter";
    case "sonnet":
      return "anthropic";
    case "opus":
      return "anthropic";
    case "haiku":
      return "anthropic";
    case "gemini":
      return "google";
    case "gpt":
      return "openai";
    default:
      if (process.env.LLM_PROVIDER && PROVIDERS.includes(process.env.LLM_PROVIDER as Provider)) {
        return process.env.LLM_PROVIDER as Provider;
      }
      return "anthropic";
  }
}

export function getApiKeyForProvider(provider: Provider | string): string {
  let apiKey: string | undefined;
  switch (provider) {
    case "anthropic":
      apiKey = process.env.ANTHROPIC_API_KEY;
      break;
    case "google":
      apiKey = process.env.GOOGLE_API_KEY;
      break;
    case "openai":
      apiKey = process.env.OPENAI_API_KEY;
      break;
    case "openrouter":
      apiKey = process.env.OPENROUTER_API_KEY;
      break;
    case "claude_code":
      apiKey = process.env.CLAUDE_CODE_API_KEY;
      break;
  }
  if (!apiKey) {
    throw new Error(`API key not found for provider: ${provider}`);
  }
  return apiKey;
}

export interface GetModelOptions {
  apiKey?: string;
  modelName?: ModelName | string;
  cwd?: string;
  executablePath?: string;
  logger?: Logger;
}

function getModelForProvider(provider: Provider, modelName: ModelName): string {
  const model = MODELS[provider][modelName];
  if (!model) {
    throw new Error(
      `Model "${modelName}" is not available for provider "${provider}". ` +
        `Available models: ${Object.keys(MODELS[provider]).join(", ")}`
    );
  }
  return model;
}

interface ModelDetails {
  provider: Provider,
  apiKey: string,
  model: LanguageModel
}
export function getModelDetails(modelName?: ModelName | string, apiKeyIn?: string): ModelDetails {
  const provider = getProviderForModel(modelName);
  const apiKey = apiKeyIn || getApiKeyForProvider(provider);
  const model = getModel(provider, {
    modelName,
    apiKey,
  });
  return {model, provider, apiKey}
}

export function getModel(
  provider: Provider,
  opts?: string | GetModelOptions
): LanguageModel {
  if (typeof opts === "string") {
    opts = { apiKey: opts };
  }
  const apiKey = opts?.apiKey || getApiKeyForProvider(provider);
  
  // Handle slash format: extract modelId from "provider/model" or "provider/org/model"
  let modelId: string;
  if (opts?.modelName && opts.modelName.includes("/")) {
    const parts = opts.modelName.split("/");
    if (PROVIDERS.includes(parts[0] as Provider)) {
      modelId = parts.slice(1).join("/");
    } else {
      modelId = opts.modelName;
    }
  } else if (opts?.modelName) {
    modelId = getModelForProvider(provider, opts.modelName as ModelName);
  } else {
    modelId = DEFAULT_MODELS[provider];
  }

  if (opts?.logger) {
    opts.logger.info(
      `Getting model for provider: ${provider}, model: ${modelId}`
    );
  }
  switch (provider) {
    case "anthropic":
      const anthropic = createAnthropic({
        apiKey,
      });
      return anthropic(modelId);
    case "google":
      const google = createGoogleGenerativeAI({
        apiKey,
      });
      return google(modelId);
    case "openai":
      const openai = createOpenAI({
        apiKey,
      });
      return openai(modelId);
    case "openrouter":
      const openrouter = createOpenRouter({
        apiKey,
      });
      return openrouter(modelId);
    // case "claude_code":
    //   try {
    //     const customProvider = createClaudeCode({
    //       defaultSettings: {
    //         pathToClaudeCodeExecutable: executablePath,
    //         // Skip permission prompts for all operations
    //         permissionMode: "bypassPermissions",
    //         // Set working directory for file operations
    //         cwd: cwd,
    //       },
    //     });
    //     if (cwd) {
    //       console.log("creating claude code model at", cwd);
    //     }
    //     return customProvider(SOTA[provider]);
    //   } catch (error) {
    //     console.error("Failed to create Claude Code provider:", error);
    //     throw new Error(
    //       "Claude Code CLI not available or not properly installed. Make sure Claude Code is installed and accessible in the environment where this code runs."
    //     );
    //   }
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

const TOKEN_PRICING: Record<Provider, TokenPricing> = {
  anthropic: {
    inputTokenPrice: 3.0,
    outputTokenPrice: 15.0,
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

export function getTokenPricing(provider: Provider): TokenPricing {
  return TOKEN_PRICING[provider];
}

export type ThinkingSpeed = "thinking" | "fast";

export function getProviderOptions(
  provider: Provider,
  thinkingSpeed?: ThinkingSpeed
) {
  const fast = thinkingSpeed === "fast";
  const budget = fast ? 0 : 24000;
  switch (provider) {
    case "anthropic":
      let thinking = fast
        ? { type: "disabled" as const }
        : { type: "enabled" as const, budgetTokens: budget };
      return {
        anthropic: {
          thinking,
        } satisfies AnthropicProviderOptions,
      };
    case "google":
      return {
        google: {
          thinkingConfig: { thinkingBudget: budget },
        } satisfies GoogleGenerativeAIProviderOptions,
      };
    case "openai":
      return {
        openai: {} satisfies OpenAIResponsesProviderOptions,
      };
    case "openrouter":
      return {
        openrouter: {} satisfies OpenRouterProviderOptions,
      };
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}
