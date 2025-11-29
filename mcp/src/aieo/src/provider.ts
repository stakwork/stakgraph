import { createAnthropic, AnthropicProviderOptions } from "@ai-sdk/anthropic";
import {
  createGoogleGenerativeAI,
  GoogleGenerativeAIProviderOptions,
} from "@ai-sdk/google";
import { createOpenAI, OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { createClaudeCode } from "ai-sdk-provider-claude-code";

export type Provider = "anthropic" | "google" | "openai" | "claude_code";

export const PROVIDERS: Provider[] = [
  "anthropic",
  "google",
  "openai",
  "claude_code",
];

const SOTA = {
  anthropic: "claude-sonnet-4-5",
  google: "gemini-2.5-pro-preview-05-06",
  openai: "gpt-5",
  claude_code: "sonnet",
};

export interface TokenPricing {
  inputTokenPrice: number;
  outputTokenPrice: number;
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
  claude_code: {
    inputTokenPrice: 3.0,
    outputTokenPrice: 15.0,
  },
};

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
    case "claude_code":
      apiKey = process.env.CLAUDE_CODE_API_KEY;
      break;
  }
  if (!apiKey) {
    throw new Error(`API key not found for provider: ${provider}`);
  }
  return apiKey;
}

export async function getModel(
  provider: Provider,
  apiKey: string,
  cwd?: string,
  executablePath?: string
) {
  switch (provider) {
    case "anthropic":
      const anthropic = createAnthropic({
        apiKey,
      });
      return anthropic(SOTA[provider]);
    case "google":
      const google = createGoogleGenerativeAI({
        apiKey,
      });
      return google(SOTA[provider]);
    case "openai":
      const openai = createOpenAI({
        apiKey,
      });
      return openai(SOTA[provider]);
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
    case "claude_code":
      return;
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}
