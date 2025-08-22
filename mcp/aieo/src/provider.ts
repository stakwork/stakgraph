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
  anthropic: "claude-4-sonnet-20250514",
  google: "gemini-2.5-pro-preview-05-06",
  openai: "gpt-4.1",
  claude_code: "sonnet",
};

export async function getModel(
  provider: Provider,
  apiKey: string,
  cwd?: string
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
    case "claude_code":
      const customProvider = createClaudeCode({
        defaultSettings: {
          // Skip permission prompts for all operations
          permissionMode: "bypassPermissions",
          // Set working directory for file operations
          cwd: cwd,
        },
      });
      if (cwd) {
        console.log("creating claude code model at", cwd);
      }
      return customProvider(SOTA[provider]);
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
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
