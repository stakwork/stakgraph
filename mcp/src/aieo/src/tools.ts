import { createAnthropic } from "@ai-sdk/anthropic";
import { Provider, getGatewayBaseURL } from "./provider.js";

export type ProviderTool = "webSearch" | "bash";

export function getProviderTool(
  provider: Provider,
  apiKey: string,
  toolName: ProviderTool
) {
  switch (provider) {
    case "anthropic":
      return getAnthropicTool(apiKey, toolName);
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

function getAnthropicTool(apiKey: string, toolName: ProviderTool) {
  const baseURL = getGatewayBaseURL("anthropic");
  const anthropic = createAnthropic({
    apiKey,
    ...(baseURL && { baseURL }),
  });
  switch (toolName) {
    case "webSearch":
      return anthropic.tools.webSearch_20250305({
        maxUses: 3,
      });
    case "bash":
      return anthropic.tools.bash_20250124({});
    default:
      throw new Error(`Unsupported tool: ${toolName}`);
  }
}
