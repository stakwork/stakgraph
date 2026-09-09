import { createAnthropic } from "@ai-sdk/anthropic";
import { Provider, getGatewayBaseURL } from "./provider.js";
import { createWebSearch } from "./search.js";

export type ProviderTool = "webSearch" | "bash";

/**
 * Provider-native tool by name.
 *
 * `webSearch` is special: only Anthropic has a native one, so every
 * other provider gets the Exa-backed shim from `./search.js` instead of
 * an exception. That keeps the tool available (and named `web_search`)
 * on any model. This entry point returns the bare tool — for citation
 * indices and the matching prompt snippet, call `createWebSearch`
 * directly.
 *
 * Returns `undefined` for `webSearch` when the chosen backend has no key
 * configured; callers should drop the tool rather than fail the request.
 * Non-`webSearch` tools still throw for unsupported providers.
 */
export function getProviderTool(
  provider: Provider,
  apiKey: string,
  toolName: ProviderTool
): any {
  if (toolName === "webSearch") {
    return createWebSearch({ provider, apiKey }).tool;
  }
  switch (provider) {
    case "anthropic":
      return getAnthropicTool(apiKey, toolName);
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

function getAnthropicTool(apiKey: string, toolName: ProviderTool): any {
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
