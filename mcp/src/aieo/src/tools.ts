import { createAnthropic } from "@ai-sdk/anthropic";
import { Provider, getGatewayBaseURL } from "./provider.js";
import { createWebSearch } from "./search.js";
import { createWebFetch } from "./fetch.js";

export type ProviderTool = "webSearch" | "webFetch" | "bash";

/**
 * Provider-native tool by name.
 *
 * `webSearch` and `webFetch` are special: only Anthropic has native
 * ones, so every other provider gets a shim of the same name and result
 * shape instead of an exception — Exa-backed search from `./search.js`,
 * a guarded HTTP GET from `./fetch.js`. That keeps both tools available
 * (as `web_search` / `web_fetch`) on any model. This entry point returns
 * the bare tool — for the result bookkeeping, citation indices and
 * prompt snippet, call `createWebSearch` / `createWebFetch` directly.
 *
 * Returns `undefined` for those two when the chosen backend has no key
 * configured; callers should drop the tool rather than fail the request.
 * Other tools still throw for unsupported providers.
 */
export function getProviderTool(
  provider: Provider,
  apiKey: string,
  toolName: ProviderTool
): any {
  if (toolName === "webSearch") {
    return createWebSearch({ provider, apiKey }).tool;
  }
  if (toolName === "webFetch") {
    return createWebFetch({ provider, apiKey }).tool;
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
