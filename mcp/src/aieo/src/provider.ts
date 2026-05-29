import { createAnthropic, AnthropicProviderOptions } from "@ai-sdk/anthropic";
import {
  createGoogleGenerativeAI,
  GoogleGenerativeAIProviderOptions,
} from "@ai-sdk/google";
import { createOpenAI, OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { LanguageModel } from "ai";
import { Logger } from "./logger.js";
import { createOpenRouter, OpenRouterModelOptions } from "@openrouter/ai-sdk-provider";

export type Provider = "anthropic" | "google" | "openai" | "openrouter";

/**
 * Optional LLM gateway URL (e.g. Bifrost: https://github.com/maximhq/bifrost,
 * LiteLLM, Portkey, etc.). When set, all provider SDKs are routed through this
 * gateway using each provider's drop-in path. Leave unset to call providers directly.
 */
const LLM_GATEWAY_URL = process.env.LLM_GATEWAY_URL?.replace(/\/$/, "");

/**
 * Per-provider path Bifrost (and compatible gateways) expose for each
 * provider's drop-in SDK. Exported so spawners can build per-provider
 * URLs without hand-rolling the suffix.
 *
 * NOTE: OpenRouter has no dedicated Bifrost route and rides the OpenAI
 * path. If you spawn an agent in a way where the runtime would override
 * the model to an OpenRouter model, you want the OpenAI suffix here too.
 */
export const GATEWAY_PATHS: Record<Provider, string> = {
  anthropic: "/anthropic/v1",
  google: "/genai/v1beta",
  openai: "/openai/v1",
  openrouter: "/openai/v1",
};

export function getGatewayBaseURL(provider: Provider): string | undefined {
  return LLM_GATEWAY_URL ? `${LLM_GATEWAY_URL}${GATEWAY_PATHS[provider]}` : undefined;
}

/**
 * Build the fully-formed `<base>/<provider-path>` URL that an official
 * SDK (or Goose, or a workflow LLM node) expects.
 *
 * This is the single source of truth for "which suffix do I add to the
 * gateway root for this provider." Spawners (Hive, Stakwork, anyone
 * launching an agent) should call this when stamping `ANTHROPIC_BASE_URL`,
 * `OPENAI_BASE_URL`, `GOOGLE_BASE_URL`, etc. on a child process or
 * workflow context.
 *
 * Behavior:
 *  - Trims a trailing slash.
 *  - Returns the URL unchanged if it already ends with the provider's
 *    gateway path (`/anthropic/v1`, `/openai/v1`, `/genai/v1beta`), so
 *    the function is idempotent.
 *  - Returns the URL unchanged if it already targets a provider/version
 *    path (e.g. `.../v1`, `.../v1beta`, `.../anthropic/...`). This lets
 *    callers pass fully-qualified URLs through without surprise.
 *  - Otherwise appends `GATEWAY_PATHS[provider]`.
 *
 * Examples
 * --------
 * gatewayUrlFor("anthropic", "https://swarm38.sphinx.chat:8181")
 *   => "https://swarm38.sphinx.chat:8181/anthropic/v1"
 *
 * gatewayUrlFor("openai", "https://swarm38.sphinx.chat:8181/")
 *   => "https://swarm38.sphinx.chat:8181/openai/v1"
 *
 * gatewayUrlFor("anthropic", "https://swarm38.sphinx.chat:8181/anthropic/v1")
 *   => "https://swarm38.sphinx.chat:8181/anthropic/v1"   (idempotent)
 *
 * gatewayUrlFor("openai", "https://api.openai.com/v1")
 *   => "https://api.openai.com/v1"                       (left alone)
 */
export function gatewayUrlFor(provider: Provider, baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/$/, "");
  const providerPath = GATEWAY_PATHS[provider];
  // Already provider-suffixed with the exact path we'd add.
  if (trimmed.endsWith(providerPath)) {
    return trimmed;
  }
  // Already targets a specific provider/version path; trust the caller.
  // Common shapes: `.../anthropic/v1`, `.../openai/v1`, `.../genai/v1beta`,
  // `.../v1`, `.../v1beta`.
  if (/\/(anthropic|openai|genai)\/v\d+[a-z]*$/i.test(trimmed) ||
      /\/v\d+[a-z]*$/i.test(trimmed)) {
    return trimmed;
  }
  return `${trimmed}${providerPath}`;
}

/**
 * Like {@link gatewayUrlFor} but takes a model name (shortcut like
 * `"sonnet"`, namespaced like `"anthropic/claude-sonnet-4-6"`, or a full
 * model id like `"claude-sonnet-4-6"`) and resolves the provider for you.
 *
 * Convenient for spawners that have a model name in hand but not a
 * provider — e.g. Hive picking up a user's chosen model and needing to
 * tell the agent which Bifrost route to use.
 */
export function gatewayUrlForModel(modelName: string | undefined, baseUrl: string): string {
  return gatewayUrlFor(getProviderForModel(modelName), baseUrl);
}

/**
 * Build a `{ ANTHROPIC_BASE_URL, OPENAI_BASE_URL, GOOGLE_BASE_URL }`-style
 * env map for spawning agents (Goose, Python, raw SDK callers) that
 * don't normalize the URL themselves. Pass the gateway root once; get
 * back the per-provider URLs.
 *
 * The keys match the env var names the official SDKs read by default:
 *   - `ANTHROPIC_BASE_URL`        (anthropic-sdk-{ts,python,go})
 *   - `OPENAI_BASE_URL`           (openai SDK)
 *   - `GOOGLE_GENERATIVE_AI_BASE_URL` (google-genai SDK / `@ai-sdk/google`)
 *
 * Spawners can spread the result into the child's env. Callers that
 * only need one provider should use {@link gatewayUrlFor} directly.
 */
export function gatewayEnvForProviders(baseUrl: string): {
  ANTHROPIC_BASE_URL: string;
  OPENAI_BASE_URL: string;
  GOOGLE_GENERATIVE_AI_BASE_URL: string;
} {
  return {
    ANTHROPIC_BASE_URL: gatewayUrlFor("anthropic", baseUrl),
    OPENAI_BASE_URL: gatewayUrlFor("openai", baseUrl),
    GOOGLE_GENERATIVE_AI_BASE_URL: gatewayUrlFor("google", baseUrl),
  };
}

/**
 * @deprecated Use {@link gatewayUrlFor}. Kept as an alias for internal
 * call sites; both have identical behavior.
 */
export const normalizeCallerBaseURL = gatewayUrlFor;

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
    sonnet: "claude-sonnet-4-6",
    opus: "claude-opus-4-6",
    haiku: "claude-haiku-4-5",
  },
  google: {
    gemini: "gemini-3-pro-preview",
  },
  openai: {
    gpt: "gpt-5",
  },
  openrouter: {
    kimi: "moonshotai/kimi-k2.6"
  }
};

const DEFAULT_MODELS: Record<Provider, string> = {
  anthropic: MODELS.anthropic.sonnet!,
  google: MODELS.google.gemini!,
  openai: MODELS.openai.gpt!,
  openrouter: MODELS.openrouter.kimi!,
};

// Light/cheap models for batch operations (descriptions, learnings, etc.)
const LIGHT_MODELS: Record<Provider, string> = {
  anthropic: "claude-haiku-4-5",
  google: "gemini-2.0-flash",
  openai: "gpt-4.1-mini",
  openrouter: "moonshotai/kimi-k2.6",
};

export function getLightModelForProvider(provider: Provider): string {
  return LIGHT_MODELS[provider];
}

export interface TokenPricing {
  inputTokenPrice: number;
  outputTokenPrice: number;
  cacheReadPrice?: number;
  cacheWritePrice?: number;
}

export interface TokenUsageForCost {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
}

export function computeSessionCost(provider: Provider, usage: TokenUsageForCost): number {
  const pricing = TOKEN_PRICING[provider];
  if (!pricing) return 0;
  const inputCost = (usage.input / 1_000_000) * pricing.inputTokenPrice;
  const cacheReadCost = (usage.cache_read / 1_000_000) * (pricing.cacheReadPrice ?? pricing.inputTokenPrice);
  const cacheWriteCost = (usage.cache_write / 1_000_000) * (pricing.cacheWritePrice ?? pricing.inputTokenPrice);
  const outputCost = (usage.output / 1_000_000) * pricing.outputTokenPrice;
  return inputCost + cacheReadCost + cacheWriteCost + outputCost;
}

export function getProviderForModel(modelName?: ModelName | string): Provider {
  // Handle format like "anthropic/claude-sonnet-4-5" or "openrouter/moonshotai/kimi-k2.6"
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
    // Full model IDs
    case "claude-sonnet-4-6":
    case "claude-opus-4-6":
    case "claude-haiku-4-5":
      return "anthropic";
    case "gemini-3-pro-preview":
    case "gemini-2.0-flash":
      return "google";
    case "gpt-5":
    case "gpt-4.1-mini":
      return "openai";
    default:
      if (
        process.env.LLM_PROVIDER &&
        PROVIDERS.includes(process.env.LLM_PROVIDER as Provider)
      ) {
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
  baseUrl?: string;
  modelName?: ModelName | string;
  cwd?: string;
  executablePath?: string;
  logger?: Logger;
  /**
   * Custom HTTP headers to attach to every request the provider client makes
   * to the LLM endpoint. Useful for gateway auth, tenant IDs, etc.
   */
  headers?: Record<string, string>;
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
  model: LanguageModel,
  modelId: string,
  contextLimit: number,
}
export function getModelDetails(
  modelName?: ModelName | string,
  apiKeyIn?: string,
  baseUrl?: string,
  headers?: Record<string, string>,
): ModelDetails {
  const provider = getProviderForModel(modelName);
  const apiKey = apiKeyIn || getApiKeyForProvider(provider);
  console.log("===> getModelDetails", {
    provider,
    modelName: modelName || "(default)",
    keySource: apiKeyIn ? "request body" : "env var",
    apiKeyPrefix: apiKey ? apiKey.slice(0, 12) + "..." : "(missing)",
    baseUrl: baseUrl || "(default)",
    hasHeaders: Boolean(headers && Object.keys(headers).length > 0),
  });
  const model = getModel(provider, {
    modelName,
    apiKey,
    baseUrl,
    headers,
  });
  // Resolve the actual modelId to look up context limit
  let modelId: string;
  if (modelName && modelName.includes("/")) {
    const parts = modelName.split("/");
    if (PROVIDERS.includes(parts[0] as Provider)) {
      modelId = parts.slice(1).join("/");
    } else {
      modelId = modelName;
    }
  } else if (modelName) {
    modelId = MODELS[provider][modelName as ModelName] || DEFAULT_MODELS[provider];
  } else {
    modelId = DEFAULT_MODELS[provider];
  }
  const contextLimit = getContextLimit(modelId, provider);
  return { model, provider, apiKey, contextLimit, modelId }
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
    const knownShortcuts: string[] = [
      "sonnet",
      "opus",
      "haiku",
      "gemini",
      "gpt",
      "kimi",
    ];
    if (knownShortcuts.includes(opts.modelName)) {
      modelId = getModelForProvider(provider, opts.modelName as ModelName);
    } else {
      modelId = opts.modelName;
    }
  } else {
    modelId = DEFAULT_MODELS[provider];
  }

  if (opts?.logger) {
    opts.logger.info(
      `Getting model for provider: ${provider}, model: ${modelId}`
    );
  }
  // Explicit baseUrl from caller takes precedence over the global LLM gateway.
  // Normalize caller-supplied URLs so a bare gateway root (e.g. the swarm
  // host:port) gets the provider's drop-in path appended.
  const baseURL = opts?.baseUrl
    ? gatewayUrlFor(provider, opts.baseUrl)
    : getGatewayBaseURL(provider);
  if (baseURL) {
    const source = opts?.baseUrl ? "caller" : "LLM_GATEWAY";
    console.log(`[${source}] routing ${provider} via ${baseURL}`);
  }
  const extraHeaders = opts?.headers && Object.keys(opts.headers).length > 0
    ? opts.headers
    : undefined;
  if (extraHeaders) {
    console.log(`[headers] custom headers present for ${provider} client`);
  }
  switch (provider) {
    case "anthropic":
      const anthropic = createAnthropic({
        apiKey,
        ...(baseURL && { baseURL }),
        ...(extraHeaders && { headers: extraHeaders }),
      });
      return anthropic(modelId);
    case "google":
      const google = createGoogleGenerativeAI({
        apiKey,
        ...(baseURL && { baseURL }),
        ...(extraHeaders && { headers: extraHeaders }),
      });
      return google(modelId);
    case "openai":
      const openai = createOpenAI({
        apiKey,
        ...(baseURL && { baseURL }),
        ...(extraHeaders && { headers: extraHeaders }),
      });
      return openai(modelId);
    case "openrouter":
      const openrouter = createOpenRouter({
        apiKey,
        ...(baseURL && { baseURL }),
        ...(extraHeaders && { headers: extraHeaders }),
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

// Context window sizes (input token limits) per model.
// For models not listed here, falls back to provider default.
const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  // Anthropic
  "claude-sonnet-4-6": 1_000_000,
  "claude-opus-4-6": 1_000_000,
  "claude-haiku-4-5": 200_000,
  // Google
  "gemini-3-pro-preview": 1_000_000,
  "gemini-2.0-flash": 1_000_000,
  // OpenAI
  "gpt-5": 128_000,
  "gpt-4.1-mini": 1_000_000,
  // OpenRouter
  "moonshotai/kimi-k2.6": 128_000,
};

const DEFAULT_CONTEXT_LIMITS: Record<Provider, number> = {
  anthropic: 1_000_000,
  google: 1_000_000,
  openai: 128_000,
  openrouter: 128_000,
};

// Conservative fallback if both model and provider lookups miss
const FALLBACK_CONTEXT_LIMIT = 128_000;

/**
 * Get the context window size (max input tokens) for a given model.
 */
export function getContextLimit(modelId: string, provider: Provider): number {
  return MODEL_CONTEXT_LIMITS[modelId] ?? DEFAULT_CONTEXT_LIMITS[provider] ?? FALLBACK_CONTEXT_LIMIT;
}

const TOKEN_PRICING: Record<Provider, TokenPricing> = {
  anthropic: {
    inputTokenPrice: 3.0,
    outputTokenPrice: 15.0,
    cacheReadPrice: 0.3,
    cacheWritePrice: 3.75,
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

// Anthropic models that do NOT support `adaptive` thinking. Adaptive is only
// available on certain Sonnet/Opus tiers; Haiku and older models reject it
// with `invalid_request_error: adaptive thinking is not supported on this model`.
function anthropicSupportsAdaptiveThinking(modelName?: string): boolean {
  if (!modelName) return false;
  const m = modelName.toLowerCase();
  // Haiku family does not support adaptive thinking.
  if (m.includes("haiku")) return false;
  // Conservatively only enable adaptive thinking for known-supported families.
  // Sonnet 4.5+ and Opus 4.5+ support it.
  return m.includes("sonnet") || m.includes("opus");
}

export function getProviderOptions(
  provider: Provider,
  thinkingSpeed?: ThinkingSpeed,
  modelName?: string
) {
  const fast = thinkingSpeed === "fast";
  const explicitThinking = thinkingSpeed === "thinking";
  // Budget only applies when thinking is explicitly enabled.
  // For Google we still need a numeric value; use 0 for fast, otherwise let it think.
  const googleBudget = fast ? 0 : 24000;
  switch (provider) {
    case "anthropic":
      let thinking: AnthropicProviderOptions["thinking"];
      if (fast) {
        thinking = { type: "disabled" };
      } else if (explicitThinking) {
        thinking = { type: "enabled", budgetTokens: 24000 };
      } else if (anthropicSupportsAdaptiveThinking(modelName)) {
        // Default for capable models: let the model decide, with summarized thinking output.
        thinking = { type: "adaptive", display: "summarized" };
      } else {
        // Models that don't support adaptive thinking (e.g. Haiku): disable it.
        thinking = { type: "disabled" };
      }
      return {
        anthropic: {
          thinking,
          cacheControl: { type: "ephemeral" },
        } satisfies AnthropicProviderOptions,
      };
    case "google":
      return {
        google: {
          thinkingConfig: { thinkingBudget: googleBudget },
        } satisfies GoogleGenerativeAIProviderOptions,
      };
    case "openai":
      return {
        openai: {} satisfies OpenAIResponsesProviderOptions,
      };
    case "openrouter":
      return {
        openrouter: {} satisfies OpenRouterModelOptions,
      };
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

export interface LLMConfig {
  provider: Provider;
  apiKey: string;
  model: LanguageModel;
  modelName?: string;
}

/**
 * Resolve LLM configuration from request params with env fallback.
 * Priority: request body/query → env vars → defaults.
 * Use `light: true` for batch/cheap operations (descriptions, learnings, etc.)
 */
export function resolveLLMConfig(opts?: {
  model?: string;
  apiKey?: string;
  provider?: string;
  light?: boolean;
}): LLMConfig {
  let modelName = opts?.model;
  let providerHint = opts?.provider;
  if (providerHint && !PROVIDERS.includes(providerHint as Provider)) {
    modelName = modelName || providerHint;
    providerHint = undefined;
  }

  const provider = modelName
    ? getProviderForModel(modelName)
    : (providerHint as Provider | undefined) ||
      (process.env.LLM_PROVIDER as Provider | undefined) ||
      getProviderForModel();

  console.log(
    `[resolveLLMConfig] provider=${provider} modelName=${modelName || "(default)"} light=${!!opts?.light}`,
  );

  const apiKey = opts?.apiKey || getApiKeyForProvider(provider);

  let effectiveModelName = modelName;
  if (!effectiveModelName && opts?.light) {
    effectiveModelName = getLightModelForProvider(provider);
  }

  const model = getModel(provider, {
    apiKey,
    modelName: effectiveModelName,
  });

  return { provider, apiKey, model, modelName: effectiveModelName };
}
