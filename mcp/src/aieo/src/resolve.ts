// Keyless model-name resolution, plus a one-call resolver that goes all the
// way to a LanguageModel. ADDITIVE over provider.ts: nothing there changes
// behavior, and everything here is built from its exported pieces.
//
// Name grammar (the one getProviderForModel / getModel already implement):
// split on "/" and strip the FIRST segment only when it names an aieo
// provider. Whatever follows is the provider's own id, which may itself
// contain slashes (OpenRouter ids are "org/model"). So:
//
//   sonnet                            anthropic   claude-sonnet-5       (alias)
//   claude-sonnet-5                   anthropic   claude-sonnet-5       (inferred)
//   anthropic/claude-sonnet-5         anthropic   claude-sonnet-5
//   openrouter/moonshotai/kimi-k2.6   openrouter  moonshotai/kimi-k2.6
//   openrouter/openrouter/auto        openrouter  openrouter/auto
//   openai/gpt-5                      openai      gpt-5                 (direct, NOT via OpenRouter)
//   openrouter/openai/gpt-5           openrouter  openai/gpt-5
//   moonshotai/kimi-k2.6              rejected — no provider prefix (see canonicalModelName)
//
// The canonical form is "<provider>/<modelId>": two segments for direct
// providers, three for OpenRouter. It round-trips through getModel, and it
// is unaffected by gateway config (Bifrost's wire prefixes are added inside
// getModel and never surface).

import type { LanguageModel } from "ai";
import {
  API_KEY_ENV,
  DEFAULT_MODELS,
  MODELS,
  PROVIDERS,
  getContextLimit,
  getModel,
  getProviderForModel,
  normalizeApiKey,
  type ModelName,
  type Provider,
} from "./provider.js";

export interface ModelOption {
  provider: Provider;
  alias: ModelName;
  modelId: string;
  /** The provider's default — what you get with no model name at all. */
  default: boolean;
}

/**
 * Every alias in the MODELS table, in PROVIDERS order. Pure and keyless,
 * which is what a UI picker or a server-config endpoint needs.
 */
export function listModels(): ModelOption[] {
  const out: ModelOption[] = [];
  for (const provider of PROVIDERS) {
    for (const [alias, modelId] of Object.entries(MODELS[provider])) {
      if (!modelId) continue;
      out.push({
        provider,
        alias: alias as ModelName,
        modelId,
        default: DEFAULT_MODELS[provider] === modelId,
      });
    }
  }
  return out;
}

export interface ParsedModelName {
  /** Set only when the name carries an explicit "<provider>/" prefix. */
  provider?: Provider;
  /** The name with that prefix stripped. Aliases are NOT resolved here.
   *  undefined for an empty name (or a bare "<provider>/"). */
  modelId?: string;
}

/** Pure syntax: strip an explicit provider prefix, nothing else. */
export function parseModelName(name?: string): ParsedModelName {
  if (!name) return {};
  const slash = name.indexOf("/");
  if (slash > 0) {
    const head = name.slice(0, slash);
    if (PROVIDERS.includes(head as Provider)) {
      return { provider: head as Provider, modelId: name.slice(slash + 1) || undefined };
    }
  }
  return { modelId: name };
}

export interface ModelRef {
  provider: Provider;
  /** Concrete id as the provider knows it (alias resolved, prefix stripped). */
  modelId: string;
  /** Canonical "<provider>/<modelId>" — unambiguous, itself a valid model name. */
  name: string;
}

const ALIASES = new Set<string>(Object.values(MODELS).flatMap((m) => Object.keys(m)));

function envProvider(): Provider | undefined {
  const v = process.env.LLM_PROVIDER;
  return v && PROVIDERS.includes(v as Provider) ? (v as Provider) : undefined;
}

/**
 * Resolve a model name to its provider + concrete id WITHOUT an API key.
 * Follows getModel's rules — explicit `provider` wins, then an explicit
 * prefix, then getProviderForModel's inference; aliases map through MODELS;
 * no name → the provider's default — so what this reports is what
 * resolveModel / getModel will build. (One deliberate extra: an alias after
 * a prefix, "anthropic/sonnet", resolves too.)
 *
 * Rejects the one input getModel silently gets wrong: a slashed name whose
 * first segment is NOT a provider ("moonshotai/kimi-k2.6") with nothing
 * else naming the provider — inference falls to the default and the request
 * 404s at the wrong API. Say so up front, with the fix. LLM_PROVIDER, when
 * set, counts as "something naming the provider", exactly as it does for
 * getProviderForModel.
 */
export function canonicalModelName(model?: string, provider?: Provider | string): ModelRef {
  const parsed = parseModelName(model);
  const chosen = provider ?? parsed.provider ?? getProviderForModel(model);
  if (!PROVIDERS.includes(chosen as Provider)) {
    throw new Error(`Unknown LLM provider: "${chosen}". Supported: ${PROVIDERS.join(", ")}`);
  }
  const p = chosen as Provider;

  if (model && !provider && !parsed.provider && model.includes("/") && !envProvider()) {
    const head = model.split("/")[0];
    throw new Error(
      `Model "${model}" has no provider prefix: "${head}" is not a provider (${PROVIDERS.join(", ")}). ` +
        `For an OpenRouter-hosted model write "openrouter/${model}"; otherwise pass the provider explicitly.`,
    );
  }

  let modelId = parsed.modelId;
  if (modelId && ALIASES.has(modelId)) {
    const hit = MODELS[p][modelId as ModelName];
    if (!hit) {
      throw new Error(
        `Model "${modelId}" is not available for provider "${p}". ` +
          `Available models: ${Object.keys(MODELS[p]).join(", ")}`,
      );
    }
    modelId = hit;
  }
  if (!modelId) modelId = DEFAULT_MODELS[p];
  return { provider: p, modelId, name: `${p}/${modelId}` };
}

/**
 * Per-call output-token cap. An explicit MAX_OUTPUT_TOKENS env always wins.
 * Otherwise the default is provider-aware: Anthropic runs at 128k because
 * @ai-sdk/anthropic clamps known models down to their true per-model max
 * instead of erroring, while OpenAI-compatible hosts (OpenRouter et al)
 * reject max_tokens above the model limit rather than clamping, so they keep
 * the conservative 64k default. Thinking tokens count against this same
 * budget, so headroom matters more than the visible text length suggests.
 */
export function maxOutputTokensFor(provider?: string): number {
  const env = Number(process.env.MAX_OUTPUT_TOKENS);
  if (env > 0) return env;
  return provider === "anthropic" ? 128_000 : 64_000;
}

export interface ResolveOptions {
  /** alias | id | "provider/id" | "openrouter/org/id". Omit for the provider's default. */
  model?: string;
  /** Explicit provider; otherwise inferred from `model`. */
  provider?: Provider | string;
  apiKey?: string;
  /**
   * Async key source consulted by env-var NAME ("OPENAI_API_KEY", …) before
   * process.env — a host's own secret store. Only called when `apiKey` is
   * absent.
   */
  getSecret?: (envName: string) => Promise<string | undefined>;
  baseUrl?: string;
  headers?: Record<string, string>;
  abortSignal?: AbortSignal;
  timeoutMs?: number;
}

export interface ResolvedModel extends ModelRef {
  model: LanguageModel;
  apiKey: string;
  contextLimit: number;
  maxOutputTokens: number;
}

/**
 * Name → everything a caller needs to run: provider, concrete id, canonical
 * name, the API key that was used, the LanguageModel, its context window,
 * and the output-token cap. Key precedence: `apiKey` → `getSecret(envName)`
 * → process.env; none → an error that names the env var. Logs nothing.
 */
export async function resolveModel(opts: ResolveOptions = {}): Promise<ResolvedModel> {
  const ref = canonicalModelName(opts.model, opts.provider);
  const envName = API_KEY_ENV[ref.provider];
  const apiKey =
    normalizeApiKey(opts.apiKey) ??
    normalizeApiKey(await opts.getSecret?.(envName)) ??
    normalizeApiKey(process.env[envName]);
  if (!apiKey) {
    throw new Error(
      `No API key for provider "${ref.provider}": set ${envName} (or pass apiKey / getSecret)`,
    );
  }
  // The canonical name carries exactly one provider prefix, which getModel
  // strips — so an OpenRouter id like "openrouter/auto" survives intact.
  const model = getModel(ref.provider, {
    modelName: ref.name,
    apiKey,
    baseUrl: opts.baseUrl,
    headers: opts.headers,
    abortSignal: opts.abortSignal,
    timeoutMs: opts.timeoutMs,
  });
  return {
    ...ref,
    model,
    apiKey,
    contextLimit: getContextLimit(ref.modelId, ref.provider),
    maxOutputTokens: maxOutputTokensFor(ref.provider),
  };
}
