export type Provider = "anthropic" | "openai";

export interface ProviderData {
  name: Provider;
  model: string;
  computer_use_model: string;
  api_key_env_var_name: string;
}

export const PROVIDER_MODELS: Record<Provider, ProviderData> = {
  anthropic: {
    name: "anthropic",
    // Frontier model, matched to aieo (MODELS.anthropic.sonnet = "claude-sonnet-5").
    // aieo-format ("provider/model") so it resolves through getModelDetails.
    model: "anthropic/claude-sonnet-5",
    // Computer-use / agent model stays a native Stagehand string (used only by
    // the stagehand_agent CUA path, which does not go through aieo).
    computer_use_model: "claude-sonnet-4-20250514",
    api_key_env_var_name: "ANTHROPIC_API_KEY",
  },
  openai: {
    name: "openai",
    // Frontier model, matched to aieo (MODELS.openai.gpt = "gpt-5").
    model: "openai/gpt-5",
    computer_use_model: "computer-use-preview",
    api_key_env_var_name: "OPENAI_API_KEY",
  },
};

export function getProvider(arg?: "anthropic" | "openai"): ProviderData {
  let provider = PROVIDER_MODELS["anthropic"];
  if (arg === "openai" || process.env.LLM_PROVIDER === "openai") {
    provider = PROVIDER_MODELS["openai"];
  }
  return provider;
}

function apiKeyForModel(model: string): string {
  if (model.startsWith("openrouter/")) return process.env.OPENROUTER_API_KEY || "";
  if (model.startsWith("openai/")) return process.env.OPENAI_API_KEY || "";
  return process.env.ANTHROPIC_API_KEY || "";
}

/**
 * Single source of truth for the browser's LLM. Returns an aieo-format model
 * string + its key, resolved once and always driven through getModelDetails —
 * no separate "Stagehand built-in vs aieo" branch. The default comes from the
 * configured provider; swap it in one place (or wire a registry) here.
 */
export function resolveBrowserModel(): { model: string; apiKey: string } {
  const model = getProvider().model;
  return { model, apiKey: apiKeyForModel(model) };
}
