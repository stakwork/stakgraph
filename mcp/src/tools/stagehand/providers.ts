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
    model: "claude-3-7-sonnet-latest",
    computer_use_model: "claude-sonnet-4-20250514",
    api_key_env_var_name: "ANTHROPIC_API_KEY",
  },
  openai: {
    name: "openai",
    model: "gpt-4o",
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

export function resolveBrowserModel(): { model: string; apiKey: string } {
  const model =
    process.env.STAGEHAND_MODEL || "anthropic/claude-opus-4-8";
  const apiKey = model.startsWith("openrouter/")
    ? process.env.OPENROUTER_API_KEY || ""
    : model.startsWith("openai/")
    ? process.env.OPENAI_API_KEY || ""
    : process.env.ANTHROPIC_API_KEY || "";
  return { model, apiKey };
}
