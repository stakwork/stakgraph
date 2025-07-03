export type ProviderName = "anthropic" | "openai";

export interface Provider {
  name: ProviderName;
  model: string;
  computer_use_model: string;
  api_key_env_var_name: string;
}

export const PROVIDER_MODELS: Record<ProviderName, Provider> = {
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

export function getProvider(arg?: "anthropic" | "openai"): Provider {
  let provider = PROVIDER_MODELS["openai"];
  if (arg === "anthropic" || process.env.LLM_PROVIDER === "anthropic") {
    provider = PROVIDER_MODELS["anthropic"];
  }
  return provider;
}
